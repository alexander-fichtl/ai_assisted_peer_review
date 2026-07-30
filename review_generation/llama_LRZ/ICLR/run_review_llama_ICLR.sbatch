#!/bin/bash
#SBATCH --job-name=review_llama_ICLR
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --output=review_llama_ICLR.%j.out
#SBATCH --error=review_llama_ICLR.%j.err

set -euo pipefail

#Absolute paths
PYFILE="$HOME/reviewer_llama_ICLR.py"
OUTPUT_DIR="$HOME/reviews_llama_ICLR"
IMG="$HOME/llama_ready.sqsh"

# Optional: use token from file instead of hardcoding in Python
if [[ -f "$HOME/.hf_token" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.hf_token")"
fi

mkdir -p "$OUTPUT_DIR"
#Mount HOME at the same path inside the container so absolute paths still work
srun --container-image="$IMG" \
     --container-mounts="$HOME:$HOME" \
     bash -lc '
set -euo pipefail

# ---- Prefer $TMPDIR or $SCRATCH and avoid /tmp ----
if [[ -n "${TMPDIR:-}" ]]; then
  SCR_ROOT="$TMPDIR"
elif [[ -n "${SCRATCH:-}" ]]; then
  SCR_ROOT="$SCRATCH"
else
  echo "WARNING: Neither TMPDIR nor SCRATCH set; falling back to /tmp" >&2
  SCR_ROOT="/tmp"
fi

JOB_TMP="$SCR_ROOT/${USER}-${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP/venv" "$JOB_TMP/hf_cache" "'"$OUTPUT_DIR"'"

# ---- Route all HF caches to scratch (not HOME); Disable fast downloader to avoid OOM ----
export HF_HOME="$JOB_TMP/hf_cache"
export HF_HUB_CACHE="$JOB_TMP/hf_cache"
unset HF_HUB_ENABLE_HF_TRANSFER
unset TRANSFORMERS_CACHE

# Optional: quiet bitsandbytes CUDA probing
export BNB_CUDA_VERSION=120

# ---- Venv bootstrap ----
rm -rf "$JOB_TMP/venv"
if python3 -c "import ensurepip" >/dev/null 2>&1; then
  python3 -m venv "$JOB_TMP/venv"
else
  python3 -m pip install --quiet --upgrade pip wheel
  python3 -m pip install --quiet virtualenv
  python3 -m virtualenv "$JOB_TMP/venv"
fi

# Prove venv exists, then activate
if [[ ! -f "$JOB_TMP/venv/bin/activate" ]]; then
  echo "[fatal] venv not created at $JOB_TMP/venv" >&2
  exit 90
fi
source "$JOB_TMP/venv/bin/activate"
python -m pip install --upgrade --quiet pip wheel

# ---- Minimal deps which the script imports ----
python -m pip install --quiet \
  "transformers==4.51.2" accelerate \
  bitsandbytes pillow numpy \
  "hf_xet>=0.4.0" auto-gptq \
  --no-cache-dir

# Install torchvision from the CUDA 12.1 index
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torchvision --no-cache-dir

cd "$HOME"
echo "[cwd] $(pwd)"

# ---- Execute the Python script (outputs go to $HOME/reviews) ----
python "'"$PYFILE"'"
'