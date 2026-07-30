#!/bin/bash
#SBATCH --job-name=build_llama_env
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=build_llama_log.txt
#SBATCH --gres=none

set -euo pipefail
set -x

cd "$HOME"
export PIP_CACHE_DIR="$HOME/.cache/pip"; mkdir -p "$PIP_CACHE_DIR"

# Clean old artifacts
enroot remove --force llama_base || true
rm -f llama_base.sqsh llama_ready.sqsh

# Base image
enroot import -o llama_base.sqsh docker://nvcr.io/nvidia/pytorch:24.03-py3
enroot create --name llama_base llama_base.sqsh

# Install into a venv inside the image
enroot start --root llama_base << 'IN'
set -euo pipefail
python -m pip install --upgrade pip virtualenv
python -m virtualenv /opt/llamaenv
. /opt/llamaenv/bin/activate
pip install --upgrade \
  llama-stack vllm huggingface-hub accelerate \
  "transformers>=4.55.0" "tokenizers>=0.21.1" \
  "scikit-learn>=1.5.0" "pandas>=2.2.2" "pyarrow>=16.0.0"
deactivate
IN

# Export final image
enroot export --output "$HOME/llama_ready.sqsh" llama_base

# Cleanup
rm -f llama_base.sqsh
enroot remove --force llama_base || true