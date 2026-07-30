#!/bin/bash
#SBATCH --job-name=build_deepseek14_env
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=build14_log.txt

set -e
set -x

cd $HOME

# Clean up old containers
enroot remove --force torch_base_14 || true
rm -f torch_base_14.sqsh deepseek14_ready.sqsh

# Step 1: Get base image
enroot import -o torch_base_14.sqsh docker://nvcr.io/nvidia/pytorch:24.03-py3
enroot create --name torch_base_14 torch_base_14.sqsh

# Step 2: Install dependencies
enroot start --root torch_base_14 << 'EOF'
pip uninstall -y flash-attn transformers
pip install transformers --no-deps --no-cache-dir
pip install vllm[triton] huggingface-hub accelerate pymupdf pillow openai
EOF

# Step 3: Export patched container
enroot export --output $HOME/deepseek14_ready.sqsh torch_base_14

# Cleanup
rm -f torch_base_14.sqsh
enroot remove --force torch_base_14