#!/bin/bash
#SBATCH --job-name=gen_eval
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_eval_%j.out
#SBATCH --error=logs/gen_eval_%j.err

set -e
cd ~/Rakuten_Project
mkdir -p logs checkpoints/generative_models

PYTHON=$(which python 2>/dev/null || echo "$HOME/Rakuten_Project/.venv/bin/python")
echo "Using Python: $PYTHON"
$PYTHON --version

# Set PYTHONPATH to python3.12 only (NOT python3.10 which has incompatible binaries)
export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"

# Verify key imports
$PYTHON -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
$PYTHON -c "import datasets; print('datasets OK')"

echo "Starting generative fallback evaluation..."
$PYTHON generative_fallback.py

echo "Job complete."
