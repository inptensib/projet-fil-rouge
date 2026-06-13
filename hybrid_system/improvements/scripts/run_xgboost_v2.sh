#!/bin/bash
#SBATCH --job-name=xgb_v2
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=logs/xgb_v2_%j.out
#SBATCH --error=logs/xgb_v2_%j.err

set -e
cd ~/Rakuten_Project
mkdir -p logs

PYTHON=$HOME/Rakuten_Project/.venv/bin/python
echo "Using Python: $PYTHON"
$PYTHON --version

export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"

$PYTHON -c "import torch; print(f'torch OK, CUDA={torch.cuda.is_available()}')"
nvidia-smi

# Use concatenated L1+L2 embeddings (1536d)
export EMB_MODE=concat

echo "Starting XGBoost v2 training (EMB_MODE=$EMB_MODE) at $(date)"
$PYTHON improvements/scripts/train_xgboost_v2.py
echo "Training completed at $(date)"
