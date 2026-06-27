#!/bin/bash
#SBATCH --job-name=xgb_v3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHON="${PYTHON:-python}"
echo "Using Python: $PYTHON"
$PYTHON --version

# (PYTHONPATH hard-coding removed for portability)

$PYTHON -c "import torch; print(f'torch OK, CUDA={torch.cuda.is_available()}')"

nvidia-smi
echo "Starting XGBoost v3 (concat emb + engineered features) at $(date)"

# Reuses the cached concat embeddings from v2 (embeddings/{split}_emb_concat.npz).
# Only the 149 XGBoost models are retrained on the enriched features.
$PYTHON improvements/scripts/train_xgboost_v3.py

echo "XGBoost v3 completed at $(date)"
