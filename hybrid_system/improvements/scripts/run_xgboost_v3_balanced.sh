#!/bin/bash
#SBATCH --job-name=xgb_v3_bal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e
cd ~/Rakuten_Project
mkdir -p logs

PYTHON=$HOME/Rakuten_Project/.venv/bin/python
echo "Using Python: $PYTHON"
$PYTHON --version

export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"

$PYTHON -c "import torch; print(f'torch OK, CUDA={torch.cuda.is_available()}')"

nvidia-smi

# Rare-class booster (#4): same enriched features as v3 + inverse-frequency
# class weights. Outputs to checkpoints/l2_xgboost_v3_balanced/.
export BALANCE=1
export BALANCE_ALPHA="${BALANCE_ALPHA:-0.5}"
export BALANCE_CAP="${BALANCE_CAP:-10}"
echo "Starting XGBoost v3 BALANCED (alpha=$BALANCE_ALPHA cap=$BALANCE_CAP) at $(date)"

$PYTHON improvements/scripts/train_xgboost_v3.py

echo "XGBoost v3 balanced completed at $(date)"
