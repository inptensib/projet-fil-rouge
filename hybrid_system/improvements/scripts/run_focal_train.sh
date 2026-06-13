#!/bin/bash
#SBATCH --job-name=focal_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=logs/focal_%j.out
#SBATCH --error=logs/focal_%j.err

set -e
cd ~/Rakuten_Project
mkdir -p logs

PYTHON=$HOME/Rakuten_Project/.venv/bin/python
echo "Using Python: $PYTHON"
$PYTHON --version

export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"

$PYTHON -c "import torch; print(f'torch OK, CUDA={torch.cuda.is_available()}')"
nvidia-smi

echo "Starting CamemBERT L2 focal loss training at $(date)"
$PYTHON improvements/scripts/train_l2_focal.py --resume
echo "Training completed at $(date)"
