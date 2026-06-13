#!/bin/bash
#SBATCH --job-name=meta_learn
#SBATCH --partition=3090
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/meta_%j.out
#SBATCH --error=logs/meta_%j.err

set -e
cd ~/Rakuten_Project
mkdir -p logs

PYTHON=$HOME/Rakuten_Project/.venv/bin/python
echo "Using Python: $PYTHON"
$PYTHON --version

export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"

echo "Starting meta-learner training at $(date)"
$PYTHON improvements/scripts/train_meta_learner.py
echo "Meta-learner completed at $(date)"
