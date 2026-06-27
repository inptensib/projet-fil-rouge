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
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHON="${PYTHON:-python}"
echo "Using Python: $PYTHON"
$PYTHON --version

# (PYTHONPATH hard-coding removed for portability)

echo "Starting meta-learner training at $(date)"
$PYTHON improvements/scripts/train_meta_learner.py
echo "Meta-learner completed at $(date)"
