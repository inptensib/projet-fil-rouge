#!/bin/bash
#SBATCH --job-name=knn_tail
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# kNN / class-prototype tail expert — CPU only, reuses cached concat embeddings.
# Fit on VAL, evaluate net effect on TEST. No GPU, no retraining.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"
# (PYTHONPATH hard-coding removed for portability)
mkdir -p logs

echo "Starting kNN tail expert at $(date)"
"${PYTHON:-python}" improvements/scripts/eval_knn_tail.py --emb l2 --tail-max 100
echo "Done at $(date)"
