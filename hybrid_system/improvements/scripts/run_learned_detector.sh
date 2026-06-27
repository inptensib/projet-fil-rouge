#!/bin/bash
#SBATCH --job-name=learned_det
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

# Learned tail detector: NCM features (+ base-confidence features if val cache present),
# fit on VAL, applied on TEST. CPU only, memmap (low RAM). Actual ~30-60 min.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"
# (PYTHONPATH hard-coding removed for portability)
mkdir -p logs

echo "Starting learned detector at $(date)"
"${PYTHON:-python}" improvements/scripts/eval_learned_detector.py
echo "Done at $(date)"
