#!/bin/bash
#SBATCH --job-name=proj_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

# Evaluate the SupCon PROJECTION-HEAD embeddings with the NCM + margin tail voter.
# Question: does a learned non-linear metric on frozen run2 beat raw run2
# (recall@1 0.704 / +1.0pp f1m)?  CPU only, memmap (low RAM). Actual ~30-60 min.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"
# (PYTHONPATH hard-coding removed for portability)
mkdir -p logs

echo "Starting projection-head tail eval at $(date)"
"${PYTHON:-python}" improvements/scripts/eval_knn_tail_whiten.py \
    --emb-dir embeddings_proj --emb-name emb_proj --emb full --configs none
echo "Done at $(date)"
