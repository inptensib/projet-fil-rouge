#!/bin/bash
#SBATCH --job-name=knn_setfit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

# Evaluate the FLAT SetFit tail expert: NCM + margin tail voter on the SetFit
# contrastive embeddings (768d), vs the run2 baseline (recall@1 0.704 / +1.0pp f1m).
# CPU only, memmap (low RAM). Large time limit just for safety — actual ~30-60 min.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"
# (PYTHONPATH hard-coding removed for portability)
mkdir -p logs

echo "Starting SetFit tail eval at $(date)"
"${PYTHON:-python}" improvements/scripts/eval_knn_tail_whiten.py \
    --emb-dir embeddings_setfit --emb-name emb_setfit --emb full --configs none
echo "Done at $(date)"
