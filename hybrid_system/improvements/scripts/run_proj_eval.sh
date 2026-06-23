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

cd ~/Rakuten_Project
PYTHON=$HOME/Rakuten_Project/.venv/bin/python
export PYTHONPATH="$HOME/Rakuten_Project/.venv/lib/python3.12/site-packages"
mkdir -p logs

echo "Starting projection-head tail eval at $(date)"
$PYTHON eval_knn_tail_whiten.py \
    --emb-dir embeddings_proj --emb-name emb_proj --emb full --configs none
echo "Done at $(date)"
