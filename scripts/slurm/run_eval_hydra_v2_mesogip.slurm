#!/bin/bash
# Éval des 3 HYDRA hierarchy v2 (nested/global/local) sur le test set v2 — mesogip H100.
# Sort F1 L1/L2, HCR strict ET HCR multi-parents (mesuré), modes base + hier.
#   sbatch --dependency=afterok:<jobid_local> scripts/run_eval_hydra_v2_mesogip.slurm
#
#SBATCH --job-name=eval-hydra-v2
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "$HOME/rakuten-classification"
source "$HOME/rakuten-venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "=== Node: $(hostname) | $(date) ==="
python scripts/eval_hierarchy.py \
    --split-dir dataset/ \
    --config configs/hydra/hierarchy.yaml \
    --model-dir models/v2 \
    --eval-batch-size 2048 \
    --output-mode hier \
    --output-json results/hydra_hierarchy_v2_metrics.json
