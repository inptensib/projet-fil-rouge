#!/bin/bash
# HYDRA <variant> + hierarchy.yaml sur dataset v2 — mesogip H100 (1 GPU).
# Variante via 1er argument (defaut: nested).
#   sbatch --job-name=hydra-global-v2 scripts/run_hydra_mesogip_v2.slurm global
#   sbatch --job-name=hydra-local-v2  scripts/run_hydra_mesogip_v2.slurm local
#
#SBATCH --job-name=hydra-v2
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "$HOME/rakuten-classification"
source "$HOME/rakuten-venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

VARIANT="${1:-nested}"
echo "=== Node: $(hostname) | $(date) | variant=$VARIANT ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

NS_ARG=""
[ -n "${N_SAMPLES:-}" ] && NS_ARG="--n-samples ${N_SAMPLES}"

python scripts/run_train.py \
    --split-dir dataset/ \
    --config configs/hydra/hierarchy.yaml \
    --variant "$VARIANT" \
    ${NS_ARG}
