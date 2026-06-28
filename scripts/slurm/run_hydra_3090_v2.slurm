#!/bin/bash
# HYDRA + hierarchy.yaml sur dataset v2 — cluster A, RTX 3090 (1 GPU, mono-GPU).
# Lancé EN PARALLÈLE du H100 (variante différente) pour la campagne v2.
#
#   Variante via 1er argument (defaut: global) :
#     sbatch --job-name=hydra-global-v2-3090 scripts/run_hydra_3090_v2.slurm global
#     sbatch --job-name=hydra-local-v2-3090  scripts/run_hydra_3090_v2.slurm local
#   Smoke : sbatch --export=ALL,N_SAMPLES=30000 scripts/run_hydra_3090_v2.slurm global
#
#SBATCH --job-name=hydra-v2-3090
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV="${VENV:-$HOME/fil_rouge/venv/bin/activate}"
# v2 canonique sur A (identique a HF/B, byte-pour-byte) — PAS ~/fil_rouge/dataset (ancien)
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset_v2/datasets_finaux}"
VARIANT="${1:-global}"

source "$VENV"
cd "$PROJECT_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "=== Node: $(hostname) | $(date) | variante=$VARIANT | data=$DATA_DIR ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

NS_ARG=""
[ -n "${N_SAMPLES:-}" ] && NS_ARG="--n-samples ${N_SAMPLES}"

python scripts/run_train.py \
    --split-dir "$DATA_DIR/" \
    --config configs/hydra/hierarchy.yaml \
    --variant "$VARIANT" \
    ${NS_ARG}
