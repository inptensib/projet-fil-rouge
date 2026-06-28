#!/bin/bash
# HYDRA nested + hierarchy.yaml sur dataset v2 — mesogip H100 (1 GPU).
#
#   Smoke (rapide) : sbatch --job-name=smoke-nested-v2 --export=ALL,N_SAMPLES=30000 scripts/run_hydra_nested_v2.slurm
#   Run complet    : sbatch scripts/run_hydra_nested_v2.slurm
#
#SBATCH --job-name=hydra-nested-v2
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "$HOME/rakuten-classification"
source "$HOME/rakuten-venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "=== Node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

NS_ARG=""
if [ -n "${N_SAMPLES:-}" ]; then
    NS_ARG="--n-samples ${N_SAMPLES}"
    echo "=== SMOKE TEST : ${N_SAMPLES} échantillons ==="
else
    echo "=== RUN COMPLET : dataset v2 entier ==="
fi

python scripts/run_train.py \
    --split-dir dataset/ \
    --config configs/hydra/hierarchy.yaml \
    --variant nested \
    ${NS_ARG}
