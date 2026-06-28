#!/bin/bash
#SBATCH --job-name=hydra-eval
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err

# Évalue les 3 variantes HYDRA (local/nested/global) avec et sans conditioning
# sur 2 GPU P100 (32 GB VRAM combinés). DataParallel activé automatiquement.
#
# Usage:
#   sbatch scripts/run_eval.slurm                  # évalue models/ (ancienne approche)
#   sbatch scripts/run_eval.slurm models/v2 # évalue models/v2/

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_DIR="${1:-models}"

echo "=== Évaluation HYDRA | model_dir=${MODEL_DIR} | $(hostname) ==="
echo "GPUs disponibles : $(python3 -c 'import torch; print(torch.cuda.device_count())')"

python scripts/eval_hierarchy.py \
    --split-dir "$DATA_DIR/" \
    --config configs/hydra/default.yaml \
    --model-dir "$MODEL_DIR" \
    --eval-batch-size 2048
