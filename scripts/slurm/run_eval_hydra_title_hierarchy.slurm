#!/bin/bash
#SBATCH --job-name=eval-hydra-title-hier
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-eval-hydra-title-hier-%j.out
#SBATCH --error=slurm-eval-hydra-title-hier-%j.err

# Évalue les 3 variantes HYDRA Hierarchy Title-only (best_model.pt) sur le test set.
# Les variantes dont best_model.pt est absent (training en cours) sont skippées.
# Produit : results/hydra_eval_title_hierarchy_metrics.json
#
# Usage :
#   sbatch scripts/run_eval_hydra_title_hierarchy.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

echo "=== Évaluation HYDRA Hierarchy Title-only | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

python scripts/eval_hierarchy.py \
    --split-dir "$DATA_DIR/" \
    --config configs/hydra/title_only_hierarchy.yaml \
    --model-dir models_title_hierarchy \
    --model-suffix _title_hierarchy \
    --no-description \
    --eval-batch-size 1024 \
    --output-json results/hydra_eval_title_hierarchy_metrics.json

echo "=== Terminé : $(date) ==="
