#!/bin/bash
#SBATCH --job-name=f1-perclass-mbert
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=0-01:30:00
#SBATCH --output=slurm-f1-perclass-mbert-%j.out
#SBATCH --error=slurm-f1-perclass-mbert-%j.err

# Génère f1_per_class_l1.json + f1_per_class_l2.json pour HYDRA Local mBERT.
# Prérequis : models/v1/HYDRA_local_mBERT/best_model.pt doit exister.
#
# Usage :
#   sbatch scripts/run_f1_perclass_mbert.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

echo "=== F1 per-class HYDRA Local mBERT | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

python scripts/generate_f1_per_class.py \
    --split-dir "$DATA_DIR/" \
    --config configs/hydra/mbert_default.yaml \
    --variant local \
    --results-dir hydra-local-mbert \
    --batch-size 512

echo "=== Terminé : $(date) ==="
