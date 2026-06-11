#!/bin/bash
#SBATCH --job-name=hydra-f1-perclass
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-6:00:00
#SBATCH --output=slurm-f1-perclass-%j.out
#SBATCH --error=slurm-f1-perclass-%j.err

# Génère f1_per_class_l2.json pour les 3 variantes HYDRA (local/global/nested).
# Charge models/HYDRA_{variant}_CamemBERT/best_model.pt, prédit sur le test set
# (1,2M samples), sauvegarde dans results/hydra-{variant}-camembert/.
#
# Usage :
#   sbatch scripts/run_f1_perclass.slurm
#   sbatch scripts/run_f1_perclass.slurm --variant local   # une seule variante

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

echo "=== F1 per-class HYDRA | $(hostname) | $(date) ==="
echo "GPUs : $(python -c 'import torch; print(torch.cuda.device_count())')"

python scripts/generate_f1_per_class.py \
    --split-dir "$DATA_DIR/" \
    --config 02_hydra/configs/default.yaml \
    --batch-size 512

echo "=== Terminé : $(date) ==="
