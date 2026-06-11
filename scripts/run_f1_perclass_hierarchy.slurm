#!/bin/bash
#SBATCH --job-name=hydra-f1-hier
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0-8:00:00
#SBATCH --output=slurm-f1-hierarchy-%j.out
#SBATCH --error=slurm-f1-hierarchy-%j.err

# Génère f1_per_class_l1.json et f1_per_class_l2.json pour les 3 variantes
# HYDRA hierarchy (local/nested/global) avec masque hiérarchique actif.
# Charge models_hierarchy/HYDRA_{variant}_CamemBERT/best_model.pt,
# évalue sur le test set (1.2M samples), sauvegarde dans
# results/hydra-{variant}-hierarchy-camembert/.
#
# Usage :
#   sbatch scripts/run_f1_perclass_hierarchy.slurm             # toutes les variantes (séquentiel)
#   sbatch --job-name=f1-hier-local  scripts/run_f1_perclass_hierarchy.slurm local
#   sbatch --job-name=f1-hier-nested scripts/run_f1_perclass_hierarchy.slurm nested
#   sbatch --job-name=f1-hier-global scripts/run_f1_perclass_hierarchy.slurm global

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

VARIANT_ARG=""
if [ -n "$1" ]; then
    VARIANT_ARG="--variant $1"
fi

echo "=== F1 per-class HYDRA hierarchy ${1:-toutes variantes} | $(hostname) | $(date) ==="
echo "GPUs : $(python -c 'import torch; print(torch.cuda.device_count())')"

python scripts/generate_f1_per_class.py \
    --split-dir "$DATA_DIR/" \
    --config 02_hydra/configs/hierarchy.yaml \
    --model-dir models_hierarchy \
    --results-tag hierarchy \
    --check-hierarchy \
    --batch-size 512 \
    $VARIANT_ARG

echo "=== Terminé : $(date) ==="
