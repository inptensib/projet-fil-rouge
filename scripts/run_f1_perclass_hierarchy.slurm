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

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

VARIANT_ARG=""
if [ -n "$1" ]; then
    VARIANT_ARG="--variant $1"
fi

echo "=== F1 per-class HYDRA hierarchy ${1:-toutes variantes} | $(hostname) | $(date) ==="
echo "GPUs : $(python -c 'import torch; print(torch.cuda.device_count())')"

python scripts/generate_f1_per_class.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/hierarchy.yaml \
    --model-dir models_hierarchy \
    --results-tag hierarchy \
    --check-hierarchy \
    --batch-size 512 \
    $VARIANT_ARG

echo "=== Terminé : $(date) ==="
