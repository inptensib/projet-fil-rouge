#!/bin/bash
#SBATCH --job-name=f1-perclass-missing
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-f1-perclass-missing-%j.out
#SBATCH --error=slurm-f1-perclass-missing-%j.err

# Génère f1_per_class_l1.json + f1_per_class_l2.json pour les 2 variantes restantes :
#   - HYDRA Nested CamemBERT Title-only + Hierarchy
#   - HYDRA Global CamemBERT Title-only + Hierarchy
# (local déjà généré par le job 814493)
#
# Usage :
#   sbatch scripts/run_f1_perclass_missing.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

echo "=== F1 per-class (nested + global hier title) | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

SPLIT_DIR=/home/infres/tandriam-25/fil_rouge/dataset/

echo ""
echo "=== HYDRA Title-only + Hierarchy (models_title_hierarchy/) ==="
for VARIANT in nested global; do
    echo "  -> variante $VARIANT"
    python scripts/generate_f1_per_class.py \
        --split-dir "$SPLIT_DIR" \
        --config 02_hydra/configs/title_only_hierarchy.yaml \
        --model-dir models_title_hierarchy \
        --model-name-suffix _title_hierarchy \
        --variant "$VARIANT" \
        --results-dir "hydra-${VARIANT}-title-hier-camembert" \
        --no-description \
        --check-hierarchy \
        --batch-size 512
done

echo ""
echo "=== Termine : $(date) ==="
