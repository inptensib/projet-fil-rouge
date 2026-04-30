#!/bin/bash
#SBATCH --job-name=f1-perclass-missing
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-8:00:00
#SBATCH --output=slurm-f1-perclass-missing-%j.out
#SBATCH --error=slurm-f1-perclass-missing-%j.err

# Génère f1_per_class_l1.json + f1_per_class_l2.json pour les modèles manquants :
#   - FlatHydra Title+Description
#   - FlatHydra Title-only
#   - HYDRA {local,nested,global} CamemBERT Title-only std
#   - HYDRA {local,nested,global} CamemBERT Title-only + Hierarchy
#
# Usage :
#   sbatch scripts/run_f1_perclass_missing.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

echo "=== F1 per-class (modèles manquants) | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

SPLIT_DIR=/home/infres/tandriam-25/fil_rouge/dataset/

# --- FlatHydra Title+Description ---
# Dossier modèle : models/flat_hydra/best_model.pt
# Dossier résultats : results/flat_hydra/
echo ""
echo "=== 1/4 : FlatHydra Title+Description ==="
python scripts/generate_f1_per_class.py \
    --split-dir "$SPLIT_DIR" \
    --flat-hydra \
    --batch-size 512

# --- FlatHydra Title-only ---
# Dossier modèle : models/flat_hydra_title/best_model.pt
# Dossier résultats : results/flat_hydra_title/
echo ""
echo "=== 2/4 : FlatHydra Title-only ==="
python scripts/generate_f1_per_class.py \
    --split-dir "$SPLIT_DIR" \
    --flat-hydra \
    --no-description \
    --batch-size 512

# --- HYDRA Title-only std ---
# Dossiers modèle : models_title/HYDRA_{variant}_CamemBERT_title/best_model.pt
# Dossiers résultats : results/hydra-{variant}-title-camembert/
echo ""
echo "=== 1/2 : HYDRA Title-only std (models_title/) ==="
python scripts/generate_f1_per_class.py \
    --split-dir "$SPLIT_DIR" \
    --config 02_hydra/configs/title_only.yaml \
    --model-dir models_title \
    --model-name-suffix _title \
    --results-tag title \
    --no-description \
    --batch-size 512

# --- HYDRA Title-only + Hierarchy ---
# Dossiers modèle : models_title_hierarchy/HYDRA_{variant}_CamemBERT_title_hierarchy/best_model.pt
# Dossiers résultats : results/hydra-{variant}-title-hier-camembert/
echo ""
echo "=== 2/2 : HYDRA Title-only + Hierarchy (models_title_hierarchy/) ==="
for VARIANT in local nested global; do
    echo "  → variante $VARIANT"
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
echo "=== Terminé : $(date) ==="
