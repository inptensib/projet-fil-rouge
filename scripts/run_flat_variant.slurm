#!/bin/bash
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-%x-%A_%a.out
#SBATCH --error=slurm-flat-%x-%A_%a.err

# Job array séquentiel pour FlatHydra.
# --array=0-9%1 : 10 tentatives, 1 à la fois. Chaque tâche reprend
# le checkpoint de la précédente. Si le training finit avant la 10e,
# les tâches restantes skipperont (training_complete.txt).
#
# Usage:
#   sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_variant.slurm
#   sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_variant.slurm [config] [n_samples]
#
# Exemples :
#   Run complet (8M produits) :
#     sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_variant.slurm
#   Smoke test sur 5 000 samples stratifiés :
#     sbatch --job-name=flat-hydra-smoke --array=0-2%1 scripts/run_flat_variant.slurm 01_baselines/configs/flat_hydra.yaml 5000

PYTHON=/home/infres/tandriam-25/fil_rouge/venv/bin/python
source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/boutrouft-25/projects/IADATA-706/projet-fil-rouge

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

CONFIG="${1:-01_baselines/configs/flat_hydra.yaml}"
N_SAMPLES="${2:-}"  # optionnel — si vide, charge le dataset complet

SAVE_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/flat_hydra'))" 2>/dev/null || echo "models/flat_hydra")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | FlatHydra | $(hostname) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR | n_samples: ${N_SAMPLES:-all}"

if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé pour FlatHydra. Skip."
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

N_SAMPLES_FLAG=""
if [ -n "$N_SAMPLES" ]; then
    N_SAMPLES_FLAG="--n-samples $N_SAMPLES"
fi

$PYTHON scripts/run_flat_hydra.py \
    --split-dir /home/infres/boutrouft-25/projects/IADATA-706/dataset/ \
    --config "$CONFIG" \
    $RESUME_FLAG \
    $N_SAMPLES_FLAG
