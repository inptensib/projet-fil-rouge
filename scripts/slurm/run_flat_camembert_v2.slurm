#!/bin/bash
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-camembert-v2-%A_%a.out
#SBATCH --error=slurm-flat-camembert-v2-%A_%a.err

# FlatHydra CamemBERT sur le dataset v2 (fuite purgée), méthode naïve.
# 2 GPU P100 — batch 256 (128/GPU) = identique au CamemBERT v1 (71,7 %)
# → comparaison v1↔v2 propre. Auto-resume par checkpoint.
#
# Usage :
#   sbatch --job-name=flat-cam-v2 --array=0-9%1 scripts/run_flat_camembert_v2.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
# Dataset v2 (fuite purgée) — distinct de la v1 utilisée par les autres scripts
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset_v2/datasets_finaux}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

CONFIG="configs/baselines/flat_hydra_camembert_v2.yaml"
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models_v2/flat_hydra'))" 2>/dev/null || echo "models_v2/flat_hydra")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | FlatHydra CamemBERT v2 | $(hostname) | $(date) ==="
echo "Config: $CONFIG | Data: $DATA_DIR | Save dir: $SAVE_DIR | GPUs: $SLURM_GPUS_ON_NODE"
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé. Skip."
    cat "$COMPLETE"
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_flat_hydra.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "=== Terminé : $(date) ==="
