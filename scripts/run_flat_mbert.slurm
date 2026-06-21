#!/bin/bash
#SBATCH --partition=3090,P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-mbert-%A_%a.out
#SBATCH --error=slurm-flat-mbert-%A_%a.err

# FlatHydra mBERT — 2 GPU, batch 96 INCHANGÉ (48/GPU via DataParallel :
# batch effectif identique → scheduler/reprise cohérents). Dataset v1
# complet. Auto-resume par checkpoint (~2× plus rapide qu'en 1 GPU).
#
# Usage :
#   sbatch --job-name=flat-mbert --array=0-9%1 scripts/run_flat_mbert.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

CONFIG="configs/baselines/flat_hydra_mbert.yaml"
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/flat_hydra_mbert'))" 2>/dev/null || echo "models/flat_hydra_mbert")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | FlatHydra mBERT | $(hostname) | $(date) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR | GPUs: $SLURM_GPUS_ON_NODE"
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
    --batch-size 96 \
    $RESUME_FLAG

echo "=== Terminé : $(date) ==="
