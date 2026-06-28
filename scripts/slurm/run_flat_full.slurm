#!/bin/bash
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-full-%A_%a.out
#SBATCH --error=slurm-flat-full-%A_%a.err

# Full training FlatHydra — dataset complet, 2 GPUs (DataParallel).
# batch_size=256 (128 samples/GPU) — budget VRAM ~9 GB/16 GB sur P100.
#
# Usage :
#   sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_full.slurm
#
# Auto-resume : chaque tâche reprend le checkpoint de la précédente.
# Si le training finit avant la 10e tâche, les suivantes skippent.

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

CONFIG="configs/baselines/flat_hydra_full.yaml"
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/v1/flat_hydra'))" 2>/dev/null || echo "models/v1/flat_hydra")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | FlatHydra full | $(hostname) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR | GPUs: $SLURM_GPUS_ON_NODE"

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
