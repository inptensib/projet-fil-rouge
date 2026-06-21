#!/bin/bash
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-3090-%A_%a.out
#SBATCH --error=slurm-flat-3090-%A_%a.err

# Full training FlatHydra — dataset complet, 2× RTX 3090 (DataParallel).
# batch_size=512 (256 samples/GPU) — budget VRAM ~18 GB/24 GB sur RTX 3090.
#
# Usage :
#   sbatch --job-name=flat-3090 --array=0-9%1 scripts/run_flat_3090.slurm
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

CONFIG="configs/baselines/flat_hydra_3090.yaml"
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/flat_hydra'))" 2>/dev/null || echo "models/flat_hydra")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | FlatHydra 3090 | $(hostname) ==="
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
