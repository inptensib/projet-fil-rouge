#!/bin/bash
#SBATCH --job-name=hydra-array
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-hydra-%A_%a.out
#SBATCH --error=slurm-hydra-%A_%a.err

# Job array : chaque tâche entraîne une variante HYDRA différente.
# %A = job array ID, %a = task index (0, 1, 2)
#
# Usage:
#   sbatch scripts/run_train_array.slurm                          # défaut: default.yaml
#   sbatch scripts/run_train_array.slurm configs/hydra/hierarchy.yaml  # avec conditioning

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Mapping index → variante
VARIANTS=(local nested global)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CONFIG="${1:-configs/hydra/default.yaml}"

# Déterminer le save_dir depuis le config
SAVE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models'))" 2>/dev/null || echo "models")
MODEL_DIR="${SAVE_DIR}/HYDRA_${VARIANT}_CamemBERT"
CKPT="${MODEL_DIR}/checkpoint_latest.pt"
COMPLETE="${MODEL_DIR}/training_complete.txt"

echo "=== Array task $SLURM_ARRAY_TASK_ID: $VARIANT ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR | Node: $(hostname)"

# Si le training est déjà terminé, on skip
if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé pour $VARIANT :"
    cat "$COMPLETE"
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_train.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    --variant "$VARIANT" \
    $RESUME_FLAG
