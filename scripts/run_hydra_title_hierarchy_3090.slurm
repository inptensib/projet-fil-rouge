#!/bin/bash
#SBATCH --job-name=hydra-title-hier
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-hydra-title-hier-%x-%A_%a.out
#SBATCH --error=slurm-hydra-title-hier-%x-%A_%a.err

# HYDRA Title-only + Hierarchy — 10 tâches chaînées, resume auto.
# Partition 3090 uniquement (batch_size=640 → OOM sur P100 16GB).
#
# Usage (une soumission par variante) :
#   sbatch --job-name=hydra-title-hier-local  --array=0-9%1 scripts/run_hydra_title_hierarchy_3090.slurm local
#   sbatch --job-name=hydra-title-hier-global --array=0-9%1 scripts/run_hydra_title_hierarchy_3090.slurm global
#   sbatch --job-name=hydra-title-hier-nested --array=0-9%1 scripts/run_hydra_title_hierarchy_3090.slurm nested

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

VARIANT="${1:?Erreur : variante manquante. Usage: sbatch ... run_hydra_title_hierarchy_3090.slurm <local|global|nested>}"
CONFIG="02_hydra/configs/title_only_hierarchy.yaml"

SAVE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models_title_hierarchy'))" 2>/dev/null || echo "models_title_hierarchy")
SUFFIX=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('experiment_suffix', ''))" 2>/dev/null || echo "_title_hierarchy")
MODEL_DIR="${SAVE_DIR}/HYDRA_${VARIANT}_CamemBERT${SUFFIX}"
CKPT="${MODEL_DIR}/checkpoint_latest.pt"
COMPLETE="${MODEL_DIR}/training_complete.txt"

echo "=== Task ${SLURM_ARRAY_TASK_ID}/9 | variant=$VARIANT | $(hostname) | $(date) ==="
echo "Config: $CONFIG | Save dir: $MODEL_DIR"
echo "GPU : $(python3 -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

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

python scripts/run_train.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    --variant "$VARIANT" \
    $RESUME_FLAG

echo "=== Terminé : $(date) ==="
