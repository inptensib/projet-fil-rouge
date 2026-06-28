#!/bin/bash
#SBATCH --job-name=flat-hydra-p100
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-hydra-p100-%j.out
#SBATCH --error=slurm-flat-hydra-p100-%j.err

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CKPT="models/flat_hydra/checkpoint_latest.pt"
COMPLETE="models/flat_hydra/training_complete.txt"

if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé :"
    cat "$COMPLETE"
    echo "Rien à faire, job terminé."
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_flat_hydra.py \
    --split-dir "$DATA_DIR/" \
    --config configs/baselines/flat_hydra.yaml \
    $RESUME_FLAG
