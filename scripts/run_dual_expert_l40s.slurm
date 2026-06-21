#!/bin/bash
# ==============================================================================
# FlatHydra Dual-Expert — L40S (48 GB GDDR6, single GPU)
# ==============================================================================
# batch_size=128 — ~26 GB/48 GB (~54% VRAM)
# Temps estimé  : ~4–5 h (early stopping epoch 4–6)
#
# Usage :
#   sbatch --job-name=dual-expert-l40s --array=0-4%1 scripts/run_dual_expert_l40s.slurm
#
# Auto-resume : chaque tâche reprend le checkpoint de la précédente.
#
# Surcharger les chemins sans éditer ce fichier :
#   export PROJECT_ROOT=/mon/chemin && sbatch scripts/run_dual_expert_l40s.slurm
# ==============================================================================

#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dual-expert-l40s-%A_%a.out
#SBATCH --error=logs/dual-expert-l40s-%A_%a.err

# ------------------------------------------------------------------------------
# Variables — modifier ici ou surcharger via export avant sbatch
# ------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projet-fil-rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/projet-fil-rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/projet-fil-rouge/dataset}"
CONFIG="${CONFIG:-01_baselines/configs/flat_hydra_dual_expert_l40s.yaml}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HF_CACHE"
export PYTHONPATH=""

mkdir -p logs

# ------------------------------------------------------------------------------
# Déduire save_dir depuis la config
# ------------------------------------------------------------------------------
SAVE_DIR=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('save_dir', 'models/dual_expert_l40s'))
" 2>/dev/null || echo "models/dual_expert_l40s")

CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

# ------------------------------------------------------------------------------
# Bannière
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " FlatHydra Dual-Expert | L40S | Task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_MAX:-0}"
echo " Host     : $(hostname)"
echo " Date     : $(date)"
echo " Config   : $CONFIG"
echo " Save dir : $SAVE_DIR"
echo " GPUs     : $SLURM_GPUS_ON_NODE"
python -c "import torch; [print(f' GPU {i}    : {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null
echo "======================================================================"

# ------------------------------------------------------------------------------
# Skip si déjà terminé
# ------------------------------------------------------------------------------
if [ -f "$COMPLETE" ]; then
    echo "[Skip] Training déjà terminé — $(cat $COMPLETE)"
    exit 0
fi

# ------------------------------------------------------------------------------
# Resume si checkpoint disponible
# ------------------------------------------------------------------------------
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "[Resume] Checkpoint trouvé : $CKPT"
    RESUME_FLAG="--resume $CKPT"
fi

# ------------------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------------------
python scripts/run_flat_hydra.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "======================================================================"
echo " Terminé : $(date)"
echo "======================================================================"
