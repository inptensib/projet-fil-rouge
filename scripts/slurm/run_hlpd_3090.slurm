#!/bin/bash
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-hlpd-%A_%a.out
#SBATCH --error=slurm-hlpd-%A_%a.err

# HLPD training — 2× RTX 3090, bfloat16, intra-epoch checkpointing.
# 10 epochs sur le dataset complet (~5.57M samples).
#
# Usage :
#   # Smoke VRAM (1 tâche, pas de chaînage)
#   sbatch --job-name=hlpd-vram --array=0-0 scripts/run_hlpd_3090.slurm
#
#   # Run complet 10 epochs (chaîné, 5 tâches max × 36h)
#   sbatch --job-name=hlpd --array=0-4%1 scripts/run_hlpd_3090.slurm
#
# Auto-resume : chaque tâche reprend le checkpoint_latest.pt de la précédente.
# Si training_complete.txt existe, la tâche skipe immédiatement.

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

CONFIG="configs/hlpd/hlpd_3090.yaml"
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/v1/hlpd'))" 2>/dev/null || echo "models/v1/hlpd")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task ${SLURM_ARRAY_TASK_ID} | HLPD 3090 | $(hostname) | $(date) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR"
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

python scripts/run_hlpd.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "=== Terminé : $(date) ==="
