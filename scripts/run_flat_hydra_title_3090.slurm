#!/bin/bash
#SBATCH --job-name=flat-title
#SBATCH --partition=3090,P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-title-%A_%a.out
#SBATCH --error=slurm-flat-title-%A_%a.err

# FlatHydra Title-only — 10 tâches chaînées (%1 = sérielles), resume auto.
# Partition 3090 par défaut, fallback P100.
#
# Usage :
#   sbatch --job-name=flat-title --array=0-9%1 scripts/run_flat_hydra_title_3090.slurm
#
# Chaque tâche dure au plus 36h, reprend checkpoint_latest.pt si présent,
# et skipe immédiatement si training_complete.txt existe.

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

CONFIG="01_baselines/configs/flat_hydra_title_3090.yaml"
SAVE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models/flat_hydra_title'))" 2>/dev/null || echo "models/flat_hydra_title")
CKPT="${SAVE_DIR}/checkpoint_latest.pt"
COMPLETE="${SAVE_DIR}/training_complete.txt"

echo "=== Task ${SLURM_ARRAY_TASK_ID}/9 | FlatHydra Title-only | $(hostname) | $(date) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR"
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

python scripts/run_flat_hydra.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "=== Terminé : $(date) ==="
