#!/bin/bash
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-hydra-%x-%A_%a.out
#SBATCH --error=slurm-hydra-%x-%A_%a.err

# Job array séquentiel pour UNE variante HYDRA.
# --array=0-9%1 : 10 tentatives, 1 à la fois. Chaque tâche reprend
# le checkpoint de la précédente. Si le training finit avant la 10e,
# les tâches restantes skipperont (training_complete.txt).
#
# Usage:
#   sbatch --job-name=hydra-local  --array=0-9%1 scripts/run_train_variant.slurm local  [config]
#   sbatch --job-name=hydra-nested --array=0-9%1 scripts/run_train_variant.slurm nested [config]
#   sbatch --job-name=hydra-global --array=0-9%1 scripts/run_train_variant.slurm global [config]

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

VARIANT="${1:?Usage: sbatch --array=0-9%1 $0 <variant> [config]}"
CONFIG="${2:-02_hydra/configs/default.yaml}"

SAVE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('save_dir', 'models'))" 2>/dev/null || echo "models")
ENCODER_KEY=$(python3 -c "
import yaml
m={'camembert-base':'CamemBERT','xlm-roberta-base':'XLM-RoBERTa','bert-base-multilingual-cased':'mBERT'}
print(m.get(yaml.safe_load(open('$CONFIG')).get('encoder','camembert-base'),'CamemBERT'))
" 2>/dev/null || echo "CamemBERT")
MODEL_DIR="${SAVE_DIR}/HYDRA_${VARIANT}_${ENCODER_KEY}"
CKPT="${MODEL_DIR}/checkpoint_latest.pt"
COMPLETE="${MODEL_DIR}/training_complete.txt"

echo "=== Task $SLURM_ARRAY_TASK_ID/$(($SLURM_ARRAY_TASK_MAX)) | $VARIANT | $(hostname) ==="
echo "Config: $CONFIG | Save dir: $SAVE_DIR"

if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé pour $VARIANT. Skip."
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_train.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config "$CONFIG" \
    --variant "$VARIANT" \
    $RESUME_FLAG
