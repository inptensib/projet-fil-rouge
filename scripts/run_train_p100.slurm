#!/bin/bash
#SBATCH --job-name=hydra-p100
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-hydra-p100-%j.out
#SBATCH --error=slurm-hydra-p100-%j.err

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

# Auto-resume : détecte un checkpoint existant
VARIANT="${HYDRA_VARIANT:-${1:-local}}"
MODEL_DIR="models/HYDRA_${VARIANT}_CamemBERT"
CKPT="${MODEL_DIR}/checkpoint_latest.pt"
COMPLETE="${MODEL_DIR}/training_complete.txt"

# Si le training est déjà terminé, on skip
if [ -f "$COMPLETE" ]; then
    echo "Training déjà terminé pour $VARIANT :"
    cat "$COMPLETE"
    echo "Rien à faire, job terminé."
    exit 0
fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_train.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/default.yaml \
    --variant "$VARIANT" \
    $RESUME_FLAG
