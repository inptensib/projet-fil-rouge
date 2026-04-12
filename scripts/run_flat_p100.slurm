#!/bin/bash
#SBATCH --job-name=flat-hydra-p100
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-hydra-p100-%j.out
#SBATCH --error=slurm-flat-hydra-p100-%j.err

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

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
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 01_baselines/configs/flat_hydra.yaml \
    $RESUME_FLAG
