#!/bin/bash
#SBATCH --job-name=hydra-p100
#SBATCH --partition=P100
#SBATCH --nodelist=node46
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
VARIANT="${1:-local}"
CKPT="models/HYDRA_${VARIANT}_CamemBERT/checkpoint_latest.pt"
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
