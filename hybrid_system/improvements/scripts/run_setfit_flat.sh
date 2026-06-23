#!/bin/bash
#SBATCH --job-name=setfit_flat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Flat SetFit tail expert: train one contrastive model over the tail label space
# + extract/cache embeddings (the body) for train/val/test. One GPU job.

cd ~/Rakuten_Project
source .venv/bin/activate
mkdir -p logs embeddings_setfit

python -c "from setfit import SetFitModel; print('SetFit import OK')" \
  || { echo "SetFit import failed — install: pip install 'setfit==1.0.3' 'sentence-transformers==2.7.0' 'transformers==4.40.2' 'huggingface_hub==0.20.3'"; exit 1; }

nvidia-smi
echo "Starting flat SetFit train+extract at $(date)"
python train_setfit_flat.py
echo "Done at $(date)"
