#!/bin/bash
#SBATCH --job-name=setfit_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

cd ~/Rakuten_Project
source .venv/bin/activate
mkdir -p logs

# Install setfit with pinned versions compatible with PyTorch 2.2.2 + transformers 4.40.2
# (run manually if needed: pip install "setfit==1.0.3" "sentence-transformers==2.7.0" "transformers==4.40.2" "huggingface_hub==0.20.3")
python -c "from setfit import SetFitModel; print('SetFit import OK')" || { echo "SetFit import failed. Install manually."; exit 1; }

nvidia-smi
echo "Starting SetFit training at $(date)"

python train_setfit.py

echo "SetFit training completed at $(date)"
