#!/bin/bash
#SBATCH --job-name=proj_head
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Contrastive projection head (SupCon MLP) on FROZEN run2 embeddings.
# Trains on cached run2 vectors (no transformer fwd passes -> fast), projects
# train/val/test, saves to embeddings_proj/. Encoder frozen => cannot degrade base.

cd ~/Rakuten_Project
source .venv/bin/activate
mkdir -p logs embeddings_proj

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi
echo "Starting projection-head train at $(date)"
python train_proj_head.py
echo "Done at $(date)"
