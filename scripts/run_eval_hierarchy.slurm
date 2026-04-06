#!/bin/bash
#SBATCH --job-name=eval-hierarchy
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-eval-hierarchy-%j.out
#SBATCH --error=slurm-eval-hierarchy-%j.err

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

echo "=== Évaluation HYDRA : conditioning L2 sur L1 prédit ==="
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo ""

python scripts/eval_hierarchy.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/default.yaml

echo ""
echo "=== Terminé: $(date) ==="
