#!/bin/bash
#SBATCH --job-name=hydra-eval
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err

# Évalue les 3 variantes HYDRA (local/nested/global) avec et sans conditioning
# sur 2 GPU P100 (32 GB VRAM combinés). DataParallel activé automatiquement.
#
# Usage:
#   sbatch scripts/run_eval.slurm                  # évalue models/ (ancienne approche)
#   sbatch scripts/run_eval.slurm models_hierarchy # évalue models_hierarchy/

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

MODEL_DIR="${1:-models}"

echo "=== Évaluation HYDRA | model_dir=${MODEL_DIR} | $(hostname) ==="
echo "GPUs disponibles : $(python3 -c 'import torch; print(torch.cuda.device_count())')"

python scripts/eval_hierarchy.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/default.yaml \
    --model-dir "$MODEL_DIR" \
    --eval-batch-size 2048
