#!/bin/bash
#SBATCH --job-name=eval-hydra-local-mbert
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-eval-hydra-local-mbert-%j.out
#SBATCH --error=slurm-eval-hydra-local-mbert-%j.err

# Évalue HYDRA Local mBERT (best_model.pt) sur le test set.
# Produit : results/hydra_eval_test_metrics_mbert.json
#
# Prérequis : models/HYDRA_local_mBERT/best_model.pt doit exister.
#
# Usage :
#   sbatch scripts/run_eval_hydra_mbert.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

echo "=== Évaluation HYDRA Local mBERT | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

python scripts/eval_hierarchy.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/mbert_default.yaml \
    --variant local \
    --model-dir models \
    --eval-batch-size 512 \
    --output-json results/hydra_eval_test_metrics_mbert.json

echo "=== Terminé : $(date) ==="
