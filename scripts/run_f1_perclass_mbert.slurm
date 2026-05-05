#!/bin/bash
#SBATCH --job-name=f1-perclass-mbert
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=0-01:30:00
#SBATCH --output=slurm-f1-perclass-mbert-%j.out
#SBATCH --error=slurm-f1-perclass-mbert-%j.err

# Génère f1_per_class_l1.json + f1_per_class_l2.json pour HYDRA Local mBERT.
# Prérequis : models/HYDRA_local_mBERT/best_model.pt doit exister.
#
# Usage :
#   sbatch scripts/run_f1_perclass_mbert.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

echo "=== F1 per-class HYDRA Local mBERT | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]' 2>/dev/null)"

python scripts/generate_f1_per_class.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/mbert_default.yaml \
    --variant local \
    --results-dir hydra-local-mbert \
    --batch-size 512

echo "=== Terminé : $(date) ==="
