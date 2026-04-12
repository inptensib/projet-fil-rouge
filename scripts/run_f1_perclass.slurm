#!/bin/bash
#SBATCH --job-name=hydra-f1-perclass
#SBATCH --partition=P100,3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-6:00:00
#SBATCH --output=slurm-f1-perclass-%j.out
#SBATCH --error=slurm-f1-perclass-%j.err

# Génère f1_per_class_l2.json pour les 3 variantes HYDRA (local/global/nested).
# Charge models/HYDRA_{variant}_CamemBERT/best_model.pt, prédit sur le test set
# (1,2M samples), sauvegarde dans results/hydra-{variant}-camembert/.
#
# Usage :
#   sbatch scripts/run_f1_perclass.slurm
#   sbatch scripts/run_f1_perclass.slurm --variant local   # une seule variante

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface
export PYTHONPATH=""

echo "=== F1 per-class HYDRA | $(hostname) | $(date) ==="
echo "GPUs : $(python -c 'import torch; print(torch.cuda.device_count())')"

python scripts/generate_f1_per_class.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 02_hydra/configs/default.yaml \
    --batch-size 512

echo "=== Terminé : $(date) ==="
