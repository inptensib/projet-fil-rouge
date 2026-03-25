#!/bin/bash
#SBATCH --job-name=split-dataset
#SBATCH --partition=cpu-high
#SBATCH --nodelist=nodecpu02
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-split-%j.out
#SBATCH --error=slurm-split-%j.err

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

python split_dataset.py \
    --input /home/infres/tandriam-25/fil_rouge/dataset/train.jsonl \
    --output-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --test-size 0.15 \
    --val-size 0.15 \
    --seed 42
