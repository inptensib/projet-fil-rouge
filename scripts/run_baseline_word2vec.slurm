#!/bin/bash
#SBATCH --job-name=baseline-w2v
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=26
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-baseline-w2v-%j.out
#SBATCH --error=slurm-baseline-w2v-%j.err

# Exécute le notebook Word2Vec baseline (chargement JSONL + vectorisation + clf_l2 + métriques)
# Les modèles déjà entraînés (word2vec_model, vectors.npy, clf_l1) sont chargés depuis le disque.
# Seul clf_l2 est ré-entraîné.
# RAM élevée : matrices denses 8M × 200 dims × 3 splits ≈ 20 GB + overhead Python.

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export N_JOBS=26

echo "=== Word2Vec baseline | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

jupyter nbconvert \
    --to notebook \
    --execute \
    --output word2vec_baseline_executed.ipynb \
    --output-dir 01_baselines/notebooks/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    01_baselines/notebooks/word2vec_baseline.ipynb

echo "=== Terminé : $(date) ==="
