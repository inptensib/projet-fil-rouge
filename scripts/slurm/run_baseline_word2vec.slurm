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

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export N_JOBS=26

echo "=== Word2Vec baseline | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

jupyter nbconvert \
    --to notebook \
    --execute \
    --output word2vec_baseline_executed.ipynb \
    --output-dir notebooks/baselines/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/baselines/word2vec_baseline.ipynb

echo "=== Terminé : $(date) ==="
