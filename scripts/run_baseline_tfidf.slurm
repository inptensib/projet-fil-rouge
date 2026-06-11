#!/bin/bash
#SBATCH --job-name=baseline-tfidf
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-baseline-tfidf-%j.out
#SBATCH --error=slurm-baseline-tfidf-%j.err

# Exécute le notebook TF-IDF baseline (chargement JSONL + vectorisation + clf_l2 + métriques)
# Les modèles déjà entraînés (tfidf_vectorizer, clf_l1) sont chargés depuis le disque.
# Seul clf_l2 est ré-entraîné.

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export N_JOBS=26

echo "=== TF-IDF baseline | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

jupyter nbconvert \
    --to notebook \
    --execute \
    --output tfidf_baseline_executed.ipynb \
    --output-dir 01_baselines/notebooks/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    01_baselines/notebooks/tfidf_baseline.ipynb

echo "=== Terminé : $(date) ==="
