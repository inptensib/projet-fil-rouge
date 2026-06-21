#!/bin/bash
#SBATCH --job-name=fasttext-eval
#SBATCH --partition=CPU
#SBATCH --nodelist=lame26
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm-fasttext-eval-%j.out
#SBATCH --error=slurm-fasttext-eval-%j.err

# Ré-évalue les modèles FastText (model_l1.bin + model_l2.bin) sur le test set.
# Les modèles sont chargés depuis le disque (pas de ré-entraînement).
# Corrige le bug fasttext_predict_batch (NFC Unicode normalization).
# Pas de cache joblib : chargement direct des 8M JSONL (~30 min).

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export N_JOBS=26

echo "=== FastText eval | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

jupyter nbconvert \
    --to notebook \
    --execute \
    --output fasttext_baseline_executed.ipynb \
    --output-dir notebooks/baselines/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/baselines/fasttext_baseline.ipynb

echo "=== Terminé : $(date) ==="
