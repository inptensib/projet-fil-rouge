#!/bin/bash
# Ré-évalue les modèles FastText (model_l1.bin + model_l2.bin) sur le test set.
# Les modèles sont chargés depuis le disque (pas de ré-entraînement).
# Lancement : nohup bash scripts/run_fasttext_eval.sh > logs_fasttext_eval.out 2>&1 &

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export N_JOBS=30

echo "=== FastText eval | $(hostname) | $(date) ==="

jupyter nbconvert \
    --to notebook \
    --execute \
    --output fasttext_baseline_executed.ipynb \
    --output-dir notebooks/baselines/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/baselines/fasttext_baseline.ipynb

echo "=== Terminé : $(date) ==="
