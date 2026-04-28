#!/bin/bash
# Entraînement FastText L2 softmax sur lame10 (accès direct, 80 CPUs)
# Usage : bash scripts/run_fasttext_softmax.sh
# Ou via SSH : ssh lame10 "cd /home/infres/tandriam-25/fil_rouge/rakuten-classification && bash scripts/run_fasttext_softmax.sh"

set -e

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export N_JOBS=80

echo "=== FastText Softmax | $(hostname) | $(date) ==="
echo "CPUs: $N_JOBS"

jupyter nbconvert \
    --to notebook \
    --execute \
    --output fasttext_softmax_baseline_executed.ipynb \
    --output-dir 01_baselines/notebooks/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    01_baselines/notebooks/fasttext_softmax_baseline.ipynb

echo "=== Terminé : $(date) ==="
