#!/bin/bash
# Ré-évalue les modèles FastText (model_l1.bin + model_l2.bin) sur le test set.
# Les modèles sont chargés depuis le disque (pas de ré-entraînement).
# Lancement : nohup bash scripts/run_fasttext_eval.sh > logs_fasttext_eval.out 2>&1 &

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export N_JOBS=30

echo "=== FastText eval | $(hostname) | $(date) ==="

jupyter nbconvert \
    --to notebook \
    --execute \
    --output fasttext_baseline_executed.ipynb \
    --output-dir 01_baselines/notebooks/ \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    01_baselines/notebooks/fasttext_baseline.ipynb

echo "=== Terminé : $(date) ==="
