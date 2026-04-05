#!/bin/bash
#SBATCH --partition=cpu-high
#SBATCH --mem=512G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-baseline-%j.out
#SBATCH --error=slurm-baseline-%j.err

# Usage:
#   sbatch --job-name=tfidf_baseline scripts/run_baseline.slurm tfidf_baseline
#   sbatch --job-name=word2vec_baseline scripts/run_baseline.slurm word2vec_baseline

set -euo pipefail

NOTEBOOK=${1:?"Usage: $0 <notebook_name>  (ex: tfidf_baseline, word2vec_baseline)"}
NOTEBOOK_DIR="/home/infres/tandriam-25/fil_rouge/rakuten-classification/01_baselines/notebooks"
INPUT="${NOTEBOOK_DIR}/${NOTEBOOK}.ipynb"
OUTPUT="${NOTEBOOK_DIR}/${NOTEBOOK}_executed.ipynb"

if [ ! -f "$INPUT" ]; then
    echo "Erreur: notebook introuvable: $INPUT"
    exit 1
fi

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

echo "=== Exécution du notebook: $NOTEBOOK ==="
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo "  Node:   $(hostname)"
echo "  RAM:    $(free -h | awk '/Mem:/{print $2}')"
echo "  Start:  $(date)"
echo ""

jupyter nbconvert \
    --to notebook --execute \
    --allow-errors \
    --ExecutePreprocessor.timeout=129600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "${NOTEBOOK}_executed.ipynb" \
    "$INPUT"

echo ""
echo "=== Terminé: $(date) ==="
echo "  Output: $OUTPUT"
