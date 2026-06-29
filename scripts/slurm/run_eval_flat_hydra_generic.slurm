#!/bin/bash
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-eval-%x-%j.out
#SBATCH --error=slurm-eval-%x-%j.err

# Éval TEST générique d'un modèle FlatHydra (best_model.pt) sur un dataset donné.
# Paramétrable via variables d'environnement (passées par --export) :
#   MODEL_PT      chemin du best_model.pt        (def: models/v2/flat_hydra/best_model.pt)
#   EVAL_CONFIG   yaml d'architecture du modèle  (def: configs/baselines/flat_hydra_camembert_v2.yaml)
#   DATA_DIR      dossier des splits *_split.jsonl
#   RESULTS_DIR   où écrire test_metrics.json
#   EVAL_BATCH    batch d'éval (no grad)         (def: 1536)
#
# Exemples (depuis la racine du repo, package installé via `pip install -e .`) :
#   # CamemBERT v2 (défauts) :
#   sbatch --job-name=eval-cam-v2 scripts/run_eval_flat_hydra_generic.slurm
#   # mBERT v1 (ancien dataset) :
#   sbatch --job-name=eval-mbert-v1 --export=ALL,\
#     MODEL_PT=models/v1/flat_hydra_mbert/best_model.pt,\
#     EVAL_CONFIG=configs/baselines/flat_hydra_mbert.yaml,\
#     DATA_DIR=$HOME/fil_rouge/dataset,\
#     RESULTS_DIR=results/flat_hydra_mbert \
#     scripts/run_eval_flat_hydra_generic.slurm

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"

# Defaults (CamemBERT v2) — surchargés par --export
export MODEL_PT="${MODEL_PT:-models/v2/flat_hydra/best_model.pt}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/baselines/flat_hydra_camembert_v2.yaml}"
export DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset_v2/datasets_finaux}"
export RESULTS_DIR="${RESULTS_DIR:-results/flat_hydra_v2}"
export EVAL_BATCH="${EVAL_BATCH:-1536}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== Éval FlatHydra | $(hostname) | $(date) ==="
echo "Model: $MODEL_PT | Config: $EVAL_CONFIG | Data: $DATA_DIR | Out: $RESULTS_DIR | Batch: $EVAL_BATCH"
echo "GPU : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python3 scripts/eval_flat_hydra.py \
    --split-dir "$DATA_DIR" \
    --model-pt "$MODEL_PT" \
    --config "$EVAL_CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --eval-batch-size "$EVAL_BATCH"

echo "=== Terminé : $(date) ==="
