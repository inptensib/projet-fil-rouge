#!/bin/bash
# Entraînement HYDRA single-GPU sur le cluster mesogip (ENSTA) — H100 / L40S.
#
# Lancement (depuis ~/rakuten-classification sur mesogip) :
#   sbatch scripts/run_train_mesogip.slurm
# Surcharges possibles via --export :
#   sbatch --export=ALL,CONFIG=configs/hydra/hierarchy.yaml,VARIANT=nested scripts/run_train_mesogip.slurm
#   sbatch --export=ALL,PART=ENSTA-l40s scripts/run_train_mesogip.slurm
#
#SBATCH --job-name=rakuten-train
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# --- Paramètres (surcharge via --export=ALL,CLE=valeur) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rakuten-classification}"
VENV="${VENV:-$HOME/rakuten-venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/dataset}"
CONFIG="${CONFIG:-configs/hydra/default.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"          # ex: "--variant nested --n-samples 50000"

source "$VENV"
cd "$PROJECT_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "=== Node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "=== Config: $CONFIG | Extra: $EXTRA_ARGS ==="

python scripts/run_train.py \
    --split-dir "$DATA_DIR/" \
    --config "$CONFIG" \
    $EXTRA_ARGS
