#!/bin/bash
# ==============================================================================
# Génération des définitions de catégories L2 via LLM (offline, une seule fois)
# ==============================================================================
# Deux modes LLM :
#
#   Mode API  (défaut) — serveur LLM séparé requis :
#     export LLM_API_URL=http://<node>:8080/v1
#     sbatch scripts/run_generate_category_defs.slurm
#
#   Mode hf_local — modèle chargé en mémoire, aucun serveur requis :
#     export USE_HF_LOCAL=1
#     export LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
#     sbatch scripts/run_generate_category_defs.slurm
#
#   Mode hf_local + 4-bit (économise la VRAM) :
#     export USE_HF_LOCAL=1
#     export HF_LOAD_IN_4BIT=1
#     sbatch scripts/run_generate_category_defs.slurm
#
# Sampling par confiance (exemples difficiles, meilleure qualité) :
#   export CONFIDENCE_SAMPLING=1
#   export CHECKPOINT=models/dual_expert_h100
#   sbatch scripts/run_generate_category_defs.slurm
#
# Reprendre un run interrompu :
#   sbatch scripts/run_generate_category_defs.slurm  # --resume est toujours actif
# ==============================================================================

#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/gen-category-defs-%j.out
#SBATCH --error=logs/gen-category-defs-%j.err

# ------------------------------------------------------------------------------
# Variables — modifier ici ou surcharger via export avant sbatch
# ------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projet-fil-rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/projet-fil-rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/projet-fil-rouge/dataset}"
TAXONOMY="${TAXONOMY:-$DATA_DIR/product_classification_taxonomy.csv}"
OUTPUT="${OUTPUT:-$HOME/projet-fil-rouge/data/category_definitions/category_definitions_v1.jsonl}"

# LLM — mode API ou hf_local
USE_HF_LOCAL="${USE_HF_LOCAL:-0}"         # 1 = chargement en mémoire, 0 = API HTTP
HF_LOAD_IN_4BIT="${HF_LOAD_IN_4BIT:-0}"  # 1 = quantification 4-bit (hf_local uniquement)
LLM_API_URL="${LLM_API_URL:-http://localhost:8080/v1}"
LLM_API_KEY="${LLM_API_KEY:-none}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
LLM_TIMEOUT="${LLM_TIMEOUT:-30.0}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"

# Sampling
EXAMPLES_PER_CATEGORY="${EXAMPLES_PER_CATEGORY:-20}"
CONFIDENCE_SAMPLING="${CONFIDENCE_SAMPLING:-0}"   # 1 pour activer
CHECKPOINT="${CHECKPOINT:-models/dual_expert_h100}"
CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE:-64}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=0
export HF_HOME="$HF_CACHE"
export PYTHONPATH=""

mkdir -p logs "$(dirname $OUTPUT)"

# ------------------------------------------------------------------------------
# Bannière
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " Generate Category Definitions | $(hostname) | $(date)"
echo " Data       : $DATA_DIR"
echo " Taxonomy   : $TAXONOMY"
echo " Output     : $OUTPUT"
echo " LLM mode   : $([ "$USE_HF_LOCAL" = "1" ] && echo "hf_local (in-process)" || echo "API ($LLM_API_URL)")"
echo " LLM Model  : $LLM_MODEL"
echo " 4-bit      : $([ "$HF_LOAD_IN_4BIT" = "1" ] && echo "oui" || echo "non")"
echo " Sampling   : $([ "$CONFIDENCE_SAMPLING" = "1" ] && echo "confidence ($CHECKPOINT)" || echo "random")"
echo " GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Construction des arguments optionnels
# ------------------------------------------------------------------------------
CONFIDENCE_FLAGS=""
if [ "$CONFIDENCE_SAMPLING" = "1" ]; then
    CONFIDENCE_FLAGS="--confidence-sampling --checkpoint $CHECKPOINT --confidence-batch-size $CONFIDENCE_BATCH_SIZE"
fi

LLM_FLAGS=""
if [ "$USE_HF_LOCAL" = "1" ]; then
    LLM_FLAGS="--provider hf_local"
    [ "$HF_LOAD_IN_4BIT" = "1" ] && LLM_FLAGS="$LLM_FLAGS --hf-load-in-4bit"
else
    LLM_FLAGS="--provider openai_compatible --api-base-url $LLM_API_URL --api-key $LLM_API_KEY --timeout $LLM_TIMEOUT"
fi

# ------------------------------------------------------------------------------
# Lancement (--resume toujours actif pour permettre la reprise)
# ------------------------------------------------------------------------------
python scripts/generate_category_definitions.py \
    --split-dir "$DATA_DIR" \
    --output "$OUTPUT" \
    --taxonomy-path "$TAXONOMY" \
    --model "$LLM_MODEL" \
    --max-new-tokens "$LLM_MAX_TOKENS" \
    --temperature "$LLM_TEMPERATURE" \
    --examples-per-category "$EXAMPLES_PER_CATEGORY" \
    --representative-sampling \
    --resume \
    $LLM_FLAGS \
    $CONFIDENCE_FLAGS

echo "======================================================================"
echo " Terminé : $(date)"
echo " Définitions : $OUTPUT"
echo "======================================================================"
