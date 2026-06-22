#!/bin/bash
# ==============================================================================
# Génération des définitions de catégories L2 via LLM (offline, une seule fois)
# ==============================================================================
# 6 126 appels LLM × ~5s = ~8h — le job peut être interrompu et repris (--resume)
#
# Mode 1 — Sampling aléatoire (défaut) :
#   sbatch scripts/run_generate_category_defs.slurm
#
# Mode 2 — Sampling par confiance (exemples difficiles, meilleure qualité) :
#   export CONFIDENCE_SAMPLING=1
#   export CHECKPOINT=models/dual_expert_h100
#   sbatch scripts/run_generate_category_defs.slurm
#
# Reprendre un run interrompu :
#   sbatch scripts/run_generate_category_defs.slurm  # --resume est toujours actif
#
# Surcharger les chemins sans éditer ce fichier :
#   export LLM_API_URL=http://<node>:8080/v1 && sbatch scripts/run_generate_category_defs.slurm
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

# LLM server
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

export TRANSFORMERS_OFFLINE=1
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
echo " LLM URL    : $LLM_API_URL"
echo " LLM Model  : $LLM_MODEL"
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

# ------------------------------------------------------------------------------
# Lancement (--resume toujours actif pour permettre la reprise)
# ------------------------------------------------------------------------------
python scripts/generate_category_definitions.py \
    --split-dir "$DATA_DIR" \
    --output "$OUTPUT" \
    --taxonomy-path "$TAXONOMY" \
    --provider openai_compatible \
    --api-base-url "$LLM_API_URL" \
    --api-key "$LLM_API_KEY" \
    --model "$LLM_MODEL" \
    --max-new-tokens "$LLM_MAX_TOKENS" \
    --timeout "$LLM_TIMEOUT" \
    --temperature "$LLM_TEMPERATURE" \
    --examples-per-category "$EXAMPLES_PER_CATEGORY" \
    --representative-sampling \
    --resume \
    $CONFIDENCE_FLAGS

echo "======================================================================"
echo " Terminé : $(date)"
echo " Définitions : $OUTPUT"
echo "======================================================================"
