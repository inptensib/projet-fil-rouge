#!/bin/bash
# ==============================================================================
# Évaluation Dual-Expert — L40S (48 GB GDDR6, single GPU)
# ==============================================================================
#
# Mode 1 — Domain Expert seul (rapide, pas de LLM) :
#   sbatch scripts/run_eval_dual_expert_l40s.slurm
#
# Mode 2 — Dual Expert complet (LLM server requis) :
#   export PROVIDER=openai_compatible
#   export LLM_API_URL=http://<node>:8080/v1
#   export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/run_eval_dual_expert_l40s.slurm
#
# Surcharger les chemins sans éditer ce fichier :
#   export CHECKPOINT=/autre/chemin && sbatch scripts/run_eval_dual_expert_l40s.slurm
# ==============================================================================

#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/eval-dual-expert-l40s-%j.out
#SBATCH --error=logs/eval-dual-expert-l40s-%j.err

# ------------------------------------------------------------------------------
# Variables — modifier ici ou surcharger via export avant sbatch
# ------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projet-fil-rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/projet-fil-rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/projet-fil-rouge/dataset}"
CHECKPOINT="${CHECKPOINT:-models/dual_expert_l40s}"
TAXONOMY="${TAXONOMY:-$DATA_DIR/product_classification_taxonomy.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dual_expert_eval_l40s}"
SPLIT="${SPLIT:-test}"
TOP_K="${TOP_K:-10}"

# LLM reranker (provider=none → Domain Expert seul)
PROVIDER="${PROVIDER:-none}"
LLM_API_URL="${LLM_API_URL:-http://localhost:8080/v1}"
LLM_API_KEY="${LLM_API_KEY:-none}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
LLM_TIMEOUT="${LLM_TIMEOUT:-15.0}"

CATEGORY_DEFS="${CATEGORY_DEFS:-}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HF_CACHE"
export PYTHONPATH=""

mkdir -p logs "$OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Bannière
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " Dual-Expert Evaluation | L40S | $(hostname) | $(date)"
echo " Checkpoint : $CHECKPOINT"
echo " Taxonomy   : $TAXONOMY"
echo " Data       : $DATA_DIR ($SPLIT split)"
echo " Output     : $OUTPUT_DIR"
echo " Provider   : $PROVIDER"
if [ "$PROVIDER" != "none" ]; then
    echo " LLM URL    : $LLM_API_URL"
    echo " LLM Model  : $LLM_MODEL"
fi
echo " Top-K      : $TOP_K"
echo " GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Construction des arguments optionnels
# ------------------------------------------------------------------------------
DEFS_FLAG=""
if [ -n "$CATEGORY_DEFS" ] && [ -f "$CATEGORY_DEFS" ]; then
    DEFS_FLAG="--category-definitions $CATEGORY_DEFS"
    echo "[Eval] Category definitions : $CATEGORY_DEFS"
fi

# ------------------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------------------
python scripts/evaluate_llm_reranker.py \
    --checkpoint "$CHECKPOINT" \
    --split-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --top-k "$TOP_K" \
    --provider "$PROVIDER" \
    --api-base-url "$LLM_API_URL" \
    --api-key "$LLM_API_KEY" \
    --model "$LLM_MODEL" \
    --max-new-tokens "$LLM_MAX_TOKENS" \
    --timeout "$LLM_TIMEOUT" \
    $DEFS_FLAG

echo "======================================================================"
echo " Terminé : $(date)"
echo " Résultats : $OUTPUT_DIR/llm_routing_report.json"
echo "======================================================================"
