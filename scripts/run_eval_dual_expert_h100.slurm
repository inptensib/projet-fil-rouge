#!/bin/bash
# ==============================================================================
# Évaluation Dual-Expert — H100 (80 GB HBM3, single GPU)
# ==============================================================================
#
# Mode 1 — Domain Expert seul (rapide, pas de LLM) :
#   sbatch scripts/run_eval_dual_expert_h100.slurm
#
# Mode 2 — Dual Expert, échantillon de 10K produits :
#   export PROVIDER=hf_local
#   export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#   export N_SAMPLES=10000
#   sbatch scripts/run_eval_dual_expert_h100.slurm
#
# Mode 3 — Dual Expert, seulement les incertains (confiance < seuil) :
#   export PROVIDER=hf_local
#   export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#   export CONFIDENCE_THRESHOLD=0.70
#   sbatch scripts/run_eval_dual_expert_h100.slurm
#
# Mode 4 — Dual Expert complet via API :
#   export PROVIDER=openai_compatible
#   export LLM_API_URL=http://<node>:8080/v1
#   export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/run_eval_dual_expert_h100.slurm
#
# Surcharger les chemins sans éditer ce fichier :
#   export CHECKPOINT=/autre/chemin && sbatch scripts/run_eval_dual_expert_h100.slurm
# ==============================================================================

#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/eval-dual-expert-h100-%j.out
#SBATCH --error=logs/eval-dual-expert-h100-%j.err

# ------------------------------------------------------------------------------
# Variables — modifier ici ou surcharger via export avant sbatch
# ------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projet-fil-rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/projet-fil-rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/projet-fil-rouge/dataset}"
CHECKPOINT="${CHECKPOINT:-models/dual_expert_h100}"
TAXONOMY="${TAXONOMY:-$DATA_DIR/product_classification_taxonomy.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dual_expert_eval_h100}"
SPLIT="${SPLIT:-test}"
TOP_K="${TOP_K:-10}"

# LLM reranker (provider=none → Domain Expert seul)
PROVIDER="${PROVIDER:-hf_local}"
USE_HF_LOCAL="${USE_HF_LOCAL:-0}"
HF_LOAD_IN_4BIT="${HF_LOAD_IN_4BIT:-0}"
LLM_API_URL="${LLM_API_URL:-http://localhost:8080/v1}"
LLM_API_KEY="${LLM_API_KEY:-none}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-512}"
LLM_TIMEOUT="${LLM_TIMEOUT:-15.0}"

# Modes d'évaluation
N_SAMPLES="${N_SAMPLES:-}"                             # vide = dataset complet ; ex: 10000
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.75}"   # seuil routing LLM
SIBLING_CONFUSION_THRESHOLD="${SIBLING_CONFUSION_THRESHOLD:-5}"  # min siblings dans top-K
DISABLE_SIBLING_CONFUSION="${DISABLE_SIBLING_CONFUSION:-0}"       # 1 = désactiver la règle

CATEGORY_DEFS="${CATEGORY_DEFS:-}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=0
export HF_HOME="$HF_CACHE"
export PYTHONPATH=""

mkdir -p logs "$OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Bannière
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " Dual-Expert Evaluation | H100 | $(hostname) | $(date)"
echo " Checkpoint         : $CHECKPOINT"
echo " Data               : $DATA_DIR ($SPLIT split)"
echo " Output             : $OUTPUT_DIR"
echo " Provider           : $PROVIDER"
echo " N_SAMPLES          : ${N_SAMPLES:-complet}"
echo " Confidence seuil   : $CONFIDENCE_THRESHOLD"
echo " Sibling threshold  : $SIBLING_CONFUSION_THRESHOLD (disabled=$([ "$DISABLE_SIBLING_CONFUSION" = "1" ] && echo "oui" || echo "non"))"
if [ "$PROVIDER" != "none" ]; then
    echo " LLM Model          : $LLM_MODEL"
    echo " 4-bit              : $([ "$HF_LOAD_IN_4BIT" = "1" ] && echo "oui" || echo "non")"
fi
echo " Top-K              : $TOP_K"
echo " GPU                : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Construction des arguments optionnels
# ------------------------------------------------------------------------------
DEFS_FLAG=""
if [ -n "$CATEGORY_DEFS" ] && [ -f "$CATEGORY_DEFS" ]; then
    DEFS_FLAG="--category-definitions $CATEGORY_DEFS"
fi

SAMPLE_FLAG=""
if [ -n "$N_SAMPLES" ]; then
    SAMPLE_FLAG="--n-samples $N_SAMPLES"
fi

LLM_FLAGS=""
if [ "$PROVIDER" = "hf_local" ]; then
    LLM_FLAGS="--provider hf_local"
    [ "$HF_LOAD_IN_4BIT" = "1" ] && LLM_FLAGS="$LLM_FLAGS --hf-load-in-4bit"
elif [ "$PROVIDER" != "none" ]; then
    LLM_FLAGS="--provider $PROVIDER --api-base-url $LLM_API_URL --api-key $LLM_API_KEY --timeout $LLM_TIMEOUT"
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
    --model "$LLM_MODEL" \
    --max-new-tokens "$LLM_MAX_TOKENS" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --sibling-confusion-threshold "$SIBLING_CONFUSION_THRESHOLD" \
    $([ "$DISABLE_SIBLING_CONFUSION" = "1" ] && echo "--no-sibling-confusion") \
    $LLM_FLAGS \
    $SAMPLE_FLAG \
    $DEFS_FLAG

echo "======================================================================"
echo " Terminé : $(date)"
echo " Résultats : $OUTPUT_DIR/llm_routing_report.json"
echo "======================================================================"
