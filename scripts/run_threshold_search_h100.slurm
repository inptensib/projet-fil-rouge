#!/bin/bash
# ==============================================================================
# Threshold search séquentiel — val set uniquement
# Boucle sur plusieurs seuils de confiance dans un seul job SLURM.
# Le meilleur seuil est ensuite utilisé pour l'évaluation finale sur test.
#
# Usage :
#   sbatch scripts/run_threshold_search_h100.slurm
#
# Avec LLM local :
#   export PROVIDER=hf_local
#   export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/run_threshold_search_h100.slurm
#
# Avec définitions de catégories :
#   export CATEGORY_DEFS=models/category_definitions/category_definitions_v1.jsonl
#   sbatch scripts/run_threshold_search_h100.slurm
# ==============================================================================

#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/threshold-search-h100-%j.out
#SBATCH --error=logs/threshold-search-h100-%j.err

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projet-fil-rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/projet-fil-rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/projet-fil-rouge/dataset}"
CHECKPOINT="${CHECKPOINT:-models/dual_expert_h100}"
TAXONOMY="${TAXONOMY:-$DATA_DIR/product_classification_taxonomy.csv}"
SEARCH_DIR="${SEARCH_DIR:-results/threshold_search}"
TOP_K="${TOP_K:-10}"

PROVIDER="${PROVIDER:-hf_local}"
HF_LOAD_IN_4BIT="${HF_LOAD_IN_4BIT:-0}"
LLM_API_URL="${LLM_API_URL:-http://localhost:8080/v1}"
LLM_API_KEY="${LLM_API_KEY:-none}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-512}"
LLM_TIMEOUT="${LLM_TIMEOUT:-15.0}"

CATEGORY_DEFS="${CATEGORY_DEFS:-}"
N_SAMPLES="${N_SAMPLES:-}"            # vide = dataset complet ; ex: 5000 pour test rapide

THRESHOLDS="${THRESHOLDS:-0.50 0.60 0.70 0.75 0.80 0.90}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HF_CACHE"
export PYTHONPATH=""

mkdir -p logs "$SEARCH_DIR"

# ------------------------------------------------------------------------------
# Bannière
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " Threshold Search | H100 | $(hostname) | $(date)"
echo " Checkpoint : $CHECKPOINT"
echo " Data       : $DATA_DIR (val split)"
echo " Provider   : $PROVIDER"
echo " Seuils     : $THRESHOLDS"
echo " Output     : $SEARCH_DIR"
echo " GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Arguments optionnels communs
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
# Boucle séquentielle sur les seuils
# ------------------------------------------------------------------------------
BEST_THRESHOLD=""
BEST_F1="-1"

for THRESHOLD in $THRESHOLDS; do
    OUT_DIR="$SEARCH_DIR/thresh_${THRESHOLD}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "----------------------------------------------------------------------"
    echo " Seuil : $THRESHOLD → $OUT_DIR"
    echo "----------------------------------------------------------------------"

    python -u scripts/evaluate_llm_reranker.py \
        --checkpoint "$CHECKPOINT" \
        --split-dir "$DATA_DIR" \
        --output "$OUT_DIR" \
        --split val \
        --top-k "$TOP_K" \
        --model "$LLM_MODEL" \
        --max-new-tokens "$LLM_MAX_TOKENS" \
        --confidence-threshold "$THRESHOLD" \
        $LLM_FLAGS \
        $SAMPLE_FLAG \
        $DEFS_FLAG

    # Extraire le f1 du rapport
    REPORT="$OUT_DIR/llm_routing_report.json"
    if [ -f "$REPORT" ]; then
        F1=$(python -c "import json; r=json.load(open('$REPORT')); print(r.get('overall_f1_l2_macro', -1))")
        N_ROUTED=$(python -c "import json; r=json.load(open('$REPORT')); print(r.get('n_routed', r.get('n_evaluated', '?')))")
        echo " → f1_l2_macro=$F1  routed=$N_ROUTED"

        # Garder le meilleur
        IS_BETTER=$(python -c "print(1 if $F1 > $BEST_F1 else 0)")
        if [ "$IS_BETTER" = "1" ]; then
            BEST_F1=$F1
            BEST_THRESHOLD=$THRESHOLD
        fi
    else
        echo " → ERREUR : rapport non généré"
    fi
done

# ------------------------------------------------------------------------------
# Résumé
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " Résultats threshold search (val set)"
echo "======================================================================"
for THRESHOLD in $THRESHOLDS; do
    REPORT="$SEARCH_DIR/thresh_${THRESHOLD}/llm_routing_report.json"
    if [ -f "$REPORT" ]; then
        F1=$(python -c "import json; r=json.load(open('$REPORT')); print(f\"{r.get('overall_f1_l2_macro', -1):.4f}\")")
        N_ROUTED=$(python -c "import json; r=json.load(open('$REPORT')); print(r.get('n_routed', '?'))")
        MARKER=$([ "$THRESHOLD" = "$BEST_THRESHOLD" ] && echo " ← BEST" || echo "")
        echo "  thresh=$THRESHOLD  f1=$F1  routed=$N_ROUTED$MARKER"
    fi
done
echo ""
echo " Meilleur seuil : $BEST_THRESHOLD (f1=$BEST_F1)"
echo ""
echo " Prochain step — évaluation finale sur test :"
echo "   export CONFIDENCE_THRESHOLD=$BEST_THRESHOLD"
echo "   export SPLIT=test"
echo "   sbatch scripts/run_eval_dual_expert_h100.slurm"
echo "======================================================================"
echo " Terminé : $(date)"
echo "======================================================================"
