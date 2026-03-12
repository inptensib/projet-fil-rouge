#!/usr/bin/env bash
# ==============================================================================
# Setup de l'environnement sur le cluster SLURM (sans Docker)
# Reproduit le meme environnement que le Dockerfile
#
# Usage:
#   ./scripts/setup_cluster.sh              # Creer/maj le venv
#   ./scripts/setup_cluster.sh --check      # Verifier l'installation
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_DIR}/../venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }

# --- Activate venv if it exists ---
activate_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
        return 0
    fi
    return 1
}

# --- Check mode ---
if [[ "${1:-}" == "--check" ]]; then
    echo "=== Environment Check ==="

    if activate_venv; then
        echo -e "Venv:       ${GREEN}${VENV_DIR} (actif)${NC}"
    else
        warn "Venv non trouve a ${VENV_DIR}. Lancer d'abord: ./scripts/setup_cluster.sh"
    fi

    echo "Python:     $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
    echo "Pip:        $(pip --version 2>/dev/null || echo 'NOT FOUND')"
    echo "PyTorch:    $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
    echo "CUDA dispo: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
    echo "GPUs:       $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
    echo "MLflow:     $(python3 -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null || echo 'NOT FOUND')"

    # Verify all requirements are installed
    echo ""
    echo "=== Requirements Check ==="
    python3 -m pip check 2>/dev/null && echo -e "${GREEN}All dependencies OK${NC}" || warn "Some dependency conflicts"

    # Verify key imports
    echo ""
    echo "=== Import Check ==="
    python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from shared.data_loader import HierarchicalProductDataset
from shared.metrics import compute_all_metrics
from shared.utils import set_seed
print('shared/         OK')
sys.path.insert(0, '$PROJECT_DIR/02_hydra')
from src.models import create_hydra_model
from src.losses import HYDRALoss
print('02_hydra/src/   OK')
" 2>/dev/null && echo -e "${GREEN}All imports OK${NC}" || warn "Some imports failed"

    exit 0
fi

# --- Create or update venv ---
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creation du venv dans $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

log "Activation du venv ..."
source "$VENV_DIR/bin/activate"

log "Mise a jour pip ..."
pip install --upgrade pip

log "Installation des dependances depuis requirements.txt ..."
pip install -r "$PROJECT_DIR/requirements.txt"

log "Installation des outils de dev (pytest) ..."
pip install pytest

# --- Pre-download CamemBERT tokenizer ---
log "Pre-telechargement du tokenizer CamemBERT ..."
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('camembert-base')" 2>/dev/null || \
    warn "Impossible de telecharger CamemBERT (mode offline ?). Le tokenizer devra etre dans le cache."

log "Setup termine."
echo ""
echo "Pour activer l'environnement :"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Pour verifier :"
echo "  ./scripts/setup_cluster.sh --check"
