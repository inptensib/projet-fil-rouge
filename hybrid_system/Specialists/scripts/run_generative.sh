#!/bin/bash
#SBATCH --job-name=gen_eval
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_eval_%j.out
#SBATCH --error=logs/gen_eval_%j.err

set -e
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs checkpoints/generative_models

PYTHON="${PYTHON:-$(command -v python)}"
echo "Using Python: $PYTHON"
$PYTHON --version


# (PYTHONPATH hard-coding removed for portability)

# Verify key imports
$PYTHON -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
$PYTHON -c "import datasets; print('datasets OK')"

echo "Starting generative fallback evaluation..."
"${PYTHON:-python}" Specialists/scripts/generative_fallback.py

echo "Job complete."
