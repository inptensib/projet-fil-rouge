#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT" && mkdir -p logs

# Use explicit venv python path (activation can fail on compute nodes)
PYTHON="${PYTHON:-python}"

# Add all venv site-packages to PYTHONPATH (gateway=py3.10, compute=py3.12)
# (PYTHONPATH hard-coding removed for portability)

echo "Using Python: $PYTHON"
$PYTHON --version
$PYTHON -c "from peft import LoraConfig; print('PEFT import OK')" || {
    echo "ERROR: peft not installed."
    exit 1
}

"${PYTHON:-python}" Specialists/scripts/train_l2_lora.py
