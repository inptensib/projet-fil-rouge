#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

cd ~/Rakuten_Project && mkdir -p logs

# Use explicit venv python path (activation can fail on compute nodes)
PYTHON="$HOME/Rakuten_Project/.venv/bin/python"

# Add all venv site-packages to PYTHONPATH (gateway=py3.10, compute=py3.12)
export PYTHONPATH=$(find $HOME/Rakuten_Project/.venv/lib -name site-packages -type d | tr '\n' ':')$PYTHONPATH

echo "Using Python: $PYTHON"
$PYTHON --version
$PYTHON -c "from peft import LoraConfig; print('PEFT import OK')" || {
    echo "ERROR: peft not installed."
    exit 1
}

$PYTHON train_l2_lora.py
