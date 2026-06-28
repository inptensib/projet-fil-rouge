#!/bin/bash
# Mini-test de validation GPU sur mesogip (H100) — ~1 min.
#SBATCH --job-name=gpu-smoke
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source ~/rakuten-venv/bin/activate
echo "=== Node: $(hostname) | $(date) ==="
nvidia-smi
echo "=== Test PyTorch ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    print("capability sm_:", "".join(map(str, torch.cuda.get_device_capability(0))))
    x = torch.randn(8192, 8192, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul GPU OK | sum =", float(y.sum()))
PY
