#!/bin/bash
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-eval-flat-%j.out
#SBATCH --error=slurm-eval-flat-%j.err

# Évalue FlatHydra (best_model.pt) sur le test set complet.
#
# Usage :
#   sbatch scripts/run_eval_flat_hydra.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=""

echo "=== Évaluation FlatHydra | $(hostname) | $(date) ==="
echo "GPU : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python - <<'PYEOF'
import json
import os
import time
from pathlib import Path

import torch
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.models.baselines.flat_trainer import FlatHydraConfig, FlatHydraTrainer

CONFIG   = "configs/baselines/flat_hydra_3090.yaml"
MODEL_PT = "models/flat_hydra/best_model.pt"
SPLIT_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "fil_rouge" / "dataset")))
RESULTS_DIR = Path("results/flat_hydra")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("[Data] Chargement des splits...")
t0 = time.time()
data = prepare_hierarchical_data_from_splits(
    train_path=str(SPLIT_DIR / "train_split.jsonl"),
    val_path=str(SPLIT_DIR / "val_split.jsonl"),
    test_path=str(SPLIT_DIR / "test_split.jsonl"),
    seed=42,
)
print(f"[Data] Chargé en {time.time()-t0:.0f}s — "
      f"test: {len(data['X_test']):,} samples, "
      f"L1: {data['num_classes_l1']}, L2: {data['num_classes_l2']}")

config = FlatHydraConfig.from_yaml(CONFIG)
config.batch_size = 3072  # eval seulement, pas de gradient → 2× GPU DataParallel

trainer = FlatHydraTrainer(data=data, config=config)

print(f"[Model] Chargement {MODEL_PT}...")
ckpt = torch.load(MODEL_PT, map_location=trainer.device, weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
trainer._base_model.load_state_dict(state_dict)
print("[Model] Poids chargés.")

test_loader = trainer._make_loader(
    data["X_test"], data["y_l1_test"], data["y_l2_test"], shuffle=False
)

print("[Eval] Évaluation sur le test set...")
t1 = time.time()
metrics = trainer.evaluate(test_loader)
elapsed = time.time() - t1

f1_l1 = metrics["f1_l1_macro"]
f1_l2 = metrics["f1_l2_macro"]
hcr   = metrics.get("hcr", 0.0)
cf1   = metrics.get("constrained_f1_macro", 0.0)

print(f"\n{'='*60}")
print(f"  FlatHydra — Résultats test set ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  F1 L1 macro       : {f1_l1:.4f}  ({f1_l1*100:.2f}%)")
print(f"  F1 L2 macro       : {f1_l2:.4f}  ({f1_l2*100:.2f}%)")
print(f"  Constrained F1 L2 : {cf1:.4f}  ({cf1*100:.2f}%)")
print(f"  HCR               : {hcr:.4f}  ({hcr*100:.2f}%)")
print(f"{'='*60}")

# Sauvegarder les métriques (sans les listes _preds/_labels)
summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
summary["eval_time_s"] = round(elapsed, 1)
out_path = RESULTS_DIR / "test_metrics.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[Saved] {out_path}")

PYEOF

echo "=== Terminé : $(date) ==="
