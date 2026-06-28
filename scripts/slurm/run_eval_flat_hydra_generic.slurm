#!/bin/bash
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-eval-%x-%j.out
#SBATCH --error=slurm-eval-%x-%j.err

# Éval TEST générique d'un modèle FlatHydra (best_model.pt) sur un dataset donné.
# Paramétrable via variables d'environnement (passées par --export) :
#   MODEL_PT      chemin du best_model.pt        (def: models_v2/flat_hydra/best_model.pt)
#   EVAL_CONFIG   yaml d'architecture du modèle  (def: configs/baselines/flat_hydra_camembert_v2.yaml)
#   DATA_DIR      dossier des splits *_split.jsonl
#   RESULTS_DIR   où écrire test_metrics.json
#   EVAL_BATCH    batch d'éval (no grad)         (def: 1536)
#
# Exemples (depuis la racine du repo, package installé via `pip install -e .`) :
#   # CamemBERT v2 (défauts) :
#   sbatch --job-name=eval-cam-v2 scripts/run_eval_flat_hydra_generic.slurm
#   # mBERT v1 (ancien dataset) :
#   sbatch --job-name=eval-mbert-v1 --export=ALL,\
#     MODEL_PT=models/flat_hydra_mbert/best_model.pt,\
#     EVAL_CONFIG=configs/baselines/flat_hydra_mbert.yaml,\
#     DATA_DIR=$HOME/fil_rouge/dataset,\
#     RESULTS_DIR=results/flat_hydra_mbert \
#     scripts/run_eval_flat_hydra_generic.slurm

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"

# Defaults (CamemBERT v2) — surchargés par --export
export MODEL_PT="${MODEL_PT:-models_v2/flat_hydra/best_model.pt}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/baselines/flat_hydra_camembert_v2.yaml}"
export DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset_v2/datasets_finaux}"
export RESULTS_DIR="${RESULTS_DIR:-results/flat_hydra_v2}"
export EVAL_BATCH="${EVAL_BATCH:-1536}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== Éval FlatHydra | $(hostname) | $(date) ==="
echo "Model: $MODEL_PT | Config: $EVAL_CONFIG | Data: $DATA_DIR | Out: $RESULTS_DIR | Batch: $EVAL_BATCH"
echo "GPU : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python - <<'PYEOF'
import os, json, time
from pathlib import Path

import torch
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.models.baselines.flat_trainer import FlatHydraConfig, FlatHydraTrainer

MODEL_PT    = os.environ["MODEL_PT"]
EVAL_CONFIG = os.environ["EVAL_CONFIG"]
SPLIT_DIR   = Path(os.environ["DATA_DIR"])
RESULTS_DIR = Path(os.environ["RESULTS_DIR"])
EVAL_BATCH  = int(os.environ.get("EVAL_BATCH", "1536"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

config = FlatHydraConfig.from_yaml(EVAL_CONFIG)

print("[Data] Chargement des splits...")
t0 = time.time()
data = prepare_hierarchical_data_from_splits(
    train_path=str(SPLIT_DIR / "train_split.jsonl"),
    val_path=str(SPLIT_DIR / "val_split.jsonl"),
    test_path=str(SPLIT_DIR / "test_split.jsonl"),
    use_description=config.use_description,
    seed=42,
)
print(f"[Data] Chargé en {time.time()-t0:.0f}s — test: {len(data['X_test']):,}, "
      f"L1: {data['num_classes_l1']}, L2: {data['num_classes_l2']}")

config.batch_size = EVAL_BATCH
trainer = FlatHydraTrainer(data=data, config=config)

print(f"[Model] Chargement {MODEL_PT}...")
ckpt = torch.load(MODEL_PT, map_location=trainer.device, weights_only=False)
trainer._base_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
print("[Model] Poids chargés.")

test_loader = trainer._make_loader(
    data["X_test"], data["y_l1_test"], data["y_l2_test"], shuffle=False
)
print("[Eval] Évaluation sur le TEST set...")
t1 = time.time()
metrics = trainer.evaluate(test_loader)
elapsed = time.time() - t1

print(f"\n{'='*60}")
print(f"  FlatHydra — TEST set ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  F1 L1 macro       : {metrics['f1_l1_macro']:.4f}")
print(f"  F1 L2 macro       : {metrics['f1_l2_macro']:.4f}")
print(f"  Constrained F1 L2 : {metrics.get('constrained_f1_macro', 0.0):.4f}")
print(f"  HCR               : {metrics.get('hcr', 0.0):.4f}")
print(f"{'='*60}")

summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
summary["eval_time_s"] = round(elapsed, 1)
summary["num_classes_l2"] = data["num_classes_l2"]
out_path = RESULTS_DIR / "test_metrics.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[Saved] {out_path}")
PYEOF

echo "=== Terminé : $(date) ==="
