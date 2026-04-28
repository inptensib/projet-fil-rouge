#!/bin/bash
#SBATCH --job-name=ft-soft-title
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-ft-soft-title-%j.out
#SBATCH --error=slurm-ft-soft-title-%j.err

# FastText Softmax — Title-only (designation seule)
# L1: softmax (25 epochs) | L2: softmax (10 epochs)
#
# Usage :
#   sbatch scripts/run_baseline_fasttext_softmax_title.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export N_JOBS=16
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=""

echo "=== FastText Softmax Title-only | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

python - <<'PYEOF'
import os, sys, time, json, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

import fasttext

from shared.data_loader import prepare_hierarchical_data_from_splits
from shared.metrics import compute_all_metrics, compute_per_class_f1

N_JOBS      = int(os.environ.get("N_JOBS", 16))
SEED        = 42
DATASET_DIR = Path("/home/infres/tandriam-25/fil_rouge/dataset")
SAVE_DIR    = Path("models/baselines/fasttext-softmax-title")
RESULTS_DIR = Path("results/baselines")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("[Data] Chargement des splits (designation seule)...")
t0 = time.time()
data = prepare_hierarchical_data_from_splits(
    train_path=str(DATASET_DIR / "train_split.jsonl"),
    val_path=str(DATASET_DIR / "val_split.jsonl"),
    test_path=str(DATASET_DIR / "test_split.jsonl"),
    seed=SEED,
    use_description=False,
)
print(f"[Data] Chargé en {time.time()-t0:.0f}s — train: {len(data['X_train']):,}, test: {len(data['X_test']):,}")

def write_fasttext_file(texts, labels, encoder, filepath):
    label_names = encoder.inverse_transform(labels)
    with open(filepath, "w", encoding="utf-8") as f:
        for text, label in zip(texts, label_names):
            clean_label = str(label).replace(" ", "_").replace("\n", "_")
            clean_text  = str(text).replace("\n", " ").strip()
            f.write(f"__label__{clean_label} {clean_text}\n")

tmpdir = tempfile.mkdtemp()
train_l1_path = os.path.join(tmpdir, "train_l1.txt")
test_l1_path  = os.path.join(tmpdir, "test_l1.txt")
train_l2_path = os.path.join(tmpdir, "train_l2.txt")
test_l2_path  = os.path.join(tmpdir, "test_l2.txt")

print("[FastText] Écriture des fichiers d'entraînement...")
write_fasttext_file(data["X_train"], data["y_l1_train"], data["l1_encoder"], train_l1_path)
write_fasttext_file(data["X_test"],  data["y_l1_test"],  data["l1_encoder"], test_l1_path)
write_fasttext_file(data["X_train"], data["y_l2_train"], data["l2_encoder"], train_l2_path)
write_fasttext_file(data["X_test"],  data["y_l2_test"],  data["l2_encoder"], test_l2_path)

# --- L1 ---
_l1_path = SAVE_DIR / "model_l1.bin"
if _l1_path.exists():
    model_l1 = fasttext.load_model(str(_l1_path))
    print(f"[L1] Modèle chargé depuis {_l1_path}")
    train_time_l1 = 0
else:
    print(f"[L1] Entraînement FastText softmax ({N_JOBS} threads, 25 epochs)...")
    t0 = time.time()
    model_l1 = fasttext.train_supervised(
        input=train_l1_path, lr=0.5, epoch=25,
        wordNgrams=2, dim=100, loss="softmax",
        thread=N_JOBS, verbose=0,
    )
    train_time_l1 = time.time() - t0
    model_l1.save_model(str(_l1_path))
    print(f"[L1] Entraîné en {train_time_l1:.0f}s")

preds_l1_test = [model_l1.predict(t)[0][0].replace("__label__", "") for t in data["X_test"]]
preds_l1_test = data["l1_encoder"].transform(preds_l1_test)

# --- L2 ---
_l2_path = SAVE_DIR / "model_l2.bin"
if _l2_path.exists():
    model_l2 = fasttext.load_model(str(_l2_path))
    print(f"[L2] Modèle chargé depuis {_l2_path}")
    train_time_l2 = 0
else:
    L2_THREADS = min(N_JOBS, 10)
    print(f"[L2] Entraînement FastText softmax ({L2_THREADS} threads, 10 epochs)...")
    t0 = time.time()
    model_l2 = fasttext.train_supervised(
        input=train_l2_path, lr=0.5, epoch=10,
        wordNgrams=2, dim=100, loss="softmax",
        thread=L2_THREADS, verbose=0,
    )
    train_time_l2 = time.time() - t0
    model_l2.save_model(str(_l2_path))
    print(f"[L2] Entraîné en {train_time_l2:.0f}s")

preds_l2_test = [model_l2.predict(t)[0][0].replace("__label__", "") for t in data["X_test"]]
preds_l2_test = data["l2_encoder"].transform(preds_l2_test)

# --- Métriques ---
all_metrics = compute_all_metrics(
    labels_l1=data["y_l1_test"], preds_l1=preds_l1_test,
    labels_l2=data["y_l2_test"], preds_l2=preds_l2_test,
    l2_to_l1_mapping=data["l2_to_l1_mapping"],
)
print("\n=== Métriques FastText Softmax Title-only ===")
for k, v in all_metrics.items():
    if isinstance(v, float):
        print(f"  {k:30s} : {v:.4f}")

# --- Per-class F1 ---
per_class_l1 = compute_per_class_f1(data["y_l1_test"], preds_l1_test, data["l1_encoder"])
per_class_l2 = compute_per_class_f1(data["y_l2_test"], preds_l2_test, data["l2_encoder"])
with open(RESULTS_DIR / "per_class_f1_l1_fasttext-softmax-title.json", "w") as f:
    json.dump(per_class_l1, f, indent=2)
with open(RESULTS_DIR / "per_class_f1_l2_fasttext-softmax-title.json", "w") as f:
    json.dump(per_class_l2, f, indent=2)

# --- Sauvegarde dans baseline_results_title.json ---
result_entry = {
    "embedding": "fasttext-softmax-title",
    "classifier": "Softmax",
    "subtitle": "FastText — L1: softmax (25 epochs), L2: softmax (10 epochs) — designation seule",
    "status": "finished",
    "metrics": {
        "test_f1_l1_macro":        round(all_metrics["f1_l1_macro"], 4),
        "test_f1_l2_macro":        round(all_metrics["f1_l2_macro"], 4),
        "test_hcr":                round(all_metrics.get("hcr", 0.0), 4),
        "test_constrained_f1_macro": round(all_metrics.get("constrained_f1_macro", 0.0), 4),
    },
    "params": {
        "dim": 100,
        "wordNgrams": 2,
        "lr": 0.5,
        "loss_l1": "softmax",
        "epochs_l1": 25,
        "loss_l2": "softmax",
        "epochs_l2": 10,
        "thread": N_JOBS,
        "use_description": False,
    },
}

results_path = RESULTS_DIR / "baseline_results_title.json"
results = json.loads(results_path.read_text()) if results_path.exists() else []
results = [r for r in results if r["embedding"] != "fasttext-softmax-title"]
results.append(result_entry)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] {results_path}")

PYEOF

echo "=== Terminé : $(date) ==="
