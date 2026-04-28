#!/bin/bash
#SBATCH --job-name=tfidf-title
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-tfidf-title-%j.out
#SBATCH --error=slurm-tfidf-title-%j.err

# TF-IDF + LinearSVC/SGD — Title-only (designation seule)
#
# Usage :
#   sbatch scripts/run_baseline_tfidf_title.slurm

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export N_JOBS=10
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=""

echo "=== TF-IDF Title-only | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

python - <<'PYEOF'
import os, sys, time, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from shared.data_loader import prepare_hierarchical_data_from_splits
from shared.metrics import compute_all_metrics, compute_per_class_f1

N_JOBS      = int(os.environ.get("N_JOBS", 10))
SEED        = 42
DATASET_DIR = Path("/home/infres/tandriam-25/fil_rouge/dataset")
SAVE_DIR    = Path("models/baselines/tfidf-title")
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

# --- TF-IDF ---
_tfidf_path = SAVE_DIR / "tfidf_vectorizer.joblib"
if _tfidf_path.exists():
    tfidf = joblib.load(_tfidf_path)
    print(f"[TF-IDF] Vectorizer chargé depuis {_tfidf_path}")
else:
    print("[TF-IDF] Vectorisation (50k features, 1-2grams)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    tfidf.fit(data["X_train"])
    joblib.dump(tfidf, _tfidf_path)
    print(f"[TF-IDF] Vectorisé en {time.time()-t0:.0f}s — vocab={len(tfidf.vocabulary_):,}")

X_train_v = tfidf.transform(data["X_train"])
X_test_v  = tfidf.transform(data["X_test"])

# --- L1 ---
_clf_l1_path = SAVE_DIR / "clf_l1.joblib"
if _clf_l1_path.exists():
    clf_l1 = joblib.load(_clf_l1_path)
    print(f"[L1] Classifieur chargé depuis {_clf_l1_path}")
else:
    print(f"[L1] Entraînement OVR(LinearSVC) — {N_JOBS} workers...")
    t0 = time.time()
    clf_l1 = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced", max_iter=2000), n_jobs=N_JOBS)
    clf_l1.fit(X_train_v, data["y_l1_train"])
    joblib.dump(clf_l1, _clf_l1_path)
    print(f"[L1] Entraîné en {time.time()-t0:.0f}s")

preds_l1_test = clf_l1.predict(X_test_v)

# --- L2 ---
_clf_l2_path = SAVE_DIR / "clf_l2.joblib"
if _clf_l2_path.exists():
    clf_l2 = joblib.load(_clf_l2_path)
    print(f"[L2] Classifieur chargé depuis {_clf_l2_path}")
else:
    print(f"[L2] Entraînement SGDClassifier — {N_JOBS} workers...")
    t0 = time.time()
    clf_l2 = SGDClassifier(loss="hinge", class_weight="balanced", alpha=1e-4, max_iter=100, n_jobs=N_JOBS)
    clf_l2.fit(X_train_v, data["y_l2_train"])
    joblib.dump(clf_l2, _clf_l2_path)
    print(f"[L2] Entraîné en {time.time()-t0:.0f}s")

preds_l2_test = clf_l2.predict(X_test_v)

# --- Métriques ---
all_metrics = compute_all_metrics(
    labels_l1=data["y_l1_test"], preds_l1=preds_l1_test,
    labels_l2=data["y_l2_test"], preds_l2=preds_l2_test,
    l2_to_l1_mapping=data["l2_to_l1_mapping"],
)
print("\n=== Métriques TF-IDF Title-only ===")
for k, v in all_metrics.items():
    if isinstance(v, float):
        print(f"  {k:30s} : {v:.4f}")

# --- Per-class F1 ---
per_class_l1 = compute_per_class_f1(data["y_l1_test"], preds_l1_test, data["l1_encoder"])
per_class_l2 = compute_per_class_f1(data["y_l2_test"], preds_l2_test, data["l2_encoder"])
with open(RESULTS_DIR / "per_class_f1_l1_tfidf-title.json", "w") as f:
    json.dump(per_class_l1, f, indent=2)
with open(RESULTS_DIR / "per_class_f1_l2_tfidf-title.json", "w") as f:
    json.dump(per_class_l2, f, indent=2)

# --- Sauvegarde dans baseline_results_title.json ---
result_entry = {
    "embedding": "tfidf-title",
    "classifier": "LinearSVC",
    "subtitle": "TF-IDF (50K features, 1-2grams) + LinearSVC / SGDClassifier — designation seule",
    "status": "finished",
    "metrics": {
        "test_f1_l1_macro":        round(all_metrics["f1_l1_macro"], 4),
        "test_f1_l2_macro":        round(all_metrics["f1_l2_macro"], 4),
        "test_hcr":                round(all_metrics.get("hcr", 0.0), 4),
        "test_constrained_f1_macro": round(all_metrics.get("constrained_f1_macro", 0.0), 4),
    },
    "params": {
        "max_features": 50000,
        "ngram_range": "1-2",
        "sublinear_tf": True,
        "min_df": 3,
        "classifier_l1": "LinearSVC (C=1.0, class_weight=balanced)",
        "classifier_l2": "SGDClassifier (hinge, class_weight=balanced)",
        "use_description": False,
    },
}

results_path = RESULTS_DIR / "baseline_results_title.json"
results = json.loads(results_path.read_text()) if results_path.exists() else []
results = [r for r in results if r["embedding"] != "tfidf-title"]
results.append(result_entry)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] {results_path}")

PYEOF

echo "=== Terminé : $(date) ==="
