#!/bin/bash
#SBATCH --job-name=w2v-title
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-w2v-title-%j.out
#SBATCH --error=slurm-w2v-title-%j.err

# Word2Vec (200d, Skip-gram) + LinearSVC/SGD — Title-only (designation seule)
# RAM élevée : matrices denses 8M × 200 dims × 3 splits ≈ 20 GB + overhead Python.
#
# Usage :
#   sbatch scripts/run_baseline_word2vec_title.slurm

# --- Chemins (surcharger via export avant sbatch si besoin) ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fil_rouge/rakuten-classification}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/fil_rouge/venv/bin/activate}"
DATA_DIR="${DATA_DIR:-$HOME/fil_rouge/dataset}"

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

export N_JOBS=10
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=""

echo "=== Word2Vec Title-only | $(hostname) | $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK | RAM: $SLURM_MEM_PER_NODE MB"

python - <<'PYEOF'
import os, sys, time, json, re, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.common.metrics import compute_all_metrics, compute_per_class_f1

N_JOBS      = int(os.environ.get("N_JOBS", 10))
SEED        = 42
DATASET_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "fil_rouge" / "dataset")))
SAVE_DIR    = Path("models/baselines/word2vec-title")
RESULTS_DIR = Path("results/baselines")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def tokenize(texts):
    return [re.findall(r"\w+", t.lower()) for t in texts]

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

print("[Tokenize] Tokenisation du corpus...")
t0 = time.time()
train_tokens = tokenize(data["X_train"])
print(f"[Tokenize] Terminé en {time.time()-t0:.0f}s")

# --- Word2Vec ---
_w2v_path = SAVE_DIR / "word2vec_model.bin"
if _w2v_path.exists():
    w2v_model = Word2Vec.load(str(_w2v_path))
    print(f"[W2V] Modèle chargé depuis {_w2v_path} ({len(w2v_model.wv)} mots)")
else:
    print(f"[W2V] Entraînement (skip-gram, 200d, {N_JOBS} threads)...")
    t0 = time.time()
    w2v_model = Word2Vec(
        sentences=train_tokens, vector_size=200, window=5,
        min_count=3, sg=1, epochs=10, workers=N_JOBS, seed=SEED,
    )
    w2v_model.save(str(_w2v_path))
    print(f"[W2V] Entraîné en {time.time()-t0:.0f}s — {len(w2v_model.wv):,} mots")

def texts_to_vectors(texts, model):
    dim = model.vector_size
    vecs = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        valid  = [model.wv[t] for t in tokens if t in model.wv]
        vecs.append(np.mean(valid, axis=0) if valid else np.zeros(dim))
    return np.array(vecs, dtype=np.float32)

print("[Vect] Vectorisation train/test...")
t0 = time.time()
X_train_v = texts_to_vectors(data["X_train"], w2v_model)
X_test_v  = texts_to_vectors(data["X_test"],  w2v_model)
print(f"[Vect] Terminé en {time.time()-t0:.0f}s")

# --- L1 ---
_clf_l1_path = SAVE_DIR / "clf_l1.joblib"
if _clf_l1_path.exists():
    clf_l1 = joblib.load(_clf_l1_path)
    print(f"[L1] Classifieur chargé depuis {_clf_l1_path}")
else:
    print("[L1] Entraînement LinearSVC (mono-thread, dense matrix)...")
    t0 = time.time()
    clf_l1 = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, dual="auto")
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
    print("[L2] Entraînement SGDClassifier (mono-thread, dense matrix)...")
    t0 = time.time()
    clf_l2 = SGDClassifier(
        loss="hinge", class_weight="balanced",
        alpha=1e-4, max_iter=100, n_jobs=1,
    )
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
print("\n=== Métriques Word2Vec Title-only ===")
for k, v in all_metrics.items():
    if isinstance(v, float):
        print(f"  {k:30s} : {v:.4f}")

# --- Per-class F1 ---
per_class_l1 = compute_per_class_f1(data["y_l1_test"], preds_l1_test, data["l1_encoder"])
per_class_l2 = compute_per_class_f1(data["y_l2_test"], preds_l2_test, data["l2_encoder"])
with open(RESULTS_DIR / "per_class_f1_l1_word2vec-title.json", "w") as f:
    json.dump(per_class_l1, f, indent=2)
with open(RESULTS_DIR / "per_class_f1_l2_word2vec-title.json", "w") as f:
    json.dump(per_class_l2, f, indent=2)

# --- Sauvegarde dans baseline_results_title.json ---
result_entry = {
    "embedding": "word2vec-title",
    "classifier": "LinearSVC",
    "subtitle": "Word2Vec (200d, Skip-gram) + LinearSVC / SGDClassifier — designation seule",
    "status": "finished",
    "metrics": {
        "test_f1_l1_macro":        round(all_metrics["f1_l1_macro"], 4),
        "test_f1_l2_macro":        round(all_metrics["f1_l2_macro"], 4),
        "test_hcr":                round(all_metrics.get("hcr", 0.0), 4),
        "test_constrained_f1_macro": round(all_metrics.get("constrained_f1_macro", 0.0), 4),
    },
    "params": {
        "vector_size": 200,
        "window": 5,
        "sg": 1,
        "min_count": 3,
        "epochs": 10,
        "classifier_l1": "LinearSVC (C=1.0, class_weight=balanced)",
        "classifier_l2": "SGDClassifier (hinge, class_weight=balanced, n_jobs=1)",
        "use_description": False,
    },
}

results_path = RESULTS_DIR / "baseline_results_title.json"
results = json.loads(results_path.read_text()) if results_path.exists() else []
results = [r for r in results if r["embedding"] != "word2vec-title"]
results.append(result_entry)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] {results_path}")

PYEOF

echo "=== Terminé : $(date) ==="
