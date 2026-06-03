#!/usr/bin/env python3
"""
L2 XGBoost classifier:
1. Extract [CLS] embeddings from L1 model for all train/val/test products
2. Train one XGBoost per L1 class on its subcategory subset
"""
import os
import json
import pickle
import torch
import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# ─── Config ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
L1_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/l1_classifier")
DAPT_MODEL  = os.path.join(BASE_DIR, "checkpoints/dapt/final")
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "checkpoints/l2_xgboost")
EMB_DIR     = os.path.join(BASE_DIR, "embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Find L1 model
def find_l1():
    best = os.path.join(L1_CKPT_DIR, "best")
    if os.path.isdir(best):
        return best
    ckpts = sorted([d for d in os.listdir(L1_CKPT_DIR) if d.startswith("checkpoint-")],
                   key=lambda x: int(x.split("-")[1]))
    return os.path.join(L1_CKPT_DIR, ckpts[-1])

L1_MODEL_DIR = find_l1()
print(f"Using L1 model: {L1_MODEL_DIR}")

# Load L1 tokenizer (try checkpoint, then DAPT, then base)
tokenizer = None
for tok_path in [L1_MODEL_DIR, DAPT_MODEL, "camembert-base"]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print(f"Tokenizer from: {tok_path}")
        break
    except Exception:
        continue

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─── Step 1: Extract embeddings ───────────────────────
def extract_embeddings(split_name, jsonl_path):
    cache_file = os.path.join(EMB_DIR, f"{split_name}_emb.npz")
    if os.path.isfile(cache_file):
        print(f"Loading cached embeddings: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y_l1"], data["y_l2"]

    print(f"\nExtracting embeddings for {split_name}...")
    model = AutoModel.from_pretrained(L1_MODEL_DIR).to(device)
    model.eval()

    ds = load_dataset("json", data_files={split_name: jsonl_path})[split_name]
    n = len(ds)
    BATCH = 128

    # Pre-allocate to avoid vstack memory doubling. Use float16 to halve size.
    X = np.empty((n, 768), dtype=np.float16)
    l1_labels, l2_labels = [], []

    pos = 0
    for i in tqdm(range(0, n, BATCH), desc=f"Embedding {split_name}"):
        batch = ds[i:i+BATCH]
        inputs = tokenizer(batch["designation"], padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
        bsz = cls_emb.shape[0]
        X[pos:pos+bsz] = cls_emb
        pos += bsz
        l1_labels.extend(batch["typelabel"])
        l2_labels.extend([pt[0] if pt else None for pt in batch["producttype"]])

    X = X[:pos]
    y_l1 = np.array(l1_labels, dtype=object)
    y_l2 = np.array(l2_labels, dtype=object)

    print(f"Saving embeddings to {cache_file} (shape={X.shape}, dtype={X.dtype})")
    # Save uncompressed (compression doubles peak memory)
    np.savez(cache_file, X=X, y_l1=y_l1, y_l2=y_l2)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return X, y_l1, y_l2

X_train, y1_train, y2_train = extract_embeddings("train", os.path.join(DATA_DIR, "train_split.jsonl"))
X_val,   y1_val,   y2_val   = extract_embeddings("val",   os.path.join(DATA_DIR, "val_split.jsonl"))
X_test,  y1_test,  y2_test  = extract_embeddings("test",  os.path.join(DATA_DIR, "test_split.jsonl"))

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ─── Step 2: Train one XGBoost per L1 class ───────────
print("\n=== Training XGBoost per L1 class ===")

# GPU available for XGBoost?
USE_GPU = torch.cuda.is_available()
TREE_METHOD = "hist"  # XGBoost 2.0+ uses device="cuda" instead of gpu_hist
print(f"XGBoost tree_method: {TREE_METHOD}, device: {'cuda' if USE_GPU else 'cpu'}")

# Subsample cap: classes with >MAX_SAMPLES get stratified subsampled
MAX_SAMPLES = 100_000

# Group training data by L1 class
l1_classes = sorted(set(y1_train))
results_per_class = {}
xgb_models = {}
label_maps = {}

# Filter samples that have an L2 label
mask_train = y2_train != None
mask_val = y2_val != None
mask_test = y2_test != None

# Sort classes by size (smallest first) for faster progress visibility
class_sizes = {c: (mask_train & (y1_train == c)).sum() for c in l1_classes}
l1_classes_sorted = sorted(l1_classes, key=lambda c: class_sizes[c])

for idx, l1_class in enumerate(l1_classes_sorted):
    # Get train/val/test subsets for this L1 class
    tr_mask = mask_train & (y1_train == l1_class)
    va_mask = mask_val & (y1_val == l1_class)
    te_mask = mask_test & (y1_test == l1_class)

    n_tr = tr_mask.sum()
    if n_tr < 10:
        continue  # too few samples

    # Build subcategory label map for this L1 class
    subcats = sorted(set(y2_train[tr_mask]))
    if len(subcats) < 2:
        label_maps[l1_class] = {"single": subcats[0] if subcats else None}
        continue

    sub2id = {s: i for i, s in enumerate(subcats)}
    id2sub = {i: s for s, i in sub2id.items()}
    label_maps[l1_class] = {"sub2id": sub2id, "id2sub": id2sub}

    Xtr = X_train[tr_mask].astype(np.float32)
    ytr = np.array([sub2id[s] for s in y2_train[tr_mask]])

    # Subsample large classes (stratified)
    if len(ytr) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        # Stratified subsample
        indices = []
        for cls_id in range(len(subcats)):
            cls_idx = np.where(ytr == cls_id)[0]
            # Keep proportional share, minimum 1
            n_keep = max(1, int(MAX_SAMPLES * len(cls_idx) / len(ytr)))
            if len(cls_idx) > n_keep:
                indices.extend(rng.choice(cls_idx, n_keep, replace=False).tolist())
            else:
                indices.extend(cls_idx.tolist())
        indices = np.array(indices)
        Xtr, ytr = Xtr[indices], ytr[indices]

    # Validation/test only on subcategories seen in training
    va_valid = va_mask & np.array([s in sub2id for s in y2_val], dtype=bool)
    te_valid = te_mask & np.array([s in sub2id for s in y2_test], dtype=bool)
    Xva = X_val[va_valid].astype(np.float32) if va_valid.sum() > 0 else None
    yva = np.array([sub2id[s] for s in y2_val[va_valid]]) if va_valid.sum() > 0 else None
    Xte = X_test[te_valid].astype(np.float32) if te_valid.sum() > 0 else None
    yte = np.array([sub2id[s] for s in y2_test[te_valid]]) if te_valid.sum() > 0 else None

    # Adaptive n_estimators: more trees for smaller problems
    n_est = 300 if len(ytr) < 50_000 else 150

    # Train XGBoost with early stopping
    clf = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(subcats),
        n_jobs=8,
        tree_method=TREE_METHOD,
        device="cuda" if USE_GPU else "cpu",
        verbosity=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
    )

    # Use val for early stopping if available, else just fit
    if Xva is not None and len(yva) > 0:
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        clf.set_params(early_stopping_rounds=None)
        clf.fit(Xtr, ytr)

    xgb_models[l1_class] = clf

    # Evaluate (predict may return proba matrix on GPU, so handle both)
    metrics = {"n_train": int(n_tr), "n_subcats": len(subcats), "n_train_used": len(ytr)}
    if Xva is not None and len(yva) > 0:
        pred = clf.predict(Xva)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        metrics["val_acc"] = accuracy_score(yva, pred)
        metrics["val_f1_macro"] = f1_score(yva, pred, average="macro", zero_division=0)
    if Xte is not None and len(yte) > 0:
        pred = clf.predict(Xte)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        metrics["test_acc"] = accuracy_score(yte, pred)
        metrics["test_f1_macro"] = f1_score(yte, pred, average="macro", zero_division=0)
    results_per_class[l1_class] = metrics

    # Progress
    if (idx + 1) % 10 == 0 or n_tr > 50000:
        va_str = f", val_acc={metrics.get('val_acc', 0):.3f}" if 'val_acc' in metrics else ""
        print(f"  [{idx+1}/{len(l1_classes_sorted)}] {l1_class}: "
              f"{n_tr} samples, {len(subcats)} subcats, used {len(ytr)}{va_str}")

# ─── Save models and metrics ──────────────────────────
print("\nSaving XGBoost models...")
with open(os.path.join(OUTPUT_DIR, "xgb_models.pkl"), "wb") as f:
    pickle.dump(xgb_models, f)
with open(os.path.join(OUTPUT_DIR, "label_maps.json"), "w") as f:
    # Convert tuples/sets and ensure JSON-serializable
    serializable = {}
    for k, v in label_maps.items():
        if "single" in v:
            serializable[k] = {"single": v["single"]}
        else:
            serializable[k] = {
                "sub2id": v["sub2id"],
                "id2sub": {str(i): s for i, s in v["id2sub"].items()},
            }
    json.dump(serializable, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "results_per_class.json"), "w") as f:
    json.dump(results_per_class, f, ensure_ascii=False, indent=2)

# ─── Summary ──────────────────────────────────────────
print("\n=== L2 XGBoost Summary ===")
val_accs = [r["val_acc"] for r in results_per_class.values() if "val_acc" in r]
test_accs = [r["test_acc"] for r in results_per_class.values() if "test_acc" in r]
test_f1s = [r["test_f1_macro"] for r in results_per_class.values() if "test_f1_macro" in r]

if val_accs:
    print(f"Mean val accuracy across L1 classes:  {np.mean(val_accs):.3f}")
if test_accs:
    print(f"Mean test accuracy across L1 classes: {np.mean(test_accs):.3f}")
    print(f"Mean test F1-macro across L1 classes: {np.mean(test_f1s):.3f}")
print(f"\nClasses with XGBoost models: {len(xgb_models)}")
print(f"Classes with single subcategory (no model needed): "
      f"{sum(1 for v in label_maps.values() if 'single' in v)}")
print(f"\nSaved to: {OUTPUT_DIR}")
