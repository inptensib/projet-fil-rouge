#!/usr/bin/env python3
"""
Improved XGBoost L2 classifier using CamemBERT L2 embeddings.

Key change: Instead of using L1 model CLS embeddings (768d, trained for L1 task),
use the L2 CamemBERT model's CLS embeddings which have learned finer subcategory distinctions.

Optionally concatenates L1 + L2 embeddings (1536d) for richer features.

Expected improvement: +1-3% XGBoost accuracy on uncertain samples → +0.3-0.7% overall hybrid accuracy.
"""
import os
import json
import pickle
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# ─── Config ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR    = os.path.join(PROJECT_DIR, "dataset")

L1_CKPT_DIR      = os.path.join(PROJECT_DIR, "checkpoints/l1_classifier")
L2_CAMEMBERT_DIR = os.path.join(PROJECT_DIR, "checkpoints/l2_camembert_run2")
OUTPUT_DIR       = os.path.join(PROJECT_DIR, "checkpoints/l2_xgboost_v2")
EMB_DIR          = os.path.join(PROJECT_DIR, "embeddings")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Embedding mode: "l2_only", "l1_only", "concat"
EMB_MODE = os.environ.get("EMB_MODE", "concat")
MAX_LEN_L2 = 256
MAX_LEN_L1 = 128
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Embedding mode: {EMB_MODE}")


def find_best_ckpt(ckpt_dir):
    best = os.path.join(ckpt_dir, "best")
    if os.path.isdir(best):
        return best
    ckpts = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir


# ─── Step 1: Extract embeddings ───────────────────────
def extract_embeddings(split_name, jsonl_path):
    cache_file = os.path.join(EMB_DIR, f"{split_name}_emb_{EMB_MODE}.npz")
    if os.path.isfile(cache_file):
        print(f"Loading cached embeddings: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y_l1"], data["y_l2"]

    print(f"\nExtracting {EMB_MODE} embeddings for {split_name}...")
    ds = load_dataset("json", data_files={split_name: jsonl_path})[split_name]
    n = len(ds)

    # Load L2 model for embeddings
    l2_model_dir = find_best_ckpt(L2_CAMEMBERT_DIR)
    l2_model = AutoModelForSequenceClassification.from_pretrained(l2_model_dir).to(device)
    l2_model.eval()
    try:
        l2_tokenizer = AutoTokenizer.from_pretrained(l2_model_dir)
    except OSError:
        l2_tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # Optionally load L1 model
    l1_model = None
    l1_tokenizer = None
    if EMB_MODE in ("l1_only", "concat"):
        l1_model_dir = find_best_ckpt(L1_CKPT_DIR)
        l1_model = AutoModel.from_pretrained(l1_model_dir).to(device)
        l1_model.eval()
        try:
            l1_tokenizer = AutoTokenizer.from_pretrained(l1_model_dir)
        except OSError:
            l1_tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # Determine embedding dimension
    if EMB_MODE == "concat":
        emb_dim = 768 * 2  # L1 CLS + L2 CLS
    else:
        emb_dim = 768

    X = np.empty((n, emb_dim), dtype=np.float16)
    l1_labels, l2_labels = [], []

    pos = 0
    for i in tqdm(range(0, n, BATCH_SIZE), desc=f"Embedding {split_name}"):
        batch = ds[i:i+BATCH_SIZE]
        bsz = len(batch["designation"])

        # Build texts
        texts_l2 = []
        for j in range(bsz):
            title = batch["designation"][j] or ""
            desc = (batch.get("description") or [""] * bsz)[j] or ""
            texts_l2.append(f"{title} {desc}"[:1024] if desc else title)

        # L2 embedding (from the sequence classification model's hidden states)
        with torch.no_grad():
            inputs_l2 = l2_tokenizer(
                texts_l2, padding=True, truncation=True,
                max_length=MAX_LEN_L2, return_tensors="pt"
            ).to(device)
            outputs_l2 = l2_model(**inputs_l2, output_hidden_states=True)
            emb_l2 = outputs_l2.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float16)

        if EMB_MODE == "l2_only":
            emb = emb_l2
        elif EMB_MODE == "l1_only":
            texts_l1 = [batch["designation"][j] or "" for j in range(bsz)]
            with torch.no_grad():
                inputs_l1 = l1_tokenizer(
                    texts_l1, padding=True, truncation=True,
                    max_length=MAX_LEN_L1, return_tensors="pt"
                ).to(device)
                outputs_l1 = l1_model(**inputs_l1)
                emb = outputs_l1.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
        elif EMB_MODE == "concat":
            texts_l1 = [batch["designation"][j] or "" for j in range(bsz)]
            with torch.no_grad():
                inputs_l1 = l1_tokenizer(
                    texts_l1, padding=True, truncation=True,
                    max_length=MAX_LEN_L1, return_tensors="pt"
                ).to(device)
                outputs_l1 = l1_model(**inputs_l1)
                emb_l1 = outputs_l1.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
            emb = np.concatenate([emb_l1, emb_l2], axis=1)

        actual_bsz = emb.shape[0]
        X[pos:pos+actual_bsz] = emb
        pos += actual_bsz
        l1_labels.extend(batch["typelabel"])
        l2_labels.extend([pt[0] if pt else None for pt in batch["producttype"]])

    X = X[:pos]
    y_l1 = np.array(l1_labels, dtype=object)
    y_l2 = np.array(l2_labels, dtype=object)

    # Free GPU
    del l2_model
    if l1_model is not None:
        del l1_model
    torch.cuda.empty_cache()

    print(f"Saving embeddings to {cache_file} (shape={X.shape})")
    # Atomic write: save to a temp file then rename, so a crash mid-write
    # never leaves a corrupt (partial) cache file behind.
    tmp_file = cache_file + ".tmp.npz"
    np.savez(tmp_file, X=X, y_l1=y_l1, y_l2=y_l2)
    os.replace(tmp_file, cache_file)
    return X, y_l1, y_l2


# ─── Extract ──────────────────────────────────────────
X_train, y1_train, y2_train = extract_embeddings("train", os.path.join(DATA_DIR, "train_split.jsonl"))
X_val,   y1_val,   y2_val   = extract_embeddings("val",   os.path.join(DATA_DIR, "val_split.jsonl"))
X_test,  y1_test,  y2_test  = extract_embeddings("test",  os.path.join(DATA_DIR, "test_split.jsonl"))

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ─── Step 2: Train XGBoost per L1 class ───────────────
print("\n=== Training XGBoost v2 per L1 class ===")

USE_GPU = torch.cuda.is_available()
MAX_SAMPLES = 100_000

l1_classes = sorted(set(y1_train))
results_per_class = {}
xgb_models = {}
label_maps = {}

mask_train = y2_train != None
mask_val = y2_val != None
mask_test = y2_test != None

class_sizes = {c: (mask_train & (y1_train == c)).sum() for c in l1_classes}
l1_classes_sorted = sorted(l1_classes, key=lambda c: class_sizes[c])

for idx, l1_class in enumerate(l1_classes_sorted):
    tr_mask = mask_train & (y1_train == l1_class)
    va_mask = mask_val & (y1_val == l1_class)
    te_mask = mask_test & (y1_test == l1_class)

    n_tr = tr_mask.sum()
    if n_tr < 10:
        continue

    subcats = sorted(set(y2_train[tr_mask]))
    if len(subcats) < 2:
        label_maps[l1_class] = {"single": subcats[0] if subcats else None}
        continue

    sub2id = {s: i for i, s in enumerate(subcats)}
    id2sub = {i: s for s, i in sub2id.items()}
    label_maps[l1_class] = {"sub2id": sub2id, "id2sub": id2sub}

    Xtr = X_train[tr_mask].astype(np.float32)
    ytr = np.array([sub2id[s] for s in y2_train[tr_mask]])

    # Subsample large classes
    if len(ytr) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        indices = []
        for cls_id in range(len(subcats)):
            cls_idx = np.where(ytr == cls_id)[0]
            n_keep = max(1, int(MAX_SAMPLES * len(cls_idx) / len(ytr)))
            if len(cls_idx) > n_keep:
                indices.extend(rng.choice(cls_idx, n_keep, replace=False).tolist())
            else:
                indices.extend(cls_idx.tolist())
        indices = np.array(indices)
        Xtr, ytr = Xtr[indices], ytr[indices]

    va_valid = va_mask & np.array([s in sub2id for s in y2_val], dtype=bool)
    te_valid = te_mask & np.array([s in sub2id for s in y2_test], dtype=bool)
    Xva = X_val[va_valid].astype(np.float32) if va_valid.sum() > 0 else None
    yva = np.array([sub2id[s] for s in y2_val[va_valid]]) if va_valid.sum() > 0 else None
    Xte = X_test[te_valid].astype(np.float32) if te_valid.sum() > 0 else None
    yte = np.array([sub2id[s] for s in y2_test[te_valid]]) if te_valid.sum() > 0 else None

    n_est = 300 if len(ytr) < 50_000 else 150

    clf = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(subcats),
        n_jobs=8,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
        verbosity=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
    )

    if Xva is not None and len(yva) > 0:
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        clf.set_params(early_stopping_rounds=None)
        clf.fit(Xtr, ytr)

    xgb_models[l1_class] = clf

    metrics = {"n_train": int(n_tr), "n_subcats": len(subcats), "n_train_used": len(ytr)}
    if Xva is not None and len(yva) > 0:
        pred = clf.predict(Xva)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        metrics["val_acc"] = round(float(accuracy_score(yva, pred)), 4)
        metrics["val_f1_macro"] = round(float(f1_score(yva, pred, average="macro", zero_division=0)), 4)
    if Xte is not None and len(yte) > 0:
        pred = clf.predict(Xte)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        metrics["test_acc"] = round(float(accuracy_score(yte, pred)), 4)
        metrics["test_f1_macro"] = round(float(f1_score(yte, pred, average="macro", zero_division=0)), 4)
    results_per_class[l1_class] = metrics

    if (idx + 1) % 10 == 0 or n_tr > 50000:
        va_str = f", val_acc={metrics.get('val_acc', 0):.3f}" if 'val_acc' in metrics else ""
        print(f"  [{idx+1}/{len(l1_classes_sorted)}] {l1_class}: "
              f"{n_tr} samples, {len(subcats)} subcats, used {len(ytr)}{va_str}")

# ─── Save ─────────────────────────────────────────────
print(f"\nSaving XGBoost v2 models to {OUTPUT_DIR}...")
with open(os.path.join(OUTPUT_DIR, "xgb_models.pkl"), "wb") as f:
    pickle.dump(xgb_models, f)
with open(os.path.join(OUTPUT_DIR, "label_maps.json"), "w") as f:
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
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump({
        "embedding_mode": EMB_MODE,
        "embedding_dim": X_train.shape[1],
        "n_models": len(xgb_models),
        "n_single_class": sum(1 for v in label_maps.values() if "single" in v),
    }, f, indent=2)

# ─── Summary ──────────────────────────────────────────
print(f"\n=== XGBoost v2 Summary ({EMB_MODE} embeddings) ===")
print(f"Models trained: {len(xgb_models)}")
val_accs = [v["val_acc"] for v in results_per_class.values() if "val_acc" in v]
test_accs = [v["test_acc"] for v in results_per_class.values() if "test_acc" in v]
if val_accs:
    print(f"Mean val accuracy:  {np.mean(val_accs):.4f}")
if test_accs:
    print(f"Mean test accuracy: {np.mean(test_accs):.4f}")
print("Done!")
