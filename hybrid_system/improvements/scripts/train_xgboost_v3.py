#!/usr/bin/env python3
"""
XGBoost v3 — v2 concat embeddings (1536d) + cheap engineered text features.

Improvement #3. Trees benefit from explicit, non-linear signals that are hard to
read off a CLS embedding: text lengths, digit/uppercase ratios, whether a
description exists, etc. We APPEND these to the exact 1536d [emb_l1, emb_l2]
features XGBoost v2 already used, so v3 is a strict superset of v2.

Reuses the cached v2 embeddings (embeddings/{split}_emb_concat.npz) — NO GPU is
needed for feature extraction, only (optionally) for XGBoost training. If the
concat caches are missing, run train_xgboost_v2.py once first to create them.

Engineered features (per sample, computed from the raw text, CPU-only):
  title_len_chars, title_len_words,
  desc_len_chars,  desc_len_words,
  has_desc, total_len_chars,
  digit_ratio, upper_ratio, punct_ratio, n_tokens_unique_ratio

Output: checkpoints/l2_xgboost_v3/  (same layout as v2: xgb_models.pkl,
label_maps.json, results_per_class.json, config.json)

Rare-class booster (improvement #4)
-----------------------------------
Set BALANCE=1 to fit each per-L1 XGBoost with inverse-frequency sample weights
(w_subcat = (N / n_subcat) ** BALANCE_ALPHA, normalized to mean 1, capped at
BALANCE_CAP). This amplifies rare L2 subcategories the same way focal loss does
for the CamemBERT branch, directly targeting F1-macro. Output then goes to
checkpoints/l2_xgboost_v3_balanced/ so the two variants never overwrite.

  EMB_MODE is fixed to concat. Env knobs:
    BALANCE=0|1         (default 0)  enable rare-class weighting
    BALANCE_ALPHA=0.5   weighting strength (0 = none, 1 = full inverse-freq)
    BALANCE_CAP=10      max weight per sample after normalization
"""
import os
import re
import json
import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except Exception:
    USE_GPU = False

# ─── Config ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR    = os.path.join(PROJECT_DIR, "dataset")
EMB_DIR     = os.path.join(PROJECT_DIR, "embeddings")
EMB_MODE    = "concat"          # v3 builds on the 1536d concat features
MAX_SAMPLES = 100_000

# Rare-class booster (#4)
BALANCE       = os.environ.get("BALANCE", "0") == "1"
BALANCE_ALPHA = float(os.environ.get("BALANCE_ALPHA", "0.5"))
BALANCE_CAP   = float(os.environ.get("BALANCE_CAP", "10"))
OUTPUT_DIR    = os.path.join(
    PROJECT_DIR,
    "checkpoints/l2_xgboost_v3_balanced" if BALANCE else "checkpoints/l2_xgboost_v3",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ENG_FEATURES = [
    "title_len_chars", "title_len_words",
    "desc_len_chars", "desc_len_words",
    "has_desc", "total_len_chars",
    "digit_ratio", "upper_ratio", "punct_ratio", "uniq_word_ratio",
]
N_ENG = len(ENG_FEATURES)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


# ─── Engineered text features ─────────────────────────
def _row_features(title, desc):
    title = title or ""
    desc = desc or ""
    t_chars = len(title)
    t_words = title.split()
    d_chars = len(desc)
    d_words = desc.split()
    combined = f"{title} {desc}".strip()
    n_chars = len(combined) or 1
    n_digits = sum(c.isdigit() for c in combined)
    n_upper = sum(c.isupper() for c in combined)
    n_punct = len(_PUNCT_RE.findall(combined))
    all_words = t_words + d_words
    uniq_ratio = (len(set(w.lower() for w in all_words)) / len(all_words)) if all_words else 0.0
    return (
        float(t_chars),
        float(len(t_words)),
        float(d_chars),
        float(len(d_words)),
        1.0 if desc else 0.0,
        float(t_chars + d_chars),
        n_digits / n_chars,
        n_upper / n_chars,
        n_punct / n_chars,
        uniq_ratio,
    )


def engineered_features(split_name, jsonl_path):
    """Compute engineered features in dataset order (aligns with cached embeddings)."""
    cache = os.path.join(EMB_DIR, f"{split_name}_eng_feats.npz")
    if os.path.isfile(cache):
        print(f"Loading cached engineered features: {cache}")
        return np.load(cache)["F"]

    print(f"Computing engineered features for {split_name}...")
    ds = load_dataset("json", data_files={split_name: jsonl_path})[split_name]
    n = len(ds)
    F = np.empty((n, N_ENG), dtype=np.float32)
    BS = 10_000
    pos = 0
    for i in tqdm(range(0, n, BS), desc=f"eng {split_name}"):
        batch = ds[i:i + BS]
        titles = batch["designation"]
        descs = batch.get("description") or [None] * len(titles)
        for j in range(len(titles)):
            F[pos] = _row_features(titles[j], descs[j] if j < len(descs) else None)
            pos += 1
    F = F[:pos]
    tmp = cache + ".tmp.npz"
    np.savez(tmp, F=F)
    os.replace(tmp, cache)
    return F


def load_concat_embeddings(split_name):
    cache = os.path.join(EMB_DIR, f"{split_name}_emb_{EMB_MODE}.npz")
    if not os.path.isfile(cache):
        raise SystemExit(
            f"ERROR: missing {cache}\n"
            "Run improvements/scripts/train_xgboost_v2.py once to create the "
            "concat embedding caches, then re-run v3."
        )
    print(f"Loading cached embeddings: {cache}")
    data = np.load(cache, allow_pickle=True)
    return data["X"], data["y_l1"], data["y_l2"]


def build_features(split_name, jsonl_path):
    X_emb, y_l1, y_l2 = load_concat_embeddings(split_name)
    F_eng = engineered_features(split_name, jsonl_path)
    assert X_emb.shape[0] == F_eng.shape[0], (
        f"row mismatch for {split_name}: emb={X_emb.shape[0]} eng={F_eng.shape[0]} "
        "(embedding cache and dataset are out of sync — re-extract)"
    )
    # keep memory low: cast emb to float32 only when training each class
    return X_emb, F_eng, y_l1, y_l2


# ─── Build feature matrices ───────────────────────────
print("=" * 60)
print("XGBoost v3 — concat embeddings + engineered features")
print(f"  engineered: {ENG_FEATURES}")
print(f"  device for XGBoost: {'cuda' if USE_GPU else 'cpu'}")
if BALANCE:
    print(f"  rare-class booster ON  (alpha={BALANCE_ALPHA}, cap={BALANCE_CAP}) -> {OUTPUT_DIR}")
print("=" * 60)

Xe_train, Fe_train, y1_train, y2_train = build_features("train", os.path.join(DATA_DIR, "train_split.jsonl"))
Xe_val,   Fe_val,   y1_val,   y2_val   = build_features("val",   os.path.join(DATA_DIR, "val_split.jsonl"))
Xe_test,  Fe_test,  y1_test,  y2_test  = build_features("test",  os.path.join(DATA_DIR, "test_split.jsonl"))

emb_dim = Xe_train.shape[1]
print(f"\nFeature dim: {emb_dim} (emb) + {N_ENG} (eng) = {emb_dim + N_ENG}")
print(f"Train: {Xe_train.shape[0]:,}  Val: {Xe_val.shape[0]:,}  Test: {Xe_test.shape[0]:,}")


def stack(Xe, Fe, mask):
    """Concatenate embedding + engineered features for a row mask, as float32."""
    return np.concatenate(
        [Xe[mask].astype(np.float32), Fe[mask].astype(np.float32)], axis=1
    )


# ─── Train per-L1 XGBoost ─────────────────────────────
print("\n=== Training XGBoost v3 per L1 class ===")
mask_train = y2_train != None
mask_val   = y2_val   != None
mask_test  = y2_test  != None

l1_classes = sorted(set(y1_train))
class_sizes = {c: (mask_train & (y1_train == c)).sum() for c in l1_classes}
l1_classes_sorted = sorted(l1_classes, key=lambda c: class_sizes[c])

results_per_class = {}
xgb_models = {}
label_maps = {}

for idx, l1_class in enumerate(l1_classes_sorted):
    tr_mask = mask_train & (y1_train == l1_class)
    va_mask = mask_val   & (y1_val   == l1_class)
    te_mask = mask_test  & (y1_test  == l1_class)

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

    Xtr = stack(Xe_train, Fe_train, tr_mask)
    ytr = np.array([sub2id[s] for s in y2_train[tr_mask]])

    # Subsample large classes (stratified) — identical policy to v2
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

    # Rare-class sample weights (#4): inverse subcategory frequency, normalized.
    sample_weight = None
    if BALANCE:
        counts = np.bincount(ytr, minlength=len(subcats)).astype(np.float64)
        counts[counts == 0] = 1.0
        cls_w = (len(ytr) / counts) ** BALANCE_ALPHA
        cls_w = cls_w / cls_w.mean()                 # mean weight = 1
        cls_w = np.clip(cls_w, None, BALANCE_CAP)
        sample_weight = cls_w[ytr].astype(np.float32)

    va_valid = va_mask & np.array([s in sub2id for s in y2_val], dtype=bool)
    te_valid = te_mask & np.array([s in sub2id for s in y2_test], dtype=bool)
    Xva = stack(Xe_val,  Fe_val,  va_valid) if va_valid.sum() > 0 else None
    yva = np.array([sub2id[s] for s in y2_val[va_valid]])  if va_valid.sum() > 0 else None
    Xte = stack(Xe_test, Fe_test, te_valid) if te_valid.sum() > 0 else None
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
        clf.fit(Xtr, ytr, sample_weight=sample_weight, eval_set=[(Xva, yva)], verbose=False)
    else:
        clf.set_params(early_stopping_rounds=None)
        clf.fit(Xtr, ytr, sample_weight=sample_weight)

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
print(f"\nSaving XGBoost v3 models to {OUTPUT_DIR}...")
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
        "feature_dim": emb_dim + N_ENG,
        "engineered_features": ENG_FEATURES,
        "n_models": len(xgb_models),
        "max_samples_per_class": MAX_SAMPLES,
        "balance": BALANCE,
        "balance_alpha": BALANCE_ALPHA if BALANCE else None,
        "balance_cap": BALANCE_CAP if BALANCE else None,
        "base": "xgboost_v2 (1536d concat) + engineered text features"
                + (" + inverse-freq class weights" if BALANCE else ""),
    }, f, ensure_ascii=False, indent=2)

# ─── Quick aggregate report ───────────────────────────
test_accs = [m["test_acc"] for m in results_per_class.values() if "test_acc" in m]
if test_accs:
    print(f"\nMean per-class test acc: {np.mean(test_accs):.4f} over {len(test_accs)} classes")
print(f"Saved {len(xgb_models)} models. Done.")
