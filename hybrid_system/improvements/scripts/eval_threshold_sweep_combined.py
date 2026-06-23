#!/usr/bin/env python3
"""
Threshold Sweep (Combined) — BOTH improvements together.

Combines:
  #1 Focal CamemBERT L2  → used for the L2 PREDICTION  (checkpoints/l2_camembert_focal)
  #2 XGBoost v2 (concat) → used for the specialist path  (checkpoints/l2_xgboost_v2)

IMPORTANT — three CamemBERT models are loaded:
  - FOCAL (l2_camembert_focal): produces cam_pred / cam_conf  (the prediction)
  - RUN2  (l2_camembert_run2):  produces emb_l2 ONLY, the feature XGBoost v2 was
                                trained on (must match train time — focal emb would
                                be mismatched features)
  - L1    (l1_classifier):      router pred + emb_l1 (768d)
  XGBoost v2 feature = concat([emb_l1, emb_l2]) = 1536d

This is the full proposed system. Compare its max_conf row to:
  v1 baseline: acc=0.8354  f1m=0.6321  f1w=0.8312

Separate output files:
  - predictions_cache_combined.pkl
  - .sweep_checkpoint_combined.pkl
  - threshold_sweep_results_combined.json

Usage:
    python eval_threshold_sweep_combined.py              # full run (test split)
    python eval_threshold_sweep_combined.py --resume     # continue from checkpoint
    python eval_threshold_sweep_combined.py --sweep-only # skip collection, use cache
    python eval_threshold_sweep_combined.py --debug      # 10K sample test
    python eval_threshold_sweep_combined.py --split val  # run on VAL split (for calibration)

The --split flag selects which dataset split to evaluate (default: test). On the
val split the output files get a "_val" suffix so they never clash with the test
cache. The val prediction cache is what the calibrated decision layer is FIT on,
while the test cache is what it is EVALUATED on.
"""
import os
import re
import sys
import json
import time
import pickle
import numpy as np
import torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DEBUG      = "--debug"      in sys.argv
SWEEP_ONLY = "--sweep-only" in sys.argv
RESUME     = "--resume"     in sys.argv


def _parse_split(argv):
    """Parse --split val / --split=val (default: test). Only test/val allowed."""
    for i, a in enumerate(argv):
        if a == "--split" and i + 1 < len(argv):
            val = argv[i + 1]
            break
        if a.startswith("--split="):
            val = a.split("=", 1)[1]
            break
    else:
        return "test"
    if val not in ("test", "val"):
        raise SystemExit(f"--split must be 'test' or 'val', got '{val}'")
    return val


SPLIT  = _parse_split(sys.argv)

# ── XGBoost variant selection (v2 default / v3 / v3_balanced) ──────────
# XGB_DIR     : checkpoint dir holding xgb_models.pkl + label_maps.json
# XGB_ENG=1   : append the 10 engineered text features (v3 / v3_balanced)
# XGB_VARIANT : output-file tag, e.g. "v3", "v3_balanced" ("" = v2, default)
XGB_DIR_REL = os.environ.get("XGB_DIR", "checkpoints/l2_xgboost_v2")
XGB_ENG     = os.environ.get("XGB_ENG", "0") == "1"
VARIANT     = os.environ.get("XGB_VARIANT", "").strip()

# Output suffix: combine variant + split (test stays unsuffixed for back-compat).
_parts = [p for p in (VARIANT, (SPLIT if SPLIT != "test" else "")) if p]
SUFFIX = ("_" + "_".join(_parts)) if _parts else ""

# Engineered text features (must EXACTLY match train_xgboost_v3.py::_row_features)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


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
    uniq = (len(set(w.lower() for w in all_words)) / len(all_words)) if all_words else 0.0
    return (
        float(t_chars), float(len(t_words)), float(d_chars), float(len(d_words)),
        1.0 if desc else 0.0, float(t_chars + d_chars),
        n_digits / n_chars, n_upper / n_chars, n_punct / n_chars, uniq,
    )

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "dataset")
L1_CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints/l1_classifier")
L2_FOCAL_DIR     = os.path.join(BASE_DIR, "checkpoints/l2_camembert_focal")  # prediction
L2_RUN2_DIR      = os.path.join(BASE_DIR, "checkpoints/l2_camembert_run2")   # emb for XGB
XGBOOST_DIR      = os.path.join(BASE_DIR, XGB_DIR_REL)

CACHE_FILE      = os.path.join(BASE_DIR, f"predictions_cache_combined{SUFFIX}.pkl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, f".sweep_checkpoint_combined{SUFFIX}.pkl")

BATCH_SIZE       = 64
MAX_LEN_L1       = 128
MAX_LEN_L2       = 256
# Only run L1 + XGBoost for samples where CamemBERT is uncertain below this ceiling
MAX_THRESHOLD    = 0.9
CHECKPOINT_EVERY = 100_000   # save progress every N samples

device = "cuda" if torch.cuda.is_available() else "cpu"


def find_best_ckpt(ckpt_dir):
    best = os.path.join(ckpt_dir, "best")
    if os.path.isdir(best):
        return best
    ckpts = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir


# ══════════════════════════════════════════════════════
# PASS 1: Collect predictions
# ══════════════════════════════════════════════════════
if not SWEEP_ONLY:
    print(f"Device: {device}")
    print("=" * 60)
    print("PASS 1: Focal L2 (pred) + run2 (emb) + XGBoost v2 (concat)")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────
    print("Loading CamemBERT L2 FOCAL (predictions)...")
    l2_focal_dir = find_best_ckpt(L2_FOCAL_DIR)
    print(f"  Focal checkpoint: {l2_focal_dir}")
    try:
        l2_focal_tokenizer = AutoTokenizer.from_pretrained(l2_focal_dir)
    except OSError:
        l2_focal_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    l2_focal_model = AutoModelForSequenceClassification.from_pretrained(l2_focal_dir).to(device)
    l2_focal_model.eval()
    l2_focal_id2label = {int(k): v for k, v in l2_focal_model.config.id2label.items()}

    print("Loading CamemBERT L2 RUN2 (embeddings for XGBoost v2)...")
    l2_run2_dir = find_best_ckpt(L2_RUN2_DIR)
    try:
        l2_run2_tokenizer = AutoTokenizer.from_pretrained(l2_run2_dir)
    except OSError:
        l2_run2_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    l2_run2_model = AutoModelForSequenceClassification.from_pretrained(l2_run2_dir).to(device)
    l2_run2_model.eval()

    print("Loading L1 model (classifier + CLS embed)...")
    l1_model_dir = find_best_ckpt(L1_CKPT_DIR)
    try:
        l1_tokenizer = AutoTokenizer.from_pretrained(l1_model_dir)
    except OSError:
        l1_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    l1_model = AutoModelForSequenceClassification.from_pretrained(l1_model_dir).to(device)
    l1_model.eval()
    l1_id2label = {int(k): v for k, v in l1_model.config.id2label.items()}

    print("Loading XGBoost v2 (concat)...")
    with open(os.path.join(XGBOOST_DIR, "xgb_models.pkl"), "rb") as f:
        xgb_models = pickle.load(f)
    with open(os.path.join(XGBOOST_DIR, "label_maps.json")) as f:
        xgb_label_maps = json.load(f)
    xgb_available = set(xgb_models.keys())
    print(f"  XGBoost dir: {XGBOOST_DIR}  (engineered feats: {XGB_ENG}, variant tag: '{VARIANT or 'v2'}')")

    print(f"Loading {SPLIT} data...")
    ds = load_dataset("json", data_files={SPLIT: os.path.join(DATA_DIR, f"{SPLIT}_split.jsonl")})
    test_ds = ds[SPLIT]
    if DEBUG:
        test_ds = test_ds.select(range(min(10_000, len(test_ds))))
    n = len(test_ds)
    print(f"  {SPLIT.capitalize()} samples: {n:,}")

    # ── Inference helpers ─────────────────────────────────
    @torch.no_grad()
    def predict_l2_focal_batch(texts):
        """Focal L2 prediction + confidence."""
        inputs = l2_focal_tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN_L2, return_tensors="pt"
        ).to(device)
        logits = l2_focal_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        labels = [l2_focal_id2label.get(int(p), "UNKNOWN") for p in preds.cpu()]
        return labels, confs.cpu().numpy()

    @torch.no_grad()
    def embed_l2_run2_batch(texts):
        """RUN2 L2 CLS embedding (the feature XGBoost v2 was trained on)."""
        inputs = l2_run2_tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN_L2, return_tensors="pt"
        ).to(device)
        outputs = l2_run2_model(**inputs, output_hidden_states=True)
        emb_l2 = outputs.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float32)
        return emb_l2

    @torch.no_grad()
    def predict_l1_and_embed_batch(texts):
        """L1 labels, L1 confs, L1 CLS embeddings (768d)."""
        inputs = l1_tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN_L1, return_tensors="pt"
        ).to(device)
        outputs = l1_model(**inputs, output_hidden_states=True)
        probs = torch.softmax(outputs.logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        emb_l1 = outputs.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float32)
        labels = [l1_id2label.get(int(p), "UNKNOWN") for p in preds.cpu()]
        return labels, confs.cpu().numpy(), emb_l1

    def xgb_predict_batch(l1_class, features):
        """Batch XGBoost v2 prediction (features = 1536d concat) for one L1 class."""
        xgb_lm = xgb_label_maps.get(l1_class, {})
        if "single" in xgb_lm:
            pred = xgb_lm["single"] or "UNKNOWN"
            return [pred] * len(features), [1.0] * len(features)
        proba = xgb_models[l1_class].predict_proba(features)
        pred_idxs = np.argmax(proba, axis=1)
        batch_confs = np.max(proba, axis=1)
        id2sub = xgb_lm.get("id2sub", {})
        preds = [id2sub.get(str(idx), "UNKNOWN") for idx in pred_idxs]
        return preds, batch_confs.tolist()

    # ── Resume from checkpoint if requested ───────────────
    all_cam_preds = []
    all_cam_confs = []
    all_xgb_preds = []
    all_xgb_confs = []
    all_true      = []
    start_sample  = 0

    if RESUME and os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from checkpoint {CHECKPOINT_FILE}...")
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        all_cam_preds = ckpt["cam_preds"]
        all_cam_confs = ckpt["cam_confs"]
        all_xgb_preds = ckpt["xgb_preds"]
        all_xgb_confs = ckpt["xgb_confs"]
        all_true      = ckpt["true_labels"]
        start_sample  = len(all_true)
        print(f"  Resumed at sample {start_sample:,} / {n:,} ({100*start_sample/n:.1f}%)")
    elif os.path.exists(CHECKPOINT_FILE):
        print("  (checkpoint exists but --resume not set — starting fresh)")

    # ── Main collection loop ──────────────────────────────
    start = time.time()
    total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    start_batch   = start_sample // BATCH_SIZE

    for batch_start in tqdm(range(start_sample, n, BATCH_SIZE),
                            desc="Collecting", initial=start_batch, total=total_batches):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch = test_ds[batch_start:batch_end]
        bsz = batch_end - batch_start

        # Build text inputs
        descs = batch.get("description") or [""] * bsz
        texts_l2 = []
        for i in range(bsz):
            title = batch["designation"][i] or ""
            desc  = descs[i] or ""
            texts_l2.append(f"{title} {desc}"[:1024] if desc else title)
        texts_l1 = [batch["designation"][i] or "" for i in range(bsz)]

        true_l2 = [
            batch["producttype"][i][0] if batch["producttype"][i] else ""
            for i in range(bsz)
        ]

        # ─ Step 1: FOCAL L2 prediction for ALL samples ──────
        cam_labels, cam_confs = predict_l2_focal_batch(texts_l2)

        # ─ Step 2: L1 + XGBoost v2 only for uncertain samples ─
        uncertain_idx = [i for i in range(bsz) if cam_confs[i] < MAX_THRESHOLD]

        xgb_preds_batch = list(cam_labels)  # default = cam prediction
        xgb_confs_batch = [0.0] * bsz       # 0.0 → XGBoost not run / unavailable

        if uncertain_idx:
            unc_texts_l1 = [texts_l1[i] for i in uncertain_idx]
            unc_texts_l2 = [texts_l2[i] for i in uncertain_idx]
            l1_labels, _, emb_l1_unc = predict_l1_and_embed_batch(unc_texts_l1)
            # emb_l2 from RUN2 (matches XGBoost v2 train-time features)
            emb_l2_unc = embed_l2_run2_batch(unc_texts_l2)
            feats_unc  = np.concatenate([emb_l1_unc, emb_l2_unc], axis=1)  # 1536d
            if XGB_ENG:
                # Append engineered text features (v3 / v3_balanced) -> 1546d.
                eng_unc = np.array(
                    [_row_features(batch["designation"][i], descs[i]) for i in uncertain_idx],
                    dtype=np.float32,
                )
                feats_unc = np.concatenate([feats_unc, eng_unc], axis=1)

            # Group by L1 class for batched XGBoost inference
            l1_groups: dict = defaultdict(list)
            for local_i, global_i in enumerate(uncertain_idx):
                l1_class = l1_labels[local_i]
                if l1_class in xgb_available:
                    l1_groups[l1_class].append((local_i, global_i))

            for l1_class, pairs in l1_groups.items():
                local_idxs  = [p[0] for p in pairs]
                global_idxs = [p[1] for p in pairs]
                feat_batch  = feats_unc[local_idxs]
                preds, confs = xgb_predict_batch(l1_class, feat_batch)
                for gi, pred, conf in zip(global_idxs, preds, confs):
                    xgb_preds_batch[gi] = pred
                    xgb_confs_batch[gi] = conf

        all_cam_preds.extend(cam_labels)
        all_cam_confs.extend(cam_confs.tolist())
        all_xgb_preds.extend(xgb_preds_batch)
        all_xgb_confs.extend(xgb_confs_batch)
        all_true.extend(true_l2)

        # ─ Checkpoint every CHECKPOINT_EVERY samples ──────────
        processed = len(all_true)
        if processed % CHECKPOINT_EVERY < BATCH_SIZE:
            ckpt = {
                "cam_preds":   all_cam_preds,
                "cam_confs":   all_cam_confs,
                "xgb_preds":   all_xgb_preds,
                "xgb_confs":   all_xgb_confs,
                "true_labels": all_true,
            }
            tmp = CHECKPOINT_FILE + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(ckpt, f)
            os.replace(tmp, CHECKPOINT_FILE)
            elapsed_so_far = time.time() - start
            rate = (processed - start_sample) / elapsed_so_far
            eta  = (n - processed) / rate if rate > 0 else 0
            print(f"\n  [Checkpoint] {processed:,}/{n:,} samples "
                  f"({100*processed/n:.1f}%) — ETA {eta/3600:.1f}h")

    elapsed = time.time() - start
    print(f"\nCollection done in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")

    # Save final cache
    cache = {
        "cam_preds":   all_cam_preds,
        "cam_confs":   all_cam_confs,
        "xgb_preds":   all_xgb_preds,
        "xgb_confs":   all_xgb_confs,
        "true_labels": all_true,
        "n_samples":   n,
    }
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, CACHE_FILE)
    print(f"Cache saved to {CACHE_FILE} ({os.path.getsize(CACHE_FILE)/1e6:.1f} MB)")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

else:
    print("Loading cached predictions...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    all_cam_preds = cache["cam_preds"]
    all_cam_confs = cache["cam_confs"]
    all_xgb_preds = cache["xgb_preds"]
    all_xgb_confs = cache["xgb_confs"]
    all_true      = cache["true_labels"]
    n             = cache["n_samples"]
    print(f"  Loaded {n:,} predictions from cache")


# ══════════════════════════════════════════════════════
# PASS 2: Sweep thresholds + modes
# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PASS 2: Threshold Sweep (Focal L2 + XGBoost v2)")
print(f"{'='*60}\n")

cam_preds   = np.array(all_cam_preds)
cam_confs   = np.array(all_cam_confs,  dtype=np.float32)
xgb_preds   = np.array(all_xgb_preds)
xgb_confs   = np.array(all_xgb_confs,  dtype=np.float32)
true_labels = np.array(all_true)

# XGBoost available where conf > 0 (only collected for uncertain samples)
xgb_has_model = xgb_confs > 0.0

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
modes      = ["gated", "always_xgb", "max_conf"]

results = []

print(f"{'Threshold':<10} {'Mode':<12} {'Accuracy':>10} {'F1-macro':>10} {'F1-wt':>10} {'%XGB':>8} {'%CAM':>8}")
print("─" * 70)

for threshold in thresholds:
    for mode in modes:
        final_preds = cam_preds.copy()
        uncertain = cam_confs < threshold

        if mode == "gated":
            override_mask = uncertain & xgb_has_model & (xgb_confs > cam_confs)
        elif mode == "always_xgb":
            override_mask = uncertain & xgb_has_model
        elif mode == "max_conf":
            override_mask = xgb_has_model & (xgb_confs > cam_confs)

        final_preds[override_mask] = xgb_preds[override_mask]

        acc  = accuracy_score(true_labels, final_preds)
        f1_m = f1_score(true_labels, final_preds, average="macro",    zero_division=0)
        f1_w = f1_score(true_labels, final_preds, average="weighted", zero_division=0)
        pct_xgb = 100.0 * override_mask.sum() / n
        pct_cam = 100.0 - pct_xgb

        print(f"  {threshold:<8} {mode:<12} {acc:>10.4f} {f1_m:>10.4f} {f1_w:>10.4f} {pct_xgb:>7.1f}% {pct_cam:>7.1f}%")
        results.append({
            "threshold":   threshold,
            "mode":        mode,
            "accuracy":    round(acc,  5),
            "f1_macro":    round(f1_m, 5),
            "f1_weighted": round(f1_w, 5),
            "pct_xgb":     round(pct_xgb, 2),
            "pct_cam":     round(pct_cam,  2),
        })

# Find best
best_by_acc = max(results, key=lambda x: x["accuracy"])
best_by_f1m = max(results, key=lambda x: x["f1_macro"])

print(f"\n{'─'*70}")
print(f"BEST by Accuracy:  threshold={best_by_acc['threshold']}  mode={best_by_acc['mode']}")
print(f"  acc={best_by_acc['accuracy']:.4f}  f1m={best_by_acc['f1_macro']:.4f}  f1w={best_by_acc['f1_weighted']:.4f}")
print(f"\nBEST by F1-macro:  threshold={best_by_f1m['threshold']}  mode={best_by_f1m['mode']}")
print(f"  acc={best_by_f1m['accuracy']:.4f}  f1m={best_by_f1m['f1_macro']:.4f}  f1w={best_by_f1m['f1_weighted']:.4f}")

# Baselines
cam_only_acc = accuracy_score(true_labels, cam_preds)
cam_only_f1m = f1_score(true_labels, cam_preds, average="macro",    zero_division=0)
cam_only_f1w = f1_score(true_labels, cam_preds, average="weighted", zero_division=0)

xgb_only_preds = np.where(xgb_has_model, xgb_preds, cam_preds)
xgb_only_acc = accuracy_score(true_labels, xgb_only_preds)
xgb_only_f1m = f1_score(true_labels, xgb_only_preds, average="macro",    zero_division=0)
xgb_only_f1w = f1_score(true_labels, xgb_only_preds, average="weighted", zero_division=0)

print(f"\n{'─'*70}")
print("BASELINES:")
print(f"  Focal CamemBERT-only: acc={cam_only_acc:.4f}  f1m={cam_only_f1m:.4f}  f1w={cam_only_f1w:.4f}")
print(f"  XGBoost v2-only:      acc={xgb_only_acc:.4f}  f1m={xgb_only_f1m:.4f}  f1w={xgb_only_f1w:.4f}")
print(f"  Best hybrid:          acc={best_by_acc['accuracy']:.4f}  f1m={best_by_acc['f1_macro']:.4f}  f1w={best_by_acc['f1_weighted']:.4f}")
print(f"  Δ vs Focal CamemBERT: +{100*(best_by_acc['accuracy']-cam_only_acc):.2f}% acc,  "
      f"+{100*(best_by_acc['f1_macro']-cam_only_f1m):.2f}% f1m")
print(f"\n  v1 baseline to beat: acc=0.8354  f1m=0.6321  f1w=0.8312")

# Save
output_file = os.path.join(BASE_DIR, f"threshold_sweep_results_combined{SUFFIX}.json")
with open(output_file, "w") as f:
    json.dump({
        "sweep_results":    results,
        "best_by_accuracy": best_by_acc,
        "best_by_f1_macro": best_by_f1m,
        "baselines": {
            "camembert_only": {"accuracy": cam_only_acc, "f1_macro": cam_only_f1m, "f1_weighted": cam_only_f1w},
            "xgboost_only":   {"accuracy": xgb_only_acc, "f1_macro": xgb_only_f1m, "f1_weighted": xgb_only_f1w},
        },
        "v1_baseline": {"accuracy": 0.8354, "f1_macro": 0.6321, "f1_weighted": 0.8312},
        "n_samples": n,
    }, f, indent=2)

print(f"\nResults saved to {output_file}")
print("Done!")
