#!/usr/bin/env python3
"""
Threshold Sweep (Focal) — evaluates the hybrid with the FOCAL-trained CamemBERT L2.

Isolates improvement #1 (focal loss + label smoothing):
  - L2 predictor = checkpoints/l2_camembert_focal  (checkpoint-65316 via find_best_ckpt)
  - XGBoost      = checkpoints/l2_xgboost (v1, L1-only 768d features — unchanged)

XGBoost v1 features are the L1 encoder CLS only (768d), independent of the L2 model,
so swapping the L2 model does not affect the XGBoost path.

Separate output files (so v1 / v2 sweep artifacts are untouched):
  - predictions_cache_focal.pkl
  - .sweep_checkpoint_focal.pkl
  - threshold_sweep_results_focal.json

Usage:
    python eval_threshold_sweep_focal.py              # full run
    python eval_threshold_sweep_focal.py --resume     # continue from checkpoint
    python eval_threshold_sweep_focal.py --sweep-only # skip collection, use cache
    python eval_threshold_sweep_focal.py --debug      # 10K sample test
"""
import os
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

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "dataset")
L1_CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints/l1_classifier")
L2_CAMEMBERT_DIR = os.path.join(BASE_DIR, "checkpoints/l2_camembert_focal")
XGBOOST_DIR      = os.path.join(BASE_DIR, "checkpoints/l2_xgboost")

CACHE_FILE      = os.path.join(BASE_DIR, "predictions_cache_focal.pkl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, ".sweep_checkpoint_focal.pkl")

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
    print("PASS 1: Collecting Focal-CamemBERT + XGBoost v1 predictions")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────
    print("Loading CamemBERT L2 (focal)...")
    l2_cam_dir = find_best_ckpt(L2_CAMEMBERT_DIR)
    print(f"  L2 checkpoint: {l2_cam_dir}")
    try:
        l2_cam_tokenizer = AutoTokenizer.from_pretrained(l2_cam_dir)
    except OSError:
        l2_cam_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    l2_cam_model = AutoModelForSequenceClassification.from_pretrained(l2_cam_dir).to(device)
    l2_cam_model.eval()
    l2_cam_id2label = {int(k): v for k, v in l2_cam_model.config.id2label.items()}

    print("Loading L1 model (classifier + CLS embed via output_hidden_states)...")
    l1_model_dir = find_best_ckpt(L1_CKPT_DIR)
    try:
        l1_tokenizer = AutoTokenizer.from_pretrained(l1_model_dir)
    except OSError:
        l1_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    l1_model = AutoModelForSequenceClassification.from_pretrained(l1_model_dir).to(device)
    l1_model.eval()
    l1_id2label = {int(k): v for k, v in l1_model.config.id2label.items()}

    print("Loading XGBoost v1...")
    with open(os.path.join(XGBOOST_DIR, "xgb_models.pkl"), "rb") as f:
        xgb_models = pickle.load(f)
    with open(os.path.join(XGBOOST_DIR, "label_maps.json")) as f:
        xgb_label_maps = json.load(f)
    xgb_available = set(xgb_models.keys())

    print("Loading test data...")
    ds = load_dataset("json", data_files={"test": os.path.join(DATA_DIR, "test_split.jsonl")})
    test_ds = ds["test"]
    if DEBUG:
        test_ds = test_ds.select(range(min(10_000, len(test_ds))))
    n = len(test_ds)
    print(f"  Test samples: {n:,}")

    # ── Inference helpers ─────────────────────────────────
    @torch.no_grad()
    def predict_l2_cam_batch(texts):
        inputs = l2_cam_tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN_L2, return_tensors="pt"
        ).to(device)
        logits = l2_cam_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        labels = [l2_cam_id2label.get(int(p), "UNKNOWN") for p in preds.cpu()]
        return labels, confs.cpu().numpy()

    @torch.no_grad()
    def predict_l1_and_embed_batch(texts):
        """Single forward pass: L1 labels, L1 confs, CLS embeddings (768d, v1 features)."""
        inputs = l1_tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN_L1, return_tensors="pt"
        ).to(device)
        outputs = l1_model(**inputs, output_hidden_states=True)
        probs = torch.softmax(outputs.logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float32)
        labels = [l1_id2label.get(int(p), "UNKNOWN") for p in preds.cpu()]
        return labels, confs.cpu().numpy(), embeddings

    def xgb_predict_batch(l1_class, embeddings):
        """Batch XGBoost v1 prediction (768d) for all samples of the same L1 class."""
        xgb_lm = xgb_label_maps.get(l1_class, {})
        if "single" in xgb_lm:
            pred = xgb_lm["single"] or "UNKNOWN"
            return [pred] * len(embeddings), [1.0] * len(embeddings)
        proba = xgb_models[l1_class].predict_proba(embeddings)
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

        # ─ Step 1: Focal CamemBERT L2 for ALL samples ───────
        cam_labels, cam_confs = predict_l2_cam_batch(texts_l2)

        # ─ Step 2: L1 + XGBoost only for uncertain samples ──
        uncertain_idx = [i for i in range(bsz) if cam_confs[i] < MAX_THRESHOLD]

        xgb_preds_batch = list(cam_labels)  # default = cam prediction
        xgb_confs_batch = [0.0] * bsz       # 0.0 → XGBoost not run / unavailable

        if uncertain_idx:
            unc_texts = [texts_l1[i] for i in uncertain_idx]
            l1_labels, _, embeddings = predict_l1_and_embed_batch(unc_texts)

            # Group by L1 class for batched XGBoost inference
            l1_groups: dict = defaultdict(list)
            for local_i, global_i in enumerate(uncertain_idx):
                l1_class = l1_labels[local_i]
                if l1_class in xgb_available:
                    l1_groups[l1_class].append((local_i, global_i))

            for l1_class, pairs in l1_groups.items():
                local_idxs  = [p[0] for p in pairs]
                global_idxs = [p[1] for p in pairs]
                emb_batch   = embeddings[local_idxs]
                preds, confs = xgb_predict_batch(l1_class, emb_batch)
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
print("PASS 2: Threshold Sweep (Focal L2 + XGBoost v1)")
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
print(f"  XGBoost-only:         acc={xgb_only_acc:.4f}  f1m={xgb_only_f1m:.4f}  f1w={xgb_only_f1w:.4f}")
print(f"  Best hybrid:          acc={best_by_acc['accuracy']:.4f}  f1m={best_by_acc['f1_macro']:.4f}  f1w={best_by_acc['f1_weighted']:.4f}")
print(f"  Δ vs Focal CamemBERT: +{100*(best_by_acc['accuracy']-cam_only_acc):.2f}% acc,  "
      f"+{100*(best_by_acc['f1_macro']-cam_only_f1m):.2f}% f1m")
print(f"\n  v1 baseline to beat: acc=0.8354  f1m=0.6321  f1w=0.8312")

# Save
output_file = os.path.join(BASE_DIR, "threshold_sweep_results_focal.json")
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
