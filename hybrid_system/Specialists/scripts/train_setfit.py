#!/usr/bin/env python3
"""
Train SetFit models for rare L2 subcategories.

Architecture:
- One SetFit model per L1 class that has rare subcats (< T_rare samples)
- Each model classifies only the RARE subcats within that L1 class
- Uses contrastive learning (sentence-transformers) + logistic head

Usage:
    python train_setfit.py                  # full run (GPU)
    python train_setfit.py --debug          # test with 2 classes
"""
import os
import sys
import json
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from datasets import Dataset

# ─── Monkey-patch setfit compatibility issues ─────────
# Fix 1: huggingface_hub removed DatasetFilter in v0.21+
import huggingface_hub
if not hasattr(huggingface_hub, "DatasetFilter"):
    huggingface_hub.DatasetFilter = None

# Fix 2: setfit model_card.py calls samples.sort(key=len) on a datasets Column
import setfit.model_card as _mc
_orig_set_widget = getattr(_mc.SetFitModelCardData, "set_widget_examples", None)
if _orig_set_widget:
    def _patched_set_widget(self, dataset):
        try:
            _orig_set_widget(self, dataset)
        except (AttributeError, TypeError):
            pass  # Skip model card widget — not needed for training
    _mc.SetFitModelCardData.set_widget_examples = _patched_set_widget

from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import accuracy_score, f1_score
import unicodedata

DEBUG = "--debug" in sys.argv

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints/setfit_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SetFit hyperparams
ST_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NUM_ITERATIONS = 20       # contrastive pairs per (anchor, pos/neg)
NUM_EPOCHS_BODY = 1       # epochs for sentence transformer body
NUM_EPOCHS_HEAD = 5       # epochs for classification head
BATCH_SIZE = 16
MAX_SAMPLES_PER_CLASS = 16  # max few-shot samples per rare subcat

# ─── Load gating config ──────────────────────────────
with open(os.path.join(ANALYSIS_DIR, "gating_config.json")) as f:
    gating = json.load(f)

T_RARE = gating["thresholds"]["T_rare"]
routing = gating["routing"]
subcat_routing = gating["subcat_routing"]

# Get L1 classes that need SetFit
setfit_classes = [c for c, route in routing.items() if route == "camembert_setfit"]
print(f"L1 classes needing SetFit: {len(setfit_classes)}")

if DEBUG:
    setfit_classes = setfit_classes[:2]
    print(f"DEBUG mode: only training {len(setfit_classes)} classes")


# ─── Encoding normalization (merge duplicates) ───────
def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


# Build canonical name mapping for rare subcats
rare_canonical = {}  # normalized → canonical form (first seen in gating config)
for l1_class in setfit_classes:
    for subcat in subcat_routing[l1_class]["rare_subcats"]:
        norm = normalize(subcat)
        if norm not in rare_canonical:
            rare_canonical[norm] = subcat


# ─── Load dataset ────────────────────────────────────
print("Loading dataset...")
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train_split.jsonl"),
    "val": os.path.join(DATA_DIR, "val_split.jsonl"),
    "test": os.path.join(DATA_DIR, "test_split.jsonl"),
})

# ─── Group samples by L1 class → rare subcats ────────
print("Grouping samples by L1 class and subcat...")


def get_text(example):
    """Combine designation + description."""
    title = example.get("designation", "") or ""
    desc = example.get("description", "") or ""
    if desc:
        return f"{title} {desc}"[:512]
    return title


def is_rare_subcat(l1_class, subcat):
    """Check if a subcat is in the rare list for this L1 class."""
    if l1_class not in subcat_routing:
        return False
    norm = normalize(subcat)
    for rare_s in subcat_routing[l1_class]["rare_subcats"]:
        if normalize(rare_s) == norm:
            return True
    return False


# Collect train and val data for rare subcats per L1 class
train_data_per_class = defaultdict(lambda: defaultdict(list))  # l1 → subcat → [texts]
val_data_per_class = defaultdict(lambda: defaultdict(list))

for split_name, split_ds, target_dict in [
    ("train", ds["train"], train_data_per_class),
    ("val", ds["val"], val_data_per_class),
]:
    print(f"  Scanning {split_name} split ({len(split_ds)} samples)...")
    for example in split_ds:
        l1_class = example.get("typelabel", "")
        if l1_class not in setfit_classes:
            continue
        subcat = example["producttype"][0] if example.get("producttype") else ""
        if not subcat:
            continue
        if is_rare_subcat(l1_class, subcat):
            # Normalize subcat name to canonical form
            norm = normalize(subcat)
            canonical = rare_canonical.get(norm, subcat)
            text = get_text(example)
            if text.strip():
                target_dict[l1_class][canonical].append(text)

# ─── Train one SetFit model per L1 class ─────────────
print(f"\n{'='*60}")
print(f"Training SetFit models for {len(setfit_classes)} L1 classes")
print(f"{'='*60}\n")

results = {}
failed = []
start_total = time.time()

for i, l1_class in enumerate(setfit_classes):
    start_class = time.time()

    # Skip if already trained (resume support)
    model_dir = os.path.join(OUTPUT_DIR, l1_class.replace("/", "_").replace(" ", "_"))
    if os.path.isfile(os.path.join(model_dir, "label_mapping.json")):
        print(f"[{i+1}/{len(setfit_classes)}] SKIP {l1_class}: already trained")
        continue

    rare_subcats_data = train_data_per_class[l1_class]
    val_subcats_data = val_data_per_class[l1_class]

    # Filter: need at least 2 subcats with at least 2 samples each
    valid_subcats = {s: texts for s, texts in rare_subcats_data.items() if len(texts) >= 2}

    if len(valid_subcats) < 2:
        print(f"[{i+1}/{len(setfit_classes)}] SKIP {l1_class}: only {len(valid_subcats)} valid subcats")
        failed.append({"class": l1_class, "reason": "too_few_subcats", "n_valid": len(valid_subcats)})
        continue

    print(f"[{i+1}/{len(setfit_classes)}] Training: {l1_class} ({len(valid_subcats)} rare subcats)")

    # Build label mapping for this class
    label_names = sorted(valid_subcats.keys())
    label2id = {name: idx for idx, name in enumerate(label_names)}

    # Build training dataset (few-shot sampling)
    train_texts = []
    train_labels = []
    for subcat, texts in valid_subcats.items():
        # Sample up to MAX_SAMPLES_PER_CLASS
        sampled = texts[:MAX_SAMPLES_PER_CLASS]
        train_texts.extend(sampled)
        train_labels.extend([label2id[subcat]] * len(sampled))

    # Build val dataset
    val_texts = []
    val_labels = []
    for subcat, texts in val_subcats_data.items():
        if subcat in label2id:
            val_texts.extend(texts[:50])  # cap val samples
            val_labels.extend([label2id[subcat]] * min(len(texts), 50))

    print(f"  Train: {len(train_texts)} samples, Val: {len(val_texts)} samples")

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}) if val_texts else None

    # Train SetFit — adaptive batch size for large classes
    effective_batch = BATCH_SIZE if len(train_texts) < 200 else 8
    effective_iters = NUM_ITERATIONS if len(valid_subcats) < 30 else 10

    model = SetFitModel.from_pretrained(ST_MODEL)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"tmp_{l1_class[:30]}"),
        batch_size=effective_batch,
        num_iterations=effective_iters,
        num_epochs=NUM_EPOCHS_BODY,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Evaluate
    if val_dataset and len(val_texts) > 0:
        preds = model.predict(val_texts)
        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        preds = np.array(preds)
        val_labels_arr = np.array(val_labels)
        acc = accuracy_score(val_labels_arr, preds)
        f1 = f1_score(val_labels_arr, preds, average="macro", zero_division=0)
    else:
        acc = f1 = -1.0

    elapsed = time.time() - start_class
    print(f"  → acc={acc:.3f}, f1_macro={f1:.3f}, time={elapsed:.1f}s")

    # Save model + label mapping
    model_dir = os.path.join(OUTPUT_DIR, l1_class.replace("/", "_").replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)

    # Save label mapping
    label_info = {
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
        "n_subcats": len(label_names),
    }
    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    results[l1_class] = {
        "n_subcats": len(valid_subcats),
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "accuracy": acc,
        "f1_macro": f1,
        "time_seconds": elapsed,
    }

    # Free GPU memory before next model
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# ─── Summary ─────────────────────────────────────────
total_time = time.time() - start_total
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time/3600:.1f}h")
print(f"Models trained: {len(results)}")
print(f"Models skipped: {len(failed)}")

if results:
    accs = [r["accuracy"] for r in results.values() if r["accuracy"] >= 0]
    f1s = [r["f1_macro"] for r in results.values() if r["f1_macro"] >= 0]
    if accs:
        print(f"Mean accuracy: {np.mean(accs):.3f}")
        print(f"Mean F1-macro: {np.mean(f1s):.3f}")

# Save results
summary = {
    "config": {
        "model": ST_MODEL,
        "T_rare": T_RARE,
        "num_iterations": NUM_ITERATIONS,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
    },
    "results": results,
    "failed": failed,
    "total_time_hours": total_time / 3600,
}

with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUTPUT_DIR}/training_results.json")
