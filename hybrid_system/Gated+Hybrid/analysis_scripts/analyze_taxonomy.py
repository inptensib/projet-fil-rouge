#!/usr/bin/env python3
"""
Analyze L1 classification results and L1→L2 taxonomy structure.
Determines which classes are easy/hard and informs L2 routing strategy.

Usage:
  python analyze_taxonomy.py          # full run (GPU cluster)
  python analyze_taxonomy.py --debug  # local debug (5k samples, no L1 model needed)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm

DEBUG = "--debug" in sys.argv
SAMPLE_SIZE = 5000 if DEBUG else None

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L1_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/l1_classifier")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# Auto-detect latest checkpoint (or "best" if it exists)
def find_l1_model_dir():
    best = os.path.join(L1_CKPT_DIR, "best")
    if os.path.isdir(best):
        return best
    # Find latest checkpoint-XXXXX
    ckpts = sorted(
        [d for d in os.listdir(L1_CKPT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    if ckpts:
        return os.path.join(L1_CKPT_DIR, ckpts[-1])
    raise FileNotFoundError(f"No model found in {L1_CKPT_DIR}")

L1_MODEL_DIR = find_l1_model_dir()
print(f"Using L1 model: {L1_MODEL_DIR}")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Part 1: L1 Per-Class Performance ─────────────────
if not DEBUG:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report, confusion_matrix

    print("Loading L1 model and data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer: try checkpoint → DAPT → camembert-base
    tokenizer = None
    for tok_path in [L1_MODEL_DIR,
                     os.path.join(BASE_DIR, "checkpoints/dapt/final"),
                     "camembert-base"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            print(f"Loaded tokenizer from: {tok_path}")
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError("Could not load tokenizer from any source")

    model = AutoModelForSequenceClassification.from_pretrained(L1_MODEL_DIR).to(device)
    model.eval()

    # Load label mapping from model config (works even without label_mapping.json)
    label_map_file = os.path.join(L1_CKPT_DIR, "label_mapping.json")
    if os.path.isfile(label_map_file):
        with open(label_map_file) as f:
            mapping = json.load(f)
        label2id = mapping["label2id"]
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        # Extract from model config
        label2id = model.config.label2id
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"Loaded {len(label2id)} labels")

    val_ds = load_dataset("json", data_files={"val": os.path.join(DATA_DIR, "val_split.jsonl")})["val"]
    print(f"Loaded {len(val_ds)} validation samples")

    print("\n=== Part 1: Analyzing L1 Per-Class Performance ===")

    y_true, y_pred, confidences = [], [], []

    for batch_start in tqdm(range(0, len(val_ds), 64), desc="Running L1 predictions"):
        batch = val_ds[batch_start:batch_start+64]

        inputs = tokenizer(batch["designation"], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values

        for i in range(len(batch["designation"])):
            true_label = label2id[str(batch["typelabel"][i])]
            y_true.append(true_label)
            y_pred.append(preds[i].item())
            confidences.append(max_probs[i].item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    # Per-class F1 scores
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_f1_scores = {}
    for label_id, label_name in id2label.items():
        if str(label_id) in report_dict:
            class_f1_scores[label_name] = report_dict[str(label_id)]['f1-score']

    sorted_by_f1 = sorted(class_f1_scores.items(), key=lambda x: x[1])

    print("\n🔴 Top 20 HARDEST L1 Classes (lowest F1):")
    for class_name, f1 in sorted_by_f1[:20]:
        print(f"  {class_name:50s} F1={f1:.3f}")

    print("\n🟢 Top 20 EASIEST L1 Classes (highest F1):")
    for class_name, f1 in sorted_by_f1[-20:]:
        print(f"  {class_name:50s} F1={f1:.3f}")

    with open(os.path.join(OUTPUT_DIR, "l1_class_difficulty.json"), "w") as f:
        json.dump(class_f1_scores, f, indent=2, ensure_ascii=False)

    # ─── Part 2: Confidence Analysis ──────────────────────
    print("\n=== Part 2: Confidence Distribution ===")

    correct_mask = (y_true == y_pred)
    conf_correct = confidences[correct_mask]
    conf_incorrect = confidences[~correct_mask]

    print(f"Correct predictions: mean conf={conf_correct.mean():.3f}, median={np.median(conf_correct):.3f}")
    print(f"Incorrect predictions: mean conf={conf_incorrect.mean():.3f}, median={np.median(conf_incorrect):.3f}")

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print("\nConfidence threshold analysis (% of data that would go to L2):")
    for thresh in thresholds:
        below_thresh = (confidences < thresh).sum()
        pct = 100 * below_thresh / len(confidences)
        # Accuracy of predictions above threshold
        above_mask = confidences >= thresh
        if above_mask.sum() > 0:
            acc_above = (y_true[above_mask] == y_pred[above_mask]).mean()
        else:
            acc_above = 0
        print(f"  Threshold {thresh:.2f}: {pct:.1f}% → L2, accuracy of kept predictions: {acc_above:.3f}")

else:
    print("=== DEBUG MODE: Skipping L1 model evaluation (no GPU) ===")
    print("Parts 1 & 2 skipped (need L1 model on GPU)")

# ─── Part 3: L1→L2 Taxonomy Mapping ───────────────────
print("\n=== Part 3: L1→L2 Subcategory Structure ===")

# Build mapping: L1 class → list of L2 subcategories
l1_to_l2 = defaultdict(set)
l2_counts = Counter()
l1_counts = Counter()

train_ds = load_dataset("json", data_files={"train": os.path.join(DATA_DIR, "train_split.jsonl")})["train"]
if SAMPLE_SIZE:
    train_ds = train_ds.shuffle(seed=42).select(range(min(SAMPLE_SIZE, len(train_ds))))
    print(f"DEBUG: Using {len(train_ds)} samples (shuffled)")
else:
    print(f"Full training set: {len(train_ds)} samples")

print("Scanning training set for L1→L2 mapping...")
for sample in tqdm(train_ds, desc="Building taxonomy"):
    l1_class = sample["typelabel"]
    l2_subcats = sample["producttype"]
    l1_counts[l1_class] += 1
    for subcat in l2_subcats:
        l1_to_l2[l1_class].add(subcat)
        l2_counts[subcat] += 1

# Convert to dict for JSON serialization
l1_to_l2_serializable = {k: list(v) for k, v in l1_to_l2.items()}

print(f"\nTotal L1 classes: {len(l1_to_l2)}")
print(f"Total unique L2 subcategories: {len(l2_counts)}")

# Subcategories per L1 class
subcats_per_l1 = {k: len(v) for k, v in l1_to_l2.items()}
sorted_by_subcats = sorted(subcats_per_l1.items(), key=lambda x: x[1], reverse=True)

print("\n🔢 L1 classes with MOST subcategories:")
for l1_class, count in sorted_by_subcats[:15]:
    print(f"  {l1_class:50s} → {count} subcategories")

print("\n🔢 L1 classes with FEWEST subcategories:")
for l1_class, count in sorted_by_subcats[-15:]:
    print(f"  {l1_class:50s} → {count} subcategories")

# Subcategory frequency distribution
sorted_l2_by_freq = l2_counts.most_common()
freq_bins = {
    "ultra_rare (<10)": sum(1 for _, c in sorted_l2_by_freq if c < 10),
    "very_rare (10-50)": sum(1 for _, c in sorted_l2_by_freq if 10 <= c < 50),
    "rare (50-100)": sum(1 for _, c in sorted_l2_by_freq if 50 <= c < 100),
    "medium (100-500)": sum(1 for _, c in sorted_l2_by_freq if 100 <= c < 500),
    "frequent (500-1000)": sum(1 for _, c in sorted_l2_by_freq if 500 <= c < 1000),
    "very_frequent (>1000)": sum(1 for _, c in sorted_l2_by_freq if c >= 1000),
}

print("\n📊 L2 Subcategory Frequency Distribution:")
for bin_name, count in freq_bins.items():
    pct = 100 * count / len(l2_counts)
    print(f"  {bin_name:25s}: {count:4d} subcategories ({pct:.1f}%)")

# Save mappings
with open(os.path.join(OUTPUT_DIR, "l1_to_l2_mapping.json"), "w") as f:
    json.dump(l1_to_l2_serializable, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "l2_subcategory_counts.json"), "w") as f:
    json.dump(dict(l2_counts), f, indent=2, ensure_ascii=False)

# ─── Part 4: Recommendations ──────────────────────────
print("\n=== Part 4: L2 Routing Recommendations ===")

# Define difficulty tiers (only available in full mode)
if not DEBUG:
    hard_classes = [name for name, f1 in sorted_by_f1 if f1 < 0.80]
    medium_classes = [name for name, f1 in sorted_by_f1 if 0.80 <= f1 < 0.90]
    easy_classes = [name for name, f1 in sorted_by_f1 if f1 >= 0.90]

    print(f"\n📋 L1 Class Difficulty Tiers:")
    print(f"  Hard classes (F1 < 0.80): {len(hard_classes)} classes")
    print(f"  Medium classes (0.80-0.90): {len(medium_classes)} classes")
    print(f"  Easy classes (F1 >= 0.90): {len(easy_classes)} classes")
else:
    hard_classes, medium_classes, easy_classes = [], [], []
    print("\n📋 L1 difficulty tiers: skipped in debug mode (need L1 model)")

# Categorize subcategories
rare_subcats = [subcat for subcat, count in l2_counts.items() if count < 100]
medium_subcats = [subcat for subcat, count in l2_counts.items() if 100 <= count < 1000]
frequent_subcats = [subcat for subcat, count in l2_counts.items() if count >= 1000]

print(f"\n📋 L2 Subcategory Frequency Tiers:")
print(f"  Rare (<100 examples): {len(rare_subcats)} subcategories")
print(f"  Medium (100-1000): {len(medium_subcats)} subcategories")
print(f"  Frequent (>1000): {len(frequent_subcats)} subcategories")

recommendations = {
    "l1_hard_classes": hard_classes,
    "l1_medium_classes": medium_classes,
    "l1_easy_classes": easy_classes,
    "l2_rare_subcats": rare_subcats,
    "l2_medium_subcats": medium_subcats,
    "l2_frequent_subcats": frequent_subcats,
    "confidence_threshold_suggestion": 0.85,  # Based on analysis
    "routing_strategy": {
        "xgboost": "Use for easy L1 classes + frequent subcategories + high confidence (>0.9)",
        "camembert": "Use for medium L1 classes + medium subcategories + medium confidence (0.7-0.9)",
        "setfit": "Use for hard L1 classes + rare subcategories + low confidence (<0.7)",
        "mt5_fallback": "Use for ultra-rare subcategories (<10 examples) or complete failures"
    }
}

with open(os.path.join(OUTPUT_DIR, "routing_recommendations.json"), "w") as f:
    json.dump(recommendations, f, indent=2, ensure_ascii=False)

print("\n✅ Analysis complete! Results saved to:", OUTPUT_DIR)
print("\nNext steps:")
print("1. Review analysis/routing_recommendations.json")
print("2. Train L2 models based on difficulty tiers")
print("3. Implement routing logic with recommended confidence thresholds")
