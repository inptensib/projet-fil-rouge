#!/usr/bin/env python3
"""
Train LoRA adapters for rare L2 subcategory classification.

Architecture:
- One LoRA adapter per L1 class that has rare subcats (< T_rare samples)
- Base model: camembert-base (shared, loaded once)
- Each adapter classifies only the RARE subcats within that L1 class
- Saves only adapter weights (~2-4MB per class) + label mapping

Comparison with SetFit:
- SetFit: contrastive body tuning + logistic head (~725MB full or head-only)
- LoRA:   parameter-efficient fine-tuning of CamemBERT (~2-4MB adapter)

Usage:
    python train_l2_lora.py                  # full run (GPU)
    python train_l2_lora.py --debug          # test with 2 classes
"""
import os
import sys
import json
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import unicodedata

DEBUG = "--debug" in sys.argv

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints/lora_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA hyperparams
BASE_MODEL = "camembert-base"
LORA_R = 8               # LoRA rank
LORA_ALPHA = 16          # LoRA scaling
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10           # base epochs; doubled for small datasets
NUM_EPOCHS_SMALL = 20     # for classes with < 50 total train samples
BATCH_SIZE = 16
MAX_SEQ_LEN = 256
MAX_SAMPLES_PER_CLASS = 50  # increased from 16 for better coverage

# ─── Load gating config ──────────────────────────────
with open(os.path.join(ANALYSIS_DIR, "gating_config.json")) as f:
    gating = json.load(f)

T_RARE = gating["thresholds"]["T_rare"]
routing = gating["routing"]
subcat_routing = gating["subcat_routing"]

# Get L1 classes that need per-class models
lora_classes = [c for c, route in routing.items() if route == "camembert_setfit"]
print(f"L1 classes routed to camembert_setfit: {len(lora_classes)}")

# Skip classes that already have a trained SetFit model (use LoRA only for the rest)
SETFIT_DIR = os.path.join(BASE_DIR, "checkpoints/setfit_models")
setfit_trained = set()
if os.path.isdir(SETFIT_DIR):
    for d in os.listdir(SETFIT_DIR):
        dpath = os.path.join(SETFIT_DIR, d)
        if os.path.isdir(dpath) and os.path.isfile(os.path.join(dpath, "label_mapping.json")):
            setfit_trained.add(d)

lora_classes_filtered = []
for c in lora_classes:
    dir_name = c.replace("/", "_").replace(" ", "_")
    if dir_name in setfit_trained:
        continue
    lora_classes_filtered.append(c)

print(f"Already trained by SetFit: {len(lora_classes) - len(lora_classes_filtered)} classes")
print(f"Remaining for LoRA training: {len(lora_classes_filtered)} classes")
lora_classes = lora_classes_filtered

if DEBUG:
    lora_classes = lora_classes[:2]
    print(f"DEBUG mode: only training {len(lora_classes)} classes")


# ─── Encoding normalization (merge duplicates) ───────
def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


# Build canonical name mapping for rare subcats
rare_canonical = {}
for l1_class in lora_classes:
    for subcat in subcat_routing[l1_class]["rare_subcats"]:
        norm = normalize(subcat)
        if norm not in rare_canonical:
            rare_canonical[norm] = subcat


# ─── Load dataset ────────────────────────────────────
print("Loading dataset...")
from datasets import load_dataset, Dataset

ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train_split.jsonl"),
    "val": os.path.join(DATA_DIR, "val_split.jsonl"),
})


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


# ─── Group samples by L1 class → rare subcats ────────
print("Grouping samples by L1 class and subcat...")
train_data_per_class = defaultdict(lambda: defaultdict(list))
val_data_per_class = defaultdict(lambda: defaultdict(list))

for split_name, split_ds, target_dict in [
    ("train", ds["train"], train_data_per_class),
    ("val", ds["val"], val_data_per_class),
]:
    print(f"  Scanning {split_name} split ({len(split_ds)} samples)...")
    for example in split_ds:
        l1_class = example.get("typelabel", "")
        if l1_class not in lora_classes:
            continue
        subcat = example["producttype"][0] if example.get("producttype") else ""
        if not subcat:
            continue
        if is_rare_subcat(l1_class, subcat):
            norm = normalize(subcat)
            canonical = rare_canonical.get(norm, subcat)
            text = get_text(example)
            if text.strip():
                target_dict[l1_class][canonical].append(text)


# ─── Load base model & tokenizer once ────────────────
print(f"\nLoading base model: {BASE_MODEL}")
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = CamembertTokenizer.from_pretrained(BASE_MODEL)


# ─── Custom Dataset for token classification ─────────
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─── Train one LoRA adapter per L1 class ─────────────
print(f"\n{'='*60}")
print(f"Training LoRA adapters for {len(lora_classes)} L1 classes")
print(f"{'='*60}\n")

results = {}
failed = []
start_total = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

for i, l1_class in enumerate(lora_classes):
    start_class = time.time()

    # Skip if already trained (resume support)
    model_dir = os.path.join(OUTPUT_DIR, l1_class.replace("/", "_").replace(" ", "_"))
    if os.path.isfile(os.path.join(model_dir, "label_mapping.json")):
        print(f"[{i+1}/{len(lora_classes)}] SKIP {l1_class}: already trained")
        continue

    rare_subcats_data = train_data_per_class[l1_class]
    val_subcats_data = val_data_per_class[l1_class]

    # Filter: need at least 2 subcats with at least 2 samples each
    valid_subcats = {s: texts for s, texts in rare_subcats_data.items() if len(texts) >= 2}

    if len(valid_subcats) < 2:
        print(f"[{i+1}/{len(lora_classes)}] SKIP {l1_class}: only {len(valid_subcats)} valid subcats")
        failed.append({"class": l1_class, "reason": "too_few_subcats", "n_valid": len(valid_subcats)})
        continue

    n_labels = len(valid_subcats)
    print(f"[{i+1}/{len(lora_classes)}] Training: {l1_class} ({n_labels} rare subcats)")

    # Build label mapping
    label_names = sorted(valid_subcats.keys())
    label2id = {name: idx for idx, name in enumerate(label_names)}

    # Build training data (few-shot sampling)
    train_texts = []
    train_labels = []
    for subcat, texts in valid_subcats.items():
        sampled = texts[:MAX_SAMPLES_PER_CLASS]
        train_texts.extend(sampled)
        train_labels.extend([label2id[subcat]] * len(sampled))

    # Build val data
    val_texts = []
    val_labels = []
    for subcat, texts in val_subcats_data.items():
        if subcat in label2id:
            val_texts.extend(texts[:50])
            val_labels.extend([label2id[subcat]] * min(len(texts), 50))

    print(f"  Train: {len(train_texts)} samples, Val: {len(val_texts)} samples")

    # ─── Create model with fresh classification head ──────
    model = CamembertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=n_labels
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],  # attention layers
        modules_to_save=["classifier"],     # CRITICAL: save the classification head too
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    if i == 0:
        model.print_trainable_parameters()

    # Compute class weights for imbalanced subcategories
    from collections import Counter
    label_counts = Counter(train_labels)
    n_samples = len(train_labels)
    class_weights = torch.tensor(
        [n_samples / (n_labels * label_counts[i]) for i in range(n_labels)],
        dtype=torch.float32
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Adaptive epochs: more for small datasets
    n_epochs = NUM_EPOCHS_SMALL if len(train_texts) < 50 else NUM_EPOCHS

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Use weighted loss instead of default
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    # ─── Evaluate ─────────────────────────────────────────
    if val_texts:
        model.eval()
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)

        val_labels_arr = np.array(val_labels)
        preds_arr = np.array(all_preds)
        acc = accuracy_score(val_labels_arr, preds_arr)
        f1 = f1_score(val_labels_arr, preds_arr, average="macro", zero_division=0)
    else:
        acc = f1 = -1.0

    elapsed = time.time() - start_class
    print(f"  → acc={acc:.3f}, f1_macro={f1:.3f}, time={elapsed:.1f}s")

    # ─── Save adapter + label mapping ─────────────────────
    os.makedirs(model_dir, exist_ok=True)

    # Save only LoRA adapter weights (~2-4MB) + classifier head
    model.save_pretrained(model_dir)  # PEFT saves only adapter, not base model

    # Save label mapping
    label_info = {
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
        "n_subcats": n_labels,
    }
    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    results[l1_class] = {
        "n_subcats": n_labels,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "accuracy": acc,
        "f1_macro": f1,
        "time_seconds": elapsed,
    }

    # ─── Cleanup GPU memory ───────────────────────────────
    del model, optimizer, train_dataset, train_loader
    if val_texts:
        del val_dataset, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# ─── Summary ─────────────────────────────────────────
elapsed_total = time.time() - start_total
print(f"\n{'='*60}")
print(f"LoRA Training Complete")
print(f"{'='*60}")
print(f"Total time: {elapsed_total/60:.1f} minutes")
print(f"Trained: {len(results)} classes")
print(f"Failed/Skipped: {len(failed)} classes")

if results:
    accs = [r["accuracy"] for r in results.values() if r["accuracy"] >= 0]
    f1s = [r["f1_macro"] for r in results.values() if r["f1_macro"] >= 0]
    if accs:
        print(f"\nMean accuracy: {np.mean(accs):.3f} (std={np.std(accs):.3f})")
        print(f"Mean F1-macro: {np.mean(f1s):.3f} (std={np.std(f1s):.3f})")

# Save results
with open(os.path.join(OUTPUT_DIR, "results_per_class.json"), "w") as f:
    json.dump({"results": results, "failed": failed, "total_time": elapsed_total}, f, indent=2, ensure_ascii=False)

print("\nDone!")
