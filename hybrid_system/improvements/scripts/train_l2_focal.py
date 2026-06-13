#!/usr/bin/env python3
"""
Improved CamemBERT L2 training with:
  - Focal Loss (γ=2, α=class-frequency-weighted) → boosts rare classes
  - Label Smoothing (0.1) → prevents overconfidence
  - Cosine LR schedule with warmup

Based on: train_l2_camembert.py (original run2 config)
Expected improvement: +2-4% F1-macro, +0.5-1% accuracy
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

RESUME = "--resume" in sys.argv

# ─── Config ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# Navigate up from improvements/scripts/ to project root
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR    = os.path.join(PROJECT_DIR, "dataset")
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "analysis")

# Use best checkpoint from run2 as starting point
L2_CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints/l2_camembert_run2")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "checkpoints/l2_camembert_focal")

MAX_LEN        = 256
EPOCHS         = 3
BATCH_SIZE     = 16
GRAD_ACCUM     = 16    # effective batch = 256
LR             = 1e-5  # slightly lower LR for fine-tuning from existing checkpoint
FOCAL_GAMMA    = 2.0
LABEL_SMOOTHING = 0.1

# ─── Focal Loss with Label Smoothing ─────────────────
class FocalLossWithSmoothing(nn.Module):
    """Focal loss: down-weights easy examples, up-weights hard/rare ones.
    Combined with label smoothing to prevent overconfident predictions."""

    def __init__(self, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight  # class weights (tensor)

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        # Label smoothing: soft targets
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        # For soft targets, use the probability of the true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        # Cross-entropy with soft targets
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply focal weight
        loss = focal_weight * loss

        # Apply class weights if provided
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets]
            loss = loss * class_w

        return loss.mean()


# ─── Custom Trainer with Focal Loss ──────────────────
class FocalTrainer(Trainer):
    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─── Load data ────────────────────────────────────────
print("=" * 60)
print("CamemBERT L2 Training with Focal Loss + Label Smoothing")
print("=" * 60)

# Find best checkpoint
def find_best_ckpt(ckpt_dir):
    best = os.path.join(ckpt_dir, "best")
    if os.path.isdir(best):
        return best
    ckpts = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir

MODEL_PATH = find_best_ckpt(L2_CKPT_DIR)
print(f"Starting from checkpoint: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except OSError:
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train_split.jsonl"),
    "val":   os.path.join(DATA_DIR, "val_split.jsonl"),
    "test":  os.path.join(DATA_DIR, "test_split.jsonl"),
})

# ─── Build label map ──────────────────────────────────
print("Building L2 label map...")
analysis_file = os.path.join(ANALYSIS_DIR, "l2_subcategory_counts.json")
if os.path.isfile(analysis_file):
    with open(analysis_file) as f:
        l2_counts = json.load(f)
    all_labels = sorted(l2_counts.keys())
else:
    all_labels = set()
    for split in ["train", "val", "test"]:
        for ex in ds[split]:
            if ex["producttype"]:
                all_labels.add(ex["producttype"][0])
    all_labels = sorted(all_labels)

label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(all_labels)
print(f"L2 classes: {num_classes}")

# ─── Compute class weights (inverse frequency, capped) ─
print("Computing class weights for focal loss...")
train_labels = []
for ex in ds["train"]:
    if ex["producttype"] and ex["producttype"][0] in label2id:
        train_labels.append(label2id[ex["producttype"][0]])

label_counts = Counter(train_labels)
total = sum(label_counts.values())
# Inverse frequency, capped at 10x to avoid extreme weights
class_weights = torch.ones(num_classes)
for cls_id, count in label_counts.items():
    w = total / (num_classes * count)
    class_weights[cls_id] = min(w, 10.0)
# Normalize so mean weight = 1
class_weights = class_weights / class_weights.mean()
print(f"  Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

# ─── Tokenize ─────────────────────────────────────────
def preprocess(batch):
    texts = []
    labels = []
    for title, desc, pt in zip(batch["designation"], batch["description"], batch["producttype"]):
        title = title or ""
        desc = desc or ""
        texts.append(title + " " + tokenizer.sep_token + " " + desc)
        if pt and pt[0] in label2id:
            labels.append(label2id[pt[0]])
        else:
            labels.append(-1)
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN)
    enc["labels"] = labels
    return enc

print("Tokenizing...")
tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names,
                   num_proc=4, desc="Tokenizing")
tokenized = tokenized.filter(lambda ex: ex["labels"] >= 0)

# ─── Model ────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
)

# ─── Training args ────────────────────────────────────
fp16_ok = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=fp16_ok,
    report_to="none",
    overwrite_output_dir=not RESUME,
)

# ─── Metrics ──────────────────────────────────────────
def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }

# ─── Focal Loss ───────────────────────────────────────
focal_loss = FocalLossWithSmoothing(
    gamma=FOCAL_GAMMA,
    label_smoothing=LABEL_SMOOTHING,
    weight=class_weights,
)
print(f"Loss: Focal(γ={FOCAL_GAMMA}) + LabelSmoothing({LABEL_SMOOTHING}) + ClassWeights")

# ─── Train ────────────────────────────────────────────
trainer = FocalTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    focal_loss_fn=focal_loss,
)

print("\nStarting training...")
if RESUME:
    # Find latest checkpoint to resume from
    ckpts = sorted(
        [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    resume_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1]) if ckpts else True
    print(f"Resuming from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# ─── Test eval ────────────────────────────────────────
print("\nEvaluating on test set...")
results = trainer.evaluate(tokenized["test"])
print(f"Test results: {results}")

# ─── Save ─────────────────────────────────────────────
best_dir = os.path.join(OUTPUT_DIR, "best")
os.makedirs(best_dir, exist_ok=True)
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)
with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
              f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump({
        "base_checkpoint": MODEL_PATH,
        "focal_gamma": FOCAL_GAMMA,
        "label_smoothing": LABEL_SMOOTHING,
        "epochs": EPOCHS,
        "lr": LR,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_len": MAX_LEN,
        "results": results,
    }, f, indent=2)
print(f"Model saved to {best_dir}")
print("Done!")
