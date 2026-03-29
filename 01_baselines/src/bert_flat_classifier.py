"""BERT flat classification for Rakuten product classification.

Supports two modes:
  - Feature extraction: BERT encoder frozen, only classification head trained.
  - Fine-tuning: All BERT parameters and classification head trained end-to-end.

Supports CamemBERT, XLM-RoBERTa, and mBERT backbones.

Usage:
    # Feature extraction (frozen BERT)
    python bert_flat_classifier.py --data_path dataset/train.jsonl --mode feature_extraction

    # Full fine-tuning
    python bert_flat_classifier.py --data_path dataset/train.jsonl --mode fine_tuning

    # Specific backbone
    python bert_flat_classifier.py --data_path dataset/train.jsonl --backbone camembert
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Cross-directory import from shared/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.data_loader import load_random_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_MAP = {
    "xlm-roberta": "xlm-roberta-base",
    "camembert": "camembert-base",
    "mbert": "bert-base-multilingual-cased",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProductDataset(Dataset):
    """PyTorch dataset that tokenizes product text on the fly.

    Args:
        texts: List of product text strings.
        labels: List of encoded integer labels.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BERTClassifier(nn.Module):
    """BERT encoder + linear classification head.

    Uses the [CLS] token representation followed by dropout and a linear
    projection to the number of classes.

    Args:
        model_name: HuggingFace model identifier.
        num_classes: Number of output classes.
        dropout: Dropout probability before the classifier.
        freeze_bert: If True, freeze all encoder parameters (feature extraction mode).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.freeze_bert = freeze_bert

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BERT] Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_text(df: pd.DataFrame) -> pd.Series:
    """Build text feature from designation + producttype."""
    df = df.copy()
    df["designation"] = df["designation"].fillna("")
    if "producttype" in df.columns:
        clean_type = df["producttype"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else (str(x) if x else "")
        )
        return df["designation"] + " " + clean_type
    return df["designation"]


def _filter_rare_classes(
    df: pd.DataFrame,
    label_column: str,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Remove classes with fewer than *min_samples* samples."""
    counts = df[label_column].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        n_before = len(df)
        df = df[~df[label_column].isin(rare)].copy()
        print(f"[BERT] Removed {len(rare)} rare classes ({n_before - len(df)} samples)")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bert_flat(
    data_path: str,
    n_samples: int = 50_000,
    label_column: str = "typelabel",
    backbone: str = "camembert",
    mode: str = "fine_tuning",
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    test_size: float = 0.2,
    save_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Train a BERT flat classifier.

    Args:
        data_path: Path to the JSONL training file.
        n_samples: Number of samples to load.
        label_column: Column name for labels.
        backbone: One of 'xlm-roberta', 'camembert', 'mbert'.
        mode: 'feature_extraction' (frozen BERT) or 'fine_tuning' (all params).
        max_length: Maximum token sequence length.
        batch_size: Training batch size (per GPU).
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay / L2 regularisation.
        test_size: Fraction of data for the test set.
        save_dir: Directory to save artefacts. Auto-generated if None.
        seed: Random seed.

    Returns:
        Dictionary with model, metrics, and predictions.
    """
    freeze_bert = mode == "feature_extraction"

    if save_dir is None:
        save_dir = f"models_bert_flat_{backbone}_{mode}"

    model_name = BACKBONE_MAP.get(backbone)
    if model_name is None:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONE_MAP)}")

    print("=" * 60)
    print(f"BERT Flat Classification  |  {backbone}")
    print(f"Mode: {mode}  |  Backbone: {model_name}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Load data ---
    df = load_random_sample(data_path, n_samples=n_samples, seed=seed)
    df = _filter_rare_classes(df, label_column)
    text_col = _prepare_text(df)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_column])
    num_classes = len(label_encoder.classes_)
    print(f"[BERT] Classes: {num_classes} | Samples: {len(df):,}")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        text_col, encoded_labels, test_size=test_size, random_state=seed,
        stratify=encoded_labels,
    )
    print(f"[BERT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # --- Tokenizer and datasets ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ProductDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length)
    test_dataset = ProductDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length)

    n_gpus = torch.cuda.device_count()
    effective_batch = batch_size * max(n_gpus, 1)

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=effective_batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # --- Model ---
    model = BERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=0.1,
        freeze_bert=freeze_bert,
    )

    if n_gpus > 1:
        print(f"[BERT] Using DataParallel on {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"\n[BERT] Training for {epochs} epochs (lr={lr}, wd={weight_decay})...")
    start_time = time.time()
    train_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    train_time = time.time() - start_time

    # --- Evaluation ---
    print("[BERT] Evaluating...")
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["label"]

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_batch.numpy().tolist())
    pred_time = time.time() - t0

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    y_true_labels = label_encoder.inverse_transform(all_labels)
    y_pred_labels = label_encoder.inverse_transform(all_preds)

    print(f"\n[BERT] Training time:   {train_time:.1f}s")
    print(f"[BERT] Prediction time: {pred_time:.1f}s")
    print(f"[BERT] F1-Macro:        {f1_macro:.4f}")
    print(f"[BERT] F1-Weighted:     {f1_weighted:.4f}")
    print()
    print(classification_report(y_true_labels, y_pred_labels))

    # --- Save artefacts ---
    os.makedirs(save_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.bert.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    classifier_state = {
        "classifier": model_to_save.classifier.state_dict(),
        "dropout": model_to_save.dropout.p,
        "hidden_size": model_to_save.hidden_size,
        "num_classes": num_classes,
    }
    torch.save(classifier_state, os.path.join(save_dir, "classifier_head.pt"))

    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    config = {
        "backbone": backbone,
        "base_model": model_name,
        "mode": mode,
        "freeze_bert": freeze_bert,
        "num_classes": num_classes,
        "max_length": max_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_samples": n_samples,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "train_time_s": float(train_time),
        "pred_time_s": float(pred_time),
        "train_losses": [float(l) for l in train_losses],
        "classes": label_encoder.classes_.tolist(),
    }
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    preds_df = pd.DataFrame({"y_true": y_true_labels, "y_pred": y_pred_labels})
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print(f"[BERT] Artefacts saved to {save_dir}/")

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time": train_time,
        "pred_time": pred_time,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a saved BERT flat classifier.

    Args:
        model_dir: Path to the saved artefacts directory.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Tuple of (model, tokenizer, label_encoder, config).
    """
    if device is None:
        device = DEVICE

    with open(os.path.join(model_dir, "training_config.json")) as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = BERTClassifier(
        model_name=model_dir,
        num_classes=config["num_classes"],
        dropout=0.1,
        freeze_bert=config.get("freeze_bert", False),
    )
    head_state = torch.load(
        os.path.join(model_dir, "classifier_head.pt"), map_location=device,
    )
    model.classifier.load_state_dict(head_state["classifier"])
    model = model.to(device)
    model.eval()

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder, config


def predict_text(
    text: str,
    model_dir: str,
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    """Predict category for a single text string.

    Args:
        text: Input product text.
        model_dir: Path to saved artefacts.
        device: Torch device.

    Returns:
        Dictionary with 'label' and 'confidence'.
    """
    model, tokenizer, label_encoder, config = load_model(model_dir, device)
    max_length = config.get("max_length", 128)

    encoding = tokenizer(
        text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device or DEVICE)
        attention_mask = encoding["attention_mask"].to(device or DEVICE)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    label = label_encoder.inverse_transform([pred_idx.item()])[0]
    return {"label": label, "confidence": confidence.item()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERT flat classifier for Rakuten product classification",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument(
        "--backbone", type=str, default="camembert",
        choices=list(BACKBONE_MAP.keys()),
        help="Transformer backbone",
    )
    parser.add_argument(
        "--mode", type=str, default="fine_tuning",
        choices=["feature_extraction", "fine_tuning"],
        help="Training mode: frozen BERT or full fine-tuning",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Predict mode
    parser.add_argument("--predict", type=str, default=None, help="Text to classify")
    parser.add_argument("--model_dir", type=str, default=None, help="Saved model dir (for --predict)")

    args = parser.parse_args()

    if args.predict is not None:
        if args.model_dir is None:
            raise ValueError("--model_dir is required when using --predict")
        result = predict_text(args.predict, args.model_dir)
        print(f"Prediction: {result['label']}  (confidence: {result['confidence']:.2%})")
        return

    train_bert_flat(
        data_path=args.data_path,
        n_samples=args.n_samples,
        backbone=args.backbone,
        mode=args.mode,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
