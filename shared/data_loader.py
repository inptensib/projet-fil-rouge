"""Data loading and preprocessing for the Rakuten product classification dataset.

Handles loading from large JSONL files, label encoding, hierarchical label
extraction, and train/val/test splitting.
"""

from __future__ import annotations

import gc
import json
import logging
import random
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def load_random_sample(
    file_path: str,
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a random sample of lines from a large JSONL file.

    Uses reservoir-style sampling to avoid loading the entire file into memory.

    Args:
        file_path: Path to the JSONL file.
        n_samples: Number of samples to extract.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with sampled records.
    """
    random.seed(seed)

    # Count total lines
    total_lines = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    print(f"[Data] Total lines in file: {total_lines:,}")

    # Calculate sampling probability (oversample by 10% then trim)
    probability = min(1.0, (n_samples * 1.1) / total_lines)

    print(f"[Data] Sampling ~{n_samples:,} records...")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if random.random() > probability:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    del data
    gc.collect()

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"[Data] Loaded {len(df):,} records.")
    return df


def extract_level_2(producttype) -> Optional[str]:
    """Extract the first valid L2 label from a producttype field.

    Args:
        producttype: Either a list of product types or a single string.

    Returns:
        Cleaned L2 label string, or None if invalid.
    """
    invalid_values = {"", "none", "nan", "unknown"}

    if isinstance(producttype, list) and len(producttype) > 0:
        first_elem = str(producttype[0]).strip()
        if first_elem.lower() not in invalid_values:
            return first_elem
    elif isinstance(producttype, str) and producttype.strip():
        if producttype.strip().lower() not in invalid_values:
            return producttype.strip()
    return None


def prepare_hierarchical_data(
    file_path: str,
    n_samples: int = 10_000,
    test_size: float = 0.2,
    val_ratio: float = 0.125,
    seed: int = 42,
) -> dict:
    """Load JSONL data and prepare hierarchical labels with train/val/test splits.

    Args:
        file_path: Path to the JSONL dataset.
        n_samples: Number of samples to load.
        test_size: Fraction for test set (from total).
        val_ratio: Fraction for val set (from remaining after test).
        seed: Random seed.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_l1_train, y_l1_val,
        y_l1_test, y_l2_train, y_l2_val, y_l2_test, l1_encoder, l2_encoder,
        l2_to_l1_mapping, num_classes_l1, num_classes_l2.
    """
    df = load_random_sample(file_path, n_samples=n_samples, seed=seed)

    # Extract hierarchical labels
    df["level_1"] = df["typelabel"].fillna("Unknown")
    df["level_2_raw"] = df["producttype"].apply(extract_level_2)
    df = df[df["level_2_raw"].notna()].copy()
    df["level_2"] = df["level_2_raw"]

    # Concatenate title + description as input text
    df["designation"] = df["designation"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = df["designation"] + ". " + df["description"]
    df = df[["text", "level_1", "level_2"]]

    print(f"[Data] Valid hierarchical samples: {len(df):,}")

    # Encode labels
    l1_encoder = LabelEncoder()
    l2_encoder = LabelEncoder()
    df["label_l1"] = l1_encoder.fit_transform(df["level_1"])
    df["label_l2"] = l2_encoder.fit_transform(df["level_2"])

    # Build parent-child mapping
    l2_to_l1 = build_parent_child_mapping(
        df["label_l1"].values, df["label_l2"].values
    )

    # Train/val/test split
    try:
        X_temp, X_test, y1_temp, y1_test, y2_temp, y2_test = train_test_split(
            df["text"], df["label_l1"], df["label_l2"],
            test_size=test_size, random_state=seed, stratify=df["label_l2"],
        )
        X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
            X_temp, y1_temp, y2_temp,
            test_size=val_ratio, random_state=seed, stratify=y2_temp,
        )
    except ValueError:
        # Fallback without stratification if some classes have too few samples
        X_temp, X_test, y1_temp, y1_test, y2_temp, y2_test = train_test_split(
            df["text"], df["label_l1"], df["label_l2"],
            test_size=test_size, random_state=seed,
        )
        X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
            X_temp, y1_temp, y2_temp,
            test_size=val_ratio, random_state=seed,
        )

    del df
    gc.collect()

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_l1_train": y1_train, "y_l1_val": y1_val, "y_l1_test": y1_test,
        "y_l2_train": y2_train, "y_l2_val": y2_val, "y_l2_test": y2_test,
        "l1_encoder": l1_encoder, "l2_encoder": l2_encoder,
        "l2_to_l1_mapping": l2_to_l1,
        "num_classes_l1": len(l1_encoder.classes_),
        "num_classes_l2": len(l2_encoder.classes_),
    }


def build_parent_child_mapping(
    level_1_labels, level_2_labels
) -> dict[int, int]:
    """Build a mapping from L2 class index to L1 class index.

    Args:
        level_1_labels: Array of encoded L1 labels.
        level_2_labels: Array of encoded L2 labels.

    Returns:
        Dictionary mapping L2 index -> L1 index.
    """
    logger = logging.getLogger(__name__)
    l2_to_l1 = {}
    for l1, l2 in zip(level_1_labels, level_2_labels):
        if l2 not in l2_to_l1:
            l2_to_l1[l2] = l1
        elif l2_to_l1[l2] != l1:
            logger.warning(
                "L2 class %s has multiple L1 parents: existing=%s, new=%s",
                l2, l2_to_l1[l2], l1,
            )
    return l2_to_l1


def build_parent_child_matrix(
    l2_to_l1_mapping: dict[int, int],
    num_l1_classes: int,
    num_l2_classes: int,
) -> torch.Tensor:
    """Build a binary matrix [num_l1, num_l2] indicating valid parent-child pairs.

    Args:
        l2_to_l1_mapping: Mapping from L2 index to L1 index.
        num_l1_classes: Number of L1 classes.
        num_l2_classes: Number of L2 classes.

    Returns:
        Binary tensor of shape (num_l1_classes, num_l2_classes).
    """
    matrix = torch.zeros(num_l1_classes, num_l2_classes)
    for l2_idx, l1_idx in l2_to_l1_mapping.items():
        matrix[l1_idx, l2_idx] = 1.0
    return matrix


class HierarchicalProductDataset(Dataset):
    """PyTorch Dataset for hierarchical product classification.

    Args:
        texts: List or Series of product text descriptions.
        labels_l1: List or Series of encoded L1 labels.
        labels_l2: List or Series of encoded L2 labels.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
    """

    def __init__(self, texts, labels_l1, labels_l2, tokenizer, max_length: int = 128):
        import numpy as np
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels_l1 = np.asarray(labels_l1) if not isinstance(labels_l1, np.ndarray) else labels_l1
        self.labels_l2 = np.asarray(labels_l2) if not isinstance(labels_l2, np.ndarray) else labels_l2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_l1": torch.as_tensor(self.labels_l1[idx], dtype=torch.long),
            "label_l2": torch.as_tensor(self.labels_l2[idx], dtype=torch.long),
            "sample_idx": idx,
        }
