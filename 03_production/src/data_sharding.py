"""Data sharding for distributed training.

Prepares training data into per-rank shards with validation and test splits,
using the shared data loading utilities. Called by rank 0 only before the
DDP barrier.
"""

from __future__ import annotations

import gc
import os
import pickle
import shutil
import sys
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup for cross-directory imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.data_loader import (
    build_parent_child_mapping,
    extract_level_2,
    load_random_sample,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_shards(
    filename: str,
    n_samples: int,
    world_size: int,
    shard_dir: str,
    test_size: float = 0.2,
    val_ratio: float = 0.125,
    seed: int = 42,
) -> dict:
    """Prepare data shards for distributed training.

    Loads a random sample from the JSONL dataset, encodes hierarchical labels,
    splits into train/val/test, and saves per-rank training shards plus
    shared validation and test sets.

    This function should only be called by rank 0.

    Args:
        filename: Path to the JSONL dataset file.
        n_samples: Number of samples to load.
        world_size: Number of distributed processes (determines shard count).
        shard_dir: Directory to write shard files.
        test_size: Fraction of data for test split.
        val_ratio: Fraction of remaining data for validation split.
        seed: Random seed for reproducibility.

    Returns:
        Metadata dictionary containing label encoders, class counts,
        and dataset sizes.
    """
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    print(f"[Rank 0] Loading {n_samples:,} samples from {filename}...")
    df = load_random_sample(filename, n_samples=n_samples, seed=seed)

    # Preprocess hierarchical labels
    df["level_1"] = df["typelabel"].fillna("Unknown")
    df["level_2_raw"] = df["producttype"].apply(extract_level_2)
    df = df[df["level_2_raw"].notna()].copy()
    df["level_2"] = df["level_2_raw"]
    df["designation"] = df["designation"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = df["designation"] + ". " + df["description"]
    df = df[["text", "level_1", "level_2"]]

    print(f"[Rank 0] Hierarchical samples: {len(df):,}")

    # Encode labels
    l1_encoder = LabelEncoder()
    l2_encoder = LabelEncoder()
    df["label_l1"] = l1_encoder.fit_transform(df["level_1"])
    df["label_l2"] = l2_encoder.fit_transform(df["level_2"])

    num_classes_l1 = len(l1_encoder.classes_)
    num_classes_l2 = len(l2_encoder.classes_)

    l2_to_l1_mapping = build_parent_child_mapping(
        df["label_l1"].values, df["label_l2"].values
    )

    # Train / val / test split
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
        # Fallback without stratification for rare classes
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

    # Write per-rank training shards
    X_train_list = X_train.tolist()
    y1_train_list = y1_train.tolist()
    y2_train_list = y2_train.tolist()
    n_total = len(X_train_list)

    for r in range(world_size):
        start = r * n_total // world_size
        end = (r + 1) * n_total // world_size
        shard = {
            "texts": X_train_list[start:end],
            "labels_l1": y1_train_list[start:end],
            "labels_l2": y2_train_list[start:end],
        }
        shard_path = os.path.join(shard_dir, f"train_shard_{r}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write shared validation and test sets
    val_data = {
        "texts": X_val.tolist(),
        "labels_l1": y1_val.tolist(),
        "labels_l2": y2_val.tolist(),
    }
    test_data = {
        "texts": X_test.tolist(),
        "labels_l1": y1_test.tolist(),
        "labels_l2": y2_test.tolist(),
    }

    with open(os.path.join(shard_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(shard_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write metadata
    metadata = {
        "l1_encoder": l1_encoder,
        "l2_encoder": l2_encoder,
        "num_classes_l1": num_classes_l1,
        "num_classes_l2": num_classes_l2,
        "l2_to_l1_mapping": l2_to_l1_mapping,
        "world_size": world_size,
        "n_train_total": n_total,
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    with open(os.path.join(shard_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[Rank 0] Shards ready: {world_size} train shards, "
        f"{n_total:,} train / {len(X_val):,} val / {len(X_test):,} test samples."
    )
    return metadata
