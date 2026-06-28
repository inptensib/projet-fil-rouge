"""Data loading and preprocessing for the Rakuten product classification dataset.

Handles loading from large JSONL files, label encoding, hierarchical label
extraction, and train/val/test splitting.
"""

from __future__ import annotations

import gc
import json
import logging
import random
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch

from rakuten.common.text_normalization import normalize_text

def _cpu_has_avx2() -> bool:
    """Check if CPU supports AVX2 (required by Polars)."""
    try:
        with open("/proc/cpuinfo") as f:
            return "avx2" in f.read()
    except OSError:
        return True  # assume modern CPU if can't check

_HAS_POLARS = False
if _cpu_has_avx2():
    try:
        import polars as pl
        _HAS_POLARS = True
    except ImportError:
        pass

if not _HAS_POLARS:
    print("[Data] Polars indisponible (CPU sans AVX2 ou non installé) → fallback pandas")
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


def load_jsonl_full(file_path: str):
    """Load all records from a JSONL file.

    Uses Polars (Rust/Arrow) if available, falls back to pandas.
    Polars avoids pymalloc fragmentation on large datasets.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Polars or pandas DataFrame with all valid records.
    """
    if _HAS_POLARS:
        df = pl.read_ndjson(file_path)
    else:
        df = pd.read_json(file_path, lines=True)
    print(f"[Data] Loaded {len(df):,} records from {file_path}.")
    return df


_SYNTH_PREFIX = "SYNTH-"
_SYNTH_KIND_PATTERN = r"^SYNTH-([A-Z]+)"
_OVERSAMPLED_KINDS = ("OVS", "OVL")


def _preprocess_df(df, use_description: bool = True, normalize: bool = False):
    """Extract hierarchical labels, text and synthetic flags from a raw JSONL DataFrame.

    Supports both Polars and pandas DataFrames. Rows whose productid starts
    with "SYNTH-" are flagged with their generation kind (ORPH, OVS, OVL, GEN).
    If normalize is True, normalize_text is applied to designation and
    description before building the text field.
    Returns DataFrame with columns: text, level_1, level_2, is_synthetic, synth_kind.
    """
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        invalid_l2 = ["", "none", "nan", "unknown"]
        designation = pl.col("designation").fill_null("")
        description = pl.col("description").fill_null("")
        if normalize:
            designation = designation.map_elements(normalize_text, return_dtype=pl.Utf8)
            description = description.map_elements(normalize_text, return_dtype=pl.Utf8)
        if "productid" in df.columns:
            pid = pl.col("productid").cast(pl.Utf8).fill_null("")
            is_synthetic = pid.str.starts_with(_SYNTH_PREFIX)
            synth_kind = pid.str.extract(_SYNTH_KIND_PATTERN, 1)
        else:
            is_synthetic = pl.lit(False)
            synth_kind = pl.lit(None, dtype=pl.Utf8)
        return (
            df.with_columns(
                pl.col("typelabel").fill_null("Unknown").alias("level_1"),
                designation,
                description,
                pl.col("producttype").list.first().cast(pl.Utf8).alias("level_2_raw"),
                is_synthetic.alias("is_synthetic"),
                synth_kind.alias("synth_kind"),
            )
            .filter(
                pl.col("level_2_raw").is_not_null()
                & ~pl.col("level_2_raw").str.to_lowercase().is_in(invalid_l2)
            )
            .with_columns(
                (
                    (pl.col("designation") + ". " + pl.col("description"))
                    if use_description
                    else pl.col("designation")
                ).alias("text"),
                pl.col("level_2_raw").str.strip_chars().alias("level_2"),
            )
            .select("text", "level_1", "level_2", "is_synthetic", "synth_kind")
        )
    else:
        # Pandas fallback
        df["level_1"] = df["typelabel"].fillna("Unknown")
        df["level_2_raw"] = df["producttype"].apply(extract_level_2)
        df = df[df["level_2_raw"].notna()].copy()
        df["level_2"] = df["level_2_raw"]
        df["designation"] = df["designation"].fillna("")
        df["description"] = df["description"].fillna("")
        if normalize:
            df["designation"] = df["designation"].map(normalize_text)
            df["description"] = df["description"].map(normalize_text)
        if "productid" in df.columns:
            pid = df["productid"].fillna("").astype(str)
            df["is_synthetic"] = pid.str.startswith(_SYNTH_PREFIX)
            df["synth_kind"] = pid.str.extract(_SYNTH_KIND_PATTERN, expand=False)
        else:
            df["is_synthetic"] = False
            df["synth_kind"] = None
        if use_description:
            df["text"] = df["designation"] + ". " + df["description"]
        else:
            df["text"] = df["designation"]
        return df[["text", "level_1", "level_2", "is_synthetic", "synth_kind"]]


def _drop_oversampled_rows(df_train):
    """Remove oversampled synthetic rows (SYNTH-OVS, SYNTH-OVL) from the train split."""
    n_before = len(df_train)
    if _HAS_POLARS and isinstance(df_train, pl.DataFrame):
        df_train = df_train.filter(
            ~pl.col("synth_kind").fill_null("").is_in(list(_OVERSAMPLED_KINDS))
        )
    else:
        df_train = df_train[~df_train["synth_kind"].isin(_OVERSAMPLED_KINDS)].copy()
    print(
        f"[Data] synthetic_policy=drop_oversampled: "
        f"{n_before - len(df_train):,} lignes SYNTH-OVS/OVL retirées du train"
    )
    return df_train


def prepare_hierarchical_data(
    file_path: str,
    n_samples: int = 10_000,
    test_size: float = 0.2,
    val_ratio: float = 0.125,
    seed: int = 42,
    use_description: bool = True,
    normalize: bool = False,
) -> dict:
    """Load JSONL data and prepare hierarchical labels with train/val/test splits.

    synthetic_policy is not supported on this single-file path (the split
    happens after loading); use prepare_hierarchical_data_from_splits for
    the v2 dataset.

    Args:
        file_path: Path to the JSONL dataset.
        n_samples: Number of samples to load.
        test_size: Fraction for test set (from total).
        val_ratio: Fraction for val set (from remaining after test).
        seed: Random seed.
        normalize: If True, apply normalize_text to designation and description.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_l1_train, y_l1_val,
        y_l1_test, y_l2_train, y_l2_val, y_l2_test, l1_encoder, l2_encoder,
        l2_to_l1_mapping, num_classes_l1, num_classes_l2.
    """
    df = load_random_sample(file_path, n_samples=n_samples, seed=seed)
    df = _preprocess_df(df, use_description=use_description, normalize=normalize)

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


def _ensure_l2_coverage(df_train, df_val, df_test):
    """Garantit que chaque classe L2 a au moins 1 échantillon dans chaque split.

    Supporte Polars et pandas. Utilise des sets pour O(1) lookup.
    """
    is_polars = _HAS_POLARS and isinstance(df_train, pl.DataFrame)

    def _get_unique(df, col):
        if is_polars:
            return set(df[col].unique().to_list())
        return set(df[col].unique())

    def _filter_first(df, col, value):
        if is_polars:
            return df.filter(pl.col(col) == value).head(1)
        return df[df[col] == value].iloc[[0]]

    def _concat(dfs):
        if is_polars:
            return pl.concat(dfs)
        return pd.concat(dfs, ignore_index=True)

    l2_sets = {
        "train": _get_unique(df_train, "level_2"),
        "val": _get_unique(df_val, "level_2"),
        "test": _get_unique(df_test, "level_2"),
    }
    all_l2 = l2_sets["train"] | l2_sets["val"] | l2_sets["test"]
    splits = {"train": df_train, "val": df_val, "test": df_test}

    rows_to_add: dict[str, list] = {"train": [], "val": [], "test": []}
    for l2_class in all_l2:
        present = {name for name in splits if l2_class in l2_sets[name]}
        missing = set(splits) - present
        if not missing or not present:
            continue
        source_name = next(iter(present))
        sample = _filter_first(splits[source_name], "level_2", l2_class)
        for name in missing:
            rows_to_add[name].append(sample)

    added = sum(len(v) for v in rows_to_add.values())
    if added > 0:
        for name in splits:
            if rows_to_add[name]:
                splits[name] = _concat([splits[name]] + rows_to_add[name])
        print(f"[Data] {added} échantillons dupliqués pour couvrir toutes les classes L2 dans chaque split")

    return splits["train"], splits["val"], splits["test"]


def prepare_hierarchical_data_from_splits(
    train_path: str,
    val_path: str,
    test_path: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
    use_description: bool = True,
    synthetic_policy: str = "keep_all",
    normalize: bool = False,
) -> dict:
    """Load pre-split JSONL files and prepare hierarchical labels.

    Returns the same dict format as prepare_hierarchical_data(), plus
    synthetic-data tracking keys (dataset v2: productid prefixed SYNTH-).
    LabelEncoders are fitted on the union of labels from all 3 splits.

    Args:
        train_path: Path to train_split.jsonl.
        val_path: Path to val_split.jsonl.
        test_path: Path to test_split.jsonl.
        n_samples: If set, sample this many total (proportional per split).
                   If None, load all data.
        seed: Random seed for optional sampling.
        synthetic_policy: "keep_all" keeps every row; "drop_oversampled"
            removes SYNTH-OVS and SYNTH-OVL rows from the train split only
            (SYNTH-ORPH and SYNTH-GEN are kept, val/test are never filtered).
        normalize: If True, apply normalize_text to designation and
            description of the 3 splits at load time.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_l1_train, y_l1_val,
        y_l1_test, y_l2_train, y_l2_val, y_l2_test, l1_encoder, l2_encoder,
        l2_to_l1_mapping, num_classes_l1, num_classes_l2, train_is_synthetic
        (list of bool aligned with X_train), real_class_counts_l1 and
        real_class_counts_l2 (Counter of encoded train labels on real rows
        only, the basis for class weights), and synthetic_policy (the policy
        applied, checked by the trainers before enabling class weighting).
    """
    if synthetic_policy not in ("keep_all", "drop_oversampled"):
        raise ValueError(
            f"synthetic_policy invalide: {synthetic_policy!r} "
            "(attendu: 'keep_all' ou 'drop_oversampled')"
        )

    if n_samples is not None:
        n_train = int(n_samples * 0.70)
        n_val = int(n_samples * 0.15)
        n_test = n_samples - n_train - n_val
        df_train = _preprocess_df(load_random_sample(train_path, n_train, seed), use_description=use_description, normalize=normalize)
        df_val = _preprocess_df(load_random_sample(val_path, n_val, seed), use_description=use_description, normalize=normalize)
        df_test = _preprocess_df(load_random_sample(test_path, n_test, seed), use_description=use_description, normalize=normalize)
    else:
        df_train = _preprocess_df(load_jsonl_full(train_path), use_description=use_description, normalize=normalize)
        gc.collect()
        df_val = _preprocess_df(load_jsonl_full(val_path), use_description=use_description, normalize=normalize)
        gc.collect()
        df_test = _preprocess_df(load_jsonl_full(test_path), use_description=use_description, normalize=normalize)
        gc.collect()

    print(
        f"[Data] Pre-split loaded: "
        f"{len(df_train):,} train / {len(df_val):,} val / {len(df_test):,} test"
    )

    if synthetic_policy == "drop_oversampled":
        df_train = _drop_oversampled_rows(df_train)

    # Garantir que chaque classe L2 a au moins 1 échantillon dans chaque split
    df_train, df_val, df_test = _ensure_l2_coverage(df_train, df_val, df_test)
    print(f"[Data] use_description={use_description}")

    is_polars = _HAS_POLARS and isinstance(df_train, pl.DataFrame)

    def _to_list(series):
        return series.to_list() if is_polars else list(series)

    def _concat_series(s1, s2, s3):
        if is_polars:
            return pl.concat([s1, s2, s3]).to_list()
        return pd.concat([s1, s2, s3]).tolist()

    # Fit encoders on the union of all labels
    all_l1 = _concat_series(df_train["level_1"], df_val["level_1"], df_test["level_1"])
    all_l2 = _concat_series(df_train["level_2"], df_val["level_2"], df_test["level_2"])

    l1_encoder = LabelEncoder()
    l2_encoder = LabelEncoder()
    l1_encoder.fit(all_l1)
    l2_encoder.fit(all_l2)

    del all_l1, all_l2

    # Encode each split
    y_l1_train = l1_encoder.transform(_to_list(df_train["level_1"]))
    y_l2_train = l2_encoder.transform(_to_list(df_train["level_2"]))
    y_l1_val = l1_encoder.transform(_to_list(df_val["level_1"]))
    y_l2_val = l2_encoder.transform(_to_list(df_val["level_2"]))
    y_l1_test = l1_encoder.transform(_to_list(df_test["level_1"]))
    y_l2_test = l2_encoder.transform(_to_list(df_test["level_2"]))

    # Build parent-child mapping from all data
    all_l1_enc = np.concatenate([y_l1_train, y_l1_val, y_l1_test])
    all_l2_enc = np.concatenate([y_l2_train, y_l2_val, y_l2_test])
    l2_to_l1 = build_parent_child_mapping(all_l1_enc, all_l2_enc)

    del all_l1_enc, all_l2_enc

    # Convert texts to Python lists (single point where Python objects are created)
    X_train = df_train["text"].to_list()
    X_val = df_val["text"].to_list()
    X_test = df_test["text"].to_list()

    train_is_synthetic = [bool(v) for v in _to_list(df_train["is_synthetic"])]
    real_mask = ~np.asarray(train_is_synthetic, dtype=bool)
    real_class_counts_l1 = Counter(int(v) for v in y_l1_train[real_mask])
    real_class_counts_l2 = Counter(int(v) for v in y_l2_train[real_mask])

    del df_train, df_val, df_test
    gc.collect()

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_l1_train": y_l1_train, "y_l1_val": y_l1_val, "y_l1_test": y_l1_test,
        "y_l2_train": y_l2_train, "y_l2_val": y_l2_val, "y_l2_test": y_l2_test,
        "l1_encoder": l1_encoder, "l2_encoder": l2_encoder,
        "l2_to_l1_mapping": l2_to_l1,
        "num_classes_l1": len(l1_encoder.classes_),
        "num_classes_l2": len(l2_encoder.classes_),
        "train_is_synthetic": train_is_synthetic,
        "real_class_counts_l1": real_class_counts_l1,
        "real_class_counts_l2": real_class_counts_l2,
        "synthetic_policy": synthetic_policy,
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


def build_parent_child_matrix_from_labels(
    level_1_labels, level_2_labels, num_l1_classes: int, num_l2_classes: int
) -> torch.Tensor:
    """Matrice binaire [num_l1, num_l2] marquant TOUTES les aretes (L1, L2) vues.

    Une L2 a plusieurs parents L1 reste valide sous chacun -> en teacher forcing
    la vraie classe n'est jamais masquee a -1e9 (corrige la loss qui explosait).
    """
    matrix = torch.zeros(num_l1_classes, num_l2_classes)
    l1_t = torch.as_tensor(np.asarray(level_1_labels), dtype=torch.long)
    l2_t = torch.as_tensor(np.asarray(level_2_labels), dtype=torch.long)
    matrix[l1_t, l2_t] = 1.0
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

    def __init__(self, texts, labels_l1, labels_l2, tokenizer, max_length: int = 256):
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