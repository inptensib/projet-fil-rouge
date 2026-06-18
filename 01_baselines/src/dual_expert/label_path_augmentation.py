"""Label-path augmentation for Domain Expert training.

Prepends the hierarchical category path to rare-class samples during training
to expose the model to the semantic structure of the taxonomy.

Variants:
    A — full path: [CAT] L1_NAME > L2_NAME [TEXT] designation. description
    B — masked L1: [CAT] <mask> > L2_NAME [TEXT] ...
    C — masked L2: [CAT] L1_NAME > <mask> [TEXT] ...

The augmentation is applied ONLY to training data (never val/test).
Uses CamemBERT's native <mask> token.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LabelPathAugConfig:
    """Configuration for label-path augmentation.

    Attributes:
        enabled: Toggle augmentation on/off.
        apply_to: Which samples to augment ('all', 'rare_only', 'none').
        rare_threshold: Max training count to consider a L2 class rare.
        augmentation_factor_rare: Repeat factor for classes with count < rare_threshold.
        augmentation_factor_medium: Repeat factor for classes above rare_threshold.
        mask_probability: Used only when allow_full_path=True. Prob to replace
            one hierarchy level with <mask>. Ignored when allow_full_path=False.
        prepend_probability: Prob that a selected sample gets the path prefix.
        allow_full_path: If False (default), exactly one hierarchy level is always
            masked so the model never sees the complete gold L1 > L2 path during
            augmented training. Set to True only for ablation studies.
    """

    enabled: bool = False
    apply_to: str = "rare_only"          # "all" | "rare_only" | "none"
    rare_threshold: int = 100
    augmentation_factor_rare: int = 3
    augmentation_factor_medium: int = 2
    mask_probability: float = 0.3
    prepend_probability: float = 1.0
    allow_full_path: bool = False

    def __post_init__(self) -> None:
        if self.apply_to not in ("all", "rare_only", "none"):
            raise ValueError(
                f"apply_to must be 'all', 'rare_only', or 'none', got '{self.apply_to}'"
            )


class AugmentedProductDataset(Dataset):
    """PyTorch Dataset that applies label-path augmentation for training.

    Expands rare-class samples by adding additional augmented copies.
    Each augmented copy prepends the category path (with optional masking)
    to the product text.

    Args:
        texts: Product texts (designation + description or designation only).
        labels_l1: Integer L1 labels (N,).
        labels_l2: Integer L2 labels (N,).
        tokenizer: HuggingFace tokenizer (CamemBERT expected).
        max_length: Maximum tokenisation length.
        l1_encoder: LabelEncoder for L1 (to recover class names).
        l2_encoder: LabelEncoder for L2 (to recover class names).
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        config: Augmentation configuration.
        seed: Random seed for reproducibility.
    """

    MASK_TOKEN = "<mask>"       # CamemBERT native mask token
    CAT_TOKEN = "[CAT]"
    TEXT_TOKEN = "[TEXT]"

    def __init__(
        self,
        texts: list[str],
        labels_l1,
        labels_l2,
        tokenizer,
        max_length: int = 256,
        l1_encoder=None,
        l2_encoder=None,
        l2_to_l1_mapping: Optional[dict] = None,
        config: Optional[LabelPathAugConfig] = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.l1_encoder = l1_encoder
        self.l2_encoder = l2_encoder
        self.l2_to_l1_mapping = l2_to_l1_mapping or {}
        self.config = config or LabelPathAugConfig(enabled=False)
        self.rng = random.Random(seed)

        # Convert to Python lists for uniform indexing
        self.base_texts = list(texts) if not isinstance(texts, list) else texts
        labels_l1_arr = np.asarray(labels_l1)
        labels_l2_arr = np.asarray(labels_l2)
        self.base_labels_l1 = labels_l1_arr.tolist()
        self.base_labels_l2 = labels_l2_arr.tolist()

        # Build index list: (original_idx, is_augmented)
        self._index = self._build_index(labels_l2_arr)
        self._log_augmentation_summary()

    def _l2_counts(self) -> dict[int, int]:
        return dict(Counter(self.base_labels_l2))

    def _build_index(self, labels_l2_arr: np.ndarray) -> list[tuple[int, bool]]:
        """Build expanded index with augmented copies for rare classes."""
        if not self.config.enabled or self.config.apply_to == "none":
            return [(i, False) for i in range(len(self.base_texts))]

        counts = self._l2_counts()
        index: list[tuple[int, bool]] = []

        for i, l2 in enumerate(self.base_labels_l2):
            n = counts.get(l2, 0)
            index.append((i, False))    # always include original

            if self.config.apply_to == "rare_only" and n >= self.config.rare_threshold:
                continue                # skip augmentation for non-rare

            factor = (
                self.config.augmentation_factor_rare
                if n < self.config.rare_threshold
                else self.config.augmentation_factor_medium
            )
            for _ in range(factor - 1):
                index.append((i, True))

        return index

    def _log_augmentation_summary(self) -> None:
        if not self.config.enabled:
            return
        n_orig = len(self.base_texts)
        n_total = len(self._index)
        n_aug = n_total - n_orig
        print(
            f"[LabelPathAug] Base samples: {n_orig:,} | "
            f"Augmented copies: {n_aug:,} | "
            f"Total: {n_total:,}"
        )

    def _build_text(self, idx: int, is_augmented: bool) -> str:
        base_text = str(self.base_texts[idx])

        if not is_augmented or not self.config.enabled:
            return base_text

        if self.rng.random() > self.config.prepend_probability:
            return base_text

        l1_id = int(self.base_labels_l1[idx])
        l2_id = int(self.base_labels_l2[idx])

        l1_name = (
            str(self.l1_encoder.classes_[l1_id])
            if self.l1_encoder is not None and 0 <= l1_id < len(self.l1_encoder.classes_)
            else f"L1_{l1_id}"
        )
        l2_name = (
            str(self.l2_encoder.classes_[l2_id])
            if self.l2_encoder is not None and 0 <= l2_id < len(self.l2_encoder.classes_)
            else f"L2_{l2_id}"
        )

        if not self.config.allow_full_path:
            # Enforce at least one masked level — never expose the full gold path
            if self.rng.random() < 0.5:
                l1_name = self.MASK_TOKEN
            else:
                l2_name = self.MASK_TOKEN
        elif self.rng.random() < self.config.mask_probability:
            # Ablation mode (allow_full_path=True): probabilistic masking
            if self.rng.random() < 0.5:
                l1_name = self.MASK_TOKEN
            else:
                l2_name = self.MASK_TOKEN
        # else: full path visible (ablation only)

        return f"{self.CAT_TOKEN} {l1_name} > {l2_name} {self.TEXT_TOKEN} {base_text}"

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        orig_idx, is_augmented = self._index[idx]
        text = self._build_text(orig_idx, is_augmented)

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
            "label_l1": torch.as_tensor(self.base_labels_l1[orig_idx], dtype=torch.long),
            "label_l2": torch.as_tensor(self.base_labels_l2[orig_idx], dtype=torch.long),
            "sample_idx": orig_idx,
            "is_augmented": int(is_augmented),
        }
