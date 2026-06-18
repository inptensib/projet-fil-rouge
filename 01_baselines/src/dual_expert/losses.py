"""Enhanced loss functions for Dual-Expert Domain training.

Adds focal cross-entropy with optional class weights and effective-number
class-balanced weighting for L2 (6126 classes, heavily imbalanced).

Backward-compatible: the existing HYDRALoss is unaffected.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal cross-entropy for multi-class single-label classification.

    pt is derived from the *unweighted* log-probability of the true class so
    that the focal modulation reflects the model's actual confidence, not the
    re-weighted loss magnitude.  Class weights are applied as a multiplicative
    factor on the per-sample loss after the focal term is computed.

    Formula (per sample):
        log_pt = -CE_unweighted(logits, target)
        pt     = exp(log_pt)
        loss   = (1 - pt)^gamma * (-log_pt) * class_weight[target]

    Args:
        logits: (B, C) raw logits.
        targets: (B,) integer class indices.
        class_weights: (C,) optional per-class weights; auto-moved to logits device.
        gamma: Focal modulation. gamma=0 → standard weighted CE.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (reduction != 'none') or per-sample tensor (B,).
    """
    # Unweighted CE so pt = softmax(logits)[target] — true model confidence
    ce_unweighted = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_unweighted)
    focal_weight = (1.0 - pt).clamp(min=1e-6) ** gamma

    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
        sample_weights = class_weights[targets]
        loss = focal_weight * ce_unweighted * sample_weights
    else:
        loss = focal_weight * ce_unweighted

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def build_effective_number_weights(
    labels_l2: list[int] | torch.Tensor,
    num_classes: int,
    beta: float = 0.9999,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Class-balanced effective-number weights for L2.

    Reference: Cui et al. (2019) — Class-Balanced Loss Based on Effective
    Number of Samples.

    Formula:
        effective_num_c = 1 - beta^n_c
        weight_c = (1 - beta) / effective_num_c
        weight = weight / mean(weight)     # normalize
        weight = clip(weight, max=max_weight)

    Args:
        labels_l2: Training-set L2 integer labels.
        num_classes: Total number of L2 classes.
        beta: Smoothing parameter. Higher values give more balanced weights.
        max_weight: Clip rare-class weight explosion.

    Returns:
        Float32 tensor of shape (num_classes,).
    """
    if isinstance(labels_l2, torch.Tensor):
        labels_l2 = labels_l2.tolist()

    counts = Counter(labels_l2)
    raw_weights: list[float] = []
    absent: list[int] = []

    for c in range(num_classes):
        n = counts.get(c, 0)
        if n == 0:
            # Class absent from training data — mark for max-weight assignment
            absent.append(c)
            raw_weights.append(0.0)
        else:
            effective_num = 1.0 - beta ** n
            w = (1.0 - beta) / effective_num
            raw_weights.append(w)

    wt = torch.tensor(raw_weights, dtype=torch.float32)

    # Normalize using only present classes so absent ones don't skew the mean
    present = torch.ones(num_classes, dtype=torch.bool)
    for idx in absent:
        present[idx] = False
    if present.any():
        wt = wt / wt[present].mean()

    wt = torch.clamp(wt, max=max_weight)

    # Absent classes receive the maximum weight explicitly
    for idx in absent:
        wt[idx] = max_weight

    return wt


def build_frequency_buckets(
    labels_l2: list[int],
    num_classes: int,
) -> dict[str, list[int]]:
    """Assign each L2 class index to a frequency bucket.

    Buckets (by training-set sample count):
        rare_0_10:          n < 10
        rare_10_50:    10 ≤ n < 50
        rare_50_100:   50 ≤ n < 100
        medium_100_500:100 ≤ n < 500
        frequent_500_plus: n ≥ 500

    Returns:
        Dict bucket_name → list[l2_class_index].
    """
    counts = Counter(labels_l2)
    buckets: dict[str, list[int]] = {
        "rare_0_10": [],
        "rare_10_50": [],
        "rare_50_100": [],
        "medium_100_500": [],
        "frequent_500_plus": [],
    }
    for c in range(num_classes):
        n = counts.get(c, 0)
        if n < 10:
            buckets["rare_0_10"].append(c)
        elif n < 50:
            buckets["rare_10_50"].append(c)
        elif n < 100:
            buckets["rare_50_100"].append(c)
        elif n < 500:
            buckets["medium_100_500"].append(c)
        else:
            buckets["frequent_500_plus"].append(c)
    return buckets


def save_class_weight_artifacts(
    weights: torch.Tensor,
    labels_l2: list[int],
    num_classes: int,
    output_dir: str | Path,
) -> None:
    """Persist class weights and frequency bucket summary to disk.

    Creates:
        {output_dir}/l2_class_weights.pt
        {output_dir}/l2_class_counts.json
        {output_dir}/l2_frequency_buckets.json

    Args:
        weights: Tensor (num_classes,).
        labels_l2: Training L2 integer labels.
        num_classes: Total L2 classes.
        output_dir: Destination directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(weights, out / "l2_class_weights.pt")

    counts = dict(Counter(labels_l2))
    full_counts = {str(c): counts.get(c, 0) for c in range(num_classes)}
    with open(out / "l2_class_counts.json", "w") as f:
        json.dump(full_counts, f)

    buckets = build_frequency_buckets(labels_l2, num_classes)
    bucket_summary = {k: len(v) for k, v in buckets.items()}
    with open(out / "l2_frequency_buckets.json", "w") as f:
        json.dump(bucket_summary, f, indent=2)

    print(f"[DualExpert] Class weights saved → {out}/")
    for bucket_name, count in bucket_summary.items():
        print(f"  {bucket_name}: {count} classes")
