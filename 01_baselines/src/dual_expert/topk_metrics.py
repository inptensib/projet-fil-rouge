"""Top-K recall metrics and rare-bucket evaluation for the Dual-Expert.

Top-K recall@K = fraction of samples where the true L2 is in the top-K
predicted L2 candidates.

Also computes per-frequency-bucket breakdowns and saves a JSON report.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def topk_recall(
    labels_l2: list[int],
    topk_preds: list[list[int]],
    k: int,
) -> float:
    """Compute recall@K: fraction of samples where true L2 is in top-K preds.

    Args:
        labels_l2: Ground-truth L2 integer labels, length N.
        topk_preds: Top-K predictions per sample; topk_preds[i] is a list of
                    up to K integer L2 indices sorted by score (best first).
        k: Recall threshold to check.

    Returns:
        Scalar recall in [0, 1].
    """
    if len(labels_l2) == 0:
        return 0.0
    correct = sum(
        1 for true, preds in zip(labels_l2, topk_preds)
        if true in preds[:k]
    )
    return correct / len(labels_l2)


def compute_topk_recall_dict(
    labels_l2: list[int],
    topk_preds: list[list[int]],
    k_values: list[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Compute recall@K for multiple K values.

    Args:
        labels_l2: Ground-truth L2 labels.
        topk_preds: Top-K candidate lists per sample (must have max(k_values) entries).
        k_values: List of K thresholds.

    Returns:
        Dict with keys like 'l2_top_1_accuracy', 'l2_top_3_recall', etc.
    """
    result = {}
    for k in k_values:
        key = f"l2_top_{k}_{'accuracy' if k == 1 else 'recall'}"
        result[key] = topk_recall(labels_l2, topk_preds, k)
    return result


# ---------------------------------------------------------------------------
# Frequency buckets
# ---------------------------------------------------------------------------


def assign_l2_to_buckets(
    train_labels_l2: list[int],
    num_classes: int,
) -> dict[int, str]:
    """Map each L2 class index to its frequency bucket name.

    Buckets:
        rare_0_10, rare_10_50, rare_50_100, medium_100_500, frequent_500_plus
    """
    counts = Counter(train_labels_l2)
    mapping: dict[int, str] = {}
    for c in range(num_classes):
        n = counts.get(c, 0)
        if n < 10:
            mapping[c] = "rare_0_10"
        elif n < 50:
            mapping[c] = "rare_10_50"
        elif n < 100:
            mapping[c] = "rare_50_100"
        elif n < 500:
            mapping[c] = "medium_100_500"
        else:
            mapping[c] = "frequent_500_plus"
    return mapping


def compute_topk_metrics_by_bucket(
    labels_l2: list[int],
    topk_preds: list[list[int]],
    train_labels_l2: list[int],
    num_classes: int,
    k_values: list[int] = (1, 3, 5, 10, 20),
) -> dict[str, dict]:
    """Compute top-K recall per frequency bucket.

    Args:
        labels_l2: Evaluation-set ground-truth L2 labels.
        topk_preds: Top-K candidate lists per sample.
        train_labels_l2: Training-set L2 labels (for frequency counts).
        num_classes: Total number of L2 classes.
        k_values: K thresholds to evaluate.

    Returns:
        Dict bucket_name → {metric_name: value, count_examples, count_classes}.
    """
    bucket_map = assign_l2_to_buckets(train_labels_l2, num_classes)
    train_counts = Counter(train_labels_l2)

    # Group evaluation samples by bucket
    bucket_samples: dict[str, list[tuple[int, list[int]]]] = {
        b: [] for b in (
            "rare_0_10", "rare_10_50", "rare_50_100", "medium_100_500", "frequent_500_plus"
        )
    }
    for true, preds in zip(labels_l2, topk_preds):
        bucket = bucket_map.get(true, "frequent_500_plus")
        bucket_samples[bucket].append((true, preds))

    results: dict[str, dict] = {}
    for bucket, samples in bucket_samples.items():
        if not samples:
            results[bucket] = {"count_examples": 0, "count_classes": 0}
            continue
        b_labels = [s[0] for s in samples]
        b_preds = [s[1] for s in samples]
        metrics = compute_topk_recall_dict(b_labels, b_preds, k_values)
        metrics["count_examples"] = len(samples)
        metrics["count_classes"] = len(set(b_labels))
        results[bucket] = metrics

    return results


# ---------------------------------------------------------------------------
# Logits → top-K extraction (used during evaluation loop)
# ---------------------------------------------------------------------------


def extract_topk_from_logits(
    logits_l2: torch.Tensor,
    k: int,
) -> list[list[int]]:
    """Extract top-K L2 indices from a batch of raw L2 logits.

    Args:
        logits_l2: (B, num_l2) raw L2 logits (or scores).
        k: Number of top candidates.

    Returns:
        List of B lists, each containing k L2 indices (sorted best first).
    """
    k_actual = min(k, logits_l2.shape[1])
    _, top_ids = torch.topk(logits_l2, k=k_actual, dim=-1)
    return top_ids.cpu().tolist()


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def build_topk_recall_report(
    labels_l2: list[int],
    topk_preds: list[list[int]],
    train_labels_l2: list[int],
    num_classes: int,
    k_values: list[int] = (1, 3, 5, 10, 20),
) -> dict:
    """Build a complete top-K recall report with global and bucketed metrics.

    Args:
        labels_l2: Evaluation ground-truth L2 labels.
        topk_preds: Top-K predictions per sample.
        train_labels_l2: Training L2 labels for frequency counts.
        num_classes: Total number of L2 classes.
        k_values: K values to evaluate.

    Returns:
        Report dict with keys 'global' and 'by_bucket'.
    """
    global_metrics = compute_topk_recall_dict(labels_l2, topk_preds, k_values)
    bucket_metrics = compute_topk_metrics_by_bucket(
        labels_l2, topk_preds, train_labels_l2, num_classes, k_values,
    )
    return {
        "global": global_metrics,
        "by_bucket": bucket_metrics,
        "num_samples": len(labels_l2),
        "num_classes": num_classes,
        "k_values": list(k_values),
    }


def save_topk_recall_report(report: dict, output_path: str | Path) -> None:
    """Save the top-K recall report to a JSON file.

    Args:
        report: Output of build_topk_recall_report().
        output_path: Destination file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DualExpert] Top-K recall report saved → {p}")
