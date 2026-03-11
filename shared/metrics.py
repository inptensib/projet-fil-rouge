"""Evaluation metrics for hierarchical classification.

Includes standard F1 scores and hierarchy-specific metrics inspired by
the Fatos Torba thesis (HMTC evaluation protocol).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score


def compute_f1_metrics(
    labels_l1: list,
    preds_l1: list,
    labels_l2: list,
    preds_l2: list,
) -> dict[str, float]:
    """Compute F1 scores for both hierarchy levels.

    Args:
        labels_l1: Ground truth L1 labels.
        preds_l1: Predicted L1 labels.
        labels_l2: Ground truth L2 labels.
        preds_l2: Predicted L2 labels.

    Returns:
        Dictionary with f1_l1_macro, f1_l1_weighted, f1_l2_macro, f1_l2_weighted.
    """
    return {
        "f1_l1_macro": f1_score(labels_l1, preds_l1, average="macro", zero_division=0),
        "f1_l1_weighted": f1_score(labels_l1, preds_l1, average="weighted", zero_division=0),
        "f1_l2_macro": f1_score(labels_l2, preds_l2, average="macro", zero_division=0),
        "f1_l2_weighted": f1_score(labels_l2, preds_l2, average="weighted", zero_division=0),
    }


def constrained_f1(
    labels_l1: list,
    preds_l1: list,
    labels_l2: list,
    preds_l2: list,
    l2_to_l1_mapping: dict[int, int],
    average: str = "macro",
) -> float:
    """Compute Constrained F1: L2 predictions only count as correct
    if the corresponding L1 ancestor is also correctly predicted.

    Reference: Fatos Torba thesis, Section 2.4.2.

    Args:
        labels_l1: Ground truth L1 labels.
        preds_l1: Predicted L1 labels.
        labels_l2: Ground truth L2 labels.
        preds_l2: Predicted L2 labels.
        l2_to_l1_mapping: Mapping from L2 index to L1 index.
        average: 'macro' or 'weighted'.

    Returns:
        Constrained F1 score.
    """
    constrained_preds_l2 = []
    for p_l1, true_l1, p_l2 in zip(preds_l1, labels_l1, preds_l2):
        # L2 prediction is valid only if L1 prediction matches ground truth
        if p_l1 == true_l1:
            constrained_preds_l2.append(p_l2)
        else:
            constrained_preds_l2.append(-1)  # Invalid prediction

    return f1_score(
        labels_l2, constrained_preds_l2, average=average, zero_division=0
    )


def hierarchical_consistency_rate(
    preds_l1: list,
    preds_l2: list,
    l2_to_l1_mapping: dict[int, int],
) -> float:
    """Compute Hierarchical Consistency Rate (HCR).

    Measures the proportion of predictions where the predicted L2 label
    is a valid child of the predicted L1 label.

    Reference: Fatos Torba thesis, Section 2.4.2.

    Args:
        preds_l1: Predicted L1 labels.
        preds_l2: Predicted L2 labels.
        l2_to_l1_mapping: Mapping from L2 index to L1 index.

    Returns:
        HCR score between 0 and 1.
    """
    if len(preds_l1) == 0:
        return 0.0

    consistent = 0
    for p_l1, p_l2 in zip(preds_l1, preds_l2):
        expected_l1 = l2_to_l1_mapping.get(p_l2)
        if expected_l1 is not None and expected_l1 == p_l1:
            consistent += 1

    return consistent / len(preds_l1)


def compute_all_metrics(
    labels_l1: list,
    preds_l1: list,
    labels_l2: list,
    preds_l2: list,
    l2_to_l1_mapping: Optional[dict[int, int]] = None,
) -> dict[str, float]:
    """Compute all available metrics including hierarchy-specific ones.

    Args:
        labels_l1: Ground truth L1 labels.
        preds_l1: Predicted L1 labels.
        labels_l2: Ground truth L2 labels.
        preds_l2: Predicted L2 labels.
        l2_to_l1_mapping: Optional mapping for hierarchy metrics.

    Returns:
        Dictionary with all computed metrics.
    """
    metrics = compute_f1_metrics(labels_l1, preds_l1, labels_l2, preds_l2)

    if l2_to_l1_mapping is not None:
        metrics["constrained_f1_macro"] = constrained_f1(
            labels_l1, preds_l1, labels_l2, preds_l2,
            l2_to_l1_mapping, average="macro",
        )
        metrics["constrained_f1_weighted"] = constrained_f1(
            labels_l1, preds_l1, labels_l2, preds_l2,
            l2_to_l1_mapping, average="weighted",
        )
        metrics["hcr"] = hierarchical_consistency_rate(
            preds_l1, preds_l2, l2_to_l1_mapping,
        )

    return metrics
