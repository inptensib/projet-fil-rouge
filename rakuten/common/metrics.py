"""Evaluation metrics and audit visualizations for hierarchical classification.

Includes standard F1 scores, hierarchy-specific metrics inspired by
the Fatos Torba thesis (HMTC evaluation protocol), plus
plotting functions for model auditing.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

matplotlib.use("Agg")


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
        Dictionary with f1_l1_macro, f1_l2_macro.
    """
    return {
        "f1_l1_macro": f1_score(labels_l1, preds_l1, average="macro", zero_division=0),
        "f1_l2_macro": f1_score(labels_l2, preds_l2, average="macro", zero_division=0),
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
        metrics["hcr"] = hierarchical_consistency_rate(
            preds_l1, preds_l2, l2_to_l1_mapping,
        )

    return metrics


def compute_f1_by_tranche(
    y_true: list,
    y_pred: list,
    class_counts: dict,
    thresholds: tuple[int, int] = (100, 20),
) -> dict[str, dict]:
    """Compute macro F1 per class-frequency tranche (many/medium/few).

    12% of L2 classes have fewer than 5 eval samples: the global macro F1
    hides the head/tail gap. Reporting per frequency tranche is the
    standard protocol in the long-tail literature.

    Tranches are defined from training-set counts:
    many (count >= thresholds[0]), medium (thresholds[1] <= count <
    thresholds[0]), few (count < thresholds[1]). Labels absent from
    class_counts are treated as few (count 0). Classes considered are
    those appearing in y_true or y_pred, so a single non-empty tranche
    matches the global macro F1.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_counts: Mapping label -> training sample count.
        thresholds: (many_min, medium_min) frequency cutoffs.

    Returns:
        Dict with keys 'many', 'medium', 'few'; each value is
        {'f1_macro': float | None, 'n_classes': int, 'support': int},
        with f1_macro None when the tranche is empty.
    """
    many_min, medium_min = thresholds
    tranche_labels: dict[str, list] = {"many": [], "medium": [], "few": []}
    for label in sorted(set(y_true) | set(y_pred)):
        count = class_counts.get(label, 0)
        if count >= many_min:
            tranche_labels["many"].append(label)
        elif count >= medium_min:
            tranche_labels["medium"].append(label)
        else:
            tranche_labels["few"].append(label)

    y_true_arr = np.asarray(y_true)
    result = {}
    for tranche, labels in tranche_labels.items():
        if labels:
            f1 = float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            )
            support = int(np.isin(y_true_arr, labels).sum())
        else:
            f1 = None
            support = 0
        result[tranche] = {
            "f1_macro": f1,
            "n_classes": len(labels),
            "support": support,
        }
    return result


# ---------------------------------------------------------------------------
# Audit visualizations
# ---------------------------------------------------------------------------


def compute_per_class_f1(
    y_true: list,
    y_pred: list,
    encoder,
) -> dict[str, float]:
    """Compute per-class F1 scores with class names.

    Args:
        y_true: Ground truth encoded labels.
        y_pred: Predicted encoded labels.
        encoder: Fitted LabelEncoder for class names.

    Returns:
        Dictionary {class_name: f1_score} sorted by F1 ascending.
    """
    labels = list(range(len(encoder.classes_)))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result = {name: float(score) for name, score in zip(encoder.classes_, scores)}
    return dict(sorted(result.items(), key=lambda x: x[1]))


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    encoder,
    title: str = "Confusion Matrix",
    top_n: Optional[int] = None,
) -> plt.Figure:
    """Generate a confusion matrix heatmap.

    Args:
        y_true: Ground truth encoded labels.
        y_pred: Predicted encoded labels.
        encoder: Fitted LabelEncoder for class names.
        title: Plot title.
        top_n: If set, show only the top_n most frequent classes.

    Returns:
        Matplotlib Figure.
    """
    classes = encoder.classes_
    labels = list(range(len(classes)))

    if top_n and top_n < len(classes):
        # Keep only the top_n most frequent classes in y_true
        counts = np.bincount(np.asarray(y_true), minlength=len(classes))
        top_indices = np.argsort(counts)[-top_n:]
        top_indices = np.sort(top_indices)
        mask = np.isin(y_true, top_indices)
        y_true = np.asarray(y_true)[mask]
        y_pred = np.asarray(y_pred)[mask]
        labels = top_indices.tolist()
        classes = encoder.classes_[top_indices]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Normalize by row (true class)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    size = max(8, len(classes) * 0.25)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm_norm, annot=len(classes) <= 30, fmt=".2f",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "Taux de prédiction"},
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=max(4, 10 - len(classes) // 20))
    plt.yticks(fontsize=max(4, 10 - len(classes) // 20))
    plt.tight_layout()
    return fig


def plot_per_class_f1(
    per_class_f1: dict[str, float],
    title: str = "F1 par classe",
    top_n: int = 30,
) -> plt.Figure:
    """Generate a horizontal barplot of per-class F1 scores.

    Shows the top_n worst and top_n best classes.

    Args:
        per_class_f1: Dict {class_name: f1_score} (sorted ascending).
        title: Plot title.
        top_n: Number of worst and best classes to show.

    Returns:
        Matplotlib Figure.
    """
    items = list(per_class_f1.items())
    if len(items) > 2 * top_n:
        worst = items[:top_n]
        best = items[-top_n:]
        selected = worst + [("...", None)] + best
    else:
        selected = items

    names = [name for name, _ in selected]
    scores = [s if s is not None else 0 for _, s in selected]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))
    colors = ["#e74c3c" if s < 0.5 else "#f39c12" if s < 0.8 else "#27ae60" for s in scores]
    # Override separator color
    for i, (name, _) in enumerate(selected):
        if name == "...":
            colors[i] = "#bdc3c7"

    ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Generate training curves with subplots.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                 'val_f1_l1_macro', 'val_f1_l2_macro', etc.
                 Values are lists (one per epoch).

    Returns:
        Matplotlib Figure with subplots.
    """
    n_plots = 0
    has_loss = "train_loss" in history
    has_f1 = "val_f1_l2_macro" in history
    has_hcr = "val_hcr" in history
    n_plots = sum([has_loss, has_f1, has_hcr])
    if n_plots == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    epochs = range(1, len(next(iter(history.values()))) + 1)

    if has_loss:
        ax = axes[idx]; idx += 1
        ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#3498db")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if has_f1:
        ax = axes[idx]; idx += 1
        if "val_f1_l1_macro" in history:
            ax.plot(epochs, history["val_f1_l1_macro"], "o-", label="F1 L1 macro", color="#2ecc71")
        ax.plot(epochs, history["val_f1_l2_macro"], "s-", label="F1 L2 macro", color="#e67e22")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    if has_hcr:
        ax = axes[idx]; idx += 1
        ax.plot(epochs, history["val_hcr"], "D-", label="HCR", color="#1abc9c")
        ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Cible 95%")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("HCR")
        ax.set_title("Cohérence Hiérarchique")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Courbes d'Apprentissage", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
