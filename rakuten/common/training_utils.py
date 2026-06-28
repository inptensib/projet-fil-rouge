"""Common training utilities shared by the project trainers.

Provides:
- get_linear_schedule_with_warmup: linear warmup then linear decay scheduler.
- make_hierarchical_loader: HierarchicalProductDataset + DataLoader factory.
- compute_class_weights: per-class loss weights from real (non synthetic) counts.
- compute_logit_adjustment: tau * log(prior) tensor (Menon et al., 2021).
- resolve_rebalancing: guard + class weight / logit adjustment tensors from a
  loader data dict.
- EarlyStopping: patience-based early stopping state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from rakuten.common.data_loader import HierarchicalProductDataset

_WEIGHTING_SCHEMES = ("none", "inv_sqrt", "inv")


def compute_class_weights(
    real_counts: Mapping,
    classes: Sequence,
    scheme: str,
    normalize: bool = True,
) -> torch.Tensor | None:
    """Per-class loss weights from real (non synthetic) training counts.

    Classes absent from ``real_counts`` (100% synthetic, e.g. SYNTH-ORPH
    templates) get the weight of the smallest non-zero count.

    Args:
        real_counts: Mapping class -> real sample count (Counter).
        classes: Class keys ordered by encoded index; position i in the
            returned tensor is the weight of classes[i].
        scheme: "none" (returns None), "inv_sqrt" (1/sqrt(n)) or "inv" (1/n).
        normalize: Rescale weights so their mean is 1.

    Returns:
        Float tensor of shape [len(classes)], or None if scheme is "none".
    """
    if scheme == "none":
        return None
    if scheme not in _WEIGHTING_SCHEMES:
        raise ValueError(
            f"scheme invalide: {scheme!r} (attendu: {', '.join(_WEIGHTING_SCHEMES)})"
        )
    counts = [real_counts.get(c, 0) for c in classes]
    nonzero = [n for n in counts if n > 0]
    if not nonzero:
        raise ValueError("compute_class_weights: aucun compte réel non nul")
    floor = min(nonzero)
    effective = torch.tensor(
        [n if n > 0 else floor for n in counts], dtype=torch.float32
    )
    weights = effective.rsqrt() if scheme == "inv_sqrt" else effective.reciprocal()
    if normalize:
        weights = weights / weights.mean()
    return weights


def compute_logit_adjustment(
    real_counts: Mapping,
    classes: Sequence,
    tau: float,
) -> torch.Tensor | None:
    """tau * log(prior) per class, with add-one smoothed priors.

    Used as the logit offset of the logit-adjusted loss (Menon et al., 2021):
    the trainer adds it to the logits inside the loss, so raw logits at
    inference are implicitly corrected for the class priors.

    Args:
        real_counts: Mapping class -> real sample count (Counter).
        classes: Class keys ordered by encoded index.
        tau: Adjustment temperature. 0 disables (returns None).

    Returns:
        Float tensor of shape [len(classes)], or None if tau == 0.
    """
    if tau == 0:
        return None
    counts = torch.tensor(
        [real_counts.get(c, 0) for c in classes], dtype=torch.float32
    )
    priors = (counts + 1.0) / (counts + 1.0).sum()
    return tau * torch.log(priors)


def resolve_rebalancing(
    data: dict,
    num_classes_l1: int,
    num_classes_l2: int,
    class_weighting: str = "none",
    logit_adjustment_tau: float = 0.0,
) -> dict[str, torch.Tensor | None]:
    """Build the HYDRALoss rebalancing tensors from a loader data dict.

    Guard (règle « doublons OU pondération ») : enabling class weighting or
    logit adjustment on a train split still containing the oversampled
    SYNTH-OVS/OVL duplicates would double-correct the imbalance. The data
    must come from prepare_hierarchical_data_from_splits with
    synthetic_policy="drop_oversampled".

    Returns:
        Dict with keys class_weights_l1, class_weights_l2,
        logit_adjustment_l1, logit_adjustment_l2 (Tensor or None), ready to
        be passed as HYDRALoss kwargs. All None when both options are off.
    """
    tensors: dict[str, torch.Tensor | None] = {
        "class_weights_l1": None,
        "class_weights_l2": None,
        "logit_adjustment_l1": None,
        "logit_adjustment_l2": None,
    }
    if class_weighting == "none" and logit_adjustment_tau == 0.0:
        return tensors
    if data.get("synthetic_policy") != "drop_oversampled":
        raise ValueError(
            "class_weighting / logit_adjustment_tau exigent un train sans "
            "doublons oversamplés (règle « doublons OU pondération ») : "
            "rechargez les données avec synthetic_policy='drop_oversampled' "
            f"(actuel: {data.get('synthetic_policy')!r})"
        )
    if "real_class_counts_l1" not in data or "real_class_counts_l2" not in data:
        raise ValueError(
            "real_class_counts_l1/l2 absents du dict data : rechargez les "
            "données avec un rakuten/common/data_loader.py à jour "
            "(prepare_hierarchical_data_from_splits)"
        )
    classes_l1 = range(num_classes_l1)
    classes_l2 = range(num_classes_l2)
    tensors["class_weights_l1"] = compute_class_weights(
        data["real_class_counts_l1"], classes_l1, class_weighting
    )
    tensors["class_weights_l2"] = compute_class_weights(
        data["real_class_counts_l2"], classes_l2, class_weighting
    )
    tensors["logit_adjustment_l1"] = compute_logit_adjustment(
        data["real_class_counts_l1"], classes_l1, logit_adjustment_tau
    )
    tensors["logit_adjustment_l2"] = compute_logit_adjustment(
        data["real_class_counts_l2"], classes_l2, logit_adjustment_tau
    )
    return tensors


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """Linear warmup then linear decay scheduler.

    The LR multiplier rises linearly from 0 to 1 over ``num_warmup_steps``,
    then decreases linearly from 1 back to 0 at ``num_training_steps``.
    It is clamped at 0 afterwards, so the learning rate is never negative.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def make_hierarchical_loader(
    texts,
    labels_l1,
    labels_l2,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader over a HierarchicalProductDataset.

    Args:
        texts: List or Series of product text descriptions.
        labels_l1: List or Series of encoded L1 labels.
        labels_l2: List or Series of encoded L2 labels.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
        batch_size: Batch size.
        shuffle: Shuffle samples between epochs.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin batches in page-locked memory.
        drop_last: Drop the last incomplete batch.
    """
    dataset = HierarchicalProductDataset(
        texts, labels_l1, labels_l2, tokenizer, max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class EarlyStopping:
    """Patience-based early stopping on a metric to maximise.

    The ``best`` and ``counter`` attributes are public and assignable so
    trainers can inject state restored from a checkpoint.

    Args:
        patience: Epochs without improvement before ``should_stop`` is True.
        min_delta: Minimum improvement over ``best`` to count as progress.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.counter = 0

    def update(self, value: float) -> bool:
        """Record a new metric value.

        Returns:
            True if ``value`` improves ``best`` by more than ``min_delta``
            (counter is reset), False otherwise (counter is incremented).
        """
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience
