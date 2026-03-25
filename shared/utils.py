"""General utilities for the Rakuten classification project."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU).

    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Supported encoder models
ENCODERS = {
    "CamemBERT": "camembert-base",
    "XLM-RoBERTa": "xlm-roberta-base",
    "mBERT": "bert-base-multilingual-cased",
}


def enforce_hierarchy(
    logits_l1: torch.Tensor,
    logits_l2: torch.Tensor,
    parent_child_matrix: torch.Tensor,
) -> torch.Tensor:
    """Mask invalid L2 predictions based on predicted L1 parent.

    Args:
        logits_l1: L1 logits of shape (batch, num_l1).
        logits_l2: L2 logits of shape (batch, num_l2).
        parent_child_matrix: Binary matrix (num_l1, num_l2).

    Returns:
        Masked L2 logits with invalid classes set to -inf.
    """
    preds_l1 = torch.argmax(logits_l1, dim=-1)
    valid_l2_mask = parent_child_matrix[preds_l1]
    masked = logits_l2.clone()
    masked[valid_l2_mask == 0] = float("-inf")
    return masked


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
