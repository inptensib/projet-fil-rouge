"""ZLPR Loss — Zero-bounded Log-sum-exp Pairwise Rank-based.

Reference: Su et al. 2022 (https://arxiv.org/abs/2208.03282).
Implementation matches the official HLPD repository (Apache 2.0 license).

Numerically stable in bfloat16: 1e12 << bfloat16 max ~3.4e38, so the masking
constants do not overflow.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ZLPRLoss(nn.Module):
    """Zero-bounded Log-sum-exp Pairwise Rank-based loss for multi-label classification.

    Operates on raw logits and binary targets. No hyperparameters.

    Args:
        y_pred: Logits of shape (B, L).
        y_true: Binary targets of shape (B, L). 1 = positive label, 0 = negative.

    Returns:
        Scalar mean loss (non-negative).
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[:, :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        return (
            torch.logsumexp(y_pred_neg, dim=-1) + torch.logsumexp(y_pred_pos, dim=-1)
        ).mean()
