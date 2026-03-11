"""Loss functions for HLPD multi-label hierarchical classification.

Includes:
- ZLPR (Zero-bounded Log-sum-exp & Pairwise Rank-based) loss from the paper.
- Standard BCE loss as a simpler alternative.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZLPRLoss(nn.Module):
    """Zero-bounded Log-sum-exp & Pairwise Rank-based loss.

    For multi-label classification where each sample can belong to
    multiple labels (or exactly one per level in hierarchical case).

    L_ZLPR = log(1 + sum_{i in pos} exp(-s_i))
           + log(1 + sum_{j in neg} exp(s_j))

    This encourages positive logits to be large and negative logits
    to be small, with a soft margin around zero.

    Args:
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ZLPR loss.

        Args:
            logits: Raw scores of shape (batch, num_labels).
            targets: Binary target matrix of shape (batch, num_labels).
                     1.0 for positive labels, 0.0 for negative.

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        # Positive part: log(1 + sum_{i in pos} exp(-s_i))
        pos_mask = targets.bool()
        neg_mask = ~pos_mask

        # For numerical stability, use logsumexp trick
        # Positive term: we want exp(-s_i) for positive labels
        neg_logits = -logits  # negate for positive positions
        # Set negative positions to -inf so they don't contribute
        pos_term_input = neg_logits.masked_fill(neg_mask, float("-inf"))
        # log(1 + sum exp(-s_i)) = log(exp(0) + sum exp(-s_i))
        #                        = logsumexp([0, -s_i1, -s_i2, ...])
        # Add a zero column for the "1" inside log(1 + ...)
        zero_col = torch.zeros(logits.size(0), 1, device=logits.device)
        pos_term = torch.logsumexp(
            torch.cat([zero_col, pos_term_input], dim=-1), dim=-1
        )

        # Negative term: log(1 + sum_{j in neg} exp(s_j))
        neg_term_input = logits.masked_fill(pos_mask, float("-inf"))
        neg_term = torch.logsumexp(
            torch.cat([zero_col, neg_term_input], dim=-1), dim=-1
        )

        loss = pos_term + neg_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalBCELoss(nn.Module):
    """Binary cross-entropy loss over all labels jointly.

    Simple alternative to ZLPR. Treats each label as an independent
    binary classification problem.

    Args:
        reduction: 'mean', 'sum', or 'none'.
        pos_weight: Optional positive class weight for imbalanced labels.
    """

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw scores of shape (batch, num_labels).
            targets: Binary target matrix of shape (batch, num_labels).

        Returns:
            Scalar loss.
        """
        if self.pos_weight is not None:
            pw = torch.tensor(
                [self.pos_weight], device=logits.device, dtype=logits.dtype
            )
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw, reduction=self.reduction
            )
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction=self.reduction
        )


def build_multi_hot_targets(
    labels_l1: torch.Tensor,
    labels_l2: torch.Tensor,
    num_l1: int,
    num_l2: int,
    include_root: bool = True,
) -> torch.Tensor:
    """Convert single-path hierarchical labels to multi-hot target vectors.

    For HLPD, the decoder outputs one logit per label node in the taxonomy.
    The target vector has 1s at the root (optional), the correct L1, and
    the correct L2 positions.

    The ordering matches the decoder: [root, L1_0, ..., L1_{n1-1}, L2_0, ..., L2_{n2-1}].

    Args:
        labels_l1: L1 class indices of shape (batch,).
        labels_l2: L2 class indices of shape (batch,).
        num_l1: Number of L1 classes.
        num_l2: Number of L2 classes.
        include_root: Whether position 0 (root) is always positive.

    Returns:
        Multi-hot tensor of shape (batch, 1 + num_l1 + num_l2).
    """
    B = labels_l1.size(0)
    total = 1 + num_l1 + num_l2
    targets = torch.zeros(B, total, device=labels_l1.device)

    if include_root:
        targets[:, 0] = 1.0  # root is always active

    # L1 labels: offset by 1 (after root)
    targets.scatter_(1, (1 + labels_l1).unsqueeze(1), 1.0)

    # L2 labels: offset by 1 + num_l1
    targets.scatter_(1, (1 + num_l1 + labels_l2).unsqueeze(1), 1.0)

    return targets
