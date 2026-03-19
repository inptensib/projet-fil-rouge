"""Loss functions for the HYDRA architecture.

Includes:
- FocalLoss: Addresses class imbalance (Lin et al., 2017).
- HYDRALoss: Combines level-specific losses with optional unified loss.

Note: The official HYDRA paper uses BCEWithLogitsLoss. We provide both
Focal Loss (better for imbalanced data) and BCE (for paper reproduction).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    When gamma=0, this reduces to standard CrossEntropyLoss.
    Default gamma=2.0 as recommended by Lin et al. (2017).

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HYDRALoss(nn.Module):
    """Combined loss function for the HYDRA architecture.

    Computes weighted sum of level-specific losses and an optional unified loss
    for global/nested variants.

    Args:
        architecture: HYDRA variant ('local', 'global', 'nested').
        alpha: Weight for the unified loss term.
        num_classes_l1: Number of L1 classes (needed for global/nested).
        num_classes_l2: Number of L2 classes (needed for global/nested).
        level_weight_l1: Weight for L1 loss relative to L2.
        focal_gamma: Gamma parameter for Focal Loss.
        use_focal: If True, use Focal Loss; if False, use CrossEntropy.
    """

    def __init__(
        self,
        architecture: str = "local",
        alpha: float = 1.0,
        num_classes_l1: int = None,
        num_classes_l2: int = None,
        level_weight_l1: float = 0.5,
        focal_gamma: float = 2.0,
        use_focal: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.alpha = alpha
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.level_weight_l1 = level_weight_l1

        if use_focal:
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        level_outputs: list[torch.Tensor],
        unified_output: torch.Tensor | None,
        labels_l1: torch.Tensor,
        labels_l2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the total HYDRA loss.

        Args:
            level_outputs: [logits_l1, logits_l2] from local heads.
            unified_output: Logits from global/nested head (or None for local).
            labels_l1: Ground truth L1 labels.
            labels_l2: Ground truth L2 labels.

        Returns:
            Tuple of (total_loss, loss_l1, loss_l2).
        """
        logits_l1, logits_l2 = level_outputs
        loss_l1 = self.loss_fn(logits_l1, labels_l1)
        loss_l2 = self.loss_fn(logits_l2, labels_l2)
        level_loss = self.level_weight_l1 * loss_l1 + loss_l2
        total_loss = level_loss

        if self.architecture in ("global", "nested") and unified_output is not None:
            unified_l1_logits = unified_output[:, :self.num_classes_l1]
            unified_l2_logits = unified_output[:, self.num_classes_l1:]
            unified_loss_l1 = self.loss_fn(unified_l1_logits, labels_l1)
            unified_loss_l2 = self.loss_fn(unified_l2_logits, labels_l2)
            unified_loss = self.level_weight_l1 * unified_loss_l1 + unified_loss_l2
            total_loss = level_loss + self.alpha * unified_loss

        return total_loss, loss_l1, loss_l2
