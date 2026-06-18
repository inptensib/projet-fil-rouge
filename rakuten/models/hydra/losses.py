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
        class_weights: Optional per-class weights [num_classes]. Same
            semantics as CrossEntropyLoss(weight=...): 'mean' divides by the
            sum of the sample weights. None keeps the loss unchanged.
        logit_adjustment: Optional tau * log(prior) tensor [num_classes],
            added to the logits before the loss (logit-adjusted loss,
            Menon et al., 2021). None keeps the loss unchanged.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        class_weights: torch.Tensor | None = None,
        logit_adjustment: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.logit_adjustment = logit_adjustment

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.logit_adjustment is not None:
            logits = logits + self.logit_adjustment
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.class_weights is not None:
            sample_weights = self.class_weights[labels]
            loss = loss * sample_weights
            if self.reduction == "mean":
                return loss.sum() / sample_weights.sum()

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
        use_focal: If True, use Focal Loss for L2 head only; L1 always uses CrossEntropy.
        class_weights_l1: Optional per-class weights for the L1 loss.
        class_weights_l2: Optional per-class weights for the L2 loss.
        logit_adjustment_l1: Optional tau * log(prior) tensor added to the L1
            logits before the loss (logit-adjusted loss, Menon et al., 2021).
        logit_adjustment_l2: Same for the L2 logits.

    All four rebalancing tensors default to None, which leaves the loss
    strictly identical to the unweighted version.
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
        class_weights_l1: torch.Tensor | None = None,
        class_weights_l2: torch.Tensor | None = None,
        logit_adjustment_l1: torch.Tensor | None = None,
        logit_adjustment_l2: torch.Tensor | None = None,
    ):
        super().__init__()
        if architecture in ("global", "nested"):
            if num_classes_l1 is None or num_classes_l2 is None:
                raise ValueError(
                    f"num_classes_l1 and num_classes_l2 are required for "
                    f"architecture='{architecture}'"
                )
        self.architecture = architecture
        self.alpha = alpha
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.level_weight_l1 = level_weight_l1
        # Appliqués aux logits dans forward (couvre les têtes locales ET unified)
        self.logit_adjustment_l1 = logit_adjustment_l1
        self.logit_adjustment_l2 = logit_adjustment_l2

        # L1 : toujours CrossEntropy (157 classes, F1~92%, calibré pour le masque hiérarchique)
        self.loss_fn_l1 = nn.CrossEntropyLoss(weight=class_weights_l1)
        # L2 : FocalLoss si use_focal (~5 777 classes L2 déséquilibrées), CE sinon
        self.loss_fn_l2 = (
            FocalLoss(gamma=focal_gamma, class_weights=class_weights_l2)
            if use_focal
            else nn.CrossEntropyLoss(weight=class_weights_l2)
        )

    @staticmethod
    def _adjust(logits: torch.Tensor, adjustment: torch.Tensor | None) -> torch.Tensor:
        """Add the logit adjustment before the loss; no-op when None."""
        return logits if adjustment is None else logits + adjustment

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
        # FlatHydra path: level_outputs est vide, toute la loss vient de unified_output
        if len(level_outputs) == 0:
            assert unified_output is not None and self.num_classes_l1 is not None
            unified_l1_logits = self._adjust(
                unified_output[:, :self.num_classes_l1], self.logit_adjustment_l1
            )
            unified_l2_logits = self._adjust(
                unified_output[:, self.num_classes_l1:], self.logit_adjustment_l2
            )
            loss_l1 = self.loss_fn_l1(unified_l1_logits, labels_l1)
            loss_l2 = self.loss_fn_l2(unified_l2_logits, labels_l2)
            total_loss = self.alpha * (self.level_weight_l1 * loss_l1 + loss_l2)
            return total_loss, loss_l1, loss_l2

        logits_l1, logits_l2 = level_outputs
        loss_l1 = self.loss_fn_l1(self._adjust(logits_l1, self.logit_adjustment_l1), labels_l1)
        loss_l2 = self.loss_fn_l2(self._adjust(logits_l2, self.logit_adjustment_l2), labels_l2)
        level_loss = self.level_weight_l1 * loss_l1 + loss_l2
        total_loss = level_loss

        if self.architecture in ("global", "nested") and unified_output is not None:
            unified_l1_logits = self._adjust(
                unified_output[:, :self.num_classes_l1], self.logit_adjustment_l1
            )
            unified_l2_logits = self._adjust(
                unified_output[:, self.num_classes_l1:], self.logit_adjustment_l2
            )
            unified_loss_l1 = self.loss_fn_l1(unified_l1_logits, labels_l1)
            unified_loss_l2 = self.loss_fn_l2(unified_l2_logits, labels_l2)
            unified_loss = self.level_weight_l1 * unified_loss_l1 + unified_loss_l2
            total_loss = level_loss + self.alpha * unified_loss

        return total_loss, loss_l1, loss_l2
