"""Hierarchy-aware L2 scoring for the Dual-Expert architecture.

Combines P(L2|text) and P(parent_L1(L2)|text) to boost L2 candidates
whose parent L1 is also confident.

score(l2) = log(P(l2|text) + eps) + alpha * log(P(parent_l1(l2)|text) + eps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HierarchyScoringConfig:
    """Configuration for hierarchy-aware L2 scoring."""

    enabled: bool = True
    alpha: float = 0.5
    eps: float = 1e-12


def make_l2_to_l1_tensor(
    l2_to_l1_mapping: dict,
    num_l2: int,
) -> torch.LongTensor:
    """Convert {l2_id → l1_id} dict to a LongTensor for vectorized indexing.

    Args:
        l2_to_l1_mapping: dict with int or str keys/values.
        num_l2: Total number of L2 classes.

    Returns:
        LongTensor (num_l2,) where tensor[l2_id] = l1_id.
        Classes missing from the mapping get value -1.
    """
    tensor = torch.full((num_l2,), -1, dtype=torch.long)
    for l2, l1 in l2_to_l1_mapping.items():
        idx = int(l2)
        if 0 <= idx < num_l2:
            tensor[idx] = int(l1)
    return tensor


def hierarchy_aware_scores(
    probs_l1: torch.Tensor,
    probs_l2: torch.Tensor,
    l2_to_l1: torch.LongTensor,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute hierarchy-aware score for every L2 candidate (vectorized).

    Args:
        probs_l1: Softmax L1 probabilities (B, num_l1).
        probs_l2: Softmax L2 probabilities (B, num_l2).
        l2_to_l1: LongTensor (num_l2,) mapping L2 idx → parent L1 idx.
                  Values of -1 are clamped to 0 (safe fallback).
        alpha: Weight for the parent L1 log-probability term.
        eps: Numerical stability constant.

    Returns:
        Scores tensor (B, num_l2).
    """
    valid = l2_to_l1.clamp(min=0)          # (num_l2,) — clamped for indexing
    parent_probs = probs_l1[:, valid]       # (B, num_l2)
    scores = (
        torch.log(probs_l2 + eps)
        + alpha * torch.log(parent_probs + eps)
    )
    return scores


class HierarchyAwareScorer:
    """Scores L2 candidates using combined P(L2) and P(parent L1).

    Args:
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        num_l2: Number of L2 classes.
        config: Scoring configuration.
    """

    def __init__(
        self,
        l2_to_l1_mapping: dict,
        num_l2: int,
        config: Optional[HierarchyScoringConfig] = None,
    ) -> None:
        self.config = config or HierarchyScoringConfig()
        self.l2_to_l1 = make_l2_to_l1_tensor(l2_to_l1_mapping, num_l2)

    def score(
        self,
        logits_l1: torch.Tensor,
        logits_l2: torch.Tensor,
    ) -> torch.Tensor:
        """Score all L2 candidates for a batch.

        If scoring is disabled, returns plain log-softmax of L2 logits.

        Args:
            logits_l1: Raw L1 logits (B, num_l1).
            logits_l2: Raw L2 logits (B, num_l2).

        Returns:
            Scores tensor (B, num_l2).
        """
        probs_l1 = torch.softmax(logits_l1, dim=-1)
        probs_l2 = torch.softmax(logits_l2, dim=-1)
        l2_to_l1 = self.l2_to_l1.to(logits_l1.device)

        if not self.config.enabled:
            return torch.log(probs_l2 + self.config.eps)

        return hierarchy_aware_scores(
            probs_l1, probs_l2, l2_to_l1,
            alpha=self.config.alpha,
            eps=self.config.eps,
        )
