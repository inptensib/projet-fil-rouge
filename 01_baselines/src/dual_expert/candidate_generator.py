"""Top-K L2 candidate generation for the Dual-Expert architecture.

Returns the K highest-scoring L2 candidates per sample with their
parent L1 information, probabilities, and scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .hierarchy_scorer import (
    HierarchyScoringConfig,
    hierarchy_aware_scores,
    make_l2_to_l1_tensor,
)


@dataclass
class CandidateGenerationConfig:
    """Configuration for top-K candidate generation."""

    top_k: int = 10
    also_compute_top_20: bool = True


def get_topk_l2_candidates(
    logits_l1: torch.Tensor,
    logits_l2: torch.Tensor,
    l2_to_l1: torch.LongTensor,
    k: int = 10,
    hierarchy_scoring_enabled: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-12,
    l1_encoder=None,
    l2_encoder=None,
) -> list[list[dict]]:
    """Generate top-K L2 candidates for a batch.

    Args:
        logits_l1: Raw L1 logits (B, num_l1).
        logits_l2: Raw L2 logits (B, num_l2).
        l2_to_l1: LongTensor (num_l2,) mapping L2 idx → parent L1 idx.
        k: Number of candidates to return.
        hierarchy_scoring_enabled: If True, use hierarchy-aware combined score.
        alpha: Hierarchy-scoring alpha.
        eps: Numerical epsilon.
        l1_encoder: Optional sklearn LabelEncoder for human-readable L1 names.
        l2_encoder: Optional sklearn LabelEncoder for human-readable L2 names.

    Returns:
        List of B candidate lists; each list contains up to k dicts with:
            rank, l2_id, l2_label (if encoder), parent_l1_id,
            parent_l1_label (if encoder), hierarchy_score, prob_l2, prob_parent_l1.

        Field semantics:
            prob_l2         — raw softmax P(L2|text), no parent correction.
            hierarchy_score — log P(L2) + alpha * log P(parent_L1); used for ranking.
            prob_parent_l1  — raw softmax P(parent_L1|text).
    """
    probs_l1 = torch.softmax(logits_l1, dim=-1)   # (B, num_l1)
    probs_l2 = torch.softmax(logits_l2, dim=-1)   # (B, num_l2)

    l2_to_l1 = l2_to_l1.to(logits_l1.device)
    valid_l2_to_l1 = l2_to_l1.clamp(min=0)

    if hierarchy_scoring_enabled:
        scores = hierarchy_aware_scores(
            probs_l1, probs_l2, l2_to_l1, alpha=alpha, eps=eps,
        )
    else:
        scores = torch.log(probs_l2 + eps)

    k_actual = min(k, scores.shape[1])
    top_scores, top_l2_ids = torch.topk(scores, k=k_actual, dim=-1)   # (B, k)
    top_l1_ids = valid_l2_to_l1[top_l2_ids]                           # (B, k)
    top_l2_probs = probs_l2.gather(1, top_l2_ids)                      # (B, k)
    top_l1_probs = probs_l1.gather(1, top_l1_ids)                      # (B, k)

    results = []
    for b in range(logits_l1.shape[0]):
        candidates = []
        for rank in range(k_actual):
            l2_id = int(top_l2_ids[b, rank].item())
            l1_id = int(top_l1_ids[b, rank].item())
            entry: dict = {
                "rank": rank + 1,
                "l2_id": l2_id,
                "parent_l1_id": l1_id,
                "hierarchy_score": float(top_scores[b, rank].item()),
                "prob_l2": float(top_l2_probs[b, rank].item()),
                "prob_parent_l1": float(top_l1_probs[b, rank].item()),
            }
            if l2_encoder is not None and 0 <= l2_id < len(l2_encoder.classes_):
                entry["l2_label"] = str(l2_encoder.classes_[l2_id])
            if l1_encoder is not None and 0 <= l1_id < len(l1_encoder.classes_):
                entry["parent_l1_label"] = str(l1_encoder.classes_[l1_id])
            candidates.append(entry)
        results.append(candidates)

    return results


class TopKCandidateGenerator:
    """Top-K L2 candidate generator with optional hierarchy-aware scoring.

    Args:
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        num_l2: Number of L2 classes.
        config: Candidate generation configuration.
        scoring_config: Hierarchy-scoring configuration.
        l1_encoder: Optional LabelEncoder for L1 names.
        l2_encoder: Optional LabelEncoder for L2 names.
    """

    def __init__(
        self,
        l2_to_l1_mapping: dict,
        num_l2: int,
        config: Optional[CandidateGenerationConfig] = None,
        scoring_config: Optional[HierarchyScoringConfig] = None,
        l1_encoder=None,
        l2_encoder=None,
    ) -> None:
        self.config = config or CandidateGenerationConfig()
        self.scoring_config = scoring_config or HierarchyScoringConfig()
        self.l2_to_l1 = make_l2_to_l1_tensor(l2_to_l1_mapping, num_l2)
        self.num_l2 = num_l2
        self.l1_encoder = l1_encoder
        self.l2_encoder = l2_encoder

    def generate(
        self,
        logits_l1: torch.Tensor,
        logits_l2: torch.Tensor,
        k: Optional[int] = None,
    ) -> list[list[dict]]:
        """Generate top-K candidates for a batch.

        Args:
            logits_l1: L1 logits (B, num_l1).
            logits_l2: L2 logits (B, num_l2).
            k: Override for number of candidates (defaults to config.top_k).

        Returns:
            List of candidate lists, one per sample in the batch.
        """
        k = k if k is not None else self.config.top_k
        return get_topk_l2_candidates(
            logits_l1, logits_l2,
            l2_to_l1=self.l2_to_l1,
            k=k,
            hierarchy_scoring_enabled=self.scoring_config.enabled,
            alpha=self.scoring_config.alpha,
            eps=self.scoring_config.eps,
            l1_encoder=self.l1_encoder,
            l2_encoder=self.l2_encoder,
        )
