"""Difficulty Router for the Dual-Expert architecture.

Decides whether a product goes to the fast path (Domain Expert top-1)
or the hard path (LLM General Expert reranking).

Routing triggers (any one is sufficient to send to LLM):
    1. Hierarchy inconsistency: argmax(L1) ≠ parent(argmax(L2))
    2. Low L2 confidence (prob_top1 < threshold)
    3. Low L2 margin (prob_top1 − prob_top2 < threshold)
    4. Rare L2 class with insufficient confidence
    5. Short or missing description text
    6. Sibling confusion in top-K candidates
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RoutingThresholds:
    """Per-signal thresholds for the difficulty router."""

    min_l2_confidence: float = 0.75
    min_l2_margin: float = 0.15
    min_parent_l1_confidence: float = 0.50
    rare_class_threshold: int = 100
    rare_class_min_confidence: float = 0.90
    min_text_chars: int = 30


@dataclass
class RoutingRules:
    """Toggle individual routing rules on/off."""

    send_if_hierarchy_inconsistent: bool = True
    send_if_low_l2_confidence: bool = True
    send_if_low_l2_margin: bool = True
    send_if_rare_and_not_very_confident: bool = True
    send_if_short_text: bool = False
    send_if_sibling_confusion: bool = True


@dataclass
class DifficultyRouterConfig:
    """Full router configuration."""

    enabled: bool = True
    max_llm_pass_through_rate: float = 0.25
    thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)
    rules: RoutingRules = field(default_factory=RoutingRules)


def should_route_to_llm(
    probs_l1: torch.Tensor,
    probs_l2: torch.Tensor,
    top_l2_ids: list[int],
    l2_to_l1_mapping: dict[int, int],
    l2_counts: dict[int, int],
    text: str,
    config: DifficultyRouterConfig,
) -> tuple[bool, list[str]]:
    """Decide whether a single product should go to the LLM hard path.

    Args:
        probs_l1: Softmax L1 probabilities (num_l1,).
        probs_l2: Softmax L2 probabilities (num_l2,).
        top_l2_ids: Ordered list of top-K L2 candidate indices (best first).
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        l2_counts: dict {l2_id → train_count}, used for rare-class detection.
        text: Product text (designation + description).
        config: Router configuration.

    Returns:
        Tuple (should_route: bool, reasons: list[str]).
    """
    if not config.enabled or not top_l2_ids:
        return False, []

    pred_l1 = int(torch.argmax(probs_l1).item())
    pred_l2 = top_l2_ids[0]
    parent_l1 = l2_to_l1_mapping.get(pred_l2, -1)

    t = config.thresholds
    rules = config.rules
    reasons: list[str] = []

    # Confidence and margin from L2 probs
    p1 = float(probs_l2[pred_l2].item()) if pred_l2 < probs_l2.shape[0] else 0.0
    top2_probs, _ = torch.topk(probs_l2, k=min(2, probs_l2.shape[0]))
    p2 = float(top2_probs[-1].item()) if top2_probs.shape[0] > 1 else 0.0
    margin = p1 - p2

    parent_conf = float(probs_l1[parent_l1].item()) if 0 <= parent_l1 < probs_l1.shape[0] else 0.0

    hierarchy_inconsistent = (parent_l1 != pred_l1) and (parent_l1 != -1)
    is_rare = l2_counts.get(pred_l2, 0) < t.rare_class_threshold
    short_text = len(text.strip()) < t.min_text_chars

    if rules.send_if_hierarchy_inconsistent and hierarchy_inconsistent:
        reasons.append("hierarchy_inconsistent")

    if rules.send_if_low_l2_confidence and p1 < t.min_l2_confidence:
        reasons.append("low_l2_confidence")

    if rules.send_if_low_l2_margin and margin < t.min_l2_margin:
        reasons.append("low_l2_margin")

    if rules.send_if_rare_and_not_very_confident and is_rare and p1 < t.rare_class_min_confidence:
        reasons.append("rare_l2_not_confident")

    if parent_conf < t.min_parent_l1_confidence:
        reasons.append("low_parent_l1_confidence")

    if rules.send_if_short_text and short_text:
        reasons.append("short_text")

    if rules.send_if_sibling_confusion and _has_sibling_confusion(top_l2_ids, l2_to_l1_mapping):
        reasons.append("sibling_confusion")

    return len(reasons) > 0, reasons


def _has_sibling_confusion(
    top_l2_ids: list[int],
    l2_to_l1_mapping: dict[int, int],
    threshold: int = 3,
) -> bool:
    """Detect if multiple top-K candidates share the same parent L1 (siblings).

    Args:
        top_l2_ids: Top-K L2 candidate indices.
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        threshold: Minimum number of sibling candidates to flag confusion.

    Returns:
        True if any L1 parent has ≥ threshold sibling L2s among top candidates.
    """
    parent_counts = Counter(
        l2_to_l1_mapping.get(l2, -1) for l2 in top_l2_ids
    )
    return any(c >= threshold for p, c in parent_counts.items() if p != -1)


class DifficultyRouter:
    """Stateful difficulty router that tracks pass-through rate.

    max_llm_pass_through_rate is *actively enforced*: once the running fraction
    of LLM-routed samples reaches the limit, subsequent samples are forced to
    the fast path regardless of routing signals.  This prevents runaway LLM costs
    in production.  Set it to 1.0 to disable the cap (monitor-only mode).

    Args:
        config: Router configuration.
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        l2_counts: dict {l2_id → train_count}.
    """

    def __init__(
        self,
        config: Optional[DifficultyRouterConfig] = None,
        l2_to_l1_mapping: Optional[dict] = None,
        l2_counts: Optional[dict] = None,
    ) -> None:
        self.config = config or DifficultyRouterConfig()
        self.l2_to_l1_mapping: dict[int, int] = {
            int(k): int(v) for k, v in (l2_to_l1_mapping or {}).items()
        }
        self.l2_counts: dict[int, int] = {
            int(k): int(v) for k, v in (l2_counts or {}).items()
        }
        self._n_total = 0
        self._n_routed = 0
        self._reason_counts: dict[str, int] = {}

    def should_route(
        self,
        probs_l1: torch.Tensor,
        probs_l2: torch.Tensor,
        top_l2_ids: list[int],
        text: str,
    ) -> tuple[bool, list[str]]:
        """Decide routing for one product and update internal counters.

        Enforces max_llm_pass_through_rate: if routing this sample would push
        the running rate above the configured cap, it is forced to fast path.

        Args:
            probs_l1: Softmax L1 probabilities (num_l1,).
            probs_l2: Softmax L2 probabilities (num_l2,).
            top_l2_ids: Top-K L2 candidate indices.
            text: Product text.

        Returns:
            (should_route_to_llm, reasons).
        """
        route, reasons = should_route_to_llm(
            probs_l1, probs_l2, top_l2_ids,
            self.l2_to_l1_mapping, self.l2_counts, text, self.config,
        )

        # Enforce the pass-through rate cap before updating counters
        if route and self.config.max_llm_pass_through_rate < 1.0:
            future_rate = (self._n_routed + 1) / (self._n_total + 1)
            if future_rate > self.config.max_llm_pass_through_rate:
                route = False
                reasons = ["capped_by_max_pass_through_rate"]

        self._n_total += 1
        if route:
            self._n_routed += 1
            for r in reasons:
                self._reason_counts[r] = self._reason_counts.get(r, 0) + 1

        return route, reasons

    @property
    def pass_through_rate(self) -> float:
        """Fraction of products sent to LLM since last reset."""
        if self._n_total == 0:
            return 0.0
        return self._n_routed / self._n_total

    def reset_counters(self) -> None:
        """Reset pass-through rate counters and reason histogram."""
        self._n_total = 0
        self._n_routed = 0
        self._reason_counts = {}

    def build_routing_report(
        self,
        routed_labels_l2: Optional[list[int]] = None,
        routed_preds_before: Optional[list[int]] = None,
        routed_preds_after: Optional[list[int]] = None,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        """Build a summary routing report (llm_routing_report.json).

        Args:
            routed_labels_l2: Ground-truth L2 labels for the routed subset only.
            routed_preds_before: Domain Expert top-1 predictions for routed samples
                (i.e., what the fast path would have predicted).
            routed_preds_after: LLM-selected L2 predictions for routed samples.
            output_path: If set, save the report as JSON.

        Returns:
            Report dict with pass_through_rate, reason breakdown, routing rates,
            and optional before/after accuracy on the routed subset.
        """
        n_routed = self._n_routed
        reason_counts = dict(self._reason_counts)

        report: dict = {
            "total_samples": self._n_total,
            "routed_to_llm": n_routed,
            "pass_through_rate": self.pass_through_rate,
            "max_llm_pass_through_rate": self.config.max_llm_pass_through_rate,
            "rate_cap_enforced": self.config.max_llm_pass_through_rate < 1.0,
            "routing_reason_counts": reason_counts,
        }

        if n_routed > 0:
            report["rare_class_routing_rate"] = (
                reason_counts.get("rare_l2_not_confident", 0) / n_routed
            )
            report["hierarchy_inconsistency_routing_rate"] = (
                reason_counts.get("hierarchy_inconsistent", 0) / n_routed
            )

        # Before/after accuracy on the routed subset
        if (
            routed_labels_l2 is not None
            and routed_preds_before is not None
            and routed_preds_after is not None
            and len(routed_labels_l2) > 0
        ):
            n = len(routed_labels_l2)
            acc_before = sum(
                p == g for p, g in zip(routed_preds_before, routed_labels_l2)
            ) / n
            acc_after = sum(
                p == g for p, g in zip(routed_preds_after, routed_labels_l2)
            ) / n
            report["routed_subset_size"] = n
            report["accuracy_before_llm"] = acc_before
            report["accuracy_after_llm"] = acc_after
            report["accuracy_delta"] = acc_after - acc_before

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[Router] Routing report saved → {p}")

        return report
