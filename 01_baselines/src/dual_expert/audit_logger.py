"""Audit logger for Dual-Expert hard-path predictions.

Every product that goes through the LLM General Expert is logged with:
- Domain Expert predictions and confidence
- Top-K candidates
- LLM selection, reason, and metadata
- Ground-truth labels (for offline evaluation)
- Correctness before and after LLM reranking

Outputs:
    hard_case_predictions.jsonl  — JSONL line-by-line
    llm_audit_samples.csv        — CSV for spreadsheet analysis
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AuditEntry:
    """One audit log entry for a product routed to the LLM."""

    timestamp: str = ""
    product_id: str = ""
    designation: str = ""
    description_hash: str = ""
    domain_pred_l1: int = -1
    domain_pred_l2: int = -1
    domain_conf_l2: float = 0.0
    top_k_candidates: list = field(default_factory=list)
    routing_reasons: list = field(default_factory=list)
    llm_selected_l2: int = -1
    llm_confidence: str = "low"
    llm_reason: str = ""
    llm_valid_json: bool = False
    llm_fallback_used: bool = False
    llm_latency_ms: float = 0.0
    ground_truth_l1: int = -1
    ground_truth_l2: int = -1
    is_correct_before: bool = False
    is_correct_after: bool = False
    decision_source: str = "domain_expert_fast_path"


_CSV_FIELDS = [
    "timestamp", "product_id", "designation", "description_hash",
    "domain_pred_l1", "domain_pred_l2", "domain_conf_l2",
    "llm_selected_l2", "llm_confidence", "llm_reason",
    "llm_valid_json", "llm_fallback_used", "llm_latency_ms",
    "ground_truth_l1", "ground_truth_l2",
    "is_correct_before", "is_correct_after", "decision_source",
    "routing_reasons",
]


def _hash_text(text: str, length: int = 16) -> str:
    """Return a short SHA-256 hex digest of the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


class AuditLogger:
    """Append-mode audit logger for hard-path predictions.

    Args:
        jsonl_path: Path for JSONL audit log.
        csv_path: Path for CSV audit log.
        log_fast_path: If True, also log fast-path predictions (sampled).
        fast_path_sample_rate: Fraction of fast-path samples to log.
    """

    def __init__(
        self,
        jsonl_path: str | Path = "logs/hard_case_predictions.jsonl",
        csv_path: str | Path = "logs/llm_audit_samples.csv",
        log_fast_path: bool = False,
        fast_path_sample_rate: float = 0.01,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.log_fast_path = log_fast_path
        self.fast_path_sample_rate = fast_path_sample_rate

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._csv_initialized = self.csv_path.exists()
        self._entries_logged = 0

    def log_hard_path(
        self,
        product: dict,
        domain_pred_l1: int,
        domain_pred_l2: int,
        domain_conf_l2: float,
        top_k_candidates: list,
        routing_reasons: list,
        llm_result,          # RerankResult or compatible dict
        ground_truth_l1: int = -1,
        ground_truth_l2: int = -1,
    ) -> None:
        """Log one hard-path product.

        Args:
            product: Dict with 'product_id', 'designation', 'description'.
            domain_pred_l1: Domain Expert predicted L1.
            domain_pred_l2: Domain Expert predicted L2 (top-1).
            domain_conf_l2: Domain Expert L2 confidence.
            top_k_candidates: List of candidate dicts from TopKCandidateGenerator.
            routing_reasons: List of routing trigger reason strings.
            llm_result: RerankResult object or dict with reranking outcome.
            ground_truth_l1: Ground-truth L1 (-1 if unknown).
            ground_truth_l2: Ground-truth L2 (-1 if unknown).
        """
        if hasattr(llm_result, "selected_l2_id"):
            llm_l2 = llm_result.selected_l2_id
            llm_conf = llm_result.confidence
            llm_reason = llm_result.reason
            llm_valid = llm_result.valid_json
            llm_fallback = llm_result.fallback_used
            llm_latency = llm_result.latency_ms
        else:
            llm_l2 = llm_result.get("selected_l2_id", -1)
            llm_conf = llm_result.get("confidence", "low")
            llm_reason = llm_result.get("reason", "")
            llm_valid = llm_result.get("valid_json", False)
            llm_fallback = llm_result.get("fallback_used", False)
            llm_latency = llm_result.get("latency_ms", 0.0)

        entry = AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            product_id=str(product.get("product_id", "")),
            designation=str(product.get("designation", ""))[:500],
            description_hash=_hash_text(str(product.get("description", ""))),
            domain_pred_l1=int(domain_pred_l1),
            domain_pred_l2=int(domain_pred_l2),
            domain_conf_l2=float(domain_conf_l2),
            top_k_candidates=[
                {"l2_id": c.get("l2_id"), "l2_label": c.get("l2_label"), "score": c.get("score")}
                for c in top_k_candidates[:20]
            ],
            routing_reasons=list(routing_reasons),
            llm_selected_l2=int(llm_l2),
            llm_confidence=str(llm_conf),
            llm_reason=str(llm_reason)[:500],
            llm_valid_json=bool(llm_valid),
            llm_fallback_used=bool(llm_fallback),
            llm_latency_ms=float(llm_latency),
            ground_truth_l1=int(ground_truth_l1),
            ground_truth_l2=int(ground_truth_l2),
            is_correct_before=int(domain_pred_l2) == int(ground_truth_l2) and ground_truth_l2 != -1,
            is_correct_after=int(llm_l2) == int(ground_truth_l2) and ground_truth_l2 != -1,
            decision_source="llm_general_expert" if not llm_fallback else "domain_expert_fallback_after_llm_error",
        )

        self._write_jsonl(entry)
        self._write_csv(entry)
        self._entries_logged += 1

    def _write_jsonl(self, entry: AuditEntry) -> None:
        record = asdict(entry)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_csv(self, entry: AuditEntry) -> None:
        record = asdict(entry)
        record["top_k_candidates"] = ""   # exclude from CSV (too long)
        record["routing_reasons"] = "|".join(entry.routing_reasons)

        write_header = not self._csv_initialized
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_CSV_FIELDS,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(record)

    @property
    def entries_logged(self) -> int:
        return self._entries_logged
