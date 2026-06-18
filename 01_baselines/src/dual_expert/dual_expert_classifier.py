"""DualExpertProductClassifier — full inference pipeline.

Wraps the Domain Expert (FlatHydra) with hierarchy-aware scoring,
top-K candidate generation, difficulty routing, and optional LLM reranking.

Fast path:
    Domain Expert top-1 with hierarchy-aware score → final prediction.

Hard path:
    Domain Expert top-K candidates + LLM General Expert → final prediction.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from .candidate_generator import CandidateGenerationConfig, TopKCandidateGenerator
from .category_definitions import CategoryDefinitionStore
from .difficulty_router import DifficultyRouter, DifficultyRouterConfig
from .hierarchy_scorer import HierarchyScoringConfig
from .llm_reranker import LLMGeneralExpertReranker, LLMRerankerConfig


@dataclass
class FastPathResult:
    """Prediction from the Domain Expert fast path."""

    product_id: str
    pred_l1_id: int
    pred_l2_id: int
    pred_l1_label: str
    pred_l2_label: str
    confidence_l1: float
    confidence_l2: float
    hierarchy_score: float
    decision_source: str = "domain_expert_fast_path"
    llm_used: bool = False
    top_k_candidates: list = None


@dataclass
class HardPathResult:
    """Prediction from the LLM General Expert hard path."""

    product_id: str
    pred_l1_id: int
    pred_l2_id: int
    pred_l1_label: str
    pred_l2_label: str
    confidence_l1: float
    confidence_l2: float
    hierarchy_score: float
    domain_top1_l2_id: int
    llm_selected_l2_id: int
    llm_confidence: str
    llm_reason: str
    llm_fallback_used: bool
    routing_reasons: list
    decision_source: str = "llm_general_expert"
    llm_used: bool = True
    top_k_candidates: list = None


class DualExpertProductClassifier:
    """Full dual-expert inference pipeline for hierarchical product classification.

    Args:
        model: Trained FlatHydra model (or nn.DataParallel wrapper).
        tokenizer: HuggingFace tokenizer.
        l1_encoder: sklearn LabelEncoder for L1.
        l2_encoder: sklearn LabelEncoder for L2.
        l2_to_l1_mapping: dict {l2_id → l1_id}.
        num_classes_l1: Number of L1 classes.
        num_classes_l2: Number of L2 classes.
        max_length: Tokenization max length.
        device: Torch device.
        scoring_config: Hierarchy scoring configuration.
        generation_config: Top-K candidate generation configuration.
        router_config: Difficulty router configuration.
        reranker_config: LLM reranker configuration.
        definition_store: Optional CategoryDefinitionStore.
        l2_counts: dict {l2_id → train_count} for rare-class detection.
        fp16: Use AMP for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        l1_encoder,
        l2_encoder,
        l2_to_l1_mapping: dict,
        num_classes_l1: int,
        num_classes_l2: int,
        max_length: int = 256,
        device: Optional[torch.device] = None,
        scoring_config: Optional[HierarchyScoringConfig] = None,
        generation_config: Optional[CandidateGenerationConfig] = None,
        router_config: Optional[DifficultyRouterConfig] = None,
        reranker_config: Optional[LLMRerankerConfig] = None,
        definition_store: Optional[CategoryDefinitionStore] = None,
        l2_counts: Optional[dict] = None,
        fp16: bool = True,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.l1_encoder = l1_encoder
        self.l2_encoder = l2_encoder
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.max_length = max_length
        self.fp16 = fp16 and self.device.type == "cuda"

        l2_to_l1_int = {int(k): int(v) for k, v in l2_to_l1_mapping.items()}
        l2_counts_int = {int(k): int(v) for k, v in (l2_counts or {}).items()}

        self.candidate_generator = TopKCandidateGenerator(
            l2_to_l1_mapping=l2_to_l1_int,
            num_l2=num_classes_l2,
            config=generation_config or CandidateGenerationConfig(),
            scoring_config=scoring_config or HierarchyScoringConfig(),
            l1_encoder=l1_encoder,
            l2_encoder=l2_encoder,
        )
        self.router = DifficultyRouter(
            config=router_config or DifficultyRouterConfig(),
            l2_to_l1_mapping=l2_to_l1_int,
            l2_counts=l2_counts_int,
        )
        self.llm_reranker = LLMGeneralExpertReranker(
            config=reranker_config or LLMRerankerConfig(enabled=False),
            category_definition_store=definition_store,
        )

    @torch.no_grad()
    def predict(self, product: dict) -> FastPathResult | HardPathResult:
        """Classify one product through the dual-expert pipeline.

        Args:
            product: Dict with keys 'designation', 'description',
                     and optionally 'product_id'.

        Returns:
            FastPathResult or HardPathResult.
        """
        product_id = str(product.get("product_id", ""))
        designation = str(product.get("designation", ""))
        description = str(product.get("description", ""))
        text = f"{designation}. {description}".strip()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with autocast("cuda", enabled=self.fp16):
            _, unified_logits = self.model(
                encoding["input_ids"], encoding["attention_mask"]
            )

        logits_l1 = unified_logits[:, :self.num_classes_l1]
        logits_l2 = unified_logits[:, self.num_classes_l1:]

        probs_l1 = torch.softmax(logits_l1, dim=-1).squeeze(0)
        probs_l2 = torch.softmax(logits_l2, dim=-1).squeeze(0)

        # Top-K candidates
        candidates = self.candidate_generator.generate(logits_l1, logits_l2)[0]
        top_l2_ids = [c["l2_id"] for c in candidates]

        top1 = candidates[0] if candidates else {}

        # Selected L2 always comes from the hierarchy-aware top-1 candidate
        pred_l2 = top1.get("l2_id", int(torch.argmax(probs_l2).item()))

        # L1 is always derived from l2_to_l1[pred_l2] for hierarchical consistency;
        # argmax(L1 head) is the fallback only when pred_l2 has no mapping entry.
        l2_to_l1 = self.router.l2_to_l1_mapping
        pred_l1 = top1.get(
            "parent_l1_id",
            l2_to_l1.get(pred_l2, int(torch.argmax(probs_l1).item())),
        )

        conf_l1 = float(probs_l1[pred_l1].item()) if pred_l1 < probs_l1.shape[0] else 0.0
        conf_l2 = float(probs_l2[pred_l2].item()) if pred_l2 < probs_l2.shape[0] else 0.0
        hier_score = float(top1.get("hierarchy_score", 0.0))

        l1_label = self._l1_name(pred_l1)
        l2_label = self._l2_name(pred_l2)

        # Route
        route, reasons = self.router.should_route(probs_l1, probs_l2, top_l2_ids, text)

        if not route or not self.llm_reranker.config.enabled:
            return FastPathResult(
                product_id=product_id,
                pred_l1_id=pred_l1,
                pred_l2_id=pred_l2,
                pred_l1_label=l1_label,
                pred_l2_label=l2_label,
                confidence_l1=conf_l1,
                confidence_l2=conf_l2,
                hierarchy_score=hier_score,
                top_k_candidates=candidates,
            )

        # LLM hard path
        llm_result = self.llm_reranker.rerank(product, candidates)
        llm_l2 = int(llm_result.selected_l2_id)
        if llm_l2 < 0 or llm_l2 >= self.num_classes_l2:
            llm_l2 = pred_l2

        # L1 is always derived from l2_to_l1[final_l2] — never from the L1 head
        final_l2 = llm_l2
        final_l1 = l2_to_l1.get(final_l2, pred_l1)

        return HardPathResult(
            product_id=product_id,
            pred_l1_id=final_l1,
            pred_l2_id=final_l2,
            pred_l1_label=self._l1_name(final_l1),
            pred_l2_label=self._l2_name(final_l2),
            confidence_l1=float(probs_l1[final_l1].item()) if final_l1 < probs_l1.shape[0] else 0.0,
            confidence_l2=float(probs_l2[final_l2].item()) if final_l2 < probs_l2.shape[0] else 0.0,
            hierarchy_score=hier_score,
            domain_top1_l2_id=pred_l2,
            llm_selected_l2_id=llm_l2,
            llm_confidence=llm_result.confidence,
            llm_reason=llm_result.reason,
            llm_fallback_used=llm_result.fallback_used,
            routing_reasons=reasons,
            top_k_candidates=candidates,
        )

    def _l1_name(self, idx: int) -> str:
        if self.l1_encoder is not None and 0 <= idx < len(self.l1_encoder.classes_):
            return str(self.l1_encoder.classes_[idx])
        return f"L1_{idx}"

    def _l2_name(self, idx: int) -> str:
        if self.l2_encoder is not None and 0 <= idx < len(self.l2_encoder.classes_):
            return str(self.l2_encoder.classes_[idx])
        return f"L2_{idx}"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        scoring_config: Optional[HierarchyScoringConfig] = None,
        generation_config: Optional[CandidateGenerationConfig] = None,
        router_config: Optional[DifficultyRouterConfig] = None,
        reranker_config: Optional[LLMRerankerConfig] = None,
        definition_store: Optional[CategoryDefinitionStore] = None,
        l2_counts: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> "DualExpertProductClassifier":
        """Load from a FlatHydra checkpoint directory.

        Expects the following files in checkpoint_dir:
            best_model.pt
            tokenizer/
            l1_label_encoder.pkl
            l2_label_encoder.pkl
            l2_to_l1_mapping.json
            config.json

        Args:
            checkpoint_dir: Directory containing FlatHydra artifacts.
            scoring_config: Override hierarchy scoring configuration.
            generation_config: Override candidate generation configuration.
            router_config: Override difficulty router configuration.
            reranker_config: Override LLM reranker configuration.
            definition_store: Optional CategoryDefinitionStore.
            l2_counts: dict {l2_id → train_count}. If None and
                       l2_class_counts.json exists, it is loaded automatically.
            device: Torch device.

        Returns:
            Initialized DualExpertProductClassifier.
        """
        import sys
        from pathlib import Path as _P

        d = _P(checkpoint_dir)

        # Load config
        with open(d / "config.json") as f:
            cfg = json.load(f)

        num_classes_l1 = cfg["num_classes_l1"]
        num_classes_l2 = cfg["num_classes_l2"]

        # Load l2_to_l1 mapping
        with open(d / "l2_to_l1_mapping.json") as f:
            l2_to_l1 = {int(k): int(v) for k, v in json.load(f).items()}

        # Load label encoders
        with open(d / "l1_label_encoder.pkl", "rb") as f:
            l1_enc = pickle.load(f)
        with open(d / "l2_label_encoder.pkl", "rb") as f:
            l2_enc = pickle.load(f)

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(d / "tokenizer"))

        # Load model
        # Add project root to path for FlatHydra import
        proj_root = d.resolve().parents[2]  # adjust depth if needed
        if str(proj_root) not in sys.path:
            sys.path.insert(0, str(proj_root))
        baselines_root = proj_root / "01_baselines"
        if str(baselines_root) not in sys.path:
            sys.path.insert(0, str(baselines_root))

        from src.flat_model import FlatHydra

        model = FlatHydra(
            encoder_name=cfg.get("encoder", "camembert-base"),
            num_classes_l1=num_classes_l1,
            num_classes_l2=num_classes_l2,
            pooling=cfg.get("pooling", "cls"),
            project_embedding=cfg.get("project_embedding", True),
            dropout=cfg.get("dropout", 0.2),
        )
        ckpt = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

        # Load l2_counts if available
        if l2_counts is None:
            counts_path = d / "l2_class_counts.json"
            if counts_path.exists():
                with open(counts_path) as f:
                    l2_counts = {int(k): int(v) for k, v in json.load(f).items()}

        return cls(
            model=model,
            tokenizer=tokenizer,
            l1_encoder=l1_enc,
            l2_encoder=l2_enc,
            l2_to_l1_mapping=l2_to_l1,
            num_classes_l1=num_classes_l1,
            num_classes_l2=num_classes_l2,
            max_length=cfg.get("max_length", 256),
            device=device,
            scoring_config=scoring_config,
            generation_config=generation_config,
            router_config=router_config,
            reranker_config=reranker_config,
            definition_store=definition_store,
            l2_counts=l2_counts,
            fp16=cfg.get("fp16", True),
        )
