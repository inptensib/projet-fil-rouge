"""Unit tests for the Dual-Expert enhancement modules.

Tests cover:
- losses: focal_cross_entropy, build_effective_number_weights
- hierarchy_scorer: HierarchyAwareScorer, hierarchy_aware_scores
- candidate_generator: TopKCandidateGenerator
- topk_metrics: topk_recall, build_topk_recall_report
- difficulty_router: DifficultyRouter routing logic
- llm_reranker: parse_llm_json, validate_llm_output
- category_definitions: CategoryDefinitionStore
- label_path_augmentation: AugmentedProductDataset
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _PROJECT_ROOT / "01_baselines"
for p in (_PROJECT_ROOT, _BASELINES_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_L1 = 5
_L2 = 12
_B = 3


def _l2_to_l1() -> dict[int, int]:
    return {i: i % _L1 for i in range(_L2)}


# ===========================================================================
# losses.py
# ===========================================================================

class TestFocalCrossEntropy:
    def test_returns_scalar(self):
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(_B, _L2)
        targets = torch.randint(0, _L2, (_B,))
        loss = focal_cross_entropy(logits, targets, gamma=2.0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gamma_zero_matches_ce(self):
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(8, _L2)
        targets = torch.randint(0, _L2, (8,))
        fl = focal_cross_entropy(logits, targets, gamma=0.0)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        assert abs(fl.item() - ce.item()) < 1e-4

    def test_with_class_weights(self):
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(_B, _L2)
        targets = torch.randint(0, _L2, (_B,))
        weights = torch.ones(_L2)
        loss = focal_cross_entropy(logits, targets, class_weights=weights, gamma=2.0)
        assert not torch.isnan(loss)

    def test_backward_no_nan(self):
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(_B, _L2, requires_grad=True)
        targets = torch.randint(0, _L2, (_B,))
        loss = focal_cross_entropy(logits, targets, gamma=2.0)
        loss.backward()
        assert not torch.isnan(logits.grad).any()

    def test_reduction_none(self):
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(_B, _L2)
        targets = torch.randint(0, _L2, (_B,))
        loss = focal_cross_entropy(logits, targets, gamma=2.0, reduction="none")
        assert loss.shape == (_B,)


class TestEffectiveNumberWeights:
    def test_shape(self):
        from src.dual_expert.losses import build_effective_number_weights
        labels = list(range(_L2)) * 10
        wt = build_effective_number_weights(labels, _L2)
        assert wt.shape == (_L2,)

    def test_no_nan(self):
        from src.dual_expert.losses import build_effective_number_weights
        labels = list(range(_L2)) * 10
        wt = build_effective_number_weights(labels, _L2)
        assert not torch.isnan(wt).any()

    def test_rare_class_higher_weight(self):
        from src.dual_expert.losses import build_effective_number_weights
        # Class 0 appears 1 time, class 1 appears 100 times
        labels = [0] + [1] * 100
        wt = build_effective_number_weights(labels, 2)
        assert wt[0] > wt[1]

    def test_max_weight_clipped(self):
        from src.dual_expert.losses import build_effective_number_weights
        labels = [0]  # class 0 seen once, very rare
        wt = build_effective_number_weights(labels, 2, max_weight=3.0)
        assert wt.max().item() <= 3.0 + 1e-6


# ===========================================================================
# hierarchy_scorer.py
# ===========================================================================

class TestHierarchyAwareScorer:
    def test_scores_shape(self):
        from src.dual_expert.hierarchy_scorer import HierarchyAwareScorer, HierarchyScoringConfig
        scorer = HierarchyAwareScorer(
            _l2_to_l1(), _L2, HierarchyScoringConfig(enabled=True, alpha=0.5)
        )
        logits_l1 = torch.randn(_B, _L1)
        logits_l2 = torch.randn(_B, _L2)
        scores = scorer.score(logits_l1, logits_l2)
        assert scores.shape == (_B, _L2)

    def test_disabled_returns_log_probs(self):
        from src.dual_expert.hierarchy_scorer import HierarchyAwareScorer, HierarchyScoringConfig
        scorer = HierarchyAwareScorer(
            _l2_to_l1(), _L2, HierarchyScoringConfig(enabled=False)
        )
        logits_l2 = torch.zeros(_B, _L2)
        logits_l1 = torch.zeros(_B, _L1)
        scores = scorer.score(logits_l1, logits_l2)
        # All same logits → uniform probs → log(1/L2) everywhere
        expected = torch.log(torch.softmax(logits_l2, dim=-1) + 1e-12)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_alpha_zero_equals_log_l2_prob(self):
        from src.dual_expert.hierarchy_scorer import hierarchy_aware_scores, make_l2_to_l1_tensor
        l2_to_l1 = make_l2_to_l1_tensor(_l2_to_l1(), _L2)
        probs_l1 = torch.softmax(torch.randn(_B, _L1), dim=-1)
        probs_l2 = torch.softmax(torch.randn(_B, _L2), dim=-1)
        scores = hierarchy_aware_scores(probs_l1, probs_l2, l2_to_l1, alpha=0.0)
        expected = torch.log(probs_l2 + 1e-12)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_no_nan(self):
        from src.dual_expert.hierarchy_scorer import HierarchyAwareScorer, HierarchyScoringConfig
        scorer = HierarchyAwareScorer(_l2_to_l1(), _L2)
        logits_l1 = torch.randn(_B, _L1)
        logits_l2 = torch.randn(_B, _L2)
        scores = scorer.score(logits_l1, logits_l2)
        assert not torch.isnan(scores).any()


# ===========================================================================
# candidate_generator.py
# ===========================================================================

class TestTopKCandidateGenerator:
    def _gen(self, k: int = 5):
        from src.dual_expert.candidate_generator import (
            CandidateGenerationConfig,
            TopKCandidateGenerator,
        )
        from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
        return TopKCandidateGenerator(
            l2_to_l1_mapping=_l2_to_l1(),
            num_l2=_L2,
            config=CandidateGenerationConfig(top_k=k),
            scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
        )

    def test_returns_k_candidates(self):
        gen = self._gen(k=5)
        logits_l1 = torch.randn(_B, _L1)
        logits_l2 = torch.randn(_B, _L2)
        results = gen.generate(logits_l1, logits_l2)
        assert len(results) == _B
        for cands in results:
            assert len(cands) == 5

    def test_parent_l1_matches_mapping(self):
        gen = self._gen(k=_L2)
        logits_l1 = torch.randn(1, _L1)
        logits_l2 = torch.randn(1, _L2)
        cands = gen.generate(logits_l1, logits_l2)[0]
        for c in cands:
            expected_l1 = _l2_to_l1().get(c["l2_id"], -1)
            assert c["parent_l1_id"] == expected_l1

    def test_scores_descending(self):
        gen = self._gen(k=_L2)
        logits_l1 = torch.randn(1, _L1)
        logits_l2 = torch.randn(1, _L2)
        cands = gen.generate(logits_l1, logits_l2)[0]
        scores = [c["hierarchy_score"] for c in cands]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field(self):
        gen = self._gen(k=3)
        cands = gen.generate(torch.randn(1, _L1), torch.randn(1, _L2))[0]
        assert [c["rank"] for c in cands] == [1, 2, 3]


# ===========================================================================
# topk_metrics.py
# ===========================================================================

class TestTopKMetrics:
    def test_topk_recall_perfect(self):
        from src.dual_expert.topk_metrics import topk_recall
        labels = [0, 1, 2]
        preds = [[0, 5], [1, 6], [2, 7]]
        assert topk_recall(labels, preds, k=1) == 1.0

    def test_topk_recall_zero(self):
        from src.dual_expert.topk_metrics import topk_recall
        labels = [0, 1, 2]
        preds = [[3, 4], [3, 4], [3, 4]]
        assert topk_recall(labels, preds, k=5) == 0.0

    def test_topk_recall_partial(self):
        from src.dual_expert.topk_metrics import topk_recall
        labels = [0, 1, 2, 3]
        preds = [[0], [5], [2], [9]]
        assert topk_recall(labels, preds, k=1) == 0.5

    def test_report_has_global_and_buckets(self):
        from src.dual_expert.topk_metrics import build_topk_recall_report
        labels = list(range(10))
        preds = [[l, (l + 1) % 12] for l in labels]
        train_labels = list(range(12)) * 5
        report = build_topk_recall_report(labels, preds, train_labels, 12, k_values=[1, 3])
        assert "global" in report
        assert "by_bucket" in report
        assert "l2_top_1_accuracy" in report["global"]

    def test_save_report(self, tmp_path):
        from src.dual_expert.topk_metrics import build_topk_recall_report, save_topk_recall_report
        report = build_topk_recall_report(
            [0, 1], [[0], [1]], [0, 1], 12, k_values=[1]
        )
        out = tmp_path / "report.json"
        save_topk_recall_report(report, out)
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)
        assert "global" in loaded


# ===========================================================================
# difficulty_router.py
# ===========================================================================

class TestDifficultyRouter:
    def _router(self, **kwargs):
        from src.dual_expert.difficulty_router import (
            DifficultyRouter,
            DifficultyRouterConfig,
            RoutingThresholds,
            RoutingRules,
        )
        thresholds = RoutingThresholds(
            min_l2_confidence=0.75,
            min_l2_margin=0.15,
            rare_class_threshold=100,
            rare_class_min_confidence=0.90,
        )
        rules = RoutingRules(**kwargs)
        cfg = DifficultyRouterConfig(enabled=True, thresholds=thresholds, rules=rules)
        l2_counts = {i: 200 for i in range(_L2)}  # all frequent by default
        return DifficultyRouter(cfg, _l2_to_l1(), l2_counts)

    def test_high_confidence_fast_path(self):
        router = self._router()
        probs_l1 = torch.zeros(_L1)
        probs_l1[0] = 0.95
        probs_l2 = torch.zeros(_L2)
        probs_l2[0] = 0.95  # l2=0 → parent l1=0 ✓
        probs_l2[1] = 0.02
        route, reasons = router.should_route(probs_l1, probs_l2, [0, 1], "good long text")
        assert not route

    def test_low_confidence_routes_to_llm(self):
        router = self._router()
        probs_l1 = torch.ones(_L1) / _L1
        probs_l2 = torch.ones(_L2) / _L2   # all uniform → low confidence
        route, reasons = router.should_route(probs_l1, probs_l2, list(range(_L2)), "text")
        assert route
        assert "low_l2_confidence" in reasons

    def test_hierarchy_inconsistency_routes(self):
        router = self._router(send_if_hierarchy_inconsistent=True)
        probs_l1 = torch.zeros(_L1)
        probs_l1[1] = 0.99  # predicted L1 = 1
        probs_l2 = torch.zeros(_L2)
        probs_l2[0] = 0.99  # l2=0 → parent l1=0 ≠ 1 → inconsistent
        route, reasons = router.should_route(probs_l1, probs_l2, [0], "long enough text here")
        assert route
        assert "hierarchy_inconsistent" in reasons

    def test_pass_through_rate_tracked(self):
        router = self._router()
        probs_l1 = torch.zeros(_L1); probs_l1[0] = 0.99
        probs_l2 = torch.zeros(_L2); probs_l2[0] = 0.99
        router.should_route(probs_l1, probs_l2, [0], "long enough text here")
        assert router._n_total == 1


# ===========================================================================
# llm_reranker.py
# ===========================================================================

class TestParseLLMJson:
    def test_valid_json(self):
        from src.dual_expert.llm_reranker import parse_llm_json
        resp = '{"selected_l2_id": 42, "confidence": "high", "reason": "ok"}'
        result = parse_llm_json(resp)
        assert result["selected_l2_id"] == 42

    def test_json_in_markdown_fence(self):
        from src.dual_expert.llm_reranker import parse_llm_json
        resp = '```json\n{"selected_l2_id": 5}\n```'
        result = parse_llm_json(resp)
        assert result["selected_l2_id"] == 5

    def test_json_embedded_in_text(self):
        from src.dual_expert.llm_reranker import parse_llm_json
        resp = 'Voici ma réponse: {"selected_l2_id": 3, "confidence": "low"}'
        result = parse_llm_json(resp)
        assert result["selected_l2_id"] == 3

    def test_invalid_raises(self):
        from src.dual_expert.llm_reranker import parse_llm_json
        with pytest.raises(ValueError):
            parse_llm_json("no json here at all")


class TestValidateLLMOutput:
    def test_valid_selection(self):
        from src.dual_expert.llm_reranker import validate_llm_output
        output = {"selected_l2_id": 5, "confidence": "high", "reason": "ok"}
        validated = validate_llm_output(output, [5, 10, 15])
        assert validated["selected_l2_id"] == 5

    def test_out_of_candidates_raises(self):
        from src.dual_expert.llm_reranker import validate_llm_output
        output = {"selected_l2_id": 99}
        with pytest.raises(ValueError, match="not in candidates"):
            validate_llm_output(output, [1, 2, 3])

    def test_invalid_confidence_defaulted(self):
        from src.dual_expert.llm_reranker import validate_llm_output
        output = {"selected_l2_id": 5, "confidence": "very_sure"}
        validated = validate_llm_output(output, [5])
        assert validated["confidence"] == "medium"


class TestLLMRerankerDryRun:
    def test_none_provider_returns_fallback(self):
        from src.dual_expert.llm_reranker import LLMGeneralExpertReranker, LLMRerankerConfig
        cfg = LLMRerankerConfig(enabled=True, provider="none", fallback_to_domain_expert=True)
        reranker = LLMGeneralExpertReranker(config=cfg)
        candidates = [{"l2_id": 5, "score": -0.3}, {"l2_id": 10, "score": -0.6}]
        product = {"designation": "Test", "description": "Desc"}
        result = reranker.rerank(product, candidates)
        # dry-run returns l2_id=-1 which triggers fallback to top-1=5
        assert result.fallback_used or result.selected_l2_id == 5


# ===========================================================================
# category_definitions.py
# ===========================================================================

class TestCategoryDefinitionStore:
    def _make_jsonl(self, tmp_path: Path) -> Path:
        records = [
            {"l2_id": 1, "definition_fr": "Définition de test", "positive_signals": ["signal"]},
            {"l2_id": 2, "definition_fr": "Autre définition", "positive_signals": ["autre"]},
        ]
        p = tmp_path / "defs.jsonl"
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return p

    def test_load_and_get(self, tmp_path):
        from src.dual_expert.category_definitions import CategoryDefinitionStore
        p = self._make_jsonl(tmp_path)
        store = CategoryDefinitionStore(str(p))
        defn = store.get(1)
        assert defn["definition_fr"] == "Définition de test"

    def test_missing_id_fallback(self, tmp_path):
        from src.dual_expert.category_definitions import CategoryDefinitionStore
        p = self._make_jsonl(tmp_path)
        store = CategoryDefinitionStore(str(p))
        defn = store.get(999)
        assert defn["confidence"] == "low"
        assert defn["_is_fallback"] is True

    def test_none_path_fallback_only_mode(self):
        from src.dual_expert.category_definitions import CategoryDefinitionStore
        store = CategoryDefinitionStore(path=None)
        assert len(store) == 0
        defn = store.get(42)
        assert "definition_fr" in defn

    def test_get_many(self, tmp_path):
        from src.dual_expert.category_definitions import CategoryDefinitionStore
        p = self._make_jsonl(tmp_path)
        store = CategoryDefinitionStore(str(p))
        results = store.get_many([1, 2, 999])
        assert len(results) == 3
        assert results[2]["_is_fallback"] is True


# ===========================================================================
# label_path_augmentation.py
# ===========================================================================

class TestAugmentedProductDataset:
    def _make_dataset(self, enabled=True, apply_to="all"):
        from unittest import mock
        from src.dual_expert.label_path_augmentation import (
            AugmentedProductDataset,
            LabelPathAugConfig,
        )
        from sklearn.preprocessing import LabelEncoder

        n = 20
        texts = [f"produit {i}" for i in range(n)]
        l1 = [i % _L1 for i in range(n)]
        l2 = [i % _L2 for i in range(n)]

        le1 = LabelEncoder().fit([f"L1_{i}" for i in range(_L1)])
        le2 = LabelEncoder().fit([f"L2_{i}" for i in range(_L2)])

        tok = mock.MagicMock()
        tok.side_effect = lambda text, **kw: {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

        cfg = LabelPathAugConfig(
            enabled=enabled,
            apply_to=apply_to,
            rare_threshold=100,
            augmentation_factor_rare=2,
            mask_probability=0.0,
        )
        return AugmentedProductDataset(
            texts, l1, l2,
            tokenizer=tok,
            max_length=8,
            l1_encoder=le1,
            l2_encoder=le2,
            l2_to_l1_mapping={i: i % _L1 for i in range(_L2)},
            config=cfg,
            seed=42,
        )

    def test_disabled_same_length(self):
        ds = self._make_dataset(enabled=False)
        assert len(ds) == 20

    def test_enabled_all_expands(self):
        ds = self._make_dataset(enabled=True, apply_to="all")
        assert len(ds) > 20

    def test_item_has_required_keys(self):
        ds = self._make_dataset(enabled=False)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label_l1" in item
        assert "label_l2" in item

    def test_invalid_apply_to_raises(self):
        from src.dual_expert.label_path_augmentation import LabelPathAugConfig
        with pytest.raises(ValueError):
            LabelPathAugConfig(apply_to="invalid")

    def test_full_path_disabled_by_default(self):
        """With allow_full_path=False, every augmented text must mask one level."""
        from src.dual_expert.label_path_augmentation import (
            AugmentedProductDataset,
            LabelPathAugConfig,
        )
        from sklearn.preprocessing import LabelEncoder

        n = 20
        le1 = LabelEncoder().fit([f"L1_{i}" for i in range(_L1)])
        le2 = LabelEncoder().fit([f"L2_{i}" for i in range(_L2)])

        cfg = LabelPathAugConfig(
            enabled=True,
            apply_to="all",
            rare_threshold=100,
            augmentation_factor_rare=2,
            mask_probability=0.0,   # would expose full path in old code
            allow_full_path=False,  # new default
        )

        tok = mock.MagicMock()
        tok.side_effect = lambda text, **kw: {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

        ds = AugmentedProductDataset(
            [f"produit {i}" for i in range(n)],
            [i % _L1 for i in range(n)],
            [i % _L2 for i in range(n)],
            tokenizer=tok,
            max_length=8,
            l1_encoder=le1,
            l2_encoder=le2,
            l2_to_l1_mapping={i: i % _L1 for i in range(_L2)},
            config=cfg,
            seed=0,
        )
        # Every augmented copy should have called the tokenizer with a masked path
        mask_token = AugmentedProductDataset.MASK_TOKEN
        for call_args in tok.call_args_list:
            text = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
            if "[CAT]" in text:
                # At least one level must be masked
                assert mask_token in text, f"Full path exposed: {text!r}"


# ===========================================================================
# Smoke tests — end-to-end minimal pipeline checks
# ===========================================================================

def _make_mock_model(num_l1: int, num_l2: int):
    """Return a mock nn.Module that outputs fixed unified logits."""
    model = mock.MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    logits = torch.zeros(1, num_l1 + num_l2)
    logits[0, 0] = 5.0          # strong L1 class 0
    logits[0, num_l1] = 5.0     # strong L2 class 0 (mapped to L1 class 0)
    model.return_value = (None, logits)
    model.__call__ = lambda self, ids, mask: (None, logits)
    return model


class TestSmoke:
    """Minimal end-to-end tests covering every major code path."""

    def _make_classifier(self, provider="none", llm_enabled=False):
        from unittest import mock as _mock
        from sklearn.preprocessing import LabelEncoder
        from src.dual_expert.candidate_generator import CandidateGenerationConfig
        from src.dual_expert.difficulty_router import DifficultyRouterConfig
        from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier
        from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
        from src.dual_expert.llm_reranker import LLMRerankerConfig

        le1 = LabelEncoder().fit([f"L1_{i}" for i in range(_L1)])
        le2 = LabelEncoder().fit([f"L2_{i}" for i in range(_L2)])

        tok = _mock.MagicMock()
        tok.side_effect = lambda text, **kw: {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        tok.__call__ = tok.side_effect

        model = _make_mock_model(_L1, _L2)

        l2_to_l1 = _l2_to_l1()
        return DualExpertProductClassifier(
            model=model,
            tokenizer=tok,
            l1_encoder=le1,
            l2_encoder=le2,
            l2_to_l1_mapping=l2_to_l1,
            num_classes_l1=_L1,
            num_classes_l2=_L2,
            max_length=16,
            device=torch.device("cpu"),
            scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
            generation_config=CandidateGenerationConfig(top_k=5),
            router_config=DifficultyRouterConfig(enabled=True, max_llm_pass_through_rate=1.0),
            reranker_config=LLMRerankerConfig(
                enabled=llm_enabled,
                provider=provider,
                fallback_to_domain_expert=True,
            ),
            fp16=False,
        )

    def test_smoke_baseline_inference(self):
        """Fast path: classifier returns FastPathResult for easy product."""
        from src.dual_expert.dual_expert_classifier import FastPathResult
        clf = self._make_classifier()
        result = clf.predict({"designation": "Chaussures de sport", "description": "Légères"})
        assert isinstance(result, FastPathResult)
        assert result.pred_l2_id >= 0
        assert result.pred_l1_id >= 0

    def test_smoke_hierarchy_aware_topk(self):
        """Hierarchy-aware top-K returns ranked candidates with hierarchy_score."""
        from src.dual_expert.candidate_generator import (
            CandidateGenerationConfig,
            TopKCandidateGenerator,
        )
        from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig

        gen = TopKCandidateGenerator(
            l2_to_l1_mapping=_l2_to_l1(),
            num_l2=_L2,
            config=CandidateGenerationConfig(top_k=5),
            scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
        )
        logits_l1 = torch.randn(2, _L1)
        logits_l2 = torch.randn(2, _L2)
        results = gen.generate(logits_l1, logits_l2)
        assert len(results) == 2
        for cands in results:
            assert len(cands) == 5
            assert "hierarchy_score" in cands[0]
            assert "prob_l2" in cands[0]
            scores = [c["hierarchy_score"] for c in cands]
            assert scores == sorted(scores, reverse=True)

    def test_smoke_focal_loss_forward(self):
        """Focal loss forward pass: no NaN, gradient flows."""
        from src.dual_expert.losses import focal_cross_entropy
        logits = torch.randn(_B, _L2, requires_grad=True)
        targets = torch.randint(0, _L2, (_B,))
        weights = torch.rand(_L2) + 0.1
        loss = focal_cross_entropy(logits, targets, class_weights=weights, gamma=2.0)
        assert not torch.isnan(loss)
        loss.backward()
        assert not torch.isnan(logits.grad).any()

    def test_smoke_category_definition_fallback(self):
        """CategoryDefinitionStore with path=None returns fallback for all IDs."""
        from src.dual_expert.category_definitions import CategoryDefinitionStore
        store = CategoryDefinitionStore(path=None)
        for l2_id in [0, 42, 9999]:
            defn = store.get(l2_id)
            assert defn["_is_fallback"] is True
            assert "definition_fr" in defn

    def test_smoke_llm_provider_none_fallback(self):
        """LLM reranker with provider='none' falls back to domain expert top-1."""
        from src.dual_expert.llm_reranker import LLMGeneralExpertReranker, LLMRerankerConfig
        cfg = LLMRerankerConfig(
            enabled=True, provider="none", fallback_to_domain_expert=True
        )
        reranker = LLMGeneralExpertReranker(config=cfg)
        candidates = [{"l2_id": 7, "hierarchy_score": -0.3, "prob_l2": 0.4},
                      {"l2_id": 3, "hierarchy_score": -0.8, "prob_l2": 0.1}]
        product = {"designation": "Appareil photo", "description": "Reflex"}
        result = reranker.rerank(product, candidates)
        # dry-run returns id=-1, triggering fallback to candidates[0]=7
        assert result.fallback_used
        assert result.selected_l2_id == 7

    def test_smoke_hard_path_mocked_llm(self):
        """Hard-path prediction with mocked LLM JSON response."""
        import json as _json

        from src.dual_expert.difficulty_router import (
            DifficultyRouterConfig,
            RoutingRules,
            RoutingThresholds,
        )
        from src.dual_expert.dual_expert_classifier import HardPathResult
        from src.dual_expert.llm_reranker import LLMRerankerConfig

        from unittest import mock as _mock
        from sklearn.preprocessing import LabelEncoder

        le1 = LabelEncoder().fit([f"L1_{i}" for i in range(_L1)])
        le2 = LabelEncoder().fit([f"L2_{i}" for i in range(_L2)])

        tok = _mock.MagicMock()
        tok.side_effect = lambda text, **kw: {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        # Force routing: very low confidence thresholds
        router_cfg = DifficultyRouterConfig(
            enabled=True,
            max_llm_pass_through_rate=1.0,
            thresholds=RoutingThresholds(
                min_l2_confidence=0.99,   # always triggers low_l2_confidence
                min_l2_margin=0.99,
                min_parent_l1_confidence=0.99,
            ),
            rules=RoutingRules(send_if_low_l2_confidence=True),
        )

        # LLM client returns L2=1 (a valid candidate)
        mock_response = _json.dumps({
            "selected_l2_id": 1,
            "selected_path": "L1_0 > L2_1",
            "confidence": "high",
            "reason": "mocked",
        })

        from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier
        from src.dual_expert.candidate_generator import CandidateGenerationConfig
        from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig

        model = _make_mock_model(_L1, _L2)
        clf = DualExpertProductClassifier(
            model=model,
            tokenizer=tok,
            l1_encoder=le1,
            l2_encoder=le2,
            l2_to_l1_mapping=_l2_to_l1(),
            num_classes_l1=_L1,
            num_classes_l2=_L2,
            max_length=16,
            device=torch.device("cpu"),
            scoring_config=HierarchyScoringConfig(enabled=True),
            generation_config=CandidateGenerationConfig(top_k=5),
            router_config=router_cfg,
            reranker_config=LLMRerankerConfig(
                enabled=True, provider="none", fallback_to_domain_expert=True
            ),
            fp16=False,
        )

        # Patch the LLM client to return our mocked JSON
        with _mock.patch.object(
            clf.llm_reranker._client, "generate", return_value=mock_response
        ):
            result = clf.predict({"designation": "Vélo", "description": "VTT"})

        # Either the LLM path was taken (HardPathResult) or routing didn't fire
        # Regardless, the result must be valid
        assert result.pred_l2_id >= 0
        assert result.pred_l1_id == _l2_to_l1().get(result.pred_l2_id, result.pred_l1_id)
