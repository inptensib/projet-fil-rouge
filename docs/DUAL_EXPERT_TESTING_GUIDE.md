# Dual Expert Pipeline — Complete Testing Guide

This guide covers every test layer for the Dual-Expert architecture: unit
tests per module, integration smoke tests, manual inference, parameter
tuning, and LLM path validation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites and Setup](#2-prerequisites-and-setup)
3. [Running the Test Suite](#3-running-the-test-suite)
4. [Module-by-Module Unit Tests](#4-module-by-module-unit-tests)
5. [Smoke Tests — End-to-End Pipeline](#5-smoke-tests--end-to-end-pipeline)
6. [Manual Inference Testing](#6-manual-inference-testing)
7. [Parameters and Tunable Knobs](#7-parameters-and-tunable-knobs)
8. [LLM Provider Configuration](#8-llm-provider-configuration)
9. [Evaluation and Metrics](#9-evaluation-and-metrics)
10. [Common Failure Modes and Fixes](#10-common-failure-modes-and-fixes)

---

## 1. Architecture Overview

```
Product: designation + description
        │
        ▼
FlatHydra (CamemBERT) = Domain Expert
        │
        ├── logits L1  (157 classes)
        ├── logits L2  (6 126 classes)
        └── top-K L2 candidates  (HierarchyAwareScorer)
                │
                ▼
        DifficultyRouter
        ├── Fast path  ──► FastPathResult  (Domain Expert top-1)
        └── Hard path  ──► LLMGeneralExpertReranker ──► HardPathResult
```

The pipeline lives entirely in `01_baselines/src/dual_expert/`.

| Module                       | Responsibility                                      |
|:-----------------------------|:----------------------------------------------------|
| `dual_expert_classifier.py`  | Orchestrates the full inference pipeline            |
| `hierarchy_scorer.py`        | Combined `log P(L2) + α·log P(parent L1)` scoring  |
| `candidate_generator.py`     | Top-K ranked L2 candidate list                      |
| `difficulty_router.py`       | Fast-path vs. hard-path routing decision            |
| `llm_reranker.py`            | LLM prompt, call, JSON parse, fallback              |
| `losses.py`                  | `focal_cross_entropy`, effective-number weights     |
| `label_path_augmentation.py` | Rare-class dataset augmentation with path prefix    |
| `topk_metrics.py`            | Top-K recall and frequency-bucket metrics           |
| `category_definitions.py`    | Offline category definition store (JSONL)           |
| `audit_logger.py`            | JSONL + CSV hard-path audit log                     |

---

## 2. Prerequisites and Setup

### 2.1 Python environment

```bash
cd /home/younes/projects/projet-fil-rouge
source .venv/bin/activate      # or python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Path setup for manual scripts

```python
import sys
sys.path.insert(0, "/home/younes/projects/projet-fil-rouge")
sys.path.insert(0, "/home/younes/projects/projet-fil-rouge/01_baselines")
```

Tests bootstrap this automatically via `conftest.py` — no manual setup
needed for `pytest`.

### 2.3 Docker (full isolation)

```bash
docker compose run --rm --profile test test
```

### 2.4 Verify imports before running anything

```bash
cd 01_baselines
python -c "from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier; print('OK')"
```

---

## 3. Running the Test Suite

### 3.1 Full suite

```bash
python -m pytest tests/ -v --tb=short
```

### 3.2 Dual-expert tests only

```bash
python -m pytest tests/test_dual_expert.py -v
```

### 3.3 Single test class

```bash
# Difficulty router only
python -m pytest tests/test_dual_expert.py::TestDifficultyRouter -v

# Focal loss only
python -m pytest tests/test_dual_expert.py::TestFocalCrossEntropy -v

# Smoke tests only
python -m pytest tests/test_dual_expert.py::TestSmoke -v
```

### 3.4 Single test

```bash
python -m pytest tests/test_dual_expert.py::TestSmoke::test_smoke_baseline_inference -v
```

### 3.5 Expected output (all green)

```
tests/test_dual_expert.py::TestFocalCrossEntropy::test_returns_scalar              PASSED
tests/test_dual_expert.py::TestFocalCrossEntropy::test_gamma_zero_matches_ce       PASSED
tests/test_dual_expert.py::TestHierarchyAwareScorer::test_scores_shape             PASSED
tests/test_dual_expert.py::TestTopKCandidateGenerator::test_returns_k_candidates   PASSED
tests/test_dual_expert.py::TestDifficultyRouter::test_high_confidence_fast_path    PASSED
tests/test_dual_expert.py::TestSmoke::test_smoke_baseline_inference                PASSED
...
```

---

## 4. Module-by-Module Unit Tests

### 4.1 Focal loss — `losses.py`

**Test class:** `TestFocalCrossEntropy`

| Test                             | What it verifies                                        |
|:---------------------------------|:--------------------------------------------------------|
| `test_returns_scalar`            | Returns a scalar > 0                                    |
| `test_gamma_zero_matches_ce`     | `gamma=0` equals standard `F.cross_entropy`             |
| `test_with_class_weights`        | Class weights accepted, output not NaN                  |
| `test_backward_no_nan`           | Gradient flows without NaN                              |
| `test_reduction_none`            | `reduction="none"` returns per-sample tensor `(B,)`     |

**Test class:** `TestEffectiveNumberWeights`

| Test                           | What it verifies                                          |
|:-------------------------------|:----------------------------------------------------------|
| `test_shape`                   | Output shape `(num_classes,)`                             |
| `test_no_nan`                  | No NaN in computed weights                                |
| `test_rare_class_higher_weight`| Rare class gets strictly higher weight than frequent one  |
| `test_max_weight_clipped`      | Weights clipped to `max_weight`                           |

**Quick manual check:**

```python
from src.dual_expert.losses import focal_cross_entropy
import torch

logits  = torch.randn(4, 12)
targets = torch.randint(0, 12, (4,))
loss    = focal_cross_entropy(logits, targets, gamma=2.0)
print(f"loss={loss.item():.4f}  nan={torch.isnan(loss).item()}")
# Expected: loss > 0, nan=False
```

---

### 4.2 Hierarchy scorer — `hierarchy_scorer.py`

**Test class:** `TestHierarchyAwareScorer`

| Test                                    | What it verifies                                  |
|:----------------------------------------|:--------------------------------------------------|
| `test_scores_shape`                     | Output shape `(B, L2)`                            |
| `test_disabled_returns_log_probs`       | Disabled → plain `log softmax(L2)`                |
| `test_alpha_zero_equals_log_l2_prob`    | `alpha=0` → pure L2 log-prob                      |
| `test_no_nan`                           | No NaN in any score                               |

**Formula under test:**

```
score(l2) = log(P(l2|text) + ε) + α · log(P(parent_l1(l2)|text) + ε)
```

**Quick manual check:**

```python
from src.dual_expert.hierarchy_scorer import HierarchyAwareScorer, HierarchyScoringConfig
import torch

scorer = HierarchyAwareScorer(
    l2_to_l1_mapping={i: i % 5 for i in range(12)},
    num_l2=12,
    config=HierarchyScoringConfig(enabled=True, alpha=0.5),
)
scores = scorer.score(torch.randn(2, 5), torch.randn(2, 12))
print(f"shape={scores.shape}  nan={torch.isnan(scores).any().item()}")
# Expected: shape=(2, 12)  nan=False
```

---

### 4.3 Top-K candidate generator — `candidate_generator.py`

**Test class:** `TestTopKCandidateGenerator`

| Test                          | What it verifies                                         |
|:------------------------------|:---------------------------------------------------------|
| `test_returns_k_candidates`   | Returns exactly `k` candidates per sample               |
| `test_parent_l1_matches_mapping` | `parent_l1_id` consistent with `l2_to_l1` mapping    |
| `test_scores_descending`      | Candidates sorted by `hierarchy_score` descending        |
| `test_rank_field`             | `rank` field equals `[1, 2, ..., k]`                     |

**Candidate dict fields:**

| Field             | Description                                            |
|:------------------|:-------------------------------------------------------|
| `rank`            | 1-based position in the ranked list                    |
| `l2_id`           | L2 class index                                         |
| `l2_label`        | Human-readable L2 name (if encoder provided)           |
| `parent_l1_id`    | Parent L1 class index from `l2_to_l1` mapping          |
| `parent_l1_label` | Human-readable L1 name (if encoder provided)           |
| `hierarchy_score` | Combined ranking score (log P(L2) + α·log P(L1))       |
| `prob_l2`         | Raw softmax P(L2\|text)                                 |
| `prob_parent_l1`  | Raw softmax P(parent L1\|text)                          |

**Quick manual check:**

```python
from src.dual_expert.candidate_generator import CandidateGenerationConfig, TopKCandidateGenerator
from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
import torch

gen = TopKCandidateGenerator(
    l2_to_l1_mapping={i: i % 5 for i in range(12)},
    num_l2=12,
    config=CandidateGenerationConfig(top_k=5),
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
)
candidates = gen.generate(torch.randn(1, 5), torch.randn(1, 12))[0]
for c in candidates:
    print(f"rank={c['rank']}  l2={c['l2_id']}  parent_l1={c['parent_l1_id']}"
          f"  score={c['hierarchy_score']:.4f}  prob_l2={c['prob_l2']:.4f}")
```

---

### 4.4 Difficulty router — `difficulty_router.py`

**Test class:** `TestDifficultyRouter`

| Test                                     | What it verifies                                    |
|:-----------------------------------------|:----------------------------------------------------|
| `test_high_confidence_fast_path`         | High-conf product stays on fast path                |
| `test_low_confidence_routes_to_llm`      | Uniform probs trigger `low_l2_confidence`           |
| `test_hierarchy_inconsistency_routes`    | L1/L2 mismatch triggers `hierarchy_inconsistent`   |
| `test_pass_through_rate_tracked`         | `_n_total` counter increments each call             |

**Six routing signals — any one sends to LLM:**

| Signal                         | Rule toggle                               |
|:-------------------------------|:------------------------------------------|
| Hierarchy inconsistency        | `send_if_hierarchy_inconsistent=True`     |
| Low L2 confidence              | `send_if_low_l2_confidence=True`          |
| Low L2 margin (top1 vs top2)   | `send_if_low_l2_margin=True`              |
| Rare class + low confidence    | `send_if_rare_and_not_very_confident=True`|
| Short product text             | `send_if_short_text=False` (off by default)|
| Sibling confusion in top-K    | `send_if_sibling_confusion=True`          |

**Quick manual check:**

```python
from src.dual_expert.difficulty_router import DifficultyRouter, DifficultyRouterConfig
import torch

router = DifficultyRouter(
    config=DifficultyRouterConfig(enabled=True),
    l2_to_l1_mapping={i: i % 5 for i in range(12)},
    l2_counts={i: 200 for i in range(12)},
)

# Low confidence case → should route
probs_l1 = torch.ones(5) / 5
probs_l2 = torch.ones(12) / 12
route, reasons = router.should_route(probs_l1, probs_l2, list(range(12)), "text")
print(f"route={route}  reasons={reasons}")
# Expected: route=True, reasons=['low_l2_confidence', 'low_l2_margin', ...]

# High confidence case → should NOT route
probs_l1 = torch.zeros(5); probs_l1[0] = 0.95
probs_l2 = torch.zeros(12); probs_l2[0] = 0.95
route, reasons = router.should_route(probs_l1, probs_l2, [0], "long enough text here")
print(f"route={route}  reasons={reasons}")
# Expected: route=False, reasons=[]
```

---

### 4.5 LLM reranker — `llm_reranker.py`

**Test classes:** `TestParseLLMJson`, `TestValidateLLMOutput`,
`TestLLMRerankerDryRun`

| Test                                  | What it verifies                                      |
|:--------------------------------------|:------------------------------------------------------|
| `test_valid_json`                     | Clean JSON parsed correctly                           |
| `test_json_in_markdown_fence`         | JSON inside ` ```json ``` ` fence extracted           |
| `test_json_embedded_in_text`          | JSON embedded in narrative text extracted             |
| `test_invalid_raises`                 | Non-JSON raises `ValueError`                          |
| `test_valid_selection`                | Valid candidate ID accepted                           |
| `test_out_of_candidates_raises`       | ID outside candidate list raises `ValueError`         |
| `test_invalid_confidence_defaulted`   | Unknown confidence string defaults to `"medium"`      |
| `test_none_provider_returns_fallback` | `provider="none"` triggers domain expert fallback     |

**Expected LLM JSON output schema:**

```json
{
  "selected_l2_id": 42,
  "selected_path": "Informatique > Ordinateurs portables",
  "confidence": "high",
  "reason": "La désignation mentionne explicitement un laptop."
}
```

---

### 4.6 Category definition store — `category_definitions.py`

**Test class:** `TestCategoryDefinitionStore`

| Test                                  | What it verifies                                    |
|:--------------------------------------|:----------------------------------------------------|
| `test_load_and_get`                   | JSONL loaded, known ID returns correct definition   |
| `test_missing_id_fallback`            | Unknown ID returns `_is_fallback=True` dict         |
| `test_none_path_fallback_only_mode`   | `path=None` works (fallback-only mode)              |
| `test_get_many`                       | Batch retrieval with mixed known/unknown IDs        |

**JSONL definition format:**

```json
{
  "l2_id": 1234,
  "definition_fr": "Chaussures destinées à la course à pied.",
  "positive_signals": ["running", "course", "jogging"],
  "negative_signals": ["randonnée", "trail"]
}
```

---

### 4.7 Label path augmentation — `label_path_augmentation.py`

**Test class:** `TestAugmentedProductDataset`

| Test                                  | What it verifies                                         |
|:--------------------------------------|:---------------------------------------------------------|
| `test_disabled_same_length`           | Disabled augmentation preserves dataset length           |
| `test_enabled_all_expands`            | `apply_to="all"` expands the dataset                     |
| `test_item_has_required_keys`         | Each item has `input_ids`, `attention_mask`, `label_l1`, `label_l2` |
| `test_invalid_apply_to_raises`        | Invalid `apply_to` value raises `ValueError`             |
| `test_full_path_disabled_by_default`  | `allow_full_path=False` always masks at least one level  |

**Augmented text variants:**

```
[CAT] <mask> > Ordinateurs portables [TEXT] MacBook Pro 14 pouces...
[CAT] Informatique > <mask>           [TEXT] MacBook Pro 14 pouces...
```

---

### 4.8 Top-K metrics — `topk_metrics.py`

**Test class:** `TestTopKMetrics`

| Test                              | What it verifies                                     |
|:----------------------------------|:-----------------------------------------------------|
| `test_topk_recall_perfect`        | All true labels in top-1 → recall = 1.0              |
| `test_topk_recall_zero`           | No true labels in candidates → recall = 0.0          |
| `test_topk_recall_partial`        | Partial overlap → correct fraction                   |
| `test_report_has_global_and_buckets` | Report has `global` and `by_bucket` keys          |
| `test_save_report`                | JSON report saved to disk and loadable               |

---

## 5. Smoke Tests — End-to-End Pipeline

**Test class:** `TestSmoke` — uses a mock `FlatHydra` model, no GPU or
checkpoint required.

| Smoke test                            | Pipeline path covered                              |
|:--------------------------------------|:---------------------------------------------------|
| `test_smoke_baseline_inference`       | Full fast path → `FastPathResult`                  |
| `test_smoke_hierarchy_aware_topk`     | Hierarchy scoring + top-K generation               |
| `test_smoke_focal_loss_forward`       | Focal loss forward + backward pass                 |
| `test_smoke_category_definition_fallback` | `path=None` store returns fallbacks           |
| `test_smoke_llm_provider_none_fallback`   | `provider="none"` falls back to domain expert |
| `test_smoke_hard_path_mocked_llm`     | Full hard path with patched LLM JSON response      |

```bash
python -m pytest tests/test_dual_expert.py::TestSmoke -v
```

---

## 6. Manual Inference Testing

### 6.1 Fast path with mock model (no checkpoint needed)

```python
import sys, torch
from unittest import mock
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/home/younes/projects/projet-fil-rouge")
sys.path.insert(0, "/home/younes/projects/projet-fil-rouge/01_baselines")

from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier
from src.dual_expert.candidate_generator import CandidateGenerationConfig
from src.dual_expert.difficulty_router import DifficultyRouterConfig
from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
from src.dual_expert.llm_reranker import LLMRerankerConfig

NUM_L1, NUM_L2 = 157, 20

# --- Mock model (replace with real FlatHydra for production) ---
model = mock.MagicMock()
model.eval.return_value = model
model.to.return_value = model
logits = torch.zeros(1, NUM_L1 + NUM_L2)
logits[0, 0] = 5.0          # strong L1 class 0
logits[0, NUM_L1] = 5.0     # strong L2 class 0
model.return_value = (None, logits)

tokenizer = mock.MagicMock()
tokenizer.side_effect = lambda text, **kw: {
    "input_ids": torch.zeros(1, 16, dtype=torch.long),
    "attention_mask": torch.ones(1, 16, dtype=torch.long),
}

l1_enc = LabelEncoder().fit([f"L1_{i}" for i in range(NUM_L1)])
l2_enc = LabelEncoder().fit([f"L2_{i}" for i in range(NUM_L2)])

clf = DualExpertProductClassifier(
    model=model,
    tokenizer=tokenizer,
    l1_encoder=l1_enc,
    l2_encoder=l2_enc,
    l2_to_l1_mapping={i: i % NUM_L1 for i in range(NUM_L2)},
    num_classes_l1=NUM_L1,
    num_classes_l2=NUM_L2,
    max_length=16,
    device=torch.device("cpu"),
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
    generation_config=CandidateGenerationConfig(top_k=5),
    router_config=DifficultyRouterConfig(enabled=True),
    reranker_config=LLMRerankerConfig(enabled=False),
    fp16=False,
)

result = clf.predict({
    "product_id": "prod_001",
    "designation": "Chaussures de running homme",
    "description": "Légères et amorties pour la course sur route.",
})

print(f"L1: {result.pred_l1_label} (id={result.pred_l1_id})")
print(f"L2: {result.pred_l2_label} (id={result.pred_l2_id})")
print(f"L2 confidence : {result.confidence_l2:.4f}")
print(f"Hierarchy score: {result.hierarchy_score:.4f}")
print(f"Decision source: {result.decision_source}")
print(f"LLM used: {result.llm_used}")
for c in result.top_k_candidates:
    print(f"  [{c['rank']}] L2={c['l2_id']}  score={c['hierarchy_score']:.4f}  prob={c['prob_l2']:.4f}")
```

### 6.2 Load from a real checkpoint

```python
from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier
from src.dual_expert.candidate_generator import CandidateGenerationConfig
from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
from src.dual_expert.difficulty_router import DifficultyRouterConfig

clf = DualExpertProductClassifier.from_checkpoint(
    checkpoint_dir="01_baselines/models/flat_hydra_best",   # adjust path
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
    generation_config=CandidateGenerationConfig(top_k=10),
    router_config=DifficultyRouterConfig(enabled=True),
)

result = clf.predict({
    "product_id": "42",
    "designation": "Vélo VTT 29 pouces aluminium",
    "description": "Fourche suspendue, 21 vitesses, freins à disque.",
})
print(result)
```

**Required files in `checkpoint_dir`:**

```
best_model.pt
tokenizer/
l1_label_encoder.pkl
l2_label_encoder.pkl
l2_to_l1_mapping.json
config.json
l2_class_counts.json    ← optional but enables rare-class routing
```

### 6.3 Force hard path — dry-run (no real LLM)

```python
from src.dual_expert.difficulty_router import (
    DifficultyRouterConfig, RoutingThresholds, RoutingRules
)
from src.dual_expert.llm_reranker import LLMRerankerConfig

# Impossibly high thresholds → every product is routed to LLM
router_cfg = DifficultyRouterConfig(
    enabled=True,
    max_llm_pass_through_rate=1.0,
    thresholds=RoutingThresholds(
        min_l2_confidence=0.99,
        min_l2_margin=0.99,
        min_parent_l1_confidence=0.99,
    ),
    rules=RoutingRules(send_if_low_l2_confidence=True),
)

# provider="none" = dry-run, always falls back to domain expert top-1
reranker_cfg = LLMRerankerConfig(
    enabled=True,
    provider="none",
    fallback_to_domain_expert=True,
)
# Pass router_cfg and reranker_cfg to DualExpertProductClassifier
```

---

## 7. Parameters and Tunable Knobs

### 7.1 `HierarchyScoringConfig`

File: `01_baselines/src/dual_expert/hierarchy_scorer.py`

| Parameter | Default | Effect                                                          |
|:----------|:-------:|:----------------------------------------------------------------|
| `enabled` | `True`  | `False` → ranking falls back to plain `log P(L2)`              |
| `alpha`   | `0.5`   | Weight of parent-L1 term. `0` = ignore L1, `1` = equal weight  |
| `eps`     | `1e-12` | Numerical stability constant in `log`                           |

> Increase `alpha` when L1 accuracy is high and hierarchical consistency matters.
> Decrease `alpha` when the L1 head is noisy and hurts L2 ranking.

---

### 7.2 `CandidateGenerationConfig`

File: `01_baselines/src/dual_expert/candidate_generator.py`

| Parameter            | Default | Effect                                              |
|:---------------------|:-------:|:----------------------------------------------------|
| `top_k`              | `10`    | Number of candidates forwarded to the router / LLM  |
| `also_compute_top_20`| `True`  | Compute top-20 for extended metrics (no extra cost)  |

> Increase `top_k` to 20 if top-10 recall on rare classes is below 85 %.
> Decrease to 5 to reduce LLM prompt size and latency.

---

### 7.3 `DifficultyRouterConfig` / `RoutingThresholds` / `RoutingRules`

File: `01_baselines/src/dual_expert/difficulty_router.py`

**`DifficultyRouterConfig`**

| Parameter                   | Default | Effect                                               |
|:----------------------------|:-------:|:-----------------------------------------------------|
| `enabled`                   | `True`  | `False` → always use fast path (benchmark mode)      |
| `max_llm_pass_through_rate` | `0.25`  | Hard cap: at most 25 % of traffic goes to LLM        |

**`RoutingThresholds`**

| Parameter                   | Default | Effect                                               |
|:----------------------------|:-------:|:-----------------------------------------------------|
| `min_l2_confidence`         | `0.75`  | Below this → route to LLM                            |
| `min_l2_margin`             | `0.15`  | `top1 − top2` below this → route                     |
| `min_parent_l1_confidence`  | `0.50`  | Parent L1 probability below this → route             |
| `rare_class_threshold`      | `100`   | Classes with fewer training samples are "rare"        |
| `rare_class_min_confidence` | `0.90`  | Rare class needs this confidence to stay on fast path |
| `min_text_chars`            | `30`    | Texts shorter than this trigger routing (off by default)|

**`RoutingRules`**

| Rule toggle                            | Default | Trigger condition                                 |
|:---------------------------------------|:-------:|:--------------------------------------------------|
| `send_if_hierarchy_inconsistent`       | `True`  | argmax(L1) ≠ parent(argmax(L2))                   |
| `send_if_low_l2_confidence`            | `True`  | Top-1 L2 probability < `min_l2_confidence`        |
| `send_if_low_l2_margin`                | `True`  | top1 − top2 probability gap < `min_l2_margin`     |
| `send_if_rare_and_not_very_confident`  | `True`  | Rare class with confidence < `rare_class_min_confidence` |
| `send_if_short_text`                   | `False` | Product text shorter than `min_text_chars`        |
| `send_if_sibling_confusion`            | `True`  | ≥ 3 top-K candidates share the same parent L1     |

---

### 7.4 `LLMRerankerConfig`

File: `01_baselines/src/dual_expert/llm_reranker.py`

| Parameter                   | Default                         | Effect                                              |
|:----------------------------|:--------------------------------|:----------------------------------------------------|
| `enabled`                   | `False`                         | Must be `True` to activate the hard path            |
| `provider`                  | `"none"`                        | `none` / `local` / `openai_compatible` / `mistral_api` |
| `model_name`                | `"mixtral-8x7b-instruct"`       | LLM model identifier sent to the API                |
| `temperature`               | `0.0`                           | Deterministic output — keep at `0.0`                |
| `max_new_tokens`            | `512`                           | Max LLM output tokens                               |
| `top_k`                     | `10`                            | Number of candidates included in the prompt         |
| `prompt_path`               | `"prompts/llm_reranker_fr.txt"` | Custom French reranking template path               |
| `require_json`              | `True`                          | Strict JSON output validation                       |
| `fallback_to_domain_expert` | `True`                          | On any error, fall back to domain top-1             |
| `timeout_seconds`           | `15.0`                          | Per-request LLM timeout                             |
| `retry_count`               | `1`                             | Retry attempts before triggering fallback           |
| `api_base_url`              | `"http://localhost:8080/v1"`    | Endpoint for local / OpenAI-compatible servers      |
| `api_key`                   | `"none"`                        | API key (`"none"` for local servers)                |

---

### 7.5 `LabelPathAugConfig`

File: `01_baselines/src/dual_expert/label_path_augmentation.py`

| Parameter                     | Default      | Effect                                              |
|:------------------------------|:------------:|:----------------------------------------------------|
| `enabled`                     | `False`      | Must be `True` to activate augmentation             |
| `apply_to`                    | `"rare_only"`| `"all"` / `"rare_only"` / `"none"`                  |
| `rare_threshold`              | `100`        | Classes with fewer samples are "rare"               |
| `augmentation_factor_rare`    | `3`          | Rare-class samples duplicated this many times       |
| `augmentation_factor_medium`  | `2`          | Medium-class copies when `apply_to="all"`           |
| `mask_probability`            | `0.3`        | Used only when `allow_full_path=True` (ablation)    |
| `prepend_probability`         | `1.0`        | Fraction of augmented copies that get the prefix    |
| `allow_full_path`             | `False`      | If `False`, always masks one hierarchy level        |

---

### 7.6 Focal loss parameters

File: `01_baselines/src/dual_expert/losses.py`

**`focal_cross_entropy`**

| Parameter       | Default  | Effect                                           |
|:----------------|:--------:|:-------------------------------------------------|
| `gamma`         | `2.0`    | Focal modulation. `0` = standard CE              |
| `class_weights` | `None`   | Per-class weights from `build_effective_number_weights` |
| `reduction`     | `"mean"` | `"mean"` / `"sum"` / `"none"`                    |

**`build_effective_number_weights`**

| Parameter    | Default  | Effect                                               |
|:-------------|:--------:|:-----------------------------------------------------|
| `beta`       | `0.9999` | Higher = more balanced. `0.9` = less aggressive      |
| `max_weight` | `10.0`   | Clips rare-class weight explosion                    |

---

## 8. LLM Provider Configuration

### 8.1 Dry-run mode (testing, no real LLM)

```python
LLMRerankerConfig(enabled=True, provider="none", fallback_to_domain_expert=True)
```

`_NoneClient` always returns `selected_l2_id=-1`, which triggers fallback
to domain expert top-1.

### 8.2 Local server (llama.cpp / vLLM / Ollama)

```python
LLMRerankerConfig(
    enabled=True,
    provider="local",
    model_name="mistral-7b-instruct",
    api_base_url="http://localhost:8080/v1",
    api_key="none",
    temperature=0.0,
    timeout_seconds=10.0,
)
```

Test connectivity:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":10}' \
  | python -m json.tool
```

### 8.3 Mistral API / OpenAI-compatible cloud

```python
LLMRerankerConfig(
    enabled=True,
    provider="mistral_api",
    model_name="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
    api_key="YOUR_API_KEY",
    temperature=0.0,
    timeout_seconds=15.0,
)
```

### 8.4 Custom reranking prompt

Create `01_baselines/prompts/llm_reranker_fr.txt` keeping the placeholders
`{designation}`, `{description}`, and `{candidate_blocks}` from the default
template. Pass the path to:

```python
LLMRerankerConfig(prompt_path="prompts/llm_reranker_fr.txt")
```

---

## 9. Evaluation and Metrics

### 9.1 Top-K recall report

```python
from src.dual_expert.topk_metrics import build_topk_recall_report, save_topk_recall_report

report = build_topk_recall_report(
    labels_l2=val_labels_l2,            # ground-truth L2 labels
    topk_preds=topk_preds,              # list[list[int]] from candidate_generator
    train_labels_l2=train_labels_l2,    # for frequency bucket assignment
    num_classes=6126,
    k_values=[1, 3, 5, 10, 20],
)
save_topk_recall_report(report, "results/topk_recall_report.json")
print(report["global"])
```

**Example output:**

```json
{
  "global": {
    "l2_top_1_accuracy":  0.61,
    "l2_top_3_recall":    0.76,
    "l2_top_5_recall":    0.83,
    "l2_top_10_recall":   0.90
  },
  "by_bucket": {
    "rare_0_10":         {"l2_top_10_recall": 0.71, "count_examples": 180},
    "rare_10_50":        {"l2_top_10_recall": 0.82, "count_examples": 540},
    "frequent_500_plus": {"l2_top_10_recall": 0.96, "count_examples": 8200}
  }
}
```

### 9.2 Routing report

```python
report = router.build_routing_report(
    routed_labels_l2=routed_gt,
    routed_preds_before=domain_preds,
    routed_preds_after=llm_preds,
    output_path="results/llm_routing_report.json",
)
print(f"LLM pass-through : {report['pass_through_rate']:.1%}")
print(f"Accuracy before LLM: {report['accuracy_before_llm']:.4f}")
print(f"Accuracy after  LLM: {report['accuracy_after_llm']:.4f}")
print(f"Accuracy delta      : {report['accuracy_delta']:+.4f}")
```

### 9.3 Audit log analysis

Hard-path predictions are automatically written to:

- `logs/hard_case_predictions.jsonl` — full structured records
- `logs/llm_audit_samples.csv` — tabular format for spreadsheet analysis

```bash
# Count LLM fallbacks
grep '"llm_fallback_used": true' logs/hard_case_predictions.jsonl | wc -l

# Routing reasons distribution
python -c "
import json
from collections import Counter
reasons = []
for line in open('logs/hard_case_predictions.jsonl'):
    reasons.extend(json.loads(line)['routing_reasons'])
print(Counter(reasons).most_common())
"
```

### 9.4 Target metrics

| Metric                      | Minimum | Ideal       |
|:----------------------------|:-------:|:-----------:|
| `L2 top-1 accuracy`         | > 60 %  | > 65 %      |
| `L2 top-10 recall`          | > 85 %  | > 90 %      |
| `L2 top-10 recall (rare)`   | > 70 %  | > 80 %      |
| LLM pass-through rate       | < 30 %  | 10 – 20 %   |
| Accuracy delta after LLM    | > 0     | +3 – +8 pp  |

---

## 10. Common Failure Modes and Fixes

### 10.1 `ImportError: No module named 'src.dual_expert'`

**Cause:** project root not in `sys.path`.

**Fix:**

```bash
export PYTHONPATH=/home/younes/projects/projet-fil-rouge:\
/home/younes/projects/projet-fil-rouge/01_baselines
```

### 10.2 `KeyError` in `l2_to_l1_mapping`

**Cause:** mapping keys not cast to `int`.

**Fix:** the constructor already applies `{int(k): int(v) for k, v in ...}`.
Verify the source JSON uses string keys (standard JSON requirement) — the
constructor handles the cast.

### 10.3 All products routed to LLM (100 % pass-through)

**Cause:** thresholds too tight for the model's current calibration.

**Fix:** relax thresholds or enforce the rate cap:

```python
DifficultyRouterConfig(
    max_llm_pass_through_rate=0.20,
    thresholds=RoutingThresholds(
        min_l2_confidence=0.60,   # relaxed from 0.75
        min_l2_margin=0.10,       # relaxed from 0.15
    ),
)
```

### 10.4 LLM selects an ID outside the candidate list

**Expected behavior:** `validate_llm_output` raises `ValueError` → caught →
fallback to domain expert top-1 (when `fallback_to_domain_expert=True`).

**Verify this path:**

```python
from src.dual_expert.llm_reranker import validate_llm_output
import pytest
with pytest.raises(ValueError, match="not in candidates"):
    validate_llm_output({"selected_l2_id": 9999}, [1, 2, 3])
```

### 10.5 NaN gradients during training with focal loss

**Cause:** extreme class imbalance + high `gamma` value.

**Fix:** keep `gamma=2.0` (default) and `max_weight=10.0` for
effective-number weights to prevent rare-class multiplier explosion.

### 10.6 `AugmentedProductDataset` length unchanged despite `enabled=True`

**Cause:** `apply_to="rare_only"` but all classes have count ≥ `rare_threshold`.

**Fix:**

```python
LabelPathAugConfig(enabled=True, apply_to="all", augmentation_factor_medium=2)
```

Or lower `rare_threshold` to match the actual distribution of your dataset.

---

## Next Steps

1. Run `python -m pytest tests/test_dual_expert.py -v` — all tests must pass.
2. Run smoke tests with a real checkpoint via `DualExpertProductClassifier.from_checkpoint(...)`.
3. Generate a top-K recall report on the validation split and confirm
   `top-10 recall > 85 %` before enabling the LLM path.
4. Start with `provider="none"` to validate routing logic and the routing
   report, then switch to a real LLM provider.
5. Run `/pre-commit` before any commit that touches dual-expert modules.
