# Dual-Expert Pipeline — Test Guide

This guide covers every layer of verification for the Dual-Expert enhancement of
Flat HYDRA: unit tests, component integration checks, end-to-end inference, training
with the new config, and the two evaluation scripts.

## Table of Contents

- [Architecture Recap](#architecture-recap)
- [Prerequisites](#prerequisites)
- [Unit Tests](#unit-tests)
- [Component Testing](#component-testing)
  - [Focal Loss and Class Weights](#focal-loss-and-class-weights)
  - [Hierarchy-Aware Scoring](#hierarchy-aware-scoring)
  - [Top-K Candidate Generator](#top-k-candidate-generator)
  - [Top-K Metrics](#top-k-metrics)
  - [Difficulty Router](#difficulty-router)
  - [LLM Reranker (Dry-Run)](#llm-reranker-dry-run)
  - [Category Definition Store](#category-definition-store)
  - [Label-Path Augmentation](#label-path-augmentation)
  - [Audit Logger](#audit-logger)
- [End-to-End Inference (No Real Data)](#end-to-end-inference-no-real-data)
- [Training with the Dual-Expert Config](#training-with-the-dual-expert-config)
- [Evaluation Scripts](#evaluation-scripts)
  - [Top-K Recall Evaluation](#top-k-recall-evaluation)
  - [LLM Reranker Evaluation](#llm-reranker-evaluation)
- [LLM Hard-Path Testing (With a Real Provider)](#llm-hard-path-testing-with-a-real-provider)
- [Checklist Before a Training Run](#checklist-before-a-training-run)

---

## Architecture Recap

```text
Product: designation + description
        │
        ▼
Flat HYDRA (CamemBERT) = Domain Expert
  ├── logits_l1  (B, 157)
  └── logits_l2  (B, 6 126)
        │
        ▼
HierarchyAwareScorer
  score(l2) = log P(l2|x) + α · log P(parent_l1(l2)|x)
        │
        ▼
TopKCandidateGenerator  →  top-K ranked L2 candidates
        │
        ▼
DifficultyRouter
  ├── Fast path  →  FastPathResult    (domain expert top-1)
  └── Hard path  →  LLMGeneralExpertReranker  →  HardPathResult
```

**Source files**:

| Module                     | Path                                                    |
|:---------------------------|:--------------------------------------------------------|
| `losses.py`                | `01_baselines/src/dual_expert/losses.py`                |
| `hierarchy_scorer.py`      | `01_baselines/src/dual_expert/hierarchy_scorer.py`      |
| `candidate_generator.py`   | `01_baselines/src/dual_expert/candidate_generator.py`   |
| `topk_metrics.py`          | `01_baselines/src/dual_expert/topk_metrics.py`          |
| `difficulty_router.py`     | `01_baselines/src/dual_expert/difficulty_router.py`     |
| `llm_reranker.py`          | `01_baselines/src/dual_expert/llm_reranker.py`          |
| `category_definitions.py`  | `01_baselines/src/dual_expert/category_definitions.py`  |
| `label_path_augmentation.py` | `01_baselines/src/dual_expert/label_path_augmentation.py` |
| `audit_logger.py`          | `01_baselines/src/dual_expert/audit_logger.py`          |
| `dual_expert_classifier.py`| `01_baselines/src/dual_expert/dual_expert_classifier.py`|
| Config                     | `01_baselines/configs/flat_hydra_dual_expert.yaml`      |

---

## Prerequisites

All commands below assume the project virtualenv is active and run from the
**project root** (`/home/younes/projects/projet-fil-rouge`).

```bash
# Activate the project venv (adjust path if different)
source .venv/bin/activate        # or: conda activate <env>

# Verify imports resolve correctly
python -c "from src.dual_expert import dual_expert_classifier; print('OK')"
# Run from 01_baselines/ if the above fails:
cd 01_baselines && python -c "from src.dual_expert import dual_expert_classifier; print('OK')"
```

Docker alternative (no local GPU needed):

```bash
docker compose run --rm --profile test test
```

---

## Unit Tests

All 164 project tests plus the dual-expert suite live in `tests/`.

```bash
# Run only the dual-expert test module
python -m pytest tests/test_dual_expert.py -v --tb=short

# Run with coverage
python -m pytest tests/test_dual_expert.py -v --tb=short \
    --cov=01_baselines/src/dual_expert --cov-report=term-missing

# Run the full suite (includes dual-expert tests)
python -m pytest tests/ -v --tb=short
```

**Expected output**: every class below should show `PASSED`.

| Test class                    | What it covers                                       |
|:------------------------------|:-----------------------------------------------------|
| `TestFocalCrossEntropy`       | Scalar output, gamma=0 ≡ CE, class weights, backward |
| `TestEffectiveNumberWeights`  | Shape, no NaN, rare > frequent weight, max clipping  |
| `TestHierarchyAwareScorer`    | Score shape, disabled mode, alpha=0 edge case        |
| `TestTopKCandidateGenerator`  | K candidates, parent mapping, descending scores      |
| `TestTopKMetrics`             | Perfect/zero/partial recall, report structure        |
| `TestDifficultyRouter`        | Fast path, low-confidence route, hierarchy mismatch  |
| `TestParseLLMJson`            | JSON fence, embedded JSON, invalid input             |
| `TestValidateLLMOutput`       | Valid selection, out-of-candidates, invalid confidence |
| `TestLLMRerankerDryRun`       | provider=none returns fallback                       |
| `TestCategoryDefinitionStore` | Load/get, missing ID fallback, none-path mode        |
| `TestAugmentedProductDataset` | Disabled/enabled length, required keys, masking      |
| `TestSmoke`                   | Full pipeline fast path, hard path with mocked LLM   |

---

## Component Testing

Each section below shows how to exercise one module in isolation from a Python
REPL or a standalone script (no dataset required).

### Focal Loss and Class Weights

```python
import torch
from src.dual_expert.losses import focal_cross_entropy, build_effective_number_weights

# --- focal_cross_entropy ---
logits  = torch.randn(4, 6126)
targets = torch.randint(0, 6126, (4,))

loss_fl = focal_cross_entropy(logits, targets, gamma=2.0)
loss_ce = torch.nn.functional.cross_entropy(logits, targets)

# gamma=0 must match standard CE (within 1e-4)
loss_fl0 = focal_cross_entropy(logits, targets, gamma=0.0)
assert abs(loss_fl0.item() - loss_ce.item()) < 1e-4

# backward must not NaN
logits.requires_grad_(True)
focal_cross_entropy(logits, targets, gamma=2.0).backward()
assert not torch.isnan(logits.grad).any()
print("focal_cross_entropy OK")

# --- build_effective_number_weights ---
# Simulate a very skewed L2 distribution
train_labels = [0] * 5 + [1] * 200 + list(range(2, 50))
weights = build_effective_number_weights(train_labels, num_classes=6126)
assert weights.shape == (6126,)
assert weights[0] > weights[1]   # rare class gets higher weight
assert weights.max() <= 10.0     # default max_weight cap
print("build_effective_number_weights OK")
```

### Hierarchy-Aware Scoring

```python
import torch
from src.dual_expert.hierarchy_scorer import (
    HierarchyAwareScorer,
    HierarchyScoringConfig,
    make_l2_to_l1_tensor,
    hierarchy_aware_scores,
)

L1, L2, B = 157, 6126, 8
l2_to_l1_dict = {i: i % L1 for i in range(L2)}
l2_to_l1_tensor = make_l2_to_l1_tensor(l2_to_l1_dict, L2)

logits_l1 = torch.randn(B, L1)
logits_l2 = torch.randn(B, L2)

scorer = HierarchyAwareScorer(l2_to_l1_dict, L2, HierarchyScoringConfig(enabled=True, alpha=0.5))
scores = scorer.score(logits_l1, logits_l2)

assert scores.shape == (B, L2)
assert not torch.isnan(scores).any()

# alpha=0 must equal log(P(l2))
probs_l1 = torch.softmax(logits_l1, dim=-1)
probs_l2 = torch.softmax(logits_l2, dim=-1)
scores0 = hierarchy_aware_scores(probs_l1, probs_l2, l2_to_l1_tensor, alpha=0.0)
expected = torch.log(probs_l2 + 1e-12)
assert torch.allclose(scores0, expected, atol=1e-5)
print("hierarchy_scorer OK")
```

### Top-K Candidate Generator

```python
import torch
from src.dual_expert.candidate_generator import CandidateGenerationConfig, TopKCandidateGenerator
from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig

L1, L2, B = 157, 6126, 4
l2_to_l1 = {i: i % L1 for i in range(L2)}

gen = TopKCandidateGenerator(
    l2_to_l1_mapping=l2_to_l1,
    num_l2=L2,
    config=CandidateGenerationConfig(top_k=10),
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
)

logits_l1 = torch.randn(B, L1)
logits_l2 = torch.randn(B, L2)
results = gen.generate(logits_l1, logits_l2)

assert len(results) == B
for cands in results:
    assert len(cands) == 10
    # Scores must be descending
    scores = [c["hierarchy_score"] for c in cands]
    assert scores == sorted(scores, reverse=True)
    # Every candidate must have its correct parent
    for c in cands:
        assert c["parent_l1_id"] == l2_to_l1[c["l2_id"]]
    # Rank field must start at 1
    assert [c["rank"] for c in cands] == list(range(1, 11))
print("candidate_generator OK")
```

### Top-K Metrics

```python
from src.dual_expert.topk_metrics import topk_recall, build_topk_recall_report, save_topk_recall_report
from pathlib import Path
import json

# Perfect recall
labels = [0, 1, 2, 3, 4]
preds  = [[0, 9], [1, 9], [2, 9], [3, 9], [4, 9]]
assert topk_recall(labels, preds, k=1) == 1.0

# Zero recall
preds_miss = [[99], [99], [99], [99], [99]]
assert topk_recall(labels, preds_miss, k=5) == 0.0

# Full report (with frequency buckets)
n = 100
train_labels = list(range(6126)) * 5
topk_preds   = [[l, (l + 1) % 6126] for l in range(n)]
true_labels  = list(range(n))

report = build_topk_recall_report(true_labels, topk_preds, train_labels, 6126, k_values=[1, 5, 10])
assert "global" in report
assert "l2_top_1_accuracy"  in report["global"]
assert "l2_top_5_recall"    in report["global"]
assert "l2_top_10_recall"   in report["global"]
assert "by_bucket" in report

# Save and reload
out = Path("/tmp/topk_recall_report.json")
save_topk_recall_report(report, out)
loaded = json.loads(out.read_text())
assert "global" in loaded
print("topk_metrics OK")
```

### Difficulty Router

```python
import torch
from src.dual_expert.difficulty_router import (
    DifficultyRouter, DifficultyRouterConfig, RoutingThresholds, RoutingRules
)

L1, L2 = 157, 6126
l2_to_l1 = {i: i % L1 for i in range(L2)}
l2_counts = {i: 200 for i in range(L2)}  # all frequent

router = DifficultyRouter(
    config=DifficultyRouterConfig(enabled=True, max_llm_pass_through_rate=1.0),
    l2_to_l1_mapping=l2_to_l1,
    l2_counts=l2_counts,
)

# --- Fast path: high L2 confidence, consistent hierarchy ---
probs_l1 = torch.zeros(L1); probs_l1[0] = 0.95
probs_l2 = torch.zeros(L2); probs_l2[0] = 0.95   # l2=0 → parent l1=0 ✓
route, reasons = router.should_route(probs_l1, probs_l2, [0, 1], "Long enough product text here")
assert not route, f"Expected fast path, got reasons: {reasons}"

# --- Hard path: uniform probs → low confidence ---
probs_l1_u = torch.ones(L1) / L1
probs_l2_u = torch.ones(L2) / L2
route2, reasons2 = router.should_route(probs_l1_u, probs_l2_u, list(range(10)), "short")
assert route2
assert "low_l2_confidence" in reasons2

# --- Pass-through rate cap ---
capped_router = DifficultyRouter(
    config=DifficultyRouterConfig(enabled=True, max_llm_pass_through_rate=0.0),
    l2_to_l1_mapping=l2_to_l1,
    l2_counts=l2_counts,
)
route3, reasons3 = capped_router.should_route(probs_l1_u, probs_l2_u, list(range(10)), "short")
assert not route3  # cap=0 → always fast path
assert "capped_by_max_pass_through_rate" in reasons3

print(f"DifficultyRouter OK — pass_through_rate: {router.pass_through_rate:.2%}")
```

### LLM Reranker (Dry-Run)

The `provider="none"` mode lets you test the full reranker code path without a
real LLM endpoint. The dry-run client returns `selected_l2_id=-1`, which triggers
`fallback_to_domain_expert=True` and selects `candidates[0]`.

```python
from src.dual_expert.llm_reranker import (
    LLMGeneralExpertReranker, LLMRerankerConfig,
    parse_llm_json, validate_llm_output,
)

# --- JSON parsing ---
resp1 = '{"selected_l2_id": 42, "confidence": "high", "reason": "fits perfectly"}'
assert parse_llm_json(resp1)["selected_l2_id"] == 42

resp2 = '```json\n{"selected_l2_id": 5}\n```'
assert parse_llm_json(resp2)["selected_l2_id"] == 5

resp3 = 'Réponse : {"selected_l2_id": 3, "confidence": "low"}'
assert parse_llm_json(resp3)["selected_l2_id"] == 3

# --- Output validation ---
out = {"selected_l2_id": 5, "confidence": "high", "reason": "ok"}
assert validate_llm_output(out, candidate_l2_ids=[5, 10])["selected_l2_id"] == 5

try:
    validate_llm_output({"selected_l2_id": 99}, candidate_l2_ids=[1, 2])
    raise AssertionError("should have raised")
except ValueError as e:
    assert "not in candidates" in str(e)

# --- Full reranker dry-run ---
cfg = LLMRerankerConfig(enabled=True, provider="none", fallback_to_domain_expert=True)
reranker = LLMGeneralExpertReranker(config=cfg)
candidates = [
    {"l2_id": 42, "hierarchy_score": -0.2, "prob_l2": 0.5},
    {"l2_id": 17, "hierarchy_score": -0.9, "prob_l2": 0.1},
]
result = reranker.rerank({"designation": "Vélo VTT", "description": "Aluminium 26\""}, candidates)
assert result.fallback_used
assert result.selected_l2_id == 42   # fallback to candidates[0]
print("llm_reranker dry-run OK")
```

### Category Definition Store

```python
import json
from pathlib import Path
from src.dual_expert.category_definitions import CategoryDefinitionStore

# --- Load from a JSONL file ---
tmp = Path("/tmp/test_cat_defs.jsonl")
records = [
    {"l2_id": 1, "definition_fr": "Vélos de montagne tout-terrain", "positive_signals": ["VTT"]},
    {"l2_id": 2, "definition_fr": "Chaussures de sport indoor",     "positive_signals": ["basket"]},
]
tmp.write_text("\n".join(json.dumps(r) for r in records))

store = CategoryDefinitionStore(str(tmp))
assert len(store) == 2
defn = store.get(1)
assert defn["definition_fr"] == "Vélos de montagne tout-terrain"

# Missing ID returns a fallback stub
fallback = store.get(9999)
assert fallback["_is_fallback"] is True
assert "definition_fr" in fallback   # always has the key

# get_many — returns list in same order, with fallbacks for missing
many = store.get_many([1, 2, 999])
assert len(many) == 3
assert many[2]["_is_fallback"] is True

# None-path mode: every lookup is a fallback
store_empty = CategoryDefinitionStore(path=None)
assert len(store_empty) == 0
assert store_empty.get(0)["_is_fallback"] is True
print("category_definitions OK")
```

### Label-Path Augmentation

```python
from unittest.mock import MagicMock
from sklearn.preprocessing import LabelEncoder
import torch
from src.dual_expert.label_path_augmentation import AugmentedProductDataset, LabelPathAugConfig

L1, L2, N = 5, 12, 20

le1 = LabelEncoder().fit([f"L1_{i}" for i in range(L1)])
le2 = LabelEncoder().fit([f"L2_{i}" for i in range(L2)])

tok = MagicMock()
tok.side_effect = lambda text, **kw: {
    "input_ids":      torch.zeros(1, 32, dtype=torch.long),
    "attention_mask": torch.ones(1, 32,  dtype=torch.long),
}

# Disabled → same length as input
ds_off = AugmentedProductDataset(
    texts=[f"produit {i}" for i in range(N)],
    labels_l1=[i % L1 for i in range(N)],
    labels_l2=[i % L2 for i in range(N)],
    tokenizer=tok,
    max_length=32,
    l1_encoder=le1,
    l2_encoder=le2,
    l2_to_l1_mapping={i: i % L1 for i in range(L2)},
    config=LabelPathAugConfig(enabled=False),
    seed=42,
)
assert len(ds_off) == N

# Enabled with apply_to="all" → expands dataset
ds_on = AugmentedProductDataset(
    texts=[f"produit {i}" for i in range(N)],
    labels_l1=[i % L1 for i in range(N)],
    labels_l2=[i % L2 for i in range(N)],
    tokenizer=tok,
    max_length=32,
    l1_encoder=le1,
    l2_encoder=le2,
    l2_to_l1_mapping={i: i % L1 for i in range(L2)},
    config=LabelPathAugConfig(enabled=True, apply_to="all", augmentation_factor_rare=2),
    seed=42,
)
assert len(ds_on) > N

# Each item has the required keys
item = ds_on[0]
assert "input_ids"      in item
assert "attention_mask" in item
assert "label_l1"       in item
assert "label_l2"       in item
print("label_path_augmentation OK")
```

### Audit Logger

```python
import json, time
from pathlib import Path
from src.dual_expert.audit_logger import AuditLogger

out_dir = Path("/tmp/audit_test")
logger = AuditLogger(output_dir=str(out_dir))

logger.log(
    product_id="P001",
    designation="Casque vélo",
    description="Léger et ventilé",
    domain_pred_l1=12,
    domain_pred_l2=400,
    domain_conf_l2=0.62,
    top_k_candidates=[{"l2_id": 400, "rank": 1}, {"l2_id": 401, "rank": 2}],
    routing_reasons=["low_l2_confidence"],
    llm_selected_l2=401,
    llm_confidence="high",
    llm_reason="Plus spécifique",
    llm_latency_ms=120.5,
    true_l2=401,
)
logger.flush()

jsonl_path = out_dir / "hard_case_predictions.jsonl"
assert jsonl_path.exists()
entry = json.loads(jsonl_path.read_text().strip().split("\n")[0])
assert entry["product_id"] == "P001"
assert entry["llm_selected_l2"] == 401
print("audit_logger OK")
```

---

## End-to-End Inference (No Real Data)

This smoke test builds a full `DualExpertProductClassifier` from scratch
with a mock model and exercises both the fast path and the mocked LLM hard path.

```python
import json
import torch
from unittest.mock import MagicMock
from sklearn.preprocessing import LabelEncoder

from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier, FastPathResult, HardPathResult
from src.dual_expert.candidate_generator import CandidateGenerationConfig
from src.dual_expert.hierarchy_scorer import HierarchyScoringConfig
from src.dual_expert.difficulty_router import DifficultyRouterConfig, RoutingThresholds, RoutingRules
from src.dual_expert.llm_reranker import LLMRerankerConfig

L1, L2 = 157, 6126
l2_to_l1 = {i: i % L1 for i in range(L2)}

# Mock model: strong signal on class 0
model = MagicMock()
model.eval.return_value = model
model.to.return_value = model
logits = torch.zeros(1, L1 + L2)
logits[0, 0]    = 5.0   # L1 class 0
logits[0, L1]   = 5.0   # L2 class 0 → parent L1 class 0
model.return_value = (None, logits)

tok = MagicMock()
tok.side_effect = lambda text, **kw: {
    "input_ids":      torch.zeros(1, 256, dtype=torch.long),
    "attention_mask": torch.ones(1, 256,  dtype=torch.long),
}

le1 = LabelEncoder().fit([f"L1_{i}" for i in range(L1)])
le2 = LabelEncoder().fit([f"L2_{i}" for i in range(L2)])

clf = DualExpertProductClassifier(
    model=model,
    tokenizer=tok,
    l1_encoder=le1,
    l2_encoder=le2,
    l2_to_l1_mapping=l2_to_l1,
    num_classes_l1=L1,
    num_classes_l2=L2,
    max_length=256,
    device=torch.device("cpu"),
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
    generation_config=CandidateGenerationConfig(top_k=10),
    router_config=DifficultyRouterConfig(enabled=True),
    reranker_config=LLMRerankerConfig(enabled=False),
    fp16=False,
)

product = {"product_id": "P001", "designation": "Vélo VTT", "description": "26 pouces aluminium"}

# Fast path (LLM disabled)
result = clf.predict(product)
assert isinstance(result, FastPathResult)
assert result.pred_l1_id == l2_to_l1[result.pred_l2_id]
assert result.hierarchy_score != 0.0
print(f"Fast path → L1={result.pred_l1_label}, L2={result.pred_l2_label}, "
      f"score={result.hierarchy_score:.4f}")

# Hard path: force routing + mock LLM response
force_router_cfg = DifficultyRouterConfig(
    enabled=True,
    max_llm_pass_through_rate=1.0,
    thresholds=RoutingThresholds(min_l2_confidence=0.99, min_l2_margin=0.99),
    rules=RoutingRules(send_if_low_l2_confidence=True),
)
mock_llm_response = json.dumps({
    "selected_l2_id": L1,     # first L2 candidate
    "selected_path": "L1_0 > L2_0",
    "confidence": "high",
    "reason": "best match",
})
clf_hard = DualExpertProductClassifier(
    model=model, tokenizer=tok,
    l1_encoder=le1, l2_encoder=le2,
    l2_to_l1_mapping=l2_to_l1,
    num_classes_l1=L1, num_classes_l2=L2,
    max_length=256, device=torch.device("cpu"),
    scoring_config=HierarchyScoringConfig(enabled=True),
    generation_config=CandidateGenerationConfig(top_k=10),
    router_config=force_router_cfg,
    reranker_config=LLMRerankerConfig(enabled=True, provider="none", fallback_to_domain_expert=True),
    fp16=False,
)
from unittest.mock import patch
with patch.object(clf_hard.llm_reranker._client, "generate", return_value=mock_llm_response):
    result_hard = clf_hard.predict(product)

assert result_hard.pred_l1_id == l2_to_l1.get(result_hard.pred_l2_id, result_hard.pred_l1_id)
print(f"Hard path → L1={result_hard.pred_l1_id}, L2={result_hard.pred_l2_id}, "
      f"llm_used={result_hard.llm_used}")
```

---

## Training with the Dual-Expert Config

The dual-expert training adds **focal CE** for L2 and (optionally) **label-path
augmentation** on top of the baseline `flat_hydra.yaml`. The canonical config is
`01_baselines/configs/flat_hydra_dual_expert.yaml`.

### Differences from the Baseline Config

| Parameter               | Baseline         | Dual Expert                    |
|:------------------------|:-----------------|:-------------------------------|
| `l2_loss_type`          | — (CE)           | `focal_ce`                     |
| `focal_gamma_l2`        | —                | `2.0`                          |
| `use_l2_class_weights`  | —                | `true`                         |
| `label_path_aug_enabled`| —                | `false` (toggle for exp. E)    |
| `experiment_name`       | `flat_hydra`     | `flat_hydra_dual_expert`       |
| `save_dir`              | `models/flat_hydra` | `models/flat_hydra_dual_expert` |

### Launch Training Locally

```bash
# From project root
python scripts/run_flat_hydra.py \
    --config 01_baselines/configs/flat_hydra_dual_expert.yaml \
    --split-dir /path/to/dataset
```

### Launch on SLURM

```bash
# Check GPU availability first
/gpu-status

# Submit (adjust variant/nodes/gpu_type as needed)
sbatch scripts/run_flat_camembert_v2.slurm  # adapt or copy to dual_expert variant
```

### Verify the Dual-Expert Loss is Active

Check the training log for these lines at startup:

```
[FlatHydra] L2 loss: focal_ce (gamma=2.0)
[DualExpert] Building effective-number class weights for 6126 L2 classes…
[DualExpert] Class weights saved → models/flat_hydra_dual_expert/
  rare_0_10:          XXX classes
  rare_10_50:         XXX classes
  frequent_500_plus:  XXX classes
```

If `[DualExpert]` lines are missing, the `l2_loss_type` key is not being read —
check that `FlatHydraConfig` maps `l2_loss_type` from the YAML correctly
(`01_baselines/src/flat_trainer.py`).

---

## Evaluation Scripts

Both scripts require a trained `FlatHydra` checkpoint directory containing:

```
best_model.pt
tokenizer/
l1_label_encoder.pkl
l2_label_encoder.pkl
l2_to_l1_mapping.json
config.json
l2_class_counts.json      ← written by dual-expert training
```

### Top-K Recall Evaluation

Measures how often the true L2 class appears in the top-K candidates globally
and per frequency bucket.

```bash
python scripts/evaluate_topk.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset \
    --split val \
    --top-k 20 \
    --hierarchy-alpha 0.5 \
    --report-dir models/flat_hydra_dual_expert
```

**Key output** (`topk_recall_report.json`):

```json
{
  "global": {
    "l2_top_1_accuracy": 0.84,
    "l2_top_5_recall":   0.93,
    "l2_top_10_recall":  0.96,
    "l2_top_20_recall":  0.97
  },
  "by_bucket": {
    "rare_0_10":         { "l2_top_10_recall": 0.71, "count_examples": 412 },
    "frequent_500_plus": { "l2_top_10_recall": 0.98, "count_examples": 12034 }
  }
}
```

**What to check**:

- `l2_top_10_recall` globally should be > 0.90 for the LLM path to be useful.
- Recall for `rare_0_10` and `rare_10_50` validates that focal loss + hierarchy
  scoring improve candidate generation for rare classes.

Ablation: disable hierarchy scoring to measure its contribution:

```bash
python scripts/evaluate_topk.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset \
    --no-hierarchy-scoring \
    --report-dir models/flat_hydra_dual_expert/no_hier
```

### LLM Reranker Evaluation

Runs the full `DualExpertProductClassifier` on the test split, collects hard
cases that the router would send to the LLM, and reports before/after accuracy.

**Dry-run (no LLM endpoint needed)**:

```bash
python scripts/evaluate_llm_reranker.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset \
    --output results/llm_reranker_eval/ \
    --provider none
```

**With a local LLM server** (e.g., vLLM / llama.cpp):

```bash
python scripts/evaluate_llm_reranker.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset \
    --output results/llm_reranker_eval/ \
    --provider local \
    --api-base-url http://localhost:8080/v1 \
    --model mixtral-8x7b-instruct \
    --category-definitions data/category_definitions/category_definitions_v1.jsonl
```

**Key output** (`llm_routing_report.json`):

```json
{
  "total_samples": 50000,
  "routed_to_llm": 8700,
  "pass_through_rate": 0.174,
  "routing_reason_counts": {
    "low_l2_confidence":   4100,
    "hierarchy_inconsistent": 2900,
    "rare_l2_not_confident":  1700
  },
  "accuracy_before_llm": 0.51,
  "accuracy_after_llm":  0.67,
  "accuracy_delta": 0.16
}
```

**Sanity checks on this report**:

- `pass_through_rate` should be ≤ `max_llm_pass_through_rate` (0.25 by default).
- `accuracy_delta` > 0 confirms the LLM adds value on hard cases.
- If `pass_through_rate` is 0, the router thresholds may be too permissive —
  lower `min_l2_confidence` or `min_l2_margin`.

---

## LLM Hard-Path Testing (With a Real Provider)

To test the full hard path with an actual LLM before running at scale, use a
**single-product dry fire**:

```python
from src.dual_expert.llm_reranker import LLMGeneralExpertReranker, LLMRerankerConfig
from src.dual_expert.category_definitions import CategoryDefinitionStore

cfg = LLMRerankerConfig(
    enabled=True,
    provider="local",               # or "mistral_api" / "openai_compatible"
    model_name="mixtral-8x7b-instruct",
    api_base_url="http://localhost:8080/v1",
    api_key="none",
    temperature=0.0,
    max_new_tokens=256,
    fallback_to_domain_expert=True,
    timeout_seconds=15.0,
)

# Optionally load definitions for richer prompts
store = CategoryDefinitionStore("data/category_definitions/category_definitions_v1.jsonl")

reranker = LLMGeneralExpertReranker(config=cfg, category_definition_store=store)

product = {"designation": "Casque vélo route aéro", "description": "Coque ABS ventilation optimisée"}
candidates = [
    {"l2_id": 400, "l2_label": "Casques vélo",         "parent_l1_id": 12, "parent_l1_label": "Sport",  "hierarchy_score": -0.21, "rank": 1},
    {"l2_id": 401, "l2_label": "Accessoires cyclisme",  "parent_l1_id": 12, "parent_l1_label": "Sport",  "hierarchy_score": -0.45, "rank": 2},
    {"l2_id": 55,  "l2_label": "Protections sport",     "parent_l1_id": 12, "parent_l1_label": "Sport",  "hierarchy_score": -0.67, "rank": 3},
]

result = reranker.rerank(product, candidates)
print(f"selected_l2_id : {result.selected_l2_id}")
print(f"confidence     : {result.confidence}")
print(f"reason         : {result.reason}")
print(f"fallback_used  : {result.fallback_used}")
print(f"latency_ms     : {result.latency_ms:.0f} ms")
```

**Assertions to check**:

- `selected_l2_id` must be one of `[400, 401, 55]` (no hallucinated ID).
- `fallback_used` should be `False` when the LLM returns a valid JSON.
- `latency_ms` should be < 15 000 ms (timeout threshold).

---

## Checklist Before a Training Run

Run through this checklist before submitting a dual-expert training job.

```
[ ] GPU free                 /gpu-status
[ ] Tests pass               python -m pytest tests/test_dual_expert.py -v
[ ] Config reviewed          cat 01_baselines/configs/flat_hydra_dual_expert.yaml
                             → l2_loss_type: focal_ce
                             → use_l2_class_weights: true
[ ] MLflow baseline known    search_best_runs("f1_l2_macro")
[ ] Dataset splits present   ls /path/to/dataset/*.jsonl
[ ] No duplicate run         list_runs("flat_hydra_dual_expert")
[ ] save_dir clean           ls models/flat_hydra_dual_expert/  (should not exist or be empty)
```

After training completes:

```bash
# 1. Top-K recall
python scripts/evaluate_topk.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset --split val

# 2. Router dry-run
python scripts/evaluate_llm_reranker.py \
    --checkpoint models/flat_hydra_dual_expert \
    --split-dir /path/to/dataset --provider none \
    --output results/llm_reranker_eval/

# 3. Compare to baseline in MLflow
compare_runs([NEW_RUN_ID, BEST_BASELINE_RUN_ID], metrics=["f1_l2_macro", "f1_l1_macro", "hcr"])
```

Expected improvements over the `flat_hydra` baseline (targets, not guarantees):

| Metric           | Baseline | Dual Expert Target |
|:-----------------|:--------:|:------------------:|
| `f1_l2_macro`    | —        | +1–3 pp            |
| `f1_l1_macro`    | —        | neutral or +0.5 pp |
| `hcr`            | —        | > 95%              |
| Top-10 recall    | —        | > 93%              |
| Rare bucket top-10 | —      | > 75%              |
