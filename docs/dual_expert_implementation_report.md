# Flat HYDRA Dual-Expert — Technical Implementation Report

Architecture enhancement of the CamemBERT Flat HYDRA classifier into a
**Dual-Expert** system for hierarchical e-commerce product classification
(Rakuten France, 157 L1 × 6 126 L2 categories).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Changes](#2-repository-changes)
3. [Architecture](#3-architecture)
4. [Component Reference](#4-component-reference)
   - [4.1 Enhanced Loss Functions](#41-enhanced-loss-functions)
   - [4.2 Hierarchy-Aware Scorer](#42-hierarchy-aware-scorer)
   - [4.3 Top-K Candidate Generator](#43-top-k-candidate-generator)
   - [4.4 Top-K Recall Metrics](#44-top-k-recall-metrics)
   - [4.5 Label-Path Augmentation](#45-label-path-augmentation)
   - [4.6 Category Definition Store](#46-category-definition-store)
   - [4.7 LLM General Expert Reranker](#47-llm-general-expert-reranker)
   - [4.8 Difficulty Router](#48-difficulty-router)
   - [4.9 Audit Logger](#49-audit-logger)
   - [4.10 Dual-Expert Classifier](#410-dual-expert-classifier)
5. [Configuration Guide](#5-configuration-guide)
6. [How to Run Each Experiment](#6-how-to-run-each-experiment)
   - [6.1 Experiment A — Baseline (unchanged)](#61-experiment-a--baseline-unchanged)
   - [6.2 Experiment B — Top-K recall measurement](#62-experiment-b--top-k-recall-measurement)
   - [6.3 Experiment C — Hierarchy-aware scoring](#63-experiment-c--hierarchy-aware-scoring)
   - [6.4 Experiment D — Focal loss + class weights](#64-experiment-d--focal-loss--class-weights)
   - [6.5 Experiment E — Label-path augmentation](#65-experiment-e--label-path-augmentation)
   - [6.6 Experiment F/G — LLM reranking](#66-experiment-fg--llm-reranking)
7. [How to Build Class Weights](#7-how-to-build-class-weights)
8. [How to Generate Category Definitions](#8-how-to-generate-category-definitions)
9. [How to Disable All New Features](#9-how-to-disable-all-new-features)
10. [Expected Artifacts](#10-expected-artifacts)
11. [Known Limitations](#11-known-limitations)
12. [Suggested Next Tests](#12-suggested-next-tests)

---

## 1. Overview

The existing **Flat HYDRA** model (CamemBERT encoder + single unified
classification head outputting `157 + 6 126 = 6 283` logits simultaneously)
becomes the **Domain Expert** in a two-stage pipeline.

The enhancement adds six capabilities on top of the unchanged base model:

| # | Capability | Training impact | Inference impact |
|:-:|:-----------|:---------------:|:----------------:|
| 1 | Focal CE + effective-number class weights for L2 | Yes | No |
| 2 | Label-path augmentation for rare L2 classes | Yes | No |
| 3 | Hierarchy-aware L2 scoring | No | Yes |
| 4 | Top-K candidate generation | No | Yes |
| 5 | LLM General Expert reranker (hard cases only) | No | Optional |
| 6 | Difficulty router (fast vs. hard path) | No | Yes |

All new features are **controlled by config flags** and default to the
original baseline behaviour when disabled.

---

## 2. Repository Changes

### Files modified

| File | Change |
|:-----|:-------|
| `01_baselines/src/flat_trainer.py` | Added 10 new fields to `FlatHydraConfig`; added `_load_or_build_class_weights()`, `_compute_dual_expert_loss()`, augmented `_make_loader()`; wired focal-CE path and label-path augmentation into the training loop |

### Files created

**Dual-Expert module** (`01_baselines/src/dual_expert/`):

| File | Purpose |
|:-----|:--------|
| `__init__.py` | Package declaration |
| `losses.py` | `focal_cross_entropy`, `build_effective_number_weights`, `build_frequency_buckets`, `save_class_weight_artifacts` |
| `hierarchy_scorer.py` | `HierarchyAwareScorer`, `hierarchy_aware_scores`, `make_l2_to_l1_tensor` |
| `candidate_generator.py` | `TopKCandidateGenerator`, `get_topk_l2_candidates` |
| `topk_metrics.py` | `topk_recall`, `compute_topk_recall_dict`, `compute_topk_metrics_by_bucket`, `build_topk_recall_report`, `save_topk_recall_report` |
| `label_path_augmentation.py` | `AugmentedProductDataset`, `LabelPathAugConfig` |
| `category_definitions.py` | `CategoryDefinitionStore` |
| `llm_reranker.py` | `LLMGeneralExpertReranker`, `PromptBuilder`, `RerankResult`, `parse_llm_json`, `validate_llm_output` |
| `difficulty_router.py` | `DifficultyRouter`, `should_route_to_llm`, routing dataclasses |
| `audit_logger.py` | `AuditLogger`, `AuditEntry` |
| `dual_expert_classifier.py` | `DualExpertProductClassifier`, `FastPathResult`, `HardPathResult` |

**Config:**

| File | Purpose |
|:-----|:--------|
| `01_baselines/configs/flat_hydra_dual_expert.yaml` | Full dual-expert configuration with all parameters |

**Scripts:**

| File | Purpose |
|:-----|:--------|
| `scripts/build_l2_class_weights.py` | Compute and save L2 effective-number class weights |
| `scripts/evaluate_topk.py` | Evaluate top-K recall on val/test with optional hierarchy scoring |
| `scripts/generate_category_definitions.py` | Generate offline L2 definitions (dry-run or via LLM) |

**Prompts:**

| File | Purpose |
|:-----|:--------|
| `prompts/llm_reranker_fr.txt` | French-language reranking prompt template |
| `prompts/generate_category_definition_fr.txt` | French-language definition generation prompt |

**Tests:**

| File | Purpose |
|:-----|:--------|
| `tests/test_dual_expert.py` | Unit tests for all 10 new modules (55 test cases) |

---

## 3. Architecture

```
Product: designation + description
         │
         ▼
 ┌───────────────────────────────────────────────────┐
 │  FlatHydra CamemBERT (Domain Expert)              │
 │  tokenizer → encoder → projector → unified head   │
 │  output: logits_l1 (157) + logits_l2 (6 126)      │
 └───────────────────────────────────────────────────┘
         │                     │
   softmax(l1)           softmax(l2)
         │                     │
         └──────────┬──────────┘
                    ▼
         HierarchyAwareScorer
         score(l2) = log P(l2) + α·log P(parent_l1(l2))
                    │
                    ▼
         TopKCandidateGenerator
         → top-K l2 ids, scores, probs
                    │
                    ▼
         DifficultyRouter
         ┌───────────────────────────────┐
         │ Easy? (high conf, consistent) │──→ FastPathResult
         └───────────────────────────────┘
         │ Hard? (low conf, rare, etc.)  │
         └───────────────────────────────┘
                    │
             fetch definitions
             (CategoryDefinitionStore)
                    │
                    ▼
         LLMGeneralExpertReranker
         selects 1 of top-K candidates
                    │
                    ▼
              HardPathResult
                    │
                    ▼
              AuditLogger
         (hard_case_predictions.jsonl)
```

---

## 4. Component Reference

### 4.1 Enhanced Loss Functions

**Location:** `01_baselines/src/dual_expert/losses.py`

#### `focal_cross_entropy`

Focal loss for multi-class single-label classification, with optional
per-class weights.

```python
from dual_expert.losses import focal_cross_entropy

loss = focal_cross_entropy(
    logits,           # (B, C) raw logits
    targets,          # (B,)  integer class indices
    class_weights=wt, # (C,)  optional, auto-moved to device
    gamma=2.0,
    reduction="mean",
)
```

Formula applied per sample:

```
ce   = CrossEntropy(logits, target, weight=class_weights)  # per-sample
pt   = exp(-ce)
loss = (1 - pt)^gamma * ce
```

When `gamma=0` the result is identical to standard weighted
`CrossEntropyLoss`.

#### `build_effective_number_weights`

Class-balanced weights using the Effective Number formula
(Cui et al., CVPR 2019).

```python
from dual_expert.losses import build_effective_number_weights

weights = build_effective_number_weights(
    labels_l2,     # list of training L2 integer labels
    num_classes,   # total number of L2 classes
    beta=0.9999,
    max_weight=10.0,
)
# weights: FloatTensor (num_classes,), normalized by mean, clipped at max_weight
```

Formula:

```
effective_num_c = 1 - beta^n_c
weight_c        = (1 - beta) / effective_num_c
weight          = weight / mean(weight)   # normalize
weight          = clip(weight, max=max_weight)
```

#### `build_frequency_buckets`

Assigns each L2 class to one of five frequency buckets based on training
count. Used by top-K metrics and routing.

```
rare_0_10          n < 10
rare_10_50    10 ≤ n < 50
rare_50_100   50 ≤ n < 100
medium_100_500   100 ≤ n < 500
frequent_500_plus   n ≥ 500
```

---

### 4.2 Hierarchy-Aware Scorer

**Location:** `01_baselines/src/dual_expert/hierarchy_scorer.py`

Combines the L2 probability and its parent L1 probability to produce a
score that naturally enforces hierarchical coherence.

```
score(l2) = log(P(l2 | text) + ε) + α · log(P(parent_l1(l2) | text) + ε)
```

- `α = 0.0` — pure L2 ranking (baseline behaviour)
- `α = 0.5` — recommended default
- `α = 1.0` — equal weight to L2 and parent L1

**Usage:**

```python
from dual_expert.hierarchy_scorer import HierarchyAwareScorer, HierarchyScoringConfig

scorer = HierarchyAwareScorer(
    l2_to_l1_mapping=data["l2_to_l1_mapping"],
    num_l2=data["num_classes_l2"],
    config=HierarchyScoringConfig(enabled=True, alpha=0.5, eps=1e-12),
)
scores = scorer.score(logits_l1, logits_l2)  # (B, num_l2)
```

The mapping conversion `make_l2_to_l1_tensor(mapping, num_l2)` converts
the `{l2_id: l1_id}` dict to a `LongTensor(num_l2,)` for vectorized
batch indexing — no Python loops during inference.

---

### 4.3 Top-K Candidate Generator

**Location:** `01_baselines/src/dual_expert/candidate_generator.py`

Returns the K highest-scoring L2 candidates per sample, enriched with
parent L1 information.

```python
from dual_expert.candidate_generator import TopKCandidateGenerator, CandidateGenerationConfig
from dual_expert.hierarchy_scorer import HierarchyScoringConfig

gen = TopKCandidateGenerator(
    l2_to_l1_mapping=data["l2_to_l1_mapping"],
    num_l2=data["num_classes_l2"],
    config=CandidateGenerationConfig(top_k=10),
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
    l1_encoder=data["l1_encoder"],   # optional: adds human-readable labels
    l2_encoder=data["l2_encoder"],
)

# logits_l1: (B, 157), logits_l2: (B, 6126)
candidates_batch = gen.generate(logits_l1, logits_l2)

# candidates_batch[0] — for first sample in batch:
# [
#   {"rank": 1, "l2_id": 583, "l2_label": "Linge de lit",
#    "parent_l1_id": 31, "parent_l1_label": "Maison",
#    "score": -0.42, "prob_l2": 0.61, "prob_parent_l1": 0.87},
#   ...
# ]
```

The functional API `get_topk_l2_candidates(...)` is also exported for
direct use in evaluation scripts without instantiating the class.

---

### 4.4 Top-K Recall Metrics

**Location:** `01_baselines/src/dual_expert/topk_metrics.py`

#### Global recall

```python
from dual_expert.topk_metrics import compute_topk_recall_dict

metrics = compute_topk_recall_dict(
    labels_l2,     # list[int] — ground truth
    topk_preds,    # list[list[int]] — top-K ids per sample
    k_values=[1, 3, 5, 10, 20],
)
# {"l2_top_1_accuracy": 0.71, "l2_top_3_recall": 0.84, ...}
```

#### Bucketed recall

```python
from dual_expert.topk_metrics import compute_topk_metrics_by_bucket

bucket_report = compute_topk_metrics_by_bucket(
    labels_l2, topk_preds,
    train_labels_l2=data["y_l2_train"],
    num_classes=data["num_classes_l2"],
    k_values=[1, 3, 5, 10, 20],
)
# {"rare_0_10": {"l2_top_10_recall": 0.73, "count_examples": 142, ...}, ...}
```

#### Full report

```python
from dual_expert.topk_metrics import build_topk_recall_report, save_topk_recall_report

report = build_topk_recall_report(labels_l2, topk_preds, train_labels_l2, num_classes)
save_topk_recall_report(report, "models/flat_hydra/topk_recall_report.json")
```

---

### 4.5 Label-Path Augmentation

**Location:** `01_baselines/src/dual_expert/label_path_augmentation.py`

Wraps the training set as a PyTorch `Dataset` that generates augmented
copies of rare-class samples by prepending the category path to the text.

```
Original:     Housse de couette coton 220x240. ...
Augmented A:  [CAT] Maison > Linge de lit [TEXT] Housse de couette coton 220x240. ...
Augmented B:  [CAT] <mask> > Linge de lit [TEXT] ...
Augmented C:  [CAT] Maison > <mask> [TEXT] ...
```

`<mask>` is CamemBERT's native mask token. Variants B and C are triggered
randomly with probability `mask_probability`.

**Config flags in `FlatHydraConfig`:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `label_path_aug_enabled` | `false` | Master toggle |
| `label_path_aug_apply_to` | `rare_only` | `all`, `rare_only`, or `none` |
| `label_path_aug_rare_threshold` | `100` | Max train count to be "rare" |
| `label_path_aug_factor_rare` | `3` | Copies added for rare classes |
| `label_path_aug_factor_medium` | `2` | Copies added for medium classes |
| `label_path_aug_mask_prob` | `0.3` | Prob. of replacing L1 or L2 name |

Augmentation is applied **only to the training loader** — validation and
test sets are never augmented.

---

### 4.6 Category Definition Store

**Location:** `01_baselines/src/dual_expert/category_definitions.py`

Loads offline-generated L2 definitions from a JSONL file and serves them
by `l2_id`. Falls back to a minimal placeholder when a definition is
missing.

```python
from dual_expert.category_definitions import CategoryDefinitionStore

store = CategoryDefinitionStore(
    path="data/category_definitions/category_definitions_v1.jsonl",
)

defn = store.get(l2_id=1842)
# {
#   "l2_id": 1842,
#   "l1_name": "Maison",
#   "l2_name": "Linge de lit",
#   "definition_fr": "Produits textiles...",
#   "positive_signals": ["housse de couette", "drap"],
#   "negative_signals": ["plaid", "canapé"],
#   "confidence": "high"
# }

defns = store.get_many([1842, 900, 3021])  # list of dicts
```

If `path=None` or the file does not exist, the store operates in
fallback-only mode — all definitions return the low-confidence placeholder.
This allows the LLM reranker to run in path-only mode (no definitions).

**Expected JSONL row format:**

```json
{
  "l2_id": 1842,
  "l1_id": 31,
  "l1_name": "Maison",
  "l2_name": "Linge de lit",
  "path": "Maison > Linge de lit",
  "definition_fr": "Produits textiles destinés à équiper un lit...",
  "included_products": ["housse de couette", "drap"],
  "excluded_products": ["plaid", "coussin décoratif"],
  "positive_signals": ["housse", "drap", "220x240", "parure"],
  "negative_signals": ["plaid", "canapé"],
  "ambiguities": [],
  "confidence": "high",
  "definition_version": "v1",
  "created_at": "2026-06-19"
}
```

---

### 4.7 LLM General Expert Reranker

**Location:** `01_baselines/src/dual_expert/llm_reranker.py`

Selects exactly one L2 from the top-K candidate list using an LLM.
Never invents a category outside the candidate list.

**Supported providers:**

| Provider | Description |
|:---------|:------------|
| `none` | Dry-run — returns fallback immediately, no HTTP call |
| `local` | OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama) |
| `openai_compatible` | Same as `local` |
| `mistral_api` | Mistral AI cloud endpoint |

**Usage:**

```python
from dual_expert.llm_reranker import LLMGeneralExpertReranker, LLMRerankerConfig

cfg = LLMRerankerConfig(
    enabled=True,
    provider="local",
    model_name="mixtral-8x7b-instruct",
    api_base_url="http://localhost:8080/v1",
    temperature=0.0,
    max_new_tokens=256,
    fallback_to_domain_expert=True,
)
reranker = LLMGeneralExpertReranker(config=cfg, category_definition_store=store)

result = reranker.rerank(
    product={"designation": "Housse de couette coton 220x240", "description": "..."},
    candidates=candidates_batch[0],  # from TopKCandidateGenerator
)
# result.selected_l2_id = 1842
# result.confidence     = "high"
# result.reason         = "Le titre mentionne housse de couette 220x240..."
# result.fallback_used  = False
```

**JSON parsing robustness** (`parse_llm_json`):

1. Direct `json.loads` attempt
2. Strip markdown fences (` ```json ... ``` `)
3. Regex extraction of `{...}` block
4. Raises `ValueError` if nothing is found

**Output validation** (`validate_llm_output`):

- Rejects `selected_l2_id` not in the candidate list
- Defaults `confidence` to `"medium"` if not one of `high|medium|low`
- On any error: falls back to Domain Expert top-1 (if configured)

**Prompt templates** are stored in `prompts/llm_reranker_fr.txt`.
The `PromptBuilder` class formats candidate blocks with path, definition,
positive/negative signals, and domain score.

---

### 4.8 Difficulty Router

**Location:** `01_baselines/src/dual_expert/difficulty_router.py`

Routes each product to the fast path (Domain Expert only) or the hard
path (LLM reranker) based on a set of configurable rules.

**Routing triggers** (any one is sufficient to send to LLM):

| Trigger | Config flag |
|:--------|:------------|
| `parent(argmax L2) ≠ argmax L1` | `send_if_hierarchy_inconsistent` |
| `P(top1 L2) < threshold` | `send_if_low_l2_confidence` |
| `P(top1) - P(top2) < threshold` | `send_if_low_l2_margin` |
| Rare L2 AND confidence below rare threshold | `send_if_rare_and_not_very_confident` |
| Text shorter than `min_text_chars` characters | `send_if_short_text` |
| ≥ 3 top-K candidates share the same parent L1 | `send_if_sibling_confusion` |

**Usage:**

```python
from dual_expert.difficulty_router import DifficultyRouter, DifficultyRouterConfig

router = DifficultyRouter(
    config=DifficultyRouterConfig(),
    l2_to_l1_mapping=data["l2_to_l1_mapping"],
    l2_counts={l2_id: count, ...},  # from l2_class_counts.json
)

route, reasons = router.should_route(
    probs_l1,    # (num_l1,) softmax probs
    probs_l2,    # (num_l2,) softmax probs
    top_l2_ids,  # list[int] from TopKCandidateGenerator
    text,        # product text
)
# route   = True  → send to LLM
# reasons = ["low_l2_margin", "sibling_confusion"]
```

The router tracks total calls and LLM-routed calls internally so
`router.pass_through_rate` is always available without re-running.

---

### 4.9 Audit Logger

**Location:** `01_baselines/src/dual_expert/audit_logger.py`

Logs every hard-path prediction in append mode. Creates two output files
simultaneously:

- `hard_case_predictions.jsonl` — full record per line (JSON)
- `llm_audit_samples.csv` — tabular format for spreadsheet review

**Usage:**

```python
from dual_expert.audit_logger import AuditLogger

logger = AuditLogger(
    jsonl_path="logs/hard_case_predictions.jsonl",
    csv_path="logs/llm_audit_samples.csv",
)

logger.log_hard_path(
    product=product,
    domain_pred_l1=31,
    domain_pred_l2=900,
    domain_conf_l2=0.73,
    top_k_candidates=candidates,
    routing_reasons=reasons,
    llm_result=result,       # RerankResult
    ground_truth_l1=31,
    ground_truth_l2=1842,
)
```

For privacy: descriptions are stored as a 16-character SHA-256 hex digest
(`description_hash`), not as plaintext.

---

### 4.10 Dual-Expert Classifier

**Location:** `01_baselines/src/dual_expert/dual_expert_classifier.py`

End-to-end inference pipeline. Wraps the `FlatHydra` model with all
enhancement components.

**Loading from checkpoint:**

```python
from dual_expert.dual_expert_classifier import DualExpertProductClassifier
from dual_expert.hierarchy_scorer import HierarchyScoringConfig
from dual_expert.candidate_generator import CandidateGenerationConfig
from dual_expert.llm_reranker import LLMRerankerConfig

classifier = DualExpertProductClassifier.from_checkpoint(
    checkpoint_dir="models/flat_hydra_dual_expert",
    scoring_config=HierarchyScoringConfig(enabled=True, alpha=0.5),
    generation_config=CandidateGenerationConfig(top_k=10),
    reranker_config=LLMRerankerConfig(enabled=False),  # fast-path only
)
```

**Single-product prediction:**

```python
result = classifier.predict({
    "product_id": "abc123",
    "designation": "Housse de couette coton 220x240",
    "description": "Housse lavable 60°. Coloris blanc.",
})

print(result.pred_l1_label)   # "Maison"
print(result.pred_l2_label)   # "Linge de lit"
print(result.confidence_l2)   # 0.87
print(result.llm_used)        # False (fast path)
```

The method returns a `FastPathResult` or `HardPathResult` dataclass.
`HardPathResult` additionally contains `llm_reason`, `routing_reasons`,
`llm_confidence`, and `domain_top1_l2_id`.

---

## 5. Configuration Guide

The file `01_baselines/configs/flat_hydra_dual_expert.yaml` is the
canonical reference. All parameters below are fields of `FlatHydraConfig`
(read by `FlatHydraTrainer`) unless noted otherwise.

### Loss configuration

```yaml
# Baseline (original behaviour):
l2_loss_type: ce             # CrossEntropy, no class weights

# Dual-Expert (focal + class weights):
l2_loss_type: focal_ce
focal_gamma_l2: 2.0
use_l2_class_weights: true
# l2_class_weights_path: models/flat_hydra_dual_expert/l2_class_weights.pt
```

When `l2_loss_type: focal_ce`:

1. The trainer looks for `l2_class_weights.pt` at `l2_class_weights_path`,
   then at `{save_dir}/l2_class_weights.pt`.
2. If not found, it builds weights from training labels on the fly and
   saves them to `save_dir`.

### Label-path augmentation

```yaml
label_path_aug_enabled: true
label_path_aug_apply_to: rare_only   # all | rare_only | none
label_path_aug_rare_threshold: 100
label_path_aug_factor_rare: 3        # 2 extra copies per rare sample
label_path_aug_factor_medium: 2      # 1 extra copy per medium sample
label_path_aug_mask_prob: 0.3        # prob to mask L1 or L2 name
```

### Hierarchy-aware scoring (inference-only, not in FlatHydraConfig)

```yaml
hierarchy_scoring:
  enabled: true
  alpha: 0.5
  eps: 1.0e-12
```

Read directly by `DualExpertProductClassifier` and `evaluate_topk.py`.

### LLM reranker (inference-only)

```yaml
llm_reranker:
  enabled: false        # set to true to activate hard path
  provider: none        # none | local | openai_compatible | mistral_api
  model_name: mixtral-8x7b-instruct
  api_base_url: http://localhost:8080/v1
  api_key: none
  temperature: 0.0
  max_new_tokens: 256
  fallback_to_domain_expert: true
```

---

## 6. How to Run Each Experiment

### Prerequisites

All commands below assume the project root as working directory and that
split files exist at a `SPLIT_DIR`.

```bash
export SPLIT_DIR=/path/to/dataset   # must contain train/val/test_split.jsonl
export CKPT=models/flat_hydra_dual_expert
```

---

### 6.1 Experiment A — Baseline (unchanged)

Original CamemBERT Flat HYDRA, CrossEntropy, no new features.

```bash
python scripts/run_flat_hydra.py \
    --split-dir "$SPLIT_DIR" \
    --config 01_baselines/configs/flat_hydra.yaml \
    --save-dir models/flat_hydra_baseline
```

---

### 6.2 Experiment B — Top-K recall measurement

Runs on an **already trained** checkpoint without any retraining.

```bash
python scripts/evaluate_topk.py \
    --checkpoint models/flat_hydra_baseline \
    --split-dir "$SPLIT_DIR" \
    --split val \
    --top-k 20 \
    --hierarchy-alpha 0.0 \
    --report-dir models/flat_hydra_baseline
```

Output: `models/flat_hydra_baseline/topk_recall_report.json`.

To test with hierarchy scoring enabled:

```bash
python scripts/evaluate_topk.py \
    --checkpoint models/flat_hydra_baseline \
    --split-dir "$SPLIT_DIR" \
    --top-k 20 \
    --hierarchy-alpha 0.5 \
    --report-dir models/flat_hydra_baseline
```

---

### 6.3 Experiment C — Hierarchy-aware scoring

Same trained model, different alpha values, no retraining needed.
Run `evaluate_topk.py` multiple times:

```bash
for ALPHA in 0.0 0.25 0.5 0.75 1.0; do
    python scripts/evaluate_topk.py \
        --checkpoint models/flat_hydra_baseline \
        --split-dir "$SPLIT_DIR" \
        --top-k 20 \
        --hierarchy-alpha "$ALPHA" \
        --report-dir "models/flat_hydra_baseline/alpha_${ALPHA}"
done
```

Compare `topk_recall_report.json` files across alpha values.

---

### 6.4 Experiment D — Focal loss + class weights

**Step 1** — Pre-build class weights (optional, otherwise auto-built):

```bash
python scripts/build_l2_class_weights.py \
    --split-dir "$SPLIT_DIR" \
    --save-dir "$CKPT" \
    --beta 0.9999 \
    --max-weight 10.0
```

**Step 2** — Train with focal CE:

```bash
python scripts/run_flat_hydra.py \
    --split-dir "$SPLIT_DIR" \
    --config 01_baselines/configs/flat_hydra_dual_expert.yaml \
    --save-dir "$CKPT"
```

The config has `l2_loss_type: focal_ce` and `use_l2_class_weights: true`
by default. To run a grid search over gamma:

```bash
for GAMMA in 1.0 2.0; do
    python scripts/run_flat_hydra.py \
        --split-dir "$SPLIT_DIR" \
        --config 01_baselines/configs/flat_hydra_dual_expert.yaml \
        --save-dir "models/flat_hydra_focal_gamma${GAMMA}"
done
```

> **Note:** The `FlatHydraConfig.from_yaml` method does not yet accept
> `focal_gamma_l2` as a CLI override. Edit the YAML directly or add the
> field to the overrides dict in the script.

---

### 6.5 Experiment E — Label-path augmentation

Edit `01_baselines/configs/flat_hydra_dual_expert.yaml` and set:

```yaml
label_path_aug_enabled: true
label_path_aug_apply_to: rare_only
```

Then train normally:

```bash
python scripts/run_flat_hydra.py \
    --split-dir "$SPLIT_DIR" \
    --config 01_baselines/configs/flat_hydra_dual_expert.yaml \
    --save-dir models/flat_hydra_aug_rare
```

To compare augmenting all classes vs. rare only, change
`label_path_aug_apply_to` to `all` and re-run with a different
`--save-dir`.

---

### 6.6 Experiment F/G — LLM reranking

**Step 1** — Generate category definitions (dry-run, no LLM):

```bash
python scripts/generate_category_definitions.py \
    --split-dir "$SPLIT_DIR" \
    --output data/category_definitions/category_definitions_v1.jsonl \
    --dry-run
```

With a local LLM (llama.cpp server on port 8080):

```bash
python scripts/generate_category_definitions.py \
    --split-dir "$SPLIT_DIR" \
    --output data/category_definitions/category_definitions_v1.jsonl \
    --provider local \
    --api-base-url http://localhost:8080/v1 \
    --model mixtral-8x7b-instruct
```

Use `--resume` to continue an interrupted run.

**Step 2** — Run LLM reranking on validation hard cases:

```python
# In a notebook or script:
import sys
sys.path.insert(0, "01_baselines")

from dual_expert.dual_expert_classifier import DualExpertProductClassifier
from dual_expert.category_definitions import CategoryDefinitionStore
from dual_expert.llm_reranker import LLMRerankerConfig
from dual_expert.difficulty_router import DifficultyRouterConfig
from dual_expert.audit_logger import AuditLogger

store = CategoryDefinitionStore(
    "data/category_definitions/category_definitions_v1.jsonl"
)
classifier = DualExpertProductClassifier.from_checkpoint(
    checkpoint_dir="models/flat_hydra_dual_expert",
    reranker_config=LLMRerankerConfig(
        enabled=True,
        provider="local",
        api_base_url="http://localhost:8080/v1",
        model_name="mixtral-8x7b-instruct",
        fallback_to_domain_expert=True,
    ),
    definition_store=store,
)
logger = AuditLogger(
    jsonl_path="logs/hard_case_predictions.jsonl",
    csv_path="logs/llm_audit_samples.csv",
)

for product in validation_products:
    result = classifier.predict(product)
    if result.llm_used:
        logger.log_hard_path(
            product=product,
            domain_pred_l1=result.pred_l1_id,
            domain_pred_l2=result.domain_top1_l2_id,
            domain_conf_l2=result.confidence_l2,
            top_k_candidates=result.top_k_candidates,
            routing_reasons=result.routing_reasons,
            llm_result=result,
        )
```

---

## 7. How to Build Class Weights

```bash
python scripts/build_l2_class_weights.py \
    --split-dir /path/to/dataset \
    --save-dir models/flat_hydra_dual_expert \
    --beta 0.9999 \
    --max-weight 10.0
```

**Output files:**

| File | Description |
|:-----|:------------|
| `l2_class_weights.pt` | FloatTensor `(num_l2,)` — ready to load with `torch.load` |
| `l2_class_counts.json` | `{"l2_id": count, ...}` for all L2 classes |
| `l2_frequency_buckets.json` | `{"rare_0_10": N, ...}` — class counts per bucket |

The trainer will auto-load `l2_class_weights.pt` from `save_dir` if
`use_l2_class_weights: true` and `l2_class_weights_path` is left empty.

---

## 8. How to Generate Category Definitions

The script has two modes:

### Dry-run (no LLM, placeholder definitions)

```bash
python scripts/generate_category_definitions.py \
    --split-dir /path/to/dataset \
    --output data/category_definitions/category_definitions_v1.jsonl \
    --dry-run
```

All 6 126 L2 definitions are generated with `confidence: low` and a
formulaic `definition_fr`. Sufficient for testing the full pipeline
without a GPU or API.

### With a local LLM

```bash
python scripts/generate_category_definitions.py \
    --split-dir /path/to/dataset \
    --output data/category_definitions/category_definitions_v1.jsonl \
    --provider local \
    --api-base-url http://localhost:8080/v1 \
    --model mixtral-8x7b-instruct \
    --examples-per-category 20 \
    --version v1 \
    --resume   # skip already completed L2s
```

### Validation

After generation, check quality:

```python
import json
from pathlib import Path

records = [json.loads(l) for l in open("data/category_definitions/category_definitions_v1.jsonl")]
low_conf = [r for r in records if r.get("confidence") == "low"]
needs_review = [r for r in records if r.get("needs_review")]

print(f"Total: {len(records)}")
print(f"Low confidence: {len(low_conf)}")
print(f"Needs review: {len(needs_review)}")
```

---

## 9. How to Disable All New Features

To recover the exact original baseline behaviour, use the original config:

```bash
python scripts/run_flat_hydra.py \
    --split-dir "$SPLIT_DIR" \
    --config 01_baselines/configs/flat_hydra.yaml
```

Alternatively, in `flat_hydra_dual_expert.yaml` set:

```yaml
l2_loss_type: ce
use_l2_class_weights: false
label_path_aug_enabled: false
```

And in `DualExpertProductClassifier`:

```python
DualExpertProductClassifier.from_checkpoint(
    checkpoint_dir=...,
    scoring_config=HierarchyScoringConfig(enabled=False),
    reranker_config=LLMRerankerConfig(enabled=False),
)
```

With `scoring_config.enabled=False`, the scorer returns plain
`log(softmax(logits_l2))` — identical to `argmax` on L2 logits.

---

## 10. Expected Artifacts

After a full dual-expert training run:

```
models/flat_hydra_dual_expert/
├── tokenizer/                   HuggingFace tokenizer
├── best_model.pt                Best checkpoint by val F1 L2
├── checkpoint_latest.pt         Resume checkpoint
├── config.json                  FlatHydraConfig + num_classes
├── l1_label_encoder.pkl         sklearn LabelEncoder L1
├── l2_label_encoder.pkl         sklearn LabelEncoder L2
├── l2_to_l1_mapping.json        {l2_id: l1_id}
├── metrics.json                 Per-epoch training history
├── training_complete.txt
│
├── l2_class_weights.pt          Effective-number weights (focal_ce only)
├── l2_class_counts.json         Training counts per L2 class
├── l2_frequency_buckets.json    Class count per bucket
└── topk_recall_report.json      (after evaluate_topk.py)

data/category_definitions/
└── category_definitions_v1.jsonl   One JSON per L2 category

logs/
├── hard_case_predictions.jsonl  Hard-path audit log (JSONL)
└── llm_audit_samples.csv        Hard-path audit log (CSV)
```

---

## 11. Known Limitations

1. **Class weights require training data access.** If `l2_class_weights.pt`
   is not pre-built, the trainer reads all training labels at startup.
   On the full 8M dataset this takes a few seconds and is not a bottleneck.

2. **`l2_class_weights_path` is not a CLI flag.** Set it in the YAML or
   pass it as a constructor argument to `FlatHydraConfig`.

3. **LLM reranker requires an external service.** In `provider: none`
   (dry-run) mode, the reranker always returns a fallback. A local llama.cpp
   or vLLM server must be running before enabling `provider: local`.

4. **`generate_category_definitions.py` processes L2 classes sequentially.**
   For 6 126 classes with a 10-second LLM timeout, the full generation pass
   takes 17+ hours. Use `--resume` to checkpoint and continue.

5. **Augmented dataset size.** With `augmentation_factor_rare=3` on a
   dataset where 30% of L2 classes are rare, the train loader grows by
   roughly 60–90%. Training time increases proportionally.

6. **`HierarchyScoringConfig` is not parsed from YAML by `FlatHydraConfig`.**
   It is a separate config used only at inference time by
   `DualExpertProductClassifier` and `evaluate_topk.py`.

---

## 12. Suggested Next Tests

1. **Measure baseline top-K recall** with `evaluate_topk.py` before any
   retraining. This tells you whether the LLM reranker is worth integrating
   (requires top-10 recall ≥ 85% globally, ≥ 70% for rare classes).

2. **Alpha sweep** (Experiment C): run `evaluate_topk.py` for alpha in
   `{0.0, 0.25, 0.5, 0.75, 1.0}` and pick the value that maximises
   `rare_0_10.l2_top_10_recall` without hurting global top-1 accuracy.

3. **Focal loss gamma sweep** (Experiment D): compare gamma `{1.0, 2.0}`
   on the validation set, focusing on `rare_0_10` and `rare_10_50` buckets.

4. **Augmentation ablation** (Experiment E): compare runs with
   `apply_to=none`, `apply_to=rare_only`, `apply_to=all` using identical
   seeds and eval protocol.

5. **LLM path-only vs. definitions** (Experiments F vs. G): run
   reranking on the validation hard cases first with `CategoryDefinitionStore`
   in fallback-only mode, then with real definitions, and measure
   `is_correct_after` rate in `llm_audit_samples.csv`.

6. **Router threshold tuning**: after LLM evaluation, adjust
   `min_l2_confidence` and `min_l2_margin` to hit the 10–25% pass-through
   target while maximising hard-case accuracy gain.
