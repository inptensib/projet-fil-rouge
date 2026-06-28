# Experiments — consolidated summary

Overview of **all** the project's experiments (baselines, HYDRA, HLPD,
FlatHydra), with the **test** figures traced to their source file. For metric
definitions, the two evaluation modes and detailed provenance, see
[`METRICS.md`](METRICS.md). For the design of the selected model, see
[`flat_hydra_conception.md`](flat_hydra_conception.md). To reproduce, see
[`PIPELINE.md`](PIPELINE.md).

> **Task**: classify a Rakuten product (French text) into **157 L1 classes**
> (`typelabel`) and **5,777 L2 classes** (`producttype`). Target metric:
> `f1_l2_macro` (each class weighs 1 — the long tail is the challenge).

## Selected model

**FlatHydra — CamemBERT, dataset v2**: a single unified head, **F1 L2 macro =
78.9%** on the test set. This is the best F1 L2 measured, with the simplest
architecture (no multiple heads, no hierarchical mask). Source:
`results/flat_hydra_v2/test_metrics.json`.

| Metric | Value |
|:---------|-------:|
| F1 L1 macro | 92.47% |
| **F1 L2 macro** | **78.93%** |
| Constrained F1 | 79.90% |
| HCR | 85.17% |

## Reading this document (comparability)

Three axes make some rows **not directly comparable**:

- **Dataset `v1` vs `v2`**: `v1` contained a leak (11.2% of val/test
  duplicated in the train set); `v2` is purged and rebalanced (synthetic).
  The actual v1 inflation is **~1 pt** (measured at L1); at L2, v1 and v2 do not
  compare directly (see the warning in [`METRICS.md`](METRICS.md)).
- **Mode `base` vs `hier`**: `hier` applies a hierarchical mask → HCR ≈ 99.7%
  *by construction* and changes the F1 L2. Do not compare a `base` run to a `hier` run.
- **Full text vs title only**: `title` = `designation` only;
  `full text` = `designation + ". " + description`.

All figures are in **% on the test set**. Tables are grouped by comparability.

## 1. Classic baselines (dataset v1, base mode, full text)

| Model | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:-------|------:|------:|----:|-----------:|
| TF-IDF (50K, 1-2 n-grams) + LinearSVC | 83.26 | 42.75 | 63.41 | 42.23 |
| fastText (L1 softmax, L2 softmax) | 82.61 | 42.87 | 67.41 | 42.59 |
| fastText (L1 softmax, L2 OVA) | 82.60 | 10.01 | 29.29 | 9.79 |
| Word2Vec (200d, skip-gram) + LinearSVC | 60.24 | 22.27 | 25.06 | — |

Source: `results/baselines/baseline_results.json`. L2 plateaus at **~43%**:
bag-of-words methods cannot handle the long tail.

## 2. HYDRA — CamemBERT (dataset v1)

Encoder-only multi-head (3 variants). Mode `base` = independent heads;
mode `hier` = hierarchical mask.

**Standard, base mode** — `results/hydra_eval_test_metrics.json` (job 772505):

| Variant | F1 L1 | F1 L2 | HCR |
|:---------|------:|------:|----:|
| local | 93.50 | 66.65 | 85.82 |
| nested | 93.62 | 64.26 | 85.83 |
| global | 93.56 | 66.20 | 85.87 |

**Standard mBERT, base mode** — `results/hydra_eval_test_metrics_mbert.json` (nested/global promoted from the logs on June 27):

| Variant | F1 L1 | F1 L2 | HCR |
|:---------|------:|------:|----:|
| local | 93.04 | 66.80 | 85.81 |
| nested | 93.03 | 64.27 | 85.83 |
| global | 92.93 | 66.05 | 85.87 |

At equal architecture, mBERT is nearly equivalent to CamemBERT on HYDRA (F1 L2 ≈ 66% on both sides), consistent with the finding on the FlatHydra side (§5).

**Hierarchy, hier mode** — `results/hydra_eval_hierarchy_metrics.json`:

| Variant | F1 L1 | F1 L2 | HCR |
|:---------|------:|------:|----:|
| local | 91.33 | 61.18 | 99.74 |
| nested | 92.06 | 61.14 | 99.74 |
| global | 92.59 | 62.70 | 99.74 |

The mask pushes HCR to ~99.7% but **lowers the F1 L2** (~−4 pts vs
standard): it forbids L2s outside the predicted L1, including correct ones.

## 3. HYDRA — dataset v2 (hier mode)

Re-trained on v2 with the **multi-parent mask** fix (an L2 with several L1
parents is no longer wrongly masked). Source:
`results/hydra_hierarchy_v2_metrics.json` (2026-06-25).

| Variant | F1 L1 | F1 L2 | HCR | Multi-parent HCR | Constr. F1 |
|:---------|------:|------:|----:|------------------:|-----------:|
| local | 92.60 | 77.38 | 87.04 | 100.0 | 79.31 |
| nested | 92.59 | 76.41 | 87.02 | 100.0 | 78.56 |
| global | 92.91 | 77.29 | 86.99 | 100.0 | 79.17 |

On v2, HYDRA hierarchy rises to ~77% F1 L2 (vs ~61% on v1) and reaches a
**multi-parent HCR of 100%**, but remains **below FlatHydra v2 (78.9%)** on
F1 L2 for a much heavier architecture.

## 4. FlatHydra (unified head, base mode)

| Model | Dataset | Text | F1 L1 | F1 L2 | HCR | Constr. F1 | Source |
|:-------|:-------:|:-----:|------:|------:|----:|-----------:|:-------|
| CamemBERT | v1 | full | 93.24 | 71.70 | 85.49 | 72.19 | `flat_hydra/` |
| CamemBERT | v1 | title | 92.13 | 69.53 | 85.00 | 70.00 | `flat_hydra_title/` |
| **CamemBERT** | **v2** | **full** | **92.47** | **78.93** | **85.17** | **79.90** | `flat_hydra_v2/` ★ |
| mBERT | v1 | full | 93.38 | 71.78 | 85.61 | 72.43 | `flat_hydra_mbert/` |
| mBERT | v2 | full | 92.09 | 77.40 | 85.20 | 78.34 | `flat_hydra_mbert_v2/` |

★ = selected model. All files: `results/<dir>/test_metrics.json`.

## 5. Encoder comparison (CamemBERT vs mBERT)

With an identical method (FlatHydra), the two encoders are **nearly equivalent**:

| Dataset | CamemBERT (F1 L2) | mBERT (F1 L2) | Gap |
|:-------:|------------------:|--------------:|------:|
| v1 | 71.70 | 71.78 | −0.08 |
| v2 | 78.93 | 77.40 | +1.53 |

Multilingual (mBERT) brings no decisive advantage on a French corpus;
**CamemBERT** is selected (slight gain on v2, natively monolingual French).

## 6. HLPD (hierarchical decoder)

`results/hlpd/test_metrics.json` (dataset v1, native mode): F1 L1 = 80.70,
F1 L2 = 61.50, **HCR = 97.80**. The decoder guarantees strong hierarchical
consistency but lags on F1 (the project's target).

## Lessons learned

1. **Deep models crush the baselines** on L2: 78.9%
   (FlatHydra v2) vs ~43% (best TF-IDF/fastText baseline) — +36 pts.
2. **The description matters**: full text > title only (FlatHydra CamemBERT v1
   71.7 vs 69.5; the gap is far more pronounced on the baselines).
3. **The hierarchical mask is a trade-off**: HCR ~99.7% *by construction*
   but capped F1 L2. FlatHydra (no mask) keeps a "learned" HCR ~85%
   while maximizing F1 L2.
4. **Simple wins**: a unified head (FlatHydra) matches or beats HYDRA
   (3 heads + mask) on F1 L2, for lower complexity → selected model.
5. **CamemBERT ≈ mBERT**; CamemBERT slightly ahead on v2.
6. **v2 (purged + rebalanced)** brings FlatHydra's F1 L2 to 78.9% — the
   long-tail rebalancing (synthetic data, see
   [`../scripts/synthetic_pipeline/`](../scripts/synthetic_pipeline/)) pulls the
   macro upward.

## Appendix — "title only" variants (designation only, dataset v1)

Baselines (base mode) — `results/baselines/baseline_results_title.json`:

| Model | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:-------|------:|------:|----:|-----------:|
| TF-IDF + LinearSVC | 82.55 | 41.10 | 61.38 | 40.69 |
| fastText (softmax) | 72.94 | 34.32 | 64.29 | 32.17 |
| Word2Vec + LinearSVC | 64.02 | 28.16 | 27.55 | 21.83 |

HYDRA hierarchy (hier mode) — `results/hydra_eval_title_hierarchy_metrics.json`:

| Variant | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:---------|------:|------:|----:|-----------:|
| local | 89.72 | 58.94 | 99.75 | 60.79 |
| nested | 90.79 | 58.67 | 99.75 | 60.41 |
| global | 91.63 | 60.49 | 99.74 | 62.16 |

The title-only FlatHydra CamemBERT (92.13 / 69.53 / 85.00) appears in §4.
Everywhere, title only remains below full text — the description provides signal.

## Reproduce

- HYDRA / FlatHydra evaluations: see [`PIPELINE.md`](PIPELINE.md) (model ×
  command matrix) and [`METRICS.md`](METRICS.md) (modes, provenance).
- Comparative HTML report: regenerate with `scripts/generate_comparison.py`
  (→ `results/comparison.html`; not committed — regenerable artifact).
- Full run history: `mlflow.db` (`docker compose up mlflow -d`).
