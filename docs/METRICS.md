# Metrics and Provenance of Results

Definitions of the project's metrics, an explanation of the two evaluation
modes (the repo's classic pitfall) and traceability of every figure published
in `results/`.

> **Warning — data leakage in the v1 benchmarks (12/06/2026)**:
> a dataset audit revealed that **11.2 % of the val and test rows
> had an exact twin in the train set** (the same product listing scraped
> several times, 98 % with the same label). All the results in `results/`
> and `comparison.html` were trained and evaluated on this "v1" dataset
> (leakage not purged). Comparisons *between* models remain valid
> (the same leakage for all).
>
> The corrected dataset ("v2", leakage purged: **−314,624 net train
> rows**, 6,126 → 5,777 L2 classes) is on HuggingFace in
> `datasets_finaux/`.
>
> **Measured real effect (≠ the initial estimate of +1.5–3 pts).** FlatHydra
> was re-trained + re-evaluated on v2 (CamemBERT **and** mBERT, naive method
> identical to v1, only the data changes):
> - **L1 (same 157 classes — clean comparison)**: v2 is **0.8 to 1.3 pts
>   lower** than v1 → this is the real effect of the leakage, which inflated v1
>   by **~1 pt**.
> - **L2**: v2 is *higher* (CamemBERT 71.7 → 78.9; mBERT 71.8 → 77.4),
>   but **not directly comparable**: ~3 pts come mechanically from the 351
>   dropped classes (mean F1 0.20), and the rest from the fact that de-leaking
>   *improved* training (duplicates removed), it did not inflate the test.
>
> Conclusion: the real v1 inflation is **~1 pt** (measured at L1), well below
> the initial estimate; at L2 the v1 and v2 figures do not compare term by
> term. (Measured on FlatHydra / 2 encoders; HYDRA and HLPD were not
> re-trained on v2.) Details of the 4 evaluations: `results/flat_hydra*/`.
>
> *Note: the v2 models have **5,775** L2 heads (not 5,777) — 2 L2 classes in the
> dataset have no train sample after purging, the number of heads being derived
> from the data at runtime. Hence `num_classes_l2: 5775` in
> `results/flat_hydra_v2/test_metrics.json`.*

## Table of Contents

1. [Definitions](#definitions)
1. [The two evaluation modes](#the-two-evaluation-modes)
1. [The mBERT incident and the auto mode](#the-mbert-incident-and-the-auto-mode)
1. [Provenance of the published figures](#provenance-of-the-published-figures)
1. [Reproducing an evaluation](#reproducing-an-evaluation)
1. [Evaluating a new model](#evaluating-a-new-model)

## Definitions

All the metrics are implemented in `rakuten/common/metrics.py`:

| Metric                 | Definition                                             |
|:-----------------------|:-------------------------------------------------------|
| `f1_l1_macro`          | Macro F1 over the 157 L1 classes (each class weighs 1) |
| `f1_l2_macro`          | Macro F1 over the 5,777 L2 classes — **target metric** |
| `constrained_f1_macro` | L2 F1 counted correct only if the L1 parent is correct |
| `hcr`                  | % of predictions where L2 is a valid child of L1       |

**Macro**-averaging gives the same weight to rare classes as to frequent
classes — this is intentional: the long tail of the 5,777 L2 classes
is the heart of the project. Overall goal: maximize `f1_l2_macro` while
maintaining an HCR > 95 % for the hierarchical variants.

## The Two Evaluation Modes

A single HYDRA model can be evaluated in two ways:

| Mode   | Mechanism                                        | Typical HCR |
|:-------|:--------------------------------------------------|:-----------:|
| `base` | Independent L1 and L2 heads, no constraint         | ~0.86       |
| `hier` | Hierarchical mask: L2 restricted to children of the predicted L1 | ~0.997 |

> **Warning**: the figures from the two modes are **not comparable**.
> The `hier` mode gives an HCR ≈ 0.997 *by construction* and changes the
> L2 F1. Comparing a model evaluated in `base` with another in `hier`
> makes no sense.

The project's publication rule:

- **Standard** models (trained without hierarchy) → publish the `base` mode
- **hierarchy** models (`condition_l2_on_l1` + `check_hierarchy`) →
  publish the `hier` mode (this is their nominal operating regime)

## The mBERT Incident and the Auto Mode

Background: `scripts/eval_hierarchy.py` systematically serialized the
`hier` mode in its output JSON. For HYDRA Local mBERT (a standard model),
this published L2 F1 = 60.7 % / HCR = 99.7 % instead of the true
values 66.8 % / 85.8 % — later corrected by hand in
`results/hydra_eval_test_metrics_mbert.json` (`_note` fields).

Since then, the script chooses the right mode automatically:

```bash
python scripts/eval_hierarchy.py --split-dir "$DATA_DIR" \
    --config configs/hydra/mbert_default.yaml \
    --output-json results/eval.json
# → "base" mode because the config has neither check_hierarchy nor condition_l2_on_l1
```

The `--output-mode {auto,base,hier}` flag allows an explicit override, and
the produced JSON contains the `_mode`, `_config` and `_date` metadata.

## Provenance of the Published Figures

Each metrics file in `results/` and its source:

| File                                 | Models                | Source                      | Mode  |
|:--------------------------------------|:----------------------|:-----------------------------|:------|
| `hydra_eval_test_metrics.json`         | HYDRA std CamemBERT ×3 | SLURM job 772505             | base  |
| `hydra_eval_hierarchy_metrics.json`    | HYDRA hierarchy ×3     | Jobs 772228/229/230          | hier  |
| `hydra_eval_test_metrics_mbert.json`   | HYDRA Local mBERT      | Training log job 813697_5    | base  |
| `flat_hydra/test_metrics.json`         | FlatHydra              | `run_eval_flat_hydra.slurm`  | base  |
| `flat_hydra_title/test_metrics.json`   | FlatHydra title        | `run_eval_flat_hydra_title.slurm` | base |
| `hlpd/test_metrics.json`               | HLPD                   | Reported (best @ step 345k)  | native |
| `baselines/baseline_results.json`      | TF-IDF, W2V, fastText  | Baseline scripts/notebooks   | base  |
| `baselines/baseline_results_title.json`| Same, title-only       | Same                         | base  |

Particularities to be aware of:

- JSON files marked with a `_source` or `_note` field were **curated by
  hand** afterwards (notably the mBERT one) — do not regenerate them without
  re-reading their header.
- The **standard title-only** HYDRA models were only logged at training time
  (no dedicated evaluation JSON); their figures live in the tables of
  [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Reproducing an Evaluation

```bash
# HYDRA standard (3 variants, both modes shown in the console)
sbatch scripts/slurm/run_eval.slurm

# HYDRA hierarchy (models in models/v2/)
sbatch scripts/slurm/run_eval.slurm models/v2

# A single variant, locally with GPU
python scripts/eval_hierarchy.py --split-dir "$DATA_DIR" \
    --variant local --eval-batch-size 2048 \
    --output-json results/hydra_eval_test_metrics.json
```

The console output always shows both modes with their deltas —
only the JSON follows `--output-mode`.

## Evaluating a New Model

1. **Systematically compare** against the TF-IDF baseline (L2 F1 = 42.8 %) and
   the best model (FlatHydra, L2 F1 = 71.7 %)
1. **Check the HCR** in addition to the F1 — a hierarchical model must hold
   HCR > 95 %
1. **Publish the right mode** (`--output-mode auto` makes the right choice).
