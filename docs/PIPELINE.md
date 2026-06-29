# Experimentation Pipeline

Reference map: which script and which config to train, evaluate,
and publish each model in the project.

## Table of Contents

1. [Overview](#overview)
1. [Model × Step Matrix](#model--step-matrix)
1. [YAML Configs](#yaml-configs)
1. [SLURM Scripts](#slurm-scripts)
1. [Report Generation](#report-generation)
1. [Utilities](#utilities)
1. [Known Technical Debt](#known-technical-debt)

## Overview

```text
HF dataset (JSONL splits)
        |
        v
   training      ----->  models*/...                     (best_model.pt + checkpoint)
        |
        v
   evaluation    ----->  results/*/test_metrics.json     (test metrics)
```

## Model × Step Matrix

Commands from the repo root.

> **Prerequisites**: export `PROJECT_ROOT`, `VENV_ACTIVATE`, and `DATA_DIR`
> before any `sbatch` (see [REPRODUCE.md](../REPRODUCE.md));
> `--array=0-9%1` = automatic checkpoint resume, one task at a time.

### Standard HYDRA CamemBERT (local, nested, global)

```bash
# Train (single variant, auto-resume job array)
sbatch --job-name=hydra-local --array=0-9%1 \
    scripts/slurm/run_train_variant.slurm local configs/hydra/default.yaml

# Or all 3 variants in parallel (1 GPU each)
sbatch scripts/slurm/run_train_array.slurm configs/hydra/default.yaml

# Evaluate (3 variants, with/without mask)
sbatch scripts/slurm/run_eval.slurm

# Per-class F1
sbatch scripts/slurm/run_f1_perclass.slurm
```

### HYDRA hierarchy (teacher forcing + mask)

```bash
# Train — chain N arrays if > 36 h
./scripts/chain_submit_array.sh 3 configs/hydra/hierarchy.yaml

# Evaluate (models are in models/v2/)
sbatch scripts/slurm/run_eval.slurm models/v2

# Per-class F1
sbatch scripts/slurm/run_f1_perclass_hierarchy.slurm
```

### HYDRA mBERT

```bash
sbatch --job-name=hydra-local-mbert --array=0-9%1 \
    scripts/slurm/run_train_variant.slurm local configs/hydra/mbert_default.yaml
sbatch scripts/slurm/run_eval_hydra_mbert.slurm
sbatch scripts/slurm/run_f1_perclass_mbert.slurm
```

### HYDRA title-only (designation only)

```bash
sbatch scripts/slurm/run_hydra_title_3090.slurm             # standard
sbatch scripts/slurm/run_hydra_title_hierarchy_3090.slurm    # + hierarchy
sbatch scripts/slurm/run_eval_hydra_title_hierarchy.slurm     # evaluation
```

### FlatHydra (single L1+L2 head)

```bash
sbatch --job-name=flat --array=0-9%1 scripts/slurm/run_flat_3090.slurm     # 2×3090 (v1)
sbatch --job-name=flat --array=0-9%1 scripts/slurm/run_flat_p100.slurm     # 2×P100 (v1)
sbatch --job-name=flat-mbert --array=0-9%1 scripts/slurm/run_flat_mbert.slurm
sbatch scripts/slurm/run_flat_hydra_title_3090.slurm                       # title-only

sbatch scripts/slurm/run_eval_flat_hydra.slurm          # full-text eval (v1)
sbatch scripts/slurm/run_eval_flat_hydra_title.slurm     # title-only eval
```

**Flagship v2 model (78.9% L2 F1 reference):**

```bash
# CamemBERT v2 training (naive method — flat_hydra_camembert_v2.yaml config)
sbatch --array=0-9%1 scripts/slurm/run_flat_camembert_v2.slurm

# mBERT v2 training (77.4% L2 F1)
sbatch --array=0-9%1 scripts/slurm/run_flat_mbert_v2.slurm

# Test evaluation (generic, configurable via --export)
sbatch scripts/slurm/run_eval_flat_hydra_generic.slurm
# Example for mBERT v2:
# sbatch --job-name=eval-mbert-v2 \
#   --export=ALL,MODEL_PT=models/v2/flat_hydra_mbert/best_model.pt,\
#   EVAL_CONFIG=configs/baselines/flat_hydra_mbert_v2.yaml,\
#   RESULTS_DIR=results/flat_hydra_mbert_v2 \
#   scripts/slurm/run_eval_flat_hydra_generic.slurm
```

### HLPD

```bash
sbatch --job-name=hlpd --array=0-9%1 scripts/slurm/run_hlpd_3090.slurm
# Config: configs/hlpd/hlpd_3090.yaml
```

### Baselines

```bash
sbatch scripts/slurm/run_baseline_tfidf.slurm                    # TF-IDF
sbatch scripts/slurm/run_baseline_tfidf_title.slurm              # TF-IDF title
sbatch scripts/slurm/run_baseline_word2vec.slurm                 # Word2Vec
sbatch scripts/slurm/run_baseline_word2vec_title.slurm           # Word2Vec title
sbatch scripts/slurm/run_baseline_fasttext_softmax_title.slurm   # fastText title

# fastText full-text: direct machine (lame10, 80 CPUs), NOT SLURM
bash scripts/run_fasttext_softmax.sh
bash scripts/run_fasttext_eval.sh
```

## YAML Configs

`configs/hydra/default.yaml` is the **source of truth**. Divergences of
the other configs:

| Config                       | Batch | LR        | Key divergences                            |
|:-----------------------------|:-----:|:----------|:-------------------------------------------|
| `default.yaml`               | 64    | 3.5e-5    | Reference (dropout 0.3, hidden 256)        |
| `hierarchy.yaml`             | 128   | 5.0e-5    | `use_focal`, `condition_l2_on_l1`, mask    |
| `title_only.yaml`            | 320   | 7.8e-5    | Title only, `save_dir: models/v1/title`       |
| `title_only_hierarchy.yaml`  | 640   | 1.12e-4   | Title + hierarchy                          |
| `mbert_default.yaml`         | 64    | 3.5e-5    | mBERT encoder, rest identical              |

> **Note — scaling rule**: when the batch grows, the LR follows the
> square root of the factor: `lr = 3.5e-5 × √(batch/64)`. Check:
> batch 128 → ×1.41 (5e-5), batch 320 → ×2.24 (7.8e-5), batch 640 → ×3.2
> (1.12e-4). The title-only configs support large batches because the
> sequences (title only) are much shorter.

FlatHydra configs (`configs/baselines/`): `dropout: 0.2` (faithful to the
paper, ≠ 0.3 HYDRA); the batch varies by GPU — `flat_hydra.yaml`
(64, CPU/debug), `flat_hydra_3090.yaml` (256), `flat_hydra_title_3090.yaml`
(512), `flat_hydra_mbert.yaml` (128, heavier mBERT).

> **Dataset v2 configs — beware of confusion**:
>
> - `configs/baselines/flat_hydra_camembert_v2.yaml` = **naive** method (identical to v1, no
>   rebalancing). Launched via `run_flat_camembert_v2.slurm` → produces the reference **78.9% L2 F1**
>   on the v2 dataset (CamemBERT).
> - `configs/baselines/flat_hydra_mbert_v2.yaml` = naive mBERT v2, launched via
>   `run_flat_mbert_v2.slurm` → 77.4% L2 F1.
> - `configs/baselines/flat_hydra_v2.yaml` and `configs/hydra/v2_default.yaml` carry the
>   **rebalancing kit** (`normalize_text`, `synthetic_policy: drop_oversampled`,
>   `class_weighting: inv_sqrt`, `logit_adjustment`) — these two configs **have never been
>   launched** and constitute a future experiment, not the 78.9% reference.
> - Configs without the `v2` suffix correspond to the historical v1 trainings.

## SLURM Scripts

All SLURM job scripts live in **`scripts/slurm/`** (e.g. `scripts/slurm/run_train_variant.slurm`).

Common conventions: `--array=0-9%1` job arrays (automatic resume from
`checkpoint_latest.pt`, skip if `training_complete.txt`), 36 h limit
per task, `P100` (2×16 GB) and `3090` (2-4×24 GB) partitions.

| Script                         | Role                            | GPU       |
|:-------------------------------|:--------------------------------|:----------|
| `run_train_variant.slurm`      | 1 HYDRA variant, 1 config       | 1-2 GPU   |
| `run_train_array.slurm`        | 3 variants in parallel          | 1 GPU/var |
| `chain_submit_array.sh`        | N chained arrays (`afterany`)   | —         |
| `run_flat_*.slurm`             | FlatHydra per GPU/encoder       | 2 GPU     |
| `run_hlpd_3090.slurm`          | HLPD                            | 2×3090    |
| `run_eval*.slurm`              | Test evaluations                | 1-2 GPU   |
| `run_f1_perclass*.slurm`       | Per-class F1                    | 1 GPU     |
| `run_baseline_*.slurm`         | CPU baselines                   | —         |
| `cluster_status.sh`            | Node GPU/CPU/RAM status         | —         |

> **Warning**: never submit jobs in a loop or recreate an
> auto-resubmission monitoring script (the 110-job incident,
> admin warning). The `%1` job arrays cover the need.

## Report Generation

```bash
# Per-class F1 of an already-trained model
python3 scripts/generate_f1_per_class.py \
    --split-dir "$DATA_DIR" --config configs/hydra/default.yaml \
    --variant local --results-dir hydra-local-camembert

# Class metadata (test support, L1 parent)
python3 scripts/build_class_metadata.py --dataset-dir "$DATA_DIR"
```

The provenance of each consumed JSON is traced in
[METRICS.md](METRICS.md).

## Utilities

| Script                    | Role                                               |
|:--------------------------|:---------------------------------------------------|
| `download_dataset.py`     | Downloads the splits from HuggingFace              |
| `split_dataset.py`        | Regenerates L2-stratified 70/15/15 splits          |
| `upload_models_hf.py`     | Archives the weights to HF (`--dry-run` available) |
| `eval_hierarchy.py`       | Eval with/without mask (`--output-mode auto`)      |

## Known Technical Debt

- **Trainer duplication**: `rakuten/models/hydra/trainer.py`,
  `rakuten/models/baselines/flat_trainer.py`, and the HLPD trainer share
  a lot of train/eval logic without common factoring.
  `rakuten/common/training_utils.py` provides the shared building blocks
  (EarlyStopping, compute_class_weights).
- **Heterogeneous JSON structures in `results/`**: three formats
  coexist depending on the model family (see [METRICS.md](METRICS.md));
  harmonizing would require regenerating all the evaluations.
- **DDP pipeline**: explored during the project but dropped from the
  deliverable in favor of single-node DataParallel. Available at the git
  tag `archive/distributed-pipeline` — see [docs/ARCHITECTURE.md](ARCHITECTURE.md).
