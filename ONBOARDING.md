# Onboarding Guide

This guide takes a new developer from zero to a first reproducible training
run. Follow the sections in order.

## Table of Contents

1. [The project in 5 minutes](#the-project-in-5-minutes)
1. [Prerequisites](#prerequisites)
1. [Installation](#installation)
1. [Getting the dataset](#getting-the-dataset)
1. [Getting the trained models](#getting-the-trained-models)
1. [First training run](#first-training-run)
1. [Working on the SLURM cluster](#working-on-the-slurm-cluster)
1. [Repository map](#repository-map)
1. [Known pitfalls](#known-pitfalls)
1. [Going further](#going-further)

## The Project in 5 Minutes

Hierarchical classification of **8M+ Rakuten France products** from their
text (`designation` + `description`, in French):

- **Level 1 (L1)**: 157 parent categories
- **Level 2 (L2)**: 5 777 leaf categories, each a child of an L1

Three model families are compared:

| Family     | Approach                                  | Best F1 L2     |
|:-----------|:------------------------------------------|:--------------:|
| Baselines  | TF-IDF, Word2Vec, fastText + classifiers  | 42.9 %         |
| FlatHydra  | CamemBERT + single L1+L2 head             | **71.7 %**     |
| HYDRA      | CamemBERT multi-head (EMNLP 2025)         | 66.8 %         |
| HLPD       | Hierarchical decoder (ECIR 2025)          | 61.5 %         |

The main metrics are `f1_l1_macro`, `f1_l2_macro` and the **HCR**
(hierarchical consistency rate) — defined in `rakuten/common/metrics.py`
and documented in [docs/METRICS.md](docs/METRICS.md). The comparison
deliverable is `results/comparison.html`.

## Prerequisites

- Python 3.11+ (validated environment: 3.13) — the 3.11 floor is imposed by
  the pinned versions of `numpy` and `scikit-learn`
- GPU optional locally (required for full training — SLURM cluster)
- **Read-access HuggingFace token** to access the private repos
  (`TahianaAndriambahoaka/rakuten-dataset` and `TahianaAndriambahoaka/rakuten-models`):
  request it from the owner (`TahianaAndriambahoaka`)

> **Without this token**: the package installs and the 312 tests pass, but it is
> impossible to reproduce the training runs or evaluations (no local data).
> This is the recommended fallback path for an initial walkthrough of the code.

## Installation

```bash
git clone <url-du-repo> rakuten-classification
cd rakuten-classification

python -m venv ../venv
source ../venv/bin/activate
# CPU or another CUDA version? Install torch first via the appropriate PyTorch index
# (see the comment at the top of requirements.txt), then:
pip install -e .

# Smoke test: the full suite passes without data or GPU
python -m pytest tests/ -q
```

All 312 tests should pass. Docker alternative:

```bash
docker compose run --rm --profile test test
```

To train locally on GPU: `docker compose run --rm --profile gpu train`

## Getting the Dataset

The pre-generated splits (70/15/15, stratified by L2 class) are hosted
on the private HuggingFace repo [`TahianaAndriambahoaka/rakuten-dataset`][hf-dataset]:

| File                | Lines     |
|:--------------------|:----------|
| `train_split.jsonl` | 5 259 035 |
| `val_split.jsonl`   | 1 194 355 |
| `test_split.jsonl`  | 1 194 356 |

> **Dataset v2**: duplicate leakage purged (−314 624 net train lines).
> See [docs/METRICS.md](docs/METRICS.md) for the v1 → v2 history and the measured effect.

```bash
export HF_TOKEN="hf_..."          # read token provided by the owner
python scripts/download_dataset.py # downloads into <repo>/dataset/
```

Each line is a JSON object with the fields `designation`,
`description`, `typelabel` (L1 label) and `producttype` (L2 label).

> **Note**: the SLURM scripts look for the data in
> `$HOME/fil_rouge/dataset` by default. If you download it elsewhere,
> export `DATA_DIR=/path/to/dataset` before any `sbatch`.

> **Warning**: check `ls -lh "$DATA_DIR/train_split.jsonl"` before
> any `sbatch` — a wrong path produces "ERROR: Split file not found".

## Getting the Trained Models

All trained weights (~63 GB, 22 models) are archived on the private repo
[`TahianaAndriambahoaka/rakuten-models`][hf-models] — they are **not**
in this git repository. The easiest way is via the dedicated script:

```bash
export HF_TOKEN="hf_..."
python scripts/download_models.py                          # all (~63 GB)
python scripts/download_models.py flat-hydra-camembert-v2  # a single model
```

Or directly in Python for a specific subfolder:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-models",
    allow_patterns="hydra-local-camembert/*",   # a single subfolder
    token="hf_...",
)
```

Mapping between the HF subfolders and the local paths expected
by the evaluation scripts:

| HF subfolder                        | Expected local path                           |
|:------------------------------------|:----------------------------------------------|
| `hydra-{variante}-camembert/`        | `models/HYDRA_{variante}_CamemBERT/`           |
| `hydra-local-mbert/`                 | `models/HYDRA_local_mBERT/`                    |
| `hydra-{v}-hierarchy-camembert/`     | `models_hierarchy/HYDRA_{v}_CamemBERT/`        |
| `hydra-{v}-title-camembert/`         | `models_title/HYDRA_{v}_CamemBERT_title/`      |
| `hydra-{v}-title-hierarchy-...`      | `models_title_hierarchy/HYDRA_{v}_..._title_hierarchy/` |
| `flat-hydra/`, `flat-hydra-title/`   | `models/flat_hydra/`, `models/flat_hydra_title/` |
| `hlpd/`                              | `models/hlpd/`                                 |
| `baselines/{tfidf,word2vec,...}/`    | `models/baselines/{...}/`                      |

Each folder contains `best_model.pt` (inference weights, ~440 MB) and
`checkpoint_latest.pt` (full state with optimizer, to resume a training
run). The HF repo card (scores, structure) is in the
`MODEL_CARD` constant of `scripts/upload_models_hf.py`.

## First Training Run

Local smoke test on a subsample — **requires the dataset** (previous section).
Duration: a few minutes to tens of minutes depending on the machine.

> **Without an HF token**: stop after the 312 tests; that's the normal fallback path.

```bash
python scripts/run_train.py \
    --split-dir dataset/ \
    --n-samples 5000 \
    --config configs/hydra/default.yaml
```

The hyperparameters come from the YAML (source of truth:
`configs/hydra/default.yaml`). Run tracking is in MLflow
(`mlflow.db` at the root, UI via `docker compose up mlflow -d`).

## Working on the SLURM Cluster

All `scripts/*.slurm` scripts use four overridable environment
variables (defaults based on `$HOME`):

```bash
export PROJECT_ROOT="$HOME/fil_rouge/rakuten-classification"
export VENV_ACTIVATE="$HOME/fil_rouge/venv/bin/activate"
export DATA_DIR="$HOME/fil_rouge/dataset"
# HF_HOME: HuggingFace cache (default ~/.cache/huggingface)
```

Typical launch of a HYDRA training run (job array with auto-resume):

```bash
sbatch --job-name=hydra-local --array=0-9%1 \
    scripts/run_train_variant.slurm local configs/hydra/default.yaml
```

The `--array=0-9%1` pattern chains up to 10 tasks of 36 h max,
one at a time: each task resumes from `checkpoint_latest.pt` and
stops cleanly when `training_complete.txt` appears.

> **Warning**: cluster rules (inherited from an admin warning) —
> always `%1` job arrays, never loop submissions nor a monitoring
> script that automatically resubmits. GPU partitions:
> `P100` (2×16 GB) and `3090` (2-4×24 GB), 36 h limit, QOS 10 CPUs/job.

> **Different infrastructure?** The `#SBATCH` directives (partition, memory, time)
> are at the top of each `.slurm` file and adapt easily.
> The `PROJECT_ROOT`, `VENV_ACTIVATE` and `DATA_DIR` variables are configurable
> via `export` before each `sbatch`.

The complete "which script for which model" matrix is in
[docs/PIPELINE.md](docs/PIPELINE.md).

## Repository Map

```text
rakuten/                Installable Python package (pip install -e .)
  common/               data_loader, metrics, training_utils, utilities
  models/baselines/     TF-IDF, Word2Vec, fastText, FlatHydra
  models/hydra/         HYDRA (EMNLP 2025): models, losses, trainer, config
  models/hlpd/          HLPD (ECIR 2025) — Mehdi Aithamma
configs/hydra/          default.yaml, hierarchy.yaml, title_only*.yaml, mbert_default.yaml
configs/baselines/      flat_hydra*.yaml
configs/hlpd/           hlpd_3090.yaml
scripts/                Python CLI + SLURM scripts (see docs/PIPELINE.md)
notebooks/eda/          Exploratory data analysis
notebooks/baselines/    fastText, TF-IDF, Word2Vec (notebooks)
tests/                  312 pytest tests (no data, no GPU)
results/                JSON metrics, HTML reports, comparison.html
docs/                   ARCHITECTURE.md, PIPELINE.md, METRICS.md
hybrid_system/          Separate contribution (CamemBERT + XGBoost)
```

Detailed architecture and mapping table: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Known Pitfalls

- **Two non-comparable evaluation modes**: with or without the hierarchical
  mask. `scripts/eval_hierarchy.py --output-mode auto` picks the
  right mode based on the config — read [docs/METRICS.md](docs/METRICS.md)
  before publishing figures.
- **`TRANSFORMERS_OFFLINE=1`** in the SLURM scripts: the GPU nodes
  have no internet access. Pre-download the encoder weights
  locally (`AutoModel.from_pretrained(...)` once) to populate the cache.
- **`_ensure_l2_coverage()`** duplicates ~923 samples (indicative
  value, recomputed on each load) to guarantee every L2 class
  in each split — normal behavior.
- **Title-only configs**: higher `batch_size`/`lr` than `default.yaml`
  (square-root scaling, shorter sequences) — justified in
  [docs/PIPELINE.md](docs/PIPELINE.md).
- **Dataset v1 vs v2**: the published figures are on v1 (leaked). Measured effect
  of the leakage: ~1 pt on L1; on L2, v1↔v2 are not comparable (6 126 → 5 777
  classes) — see [docs/METRICS.md](docs/METRICS.md).

## Going Further

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — project narrative,
  old → new mapping table, target imports
- [docs/PIPELINE.md](docs/PIPELINE.md) — model × step matrix, configs,
  SLURM scripts, known technical debt
- [docs/METRICS.md](docs/METRICS.md) — definitions, evaluation modes,
  provenance of every published figure
- [docs/flat_hydra_conception.md](docs/flat_hydra_conception.md) —
  design of the FlatHydra baseline
- [CONTRIBUTING.md](CONTRIBUTING.md) — branching, quality pipeline (tests + lint),
  Conventional Commits commit format

[hf-dataset]: https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset
[hf-models]: https://huggingface.co/TahianaAndriambahoaka/rakuten-models
