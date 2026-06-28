# Project Architecture

This document describes the project narrative, the structure of the `rakuten`
Python package, and the mapping table between the old organization and the new
one.

## Table of Contents

1. [Project Narrative](#project-narrative)
1. [Package Structure](#package-structure)
1. [Old → New Mapping Table](#old--new-mapping-table)
1. [Target Imports](#target-imports)
1. [Note on the Distributed Pipeline](#note-on-the-distributed-pipeline)
1. [Note on `hybrid_system/`](#note-on-hybrid_system)

---

## Project Narrative

The project follows a classic supervised machine learning progression: data
exploration, establishing baselines, then increasing complexity toward more
expressive architectures.

### Step 1 — Exploratory Data Analysis (EDA)

Before any model, the `notebooks/eda/` notebooks characterized the Rakuten
France dataset: distribution of the 157 L1 categories, long tail of the 5,777
L2 subcategories, text length (`designation` + `description`), and class
imbalance. The EDA motivated several technical choices: macro-averaging of the
metrics (to give equal weight to rare classes), `max_length=256` (covers 95 % of
products), and stratification by L2 when generating the splits.

### Step 2 — Baselines

The `notebooks/baselines/` notebooks and the `rakuten/models/baselines/` code
implement three families of classic baselines:

- **TF-IDF + LinearSVC**: TF-IDF bag of words, linear SVM. Strongest reference
  (F1 L2 = 42.9 %).
- **Word2Vec + LinearSVC**: pre-trained Word2Vec embeddings, SVM.
- **fastText**: supervised classification (fastText's supervised mode).
- **FlatHydra**: CamemBERT with a single flat L1+L2 head (5,934 outputs in
  v2 = 157 + 5,777; 6,283 on the original v1 dataset). It is the best baseline
  (F1 L2 = 71.7 % in v1, 78.9 % in v2) and the natural starting point for HYDRA.
  **Key finding**: FlatHydra achieves the best F1 L2 of the project, above the
  HYDRA multi-level heads. Despite its name, FlatHydra is **not** a HYDRA variant:
  it is a strong baseline (filed under `rakuten/models/baselines/`).

### Step 3 — HYDRA (EMNLP 2025)

`rakuten/models/hydra/` implements **HYDRA** (Hierarchical YDRA), a CamemBERT
encoder with two independent classification heads: one for the 157 L1 classes,
one for the 5,777 L2 classes. Three variants were compared:

| Variant | Idea | L1↔L2 consistency |
|:---------|:-----|:----------------|
| `local`  | Independent L1 and L2 heads | Post-hoc (hierarchical mask at inference) |
| `global` | + unified L1+L2 head | Implicit (shared representation) |
| `nested` | + L2 refinement MLP conditioned on L1 | By construction |

The hierarchical variant (`configs/hydra/hierarchy.yaml`) adds *teacher forcing*
(`condition_l2_on_l1`) during training and the hierarchical mask at inference
(`check_hierarchy`).

### Step 4 — HLPD (ECIR 2025)

`rakuten/models/hlpd/` implements **HLPD** (Hierarchical Label Pre-populated
Decoder), a non-autoregressive encode–decode architecture developed by
Mehdi Aithamma. The decoder takes the L2 label embeddings as input and refines
them through a hierarchical attention biased by the parent-child structure.
HLPD reaches F1 L2 = 61.5 % using the same data.

---

## Package Structure

```
rakuten-classification/
├── pyproject.toml              # setuptools packaging, dependencies, ruff
├── README.md  CONTRIBUTING.md  REPRODUCE.md
├── rakuten/                    # installable package (pip install -e .)
│   ├── __init__.py
│   ├── common/                 # building blocks shared by all models
│   │   ├── data_loader.py      # JSONL loading + splits
│   │   ├── metrics.py          # f1_l1_macro, f1_l2_macro, HCR, constrained_f1
│   │   ├── training_utils.py   # EarlyStopping, compute_class_weights
│   │   ├── text_normalization.py
│   │   └── utils.py
│   └── models/
│       ├── baselines/          # TF-IDF, Word2Vec, BERT-flat, FlatHydra
│       ├── hydra/              # HYDRA (EMNLP 2025): models, losses, trainer, config
│       └── hlpd/               # HLPD (ECIR 2025): models, losses, attention_bias, trainer
├── configs/                    # YAML outside the package (not imported in Python)
│   ├── hydra/                  # default.yaml, hierarchy.yaml, title_only*.yaml, mbert_default.yaml
│   ├── baselines/              # flat_hydra.yaml, flat_hydra_3090.yaml, …
│   └── hlpd/                   # hlpd_3090.yaml
├── scripts/                    # Python CLI entry points (root) + subfolders below
│   ├── slurm/                  # all SLURM job scripts (train, eval, baselines)
│   ├── flat_inference/         # standalone + ONNX inference (CamemBERT Flat v2)
│   ├── hydra_inference/        # ONNX inference of the 3 HYDRA variants (+ mask)
│   └── synthetic_pipeline/     # synthetic data generation
├── notebooks/                  # exploration and baselines (outside the package)
│   ├── eda/                    # exploratory analysis
│   └── baselines/              # fastText, TF-IDF, Word2Vec notebooks
├── tests/                      # syntax + import smoke tests (no data, no GPU)
├── docs/                       # technical documentation
│   ├── ARCHITECTURE.md         # this file
│   ├── PIPELINE.md             # script × step matrix
│   ├── METRICS.md              # metric definitions, figures provenance
│   └── EXPERIMENTS.md          # consolidated experiments summary
├── hybrid_system/              # separate experimental contribution (see note below)
└── results/                    # JSON metrics, HTML reports
```

---

## Old → New Mapping Table

| Old path | New path | Note |
|:--------------|:---------------|:---------|
| `shared/` | `rakuten/common/` | Shared building blocks |
| `02_hydra/src/` | `rakuten/models/hydra/` | Internal relative imports preserved |
| `01_baselines/src/` | `rakuten/models/baselines/` | FlatHydra included |
| `02b_hlpd/src/` | `rakuten/models/hlpd/` | Mehdi Aithamma's contribution |
| `03_distributed/src/` | **excluded from the deliverable** | See note below |
| `02_hydra/configs/` | `configs/hydra/` | YAML outside the package |
| `01_baselines/configs/` | `configs/baselines/` | |
| `02b_hlpd/configs/` | `configs/hlpd/` | |
| `00_eda/` | `notebooks/eda/` | Outside the package |
| `01_baselines/notebooks/` | `notebooks/baselines/` | Outside the package |

---

## Target Imports

After `pip install -e .` (or installation in the venv), imports are clean from
any working directory:

```python
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.common.metrics import compute_all_metrics

from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.models.hydra.models import HYDRALocal, create_hydra_model
from rakuten.models.hydra.losses import HYDRALoss
from rakuten.models.hydra.config import map_config

from rakuten.models.baselines.flat_trainer import FlatHydraTrainer
from rakuten.models.baselines.flat_model import FlatHydra

from rakuten.models.hlpd import HLPDConfig, HLPDTrainer
from rakuten.models.hlpd.losses import ZLPRLoss
```

---

## Adding a HYDRA Variant

To integrate a new variant (e.g. `bilinear`):

- Create the class in `rakuten/models/hydra/models.py` by inheriting from `HYDRABase`,
  then register it in the `HYDRA_VARIANTS` dictionary (line ~249).
- If the variant produces a unified output (like `global`/`nested`), handle the case
  in `HYDRALoss.__init__` and `forward` (`rakuten/models/hydra/losses.py`) via the
  `architecture` parameter.
- Add a test in `tests/test_hydra_models.py` (forward pass, output dimensions).
- Declare a YAML config in `configs/hydra/` if the hyperparameters differ from `default.yaml`.

---

## Note on the Distributed Pipeline

Distributed training (DDP, DistributedDataParallel) was explored during the
project (`03_distributed/`) but **left out of the deliverable** in favor of
single-node DataParallel, which proved sufficient on the available GPU nodes
(P100 2×16 GB, 3090 2-4×24 GB) and far simpler to maintain.

The DDP code (~2,000 lines: `trainer_ddp.py`, `ddp_utils.py`,
`data_sharding.py`, `error_analysis.py`, `predict.py`, `main.py`) is
**available at the git tag `archive/distributed-pipeline`** and can be
retrieved at any time via:

```bash
git checkout archive/distributed-pipeline -- 03_distributed/
```

---

## Note on `hybrid_system/`

The `hybrid_system/` folder is the **separate contribution of a teammate**
(CamemBERT + XGBoost), developed in parallel with the `rakuten` package. It is
**not integrated into the package** (`pip install -e .` does not expose it) and
does not depend on the `rakuten/` code. Its evaluation and results are documented
in its own files.
