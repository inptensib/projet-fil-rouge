<div align="center">

<img src="docs/screenshots/hero.jpg" alt="HieraCat" width="80%" style="border-radius:12px; margin-bottom:16px"/>

# Rakuten Hierarchical Product Classification

**8M+ products · 157 categories · 5 777 subcategories · CamemBERT**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-312%20passed-brightgreen?logo=pytest&logoColor=white)
![SLURM](https://img.shields.io/badge/SLURM-Multi--GPU-orange)

> **Télécom Paris** (IP Paris) × **Rakuten France**

</div>

---

## The project in 30 seconds

A product arrives with a title and a description in French. The model predicts
**simultaneously** its L1 category among 157 and its L2 subcategory among 5 777:

```
"iPhone 12 transparent silicone case"
          │
          ▼
    ┌─ HYDRA ─┐
    │CamemBERT│
    │ 12 layers │
    └────┬────┘
    ┌────┴────┐
    ▼         ▼
 L1: 157   L2: 5 777
Electronics  Phone case
```

Three model families compared:

| Family | Approach | Best F1 L2 (v1, test) |
|:--------|:---------|:-------------------:|
| Baselines | TF-IDF, Word2Vec, fastText + classifiers | 42.9 % |
| **FlatHydra** | CamemBERT + single L1+L2 head | **71.7 %** |
| HYDRA | CamemBERT multi-head (EMNLP 2025) | 66.8 % |
| HLPD | Hierarchical decoder (ECIR 2025) | 61.5 % |

> **v1 → v2 note (measured)**: these figures are on the **v1** dataset (train/test
> leakage not purged). On the de-leaked **v2** dataset, the re-evaluated FlatHydra
> CamemBERT yields **78.9 %** L2 F1 — not directly comparable to v1 (L2 classes
> go from 6 126 to 5 777). The leakage's own effect, measured at the L1 level
> (same 157 classes), is **~1 pt** (v1 slightly inflated).
> Details: [docs/METRICS.md](docs/METRICS.md).

---

## Architecture

Three architecture families were explored — all built on a fine-tuned
**CamemBERT** encoder: the retained **CamemBERT Flat** (single unified head), the
multi-head **HYDRA**, and a separate **hybrid** track (CamemBERT + per-L1
specialists).

### Retained model — CamemBERT Flat

A **single unified head** predicts L1 + L2 jointly. The projection is applied on
the token sequence **before** pooling (faithful to the official HYDRA baseline),
and the 5 934 output logits are sliced into L1 (`[0:157]`) and L2 (`[157:]`).

![CamemBERT Flat — retained model](docs/diagrams/arch_flat.png)

### HYDRA (multi-head) + hierarchical masking

Faithful to **HYDRA** (Karl & Scherp, EMNLP 2025): a shared **text encoder** feeds
**k local heads** (one per hierarchy level — here **k = 2**: L1 and L2). Two of the
three variants add an optional **(k+1)-th head** that predicts the flat union of
all labels (training only). A CLS pool + a 768→1536 projection build the shared
embedding the heads sit on.

![HYDRA — local heads + optional (k+1)-th head](docs/diagrams/arch_hydra.png)

**3 variants**, mapping directly to the paper's (a)/(b)/(c) — they differ only in
the optional **Head k+1** (training only); inference always uses the two local
heads + the mask below:

| Variant | Head k+1 (training only) | Paper |
|:---------|:-----|:----------------|
| `local` | none | (a) Local Heads Only |
| `global` | predicts all 5 934 labels from the embedding | (b) Local + Global Head |
| `nested` | predicts all 5 934 labels from the `[L1;L2]` logits | (c) Local + Nested Head |

At **inference** (all variants use the two local heads), a **hierarchical mask**
keeps only the L2 children of the predicted L1 — lifting the hierarchical
consistency rate (HCR) to ~100 %:

![HYDRA — hierarchical masking at inference](docs/diagrams/arch_hydra_mask.png)

The mask uses a `parent_child_matrix` (157 × 5 777), **multi-parent aware**:
545 L2 classes have more than one valid L1 parent.

### Explored alternatives — hybrid systems

A separate experimental track ([`hybrid_system/`](hybrid_system/), unaudited)
combined CamemBERT with per-L1 specialists. Two designs were explored, both ruled
out in favour of the simpler CamemBERT Flat.

**Gated routing** — route each product to a single per-L1 specialist, gated by L1
confidence + class frequency. A wrong L1 route hands the product to the wrong
specialist, so L2 collapses: the L1 prediction decides everything, no recourse —
the L1 route is the bottleneck.

![Hybrid gated routing](docs/diagrams/arch_hybrid_gated.png)

**Max-confidence** — two flat experts run in parallel (CamemBERT-L2 + per-L1
XGBoost on 1536-d embeddings), combined by keeping the more confident; a kNN tail
expert overrides on confident rare-class hits. A wrong L1 route only makes XGBoost
abstain, so the routing-free flat predictor still stands. Competitive, but heavier
for no gain over the plain CamemBERT Flat.

![Hybrid max-confidence](docs/diagrams/arch_hybrid_maxconf.png)

> **Key finding**: the flat *FlatHydra* head (CamemBERT + a single L1+L2 head)
> achieves the project's best L2 F1 — above the multi-level HYDRA heads **and the
> hybrid systems**. Despite its name, FlatHydra is **not** a HYDRA variant: it is a
> strong baseline (filed under `rakuten/models/baselines/`).

---

## Key results (v1, test)

> L1 F1 not reported for the deep models — target metric = L2 F1.

| Model | F1 L1 | F1 L2 | HCR |
|:-------|:-----:|:-----:|:---:|
| TF-IDF + LinearSVC | 83.26 % | 42.75 % | 63.41 % |
| FastText Supervised | 82.60 % | 10.01 % | 29.29 % |
| Word2Vec + LinearSVC | 60.24 % | 22.27 % | 25.06 % |
| **FlatHydra CamemBERT** | **—** | **71.7 %** | — |
| HYDRA local CamemBERT | — | 66.8 % | > 95 % |
| HLPD CamemBERT | — | 61.5 % | — |

---

## Synthetic data pipeline

Rakuten's catalogue has a heavy **long tail**: many L2 classes hold very few
products, which caps the macro-F1. The pipeline
([`scripts/synthetic_pipeline/`](scripts/synthetic_pipeline/)) rebalances **train
only** so every L1 (157) and every L2 (5 777) class reaches **≥ 100 products** —
**val and test stay frozen** (every generated row is hash-checked against them, so
nothing leaks).

The **LLM (Claude Opus) is in the loop at generation time**: it reads each class
and decides its vocabulary (keywords) and its listing shapes (templates). Python then does the
mechanical work — filling templates, duplicating real rows for raw volume, and
validating.

![Synthetic data — generation pipeline](docs/diagrams/pipeline_synth_generation.png)

![Synthetic data — template variants](docs/diagrams/pipeline_synth_templates.png)

Two strategies, composed to exactly 100 rows per class (generated first, then
duplication to top up):

| Kind | Rows | How |
|:------|-----:|:----|
| `SYNTH-OVS` | 119 510 | duplication (L2) |
| `SYNTH-GEN` | 7 044 | LLM templates, anchored in the class's real vocabulary |
| `SYNTH-ORPH` | 2 900 | LLM templates, 29 orphan classes (no real rows) × 100 |
| `SYNTH-OVL` | 187 | duplication (L1) |
| **Total** | **129 641** | added on top of the real training data |

Synthetic rows are identifiable by the non-numeric `SYNTH-` prefix on `productid`.
The whole chain runs **without an LLM** for testing (`RAKUTEN_LLM_OFFLINE=1`); see
[`scripts/synthetic_pipeline/README.md`](scripts/synthetic_pipeline/README.md) and
[`WORKED_EXAMPLE.md`](scripts/synthetic_pipeline/WORKED_EXAMPLE.md).

---

## Quick Start

> **New to the project?** Follow [ONBOARDING.md](ONBOARDING.md) — installation,
> access to data and models, first training run, SLURM.
> Detailed reference: [docs/PIPELINE.md](docs/PIPELINE.md) (which script for which model)
> and [docs/METRICS.md](docs/METRICS.md) (metrics and provenance of the results).

```bash
# 1. Clone and install
git clone <repo-url> && cd rakuten-classification
python -m venv venv && source venv/bin/activate
# CPU or another CUDA version? Install torch first via the appropriate PyTorch index
# (see the comment at the top of requirements.txt), then:
pip install -e .

# 2. Verify the installation — no data or GPU required
python -m pytest tests/ -q          # 312 tests should pass

# 3. Download the data (HF token required — see the next section)
export HF_TOKEN="hf_..."
python scripts/download_dataset.py

# 4. Local smoke test — requires the dataset (step 3 above)
# Duration: a few minutes to tens of minutes depending on the machine
# Without an HF token: stop after the 312 tests, that is the normal fallback path.
python scripts/run_train.py --split-dir dataset/ --n-samples 5000 \
    --config configs/hydra/default.yaml

# 5. Full training on the SLURM cluster
sbatch --job-name=hydra-local --array=0-9%1 scripts/run_train_variant.slurm local
```

---

## Access to data and models

> **Private HuggingFace repos** — a read HF token is required.
> Contact: `TahianaAndriambahoaka` (owner of the repos).

### Without access (fallback path)

You can install the package and run the 312 tests **without data or GPU**:

```bash
pip install -e .
python -m pytest tests/ -q   # 312 passed
```

What this covers: imports, model architecture, metrics, configs, utilities.
What this does not cover: real training and evaluation.

### Data (dataset v2)

Private HF repo: [`TahianaAndriambahoaka/rakuten-dataset`][hf-dataset]
Subfolder: `datasets_finaux/` — pre-generated splits (70/15/15, stratified by L2, seed=42).

| File | Lines |
|:--------|-------:|
| `train_split.jsonl` | 5 259 035 |
| `val_split.jsonl` | 1 194 355 |
| `test_split.jsonl` | 1 194 356 |

Each line: `{"designation": "...", "description": "...", "typelabel": "L1", "producttype": "L2"}`

```bash
export HF_TOKEN="hf_..."           # read token provided by the owner
python scripts/download_dataset.py  # downloads into <repo>/dataset/
```

> **Dataset v2**: duplicate leakage purged (−314 624 net train rows).
> Measured effect of the leakage (FlatHydra): ~1 pt at L1; at L2 the v1↔v2 figures
> do not compare term for term (6 126 → 5 777 classes). See
> [docs/METRICS.md](docs/METRICS.md).

### Trained models (~63 GB, 22 models)

Private HF repo: [`TahianaAndriambahoaka/rakuten-models`][hf-models]

The simplest way — via the dedicated script:

```bash
export HF_TOKEN="hf_..."
python scripts/download_models.py                      # all models (~63 GB)
python scripts/download_models.py flat-hydra-camembert-v2   # a single model
```

Or directly in Python for a specific subfolder:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-models",
    allow_patterns="flat-hydra/*",   # replace according to the desired model
    token="hf_...",
)
```

Mapping of HF subfolders → local paths expected by the evaluation scripts:
see [ONBOARDING.md](ONBOARDING.md), "Retrieving the trained models" section.

### Delivery bundle (ONNX — offline inference)

For a one-shot handoff, everything needed to **run** the models — plus the v2
dataset — is bundled in a single private repo:
[`TahianaAndriambahoaka/rakuten-delivery`][hf-delivery] (~9 GB).

| Folder | Content |
|:--------|:---------|
| `onnx/flat_v2/` | Retained model (CamemBERT Flat v2) in ONNX + tokenizer, label encoders, standalone `predict_onnx.py` |
| `onnx/hydra_v2/` | The 3 HYDRA hierarchy models in ONNX + hierarchical mask matrix + `predict_onnx.py` |
| `dataset_v2/` | train / val / test splits (JSONL) + `synthetic_ids.txt` |

```bash
export HF_TOKEN="hf_..."                            # read token provided by the owner
python scripts/download_delivery.py                 # everything (~9 GB)
python scripts/download_delivery.py --onnx-only     # ONNX models only (~1.8 GB)
python scripts/download_delivery.py --dataset-only  # dataset v2 only (~7.2 GB)
```

**Inference is fully offline** — `onnxruntime` only, no PyTorch and no Hugging
Face at runtime (the tokenizer and label encoders are bundled):

```bash
cd rakuten-delivery/onnx/flat_v2
pip install -r requirements.txt
python predict_onnx.py --model-dir . \
    --designation "Cage à hamster 3 étages" --description "Métal, 60x40 cm"
```

The same export/inference code lives in the repo under
[`scripts/flat_inference/`](scripts/flat_inference/) (retained model) and
[`scripts/hydra_inference/`](scripts/hydra_inference/) (HYDRA + hierarchical mask).

### SLURM cluster

The `scripts/*.slurm` scripts assume the `P100` (2×16 GB)
and `3090` (2-4×24 GB) partitions of the Télécom Paris cluster.
On a different infrastructure, adapt the `#SBATCH` directives and the variables:

```bash
export PROJECT_ROOT="/path/to/rakuten-classification"
export VENV_ACTIVATE="/path/to/venv/bin/activate"
export DATA_DIR="/path/to/dataset"
```

[hf-dataset]: https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset
[hf-models]: https://huggingface.co/TahianaAndriambahoaka/rakuten-models
[hf-delivery]: https://huggingface.co/TahianaAndriambahoaka/rakuten-delivery

---

## Metrics

All defined in `rakuten/common/metrics.py`, documented in [docs/METRICS.md](docs/METRICS.md).

| Metric | Description |
|:---------|:------------|
| `f1_l1_macro` | Macro F1 over the 157 L1 categories |
| `f1_l2_macro` | Macro F1 over the 5 777 L2 subcategories — **main metric** |
| `HCR` | Hierarchical consistency rate: % of predictions where L2 is a valid child of L1 |
| `constrained_f1` | L2 F1 counted only if the L1 parent is correct |

Goal: maximize `f1_l2_macro` with HCR > 95 %.

Each epoch also logs: L1/L2 confusion matrices, per-class F1 (sorted barplots),
and an HTML explainability report (Gradient × Input) available in MLflow
(`docker compose up mlflow -d`).

---

## HYDRA configuration

Source of truth: [`configs/hydra/default.yaml`](configs/hydra/default.yaml)

| Parameter | Value | | Parameter | Value |
|-----------|--------|-|-----------|--------|
| Encoder | `camembert-base` | | `batch_size` | 64 |
| `max_length` | 256 | | `lr` | 3.5e-5 |
| Warmup | 10% (linear) | | Weight decay | 0.01 |
| Gradient clip | 1.0 | | Early stop | patience 2 |
| `dropout` | 0.3 | | `hidden_dim` | 256 |

Hierarchical variant: [`configs/hydra/hierarchy.yaml`](configs/hydra/hierarchy.yaml) — enables `condition_l2_on_l1` (teacher forcing) and `check_hierarchy` (mask at inference)

---

## Structure

```
rakuten-classification/
├── rakuten/                   Installable Python package (pip install -e .)
│   ├── common/                data_loader, metrics, training_utils, utils
│   └── models/
│       ├── baselines/         TF-IDF, Word2Vec, fastText, FlatHydra
│       ├── hydra/             HYDRA (EMNLP 2025): models, losses, trainer, config
│       └── hlpd/              HLPD (ECIR 2025) — Mehdi Aithamma
├── configs/                   YAML outside the package (source of truth)
│   ├── hydra/                 default.yaml, hierarchy.yaml, title_only*.yaml
│   ├── baselines/             flat_hydra*.yaml
│   └── hlpd/                  hlpd_3090.yaml
├── scripts/                   Python CLI, SLURM scripts, evaluation, comparison
│   ├── flat_inference/        Standalone + ONNX inference of the selected model (CamemBERT Flat v2)
│   ├── hydra_inference/       ONNX inference of the 3 HYDRA hierarchy models (+ hierarchical mask)
│   └── synthetic_pipeline/    Synthetic data generation (LLM + templates + duplication)
├── notebooks/
│   ├── eda/                   Exploratory analysis
│   └── baselines/             fastText, TF-IDF, Word2Vec (notebooks)
├── docs/                      ARCHITECTURE.md, PIPELINE.md, METRICS.md, EXPERIMENTS.md
├── results/                   HTML reports, per-model metrics
├── tests/                     312 pytest tests (no data or GPU)
├── hybrid_system/             Separate experimental contribution — outside the package, not audited (see its own README)
├── ONBOARDING.md              Complete getting-started guide
└── docker-compose.yml         MLflow, tests, training
```

Detailed architecture and old → new mapping table:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — branching, quality pipeline (tests + lint), commit format.

---

## References

- **HYDRA** — Karl & Scherp, *Hierarchical Multi-label Classification*, EMNLP 2025 [[GitHub]](https://github.com/FKarl/HYDRA)
- **HLPD** — Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder*, ECIR 2025

---

<div align="center">

**[Tahiana H. Andriambahoaka](https://github.com/tahianahajanirina) · [Mehdi Aithamma](https://github.com/MehdiiAH) · [Younes Boutrouft](https://github.com/inptensib) · Amir H. Kashani Rahimi**

</div>
