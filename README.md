<div align="center">

# Rakuten Product Classification — Delivery

**Retained models · dataset v2 · download · inference · re-training**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/Inference-ONNX-005CED?logo=onnx&logoColor=white)

> **Télécom Paris** (IP Paris) × **Rakuten France**

</div>

---

This is the **curated handoff branch** for Rakuten: only the **retained models**,
the **v2 dataset**, and everything needed to **download, run, and re-train** them.
The exploration tracks (other architectures, classic baselines, the synthetic-data
pipeline, dev tooling, the full experiment matrix) are intentionally left out —
they live on the `main` branch.

## Retained models

Two-level classification of product listings (mostly French) into **157 L1
categories** (`typelabel`) and **5 777 L2 subcategories** (`producttype`), on the
de-leaked **v2** dataset. Encoder: `camembert-base`.

| Model | Approach | **Test F1 L2** | F1 L1 | HCR |
|:--|:--|:--:|:--:|:--:|
| **CamemBERT Flat v2** ⭐ | single unified L1+L2 head | **78.9 %** | 92.5 % | 85.2 %¹ |
| HYDRA local (+ mask) | 2 local heads + hierarchical mask | 77.4 % | 92.6 % | ≈100 %² |
| HYDRA nested (+ mask) | + nested head (training only) | 76.4 % | 92.6 % | ≈100 %² |
| HYDRA global (+ mask) | + global head (training only) | 77.3 % | 92.9 % | ≈100 %² |

¹ Flat applies **no** hierarchical mask by default → genuine HCR ≈ 85 %
  (`--enforce-hierarchy` constrains L2 to the predicted L1's children).
² HYDRA's mask keeps only valid L2 children of the predicted L1 → **every**
  prediction is hierarchically consistent (verified 100 % on real test rows). The
  single-canonical-parent figure in `test_metrics` reads 87 % — understated, as
  545 of the 5 777 L2 classes have several valid L1 parents.

**Retained production model: CamemBERT Flat v2** (best L2 macro-F1). Figures are
traced to [`results/flat_hydra_v2/test_metrics.json`](results/flat_hydra_v2/test_metrics.json)
and [`results/hydra_hierarchy_v2_metrics.json`](results/hydra_hierarchy_v2_metrics.json).

## Architecture

**CamemBERT Flat** — a single unified head: a 768→1536 projection applied before
CLS pooling, then one head producing **5 934 logits** sliced into L1 `[0:157]` and
L2 `[157:]`. Loss = `0.5·CE(L1) + CE(L2)`.

![CamemBERT Flat](docs/diagrams/arch_flat.png)

**HYDRA** — a shared encoder feeds 2 local heads (L1, L2); at inference a
**hierarchical mask** keeps only the L2 children of the predicted L1, lifting the
hierarchical consistency rate (HCR) to ~100 %. The 3 variants differ only in an
optional training-time head.

![HYDRA](docs/diagrams/arch_hydra.png)
![HYDRA — hierarchical mask](docs/diagrams/arch_hydra_mask.png)

Detailed FlatHydra design: [docs/flat_hydra_conception.md](docs/flat_hydra_conception.md).

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
# CPU or another CUDA version? Install torch first (see the top of requirements.txt), then:
pip install -e .
python -m pytest tests/ -q       # smoke tests (syntax + imports) — no data, no GPU
```
Python ≥ 3.11.

## 2. Get the dataset and models (HuggingFace, READ token)

Three **private** HuggingFace repos (owner `TahianaAndriambahoaka`) — ask the owner
for a **READ token**, then:

```bash
export HF_TOKEN="hf_..."
python scripts/download_dataset.py                       # v2 splits  → dataset/
python scripts/download_models.py flat-hydra-camembert-v2 # one model → models/  (omit name = all)
python scripts/download_delivery.py --onnx-only          # one-shot ONNX bundle (~1.8 GB)
```

| Repo | Downloader | Content |
|:--|:--|:--|
| `rakuten-dataset` | `download_dataset.py` | v2 splits — train 5 259 035 / val 1 194 355 / test 1 194 356 |
| `rakuten-models` | `download_models.py` | trained weights (`best_model.pt` + tokenizer/encoders) |
| `rakuten-delivery` | `download_delivery.py` | one-shot bundle: the 4 ONNX models + v2 dataset |

## 3. Run inference (offline, ONNX — no PyTorch, no HuggingFace at runtime)

```bash
cd scripts/flat_inference && pip install -r requirements.txt
python predict_onnx.py --model-dir <onnx_model_dir> \
    --designation "40 Pcs Automatique Poulet Buveur d'eau Volaille Abreuvoir Mamelon Élevage" \
    --description "Abreuvoir à poulet / volaille, tétine à boire, plastique + acier inoxydable"
# → level1: "Accessoires animalerie"  ·  level2: "Abreuvoir"   (confidence ≈ 1.0)
```

> **Feed real product-listing text** — terse, keyword-heavy, as on the catalogue.
> The models were trained on that style; clean hand-written sentences are
> out-of-distribution and degrade accuracy. On real v2 test listings the four
> retained models score **~90–93 %** (L1 and L2).

- **CamemBERT Flat v2** → [`scripts/flat_inference/`](scripts/flat_inference/)
- **3 HYDRA + hierarchical mask** → [`scripts/hydra_inference/`](scripts/hydra_inference/)

Both run on `onnxruntime` only. The HYDRA bundle ships its pre-computed
`parent_child_matrix_v2.npy` so the mask works out of the box.

## 4. Re-train (optional)

Full runbook with expected metrics: **[REPRODUCE.md](REPRODUCE.md)**. In short:

```bash
export DATA_DIR="$PWD/dataset"

# CamemBERT Flat v2 (the retained model)
python scripts/run_flat_hydra.py --split-dir "$DATA_DIR/" \
    --config configs/baselines/flat_hydra_camembert_v2.yaml      # → models/v2/flat_hydra/

# HYDRA v2 — local | nested | global
python scripts/run_train.py --split-dir "$DATA_DIR/" \
    --config configs/hydra/hierarchy.yaml --variant local        # → models/v2/HYDRA_local_CamemBERT/

# Evaluate
python scripts/eval_hierarchy.py --split-dir "$DATA_DIR/" --config configs/hydra/hierarchy.yaml
```
SLURM equivalents in [`scripts/slurm/`](scripts/slurm/) (`run_flat_camembert_v2.slurm`,
`run_hydra_3090_v2.slurm`).

> **HYDRA evaluation — two non-comparable modes.** `eval_hierarchy.py` reports
> either `base` (no mask, HCR ≈ 0.86) or `hier` (with mask, HCR ≈ 1.0 *by
> construction*). The two **cannot be compared**; `--output-mode auto` (default)
> picks the right one from the config. Hierarchical models are published in `hier`.

## Metrics

The target is **`f1_l2_macro`** (macro-F1 over the 5 777 L2 classes — the long tail
is the hard part), alongside `f1_l1_macro`, `constrained_f1` (an L2 prediction
counts only if its L1 parent is correct), and **HCR** (share of predictions where
L2 is a valid child of L1). All in [`rakuten/common/metrics.py`](rakuten/common/metrics.py).

## Structure

```
rakuten/      installable package — common/ + models/{baselines (FlatHydra), hydra}
configs/      retained configs — hydra/{default,hierarchy}.yaml, baselines/flat_hydra*.yaml
scripts/      download_* · run_flat_hydra · run_train · eval_hierarchy
              slurm/ · flat_inference/ (ONNX) · hydra_inference/ (ONNX + mask)
docs/         flat_hydra_conception.md + diagrams/
results/      retained test metrics (JSON)
REPRODUCE.md  re-train & re-test runbook (with expected metrics)
```

---

<div align="center">

**[Tahiana H. Andriambahoaka](https://github.com/tahianahajanirina) · [Mehdi Aithamma](https://github.com/MehdiiAH) · [Younes Boutrouft](https://github.com/inptensib) · Amir H. Kashani Rahimi**

</div>
