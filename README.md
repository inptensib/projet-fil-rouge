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
python3 -m venv venv && source venv/bin/activate
# CPU or another CUDA version? Install torch first (see the top of requirements.txt), then:
pip install -e .
python3 -m pytest tests/ -q       # smoke tests (syntax + imports) — no data, no GPU
```
Python ≥ 3.11.

## 2. Download everything in one shot (HuggingFace, READ token)

The **`rakuten-delivery`** repo (private, owner `TahianaAndriambahoaka`) is a
self-contained bundle (~9 GB): the **v2 dataset** *and* the **ONNX models** in a
**single** download. Ask the owner for a **READ token**, then:

```bash
export HF_TOKEN="hf_..."                              # READ token from the owner
python3 scripts/download_delivery.py                  # everything: dataset v2 + ONNX (~9 GB)
python3 scripts/download_delivery.py --onnx-only      # ONNX models only (~1.8 GB)
python3 scripts/download_delivery.py --dataset-only   # v2 dataset only (~7.2 GB)
```

Everything lands in `rakuten-delivery/`:

```
onnx/flat_v2/    retained model (CamemBERT Flat v2) — ONNX + tokenizer + encoders
onnx/hydra_v2/   the 3 HYDRA models — ONNX + the hierarchical mask matrix
dataset_v2/      train / val / test splits (JSONL) + synthetic_ids.txt
```

> **PyTorch checkpoints (`.pt`) are *not* in the bundle** — inference here is ONNX.
> The optional PyTorch path ([`scripts/flat_inference/predict.py`](scripts/flat_inference/))
> needs the original `.pt` from the separate `rakuten-models` repo
> (`scripts/flat_inference/download_weights.py`, requires its own access).

## 3. Run inference (offline, ONNX — no PyTorch, no HuggingFace at runtime)

```bash
cd scripts/flat_inference && pip install -r requirements.txt
python3 predict_onnx.py --model-dir ../../rakuten-delivery/onnx/flat_v2 \
    --designation "40 Pcs Automatique Poulet Buveur D'eau Volaille Abreuvoir Potable Mamelon Élevage Accessoires"
# → level1: "Accessoires animalerie" (0.9997)  ·  level2: "Abreuvoir" (0.9991)
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
export DATA_DIR="$PWD/rakuten-delivery/dataset_v2"   # the dataset from the delivery bundle (§2)

# CamemBERT Flat v2 (the retained model)
python3 scripts/run_flat_hydra.py --split-dir "$DATA_DIR/" \
    --config configs/baselines/flat_hydra_camembert_v2.yaml      # → models/v2/flat_hydra/

# HYDRA v2 — local | nested | global
python3 scripts/run_train.py --split-dir "$DATA_DIR/" \
    --config configs/hydra/hierarchy.yaml --variant local        # → models/v2/HYDRA_local_CamemBERT/

# Evaluate
python3 scripts/eval_hierarchy.py --split-dir "$DATA_DIR/" --config configs/hydra/hierarchy.yaml
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
scripts/      download_delivery · run_flat_hydra · run_train · eval_hierarchy
              slurm/ · flat_inference/ (ONNX) · hydra_inference/ (ONNX + mask)
docs/         flat_hydra_conception.md + diagrams/
results/      retained test metrics (JSON)
REPRODUCE.md  re-train & re-test runbook (with expected metrics)
```

---

<div align="center">

**[Tahiana H. Andriambahoaka](https://github.com/tahianahajanirina) · [Mehdi Aithamma](https://github.com/MehdiiAH) · [Younes Boutrouft](https://github.com/inptensib) · Amir H. Kashani Rahimi**

</div>
