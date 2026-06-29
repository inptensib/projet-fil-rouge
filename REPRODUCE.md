# Reproduce — retrain & re-test the delivered models

Short runbook for the two things needed to take the project further:
**(1) re-run our test evaluations** and **(2) re-train the retained models**.
It covers only the **delivered (retained) models**; the full experiment matrix
(baselines, title-only, mBERT, v1…) is in [docs/PIPELINE.md](docs/PIPELINE.md).

Delivered models:

- **CamemBERT Flat v2** — the retained production model (best L2 macro-F1).
- **HYDRA v2** (`local`, `nested`, `global`) — multi-head + hierarchical mask.

## 0. Prerequisites

```bash
# Python env + package
python3 -m venv venv && source venv/bin/activate
pip install -e .          # CPU / other CUDA: install torch first (see requirements.txt)

# v2 dataset (HF read token) — or reuse the delivery bundle's dataset_v2/
export HF_TOKEN="hf_..."
python3 scripts/download_dataset.py      # → dataset/  (train/val/test _split.jsonl)
export DATA_DIR="$PWD/dataset"          # SLURM scripts read $DATA_DIR (propagated by sbatch)
```

- **GPU** required for training (we used P100 2×16 GB / RTX 3090); evaluation fits on one GPU.
- Training fine-tunes **`camembert-base`** (downloaded from HF on first use). The
  scripts set `TRANSFORMERS_OFFLINE=1`, so either pre-cache the encoder
  (`python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('camembert-base')"`)
  or `unset TRANSFORMERS_OFFLINE`.
- The SLURM scripts default paths to our cluster — override `PROJECT_ROOT` /
  `VENV_ACTIVATE` / `DATA_DIR` (and adapt the `#SBATCH` directives), **or** run the
  `python3 …` entry points directly on any GPU machine.

## 1. Re-train

### CamemBERT Flat v2 (the retained model)

```bash
# SLURM (auto-resume job array)
sbatch --job-name=flat-cam-v2 --array=0-9%1 scripts/slurm/run_flat_camembert_v2.slurm
# …or directly on a GPU box:
python3 scripts/run_flat_hydra.py --split-dir "$DATA_DIR/" \
    --config configs/baselines/flat_hydra_camembert_v2.yaml
# → models/v2/flat_hydra/best_model.pt
```

### HYDRA v2 — local / nested / global

```bash
for V in local nested global; do
  sbatch --job-name=hydra-$V-v2 scripts/slurm/run_hydra_3090_v2.slurm "$V"
done
# …or directly (one variant at a time):
python3 scripts/run_train.py --split-dir "$DATA_DIR/" \
    --config configs/hydra/hierarchy.yaml --variant local      # local | nested | global
# → models/v2/HYDRA_<variant>_CamemBERT/
```

## 2. Re-test (re-run the evaluations)

```bash
# CamemBERT Flat v2  → results/flat_hydra_v2/test_metrics.json
sbatch --job-name=eval-cam-v2 scripts/slurm/run_eval_flat_hydra_generic.slurm
#   (defaults point at models/v2/flat_hydra/best_model.pt on the v2 dataset —
#    see the header of that script to evaluate another checkpoint)

# HYDRA v2 (3 variants, base + hierarchical mask) → console + JSON
python3 scripts/eval_hierarchy.py --split-dir "$DATA_DIR/" \
    --config configs/hydra/hierarchy.yaml --model-dir models/v2 --eval-batch-size 2048
```

Every evaluation reports `f1_l1_macro`, `f1_l2_macro`, `constrained_f1` and `HCR`.

## 3. Expected TEST metrics (v2) — to verify your re-run

| Model | F1 L1 | **F1 L2 macro** | Constrained F1 | HCR |
|:--|:--:|:--:|:--:|:--:|
| **CamemBERT Flat v2** (retained) | 92.47 | **78.93** | 79.90 | 85.17 |
| HYDRA local (+ mask) | 92.60 | 77.38 | 79.31 | 87.04 |
| HYDRA nested (+ mask) | 92.59 | 76.41 | 78.56 | 87.02 |
| HYDRA global (+ mask) | 92.91 | 77.29 | 79.17 | 86.99 |

HYDRA **multi-parent HCR = 100 %** by construction (the mask keeps only valid L2
children of the predicted L1; the plain HCR ≈ 87 % counts an L2 as consistent only
under its single canonical parent). Figures are traced to
`results/*/test_metrics.json` and `results/hydra_hierarchy_v2_metrics.json` — see
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/METRICS.md](docs/METRICS.md).

## 4. Just run inference (no training)

The delivered **ONNX** models run fully offline (onnxruntime, no PyTorch, no
Hugging Face) — see the `rakuten-delivery` bundle and
[`scripts/flat_inference/`](scripts/flat_inference/) /
[`scripts/hydra_inference/`](scripts/hydra_inference/).
