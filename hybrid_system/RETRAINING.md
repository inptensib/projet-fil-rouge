# hybrid_system/ — retraining guide

**Status (be honest):** this folder has been made **portable** (no machine-specific
paths) and its **dependencies are declared** (`requirements.txt`), so it is *ready
to be retrained*. It has **not** been run end-to-end here, and it is **not** the
retained product model — the deliverable is **CamemBERT Flat v2** (F1 L2 ≈ 78.9 %,
see `../scripts/flat_inference/`). This hybrid track peaks at **F1 L2 ≈ 69 %** and
is exploratory (separate contribution by Younes Boutrouft + Amir's improvements;
unaudited, untested).

## What it is

A different strategy: CamemBERT + **XGBoost** per-L1 experts + **tail experts**
(kNN/NCM) combined by `max_conf`. Best result (`improvements/`): **0.8487 accuracy
/ 0.6912 F1-macro** on the v2 test set.

## Prerequisites (must be supplied before retraining)

These are **not in the repo and not on Hugging Face** — they come from the authors
(or must be regenerated):

1. **Python env** — `pip install -r requirements.txt` (the SetFit steps need the
   pinned versions noted in that file; use a dedicated env if they conflict).
2. **Dataset** — the v2 splits `train/val/test_split.jsonl` (from the dataset hub
   repo `datasets_finaux/`). The scripts expect them under the project root the way
   the authors laid it out (`dataset/`); check each script's `BASE_DIR` / `DATA_DIR`
   constants and place the splits accordingly.
3. **Base checkpoints** — `l1_classifier/`, `l2_camembert_run2/`. Per the authors'
   own runbook these are **Step 0, "not retrained" — a prerequisite**. They are the
   biggest gap: without them the pipeline cannot run.
4. **Cached embeddings** — `embeddings/*.npz` (1536-d L1+L2 concat). Produced by an
   earlier step; every step after the first reuses them.

## Run order (authors' optimal path — see `improvements/README.md`)

| # | Step | Script | Compute |
|---|------|--------|---------|
| 0 | Base checkpoints `l1_classifier`, `l2_camembert_run2` | *(prerequisite — not retrained)* | — |
| 1 | Focal CamemBERT L2 (the L2 predictor) | `improvements/scripts/run_focal_train.sh` | 1 GPU, ~hours (only heavy step) |
| 2 | XGBoost v3_balanced (per-L1, rare-weighted) | `improvements/scripts/run_xgboost_v3_balanced.sh` | ~2 h, reuses cached embeddings |
| 3 | Evaluate `max_conf` | `improvements/scripts/eval_threshold_sweep_combined.py` | CPU |
| 4 | kNN/NCM tail voter | `improvements/scripts/run_knn_tail.sh` | CPU |

`Gated+Hybrid/` is the earlier final system (F1 L2 = 63.2 %); `Specialists/`
holds ablations (SetFit, LoRA, generative — excluded from the final).

## Portability (what was fixed here)

All `*.sh` now **auto-detect the project root** and use a portable Python:

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
```

Override with `export PROJECT_ROOT=...`, `export PYTHON=...`, or `export VENV=...`.
The old hard-coded `~/Rakuten_Project` / `python3.12` paths were removed; bare
script invocations were rewritten to their real `hybrid_system/` paths.

## Honest caveat

This pipeline is multi-step and depends on prerequisite trained artifacts that are
not shipped. Retraining is feasible only once items 2–4 above are in place. The
results were not independently verified. For anything beyond this, contact the
original authors.
