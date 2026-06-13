# Improvements — Beating the 83.54% Hybrid Baseline

Three targeted experiments to push the hybrid CamemBERT + XGBoost system past its
`max_conf` baseline. **Result: the combined system beats the baseline on every metric.**

## Final results (full 1.19M test set)

| System | Accuracy | F1-macro | F1-weighted | Best config |
|--------|----------|----------|-------------|-------------|
| **v1 baseline** (run2 + XGB v1) | 0.8354 | 0.6321 | 0.8312 | — |
| Focal L2 + XGB v1 | 0.8324 | 0.6548 | 0.8307 | gated, thr 0.8 |
| run2 + **XGB v2** (concat) | 0.8467 | 0.6608 | 0.8440 | max_conf |
| **Focal L2 + XGB v2** (final) | **0.8476** | **0.6722** | **0.8463** | max_conf |

**Δ of final system vs baseline: +1.22% accuracy, +4.01% F1-macro, +1.51% F1-weighted.**

The improvements compound: combined (0.8476) > XGB v2 alone (0.8467) > focal alone (0.8324).
XGBoost v2 drives the accuracy gain; focal adds the extra F1-macro on rare classes.
`max_conf` is the optimal decision mode at every threshold (no threshold tuning needed).

---

## Where each improvement plugs in

```
                 text
                  │
   ┌──────────────┴───────────────┐
   ▼                              ▼
[CamemBERT L2 flat]        [L1 router → XGBoost]
 (6067-way)                 (per-L1 specialist)
   │  ◄── Step 1: Focal      │  ◄── Step 2: richer
   │      loss retrain       │      embeddings (1536d)
   └──────────────┬───────────────┘
                  ▼
         max_conf decision     ◄── Step 3: meta-learner (tested, rejected)
                  ▼
            final L2 label
```

---

## Step 1 — Focal Loss + Label Smoothing (CamemBERT L2)

**Problem:** F1-macro (57→63%) lags accuracy (82→83.5%) because the flat model is weak on rare subcategories.

**Solution:** retrain CamemBERT L2 with
- **Focal Loss** (γ=2) — down-weights easy/confident examples, focuses on hard/rare ones
- **Label Smoothing** (0.1) — prevents overconfidence, improves calibration
- **Class weights** — inverse frequency, capped at 10×, normalized to mean=1

```bash
sbatch improvements/scripts/run_focal_train.sh   # ~9h, 1 GPU, --resume supported
```

Starts from `checkpoints/l2_camembert_run2/best`, outputs `checkpoints/l2_camembert_focal/checkpoint-65316`.

**Result:** standalone the focal model is weaker on accuracy (0.8100) but stronger on rare
classes. In the hybrid it lifts F1-macro by ~+1.1pp over XGB v2 alone (0.6608 → 0.6722).

---

## Step 2 — XGBoost with Better Embeddings (v2)  ⭐ biggest single win

**Problem:** XGBoost v1 uses only the L1 encoder's embeddings (768d, trained for the 157-class L1 task), which miss fine L2 distinctions.

**Solution:** feed XGBoost the **concatenation of L1 + L2 CLS embeddings (1536d)**. The BERT models stay frozen — only the ~149 per-L1 XGBoost trees are trained on the richer features.

```bash
# Default: concat (recommended)
sbatch improvements/scripts/run_xgboost_v2.sh        # extract (cached) + train 149 models

# Or test alternative feature sets:
EMB_MODE=l2_only sbatch improvements/scripts/run_xgboost_v2.sh
```

Embeddings are cached to `embeddings/{split}_emb_{mode}.npz` (atomic write — a crash never
leaves a corrupt cache). Outputs `checkpoints/l2_xgboost_v2/` (149 models).

**Result:** beats the v1 baseline on all three metrics alone — 0.8467 acc / 0.6608 f1m / 0.8440 f1w.

---

## Step 3 — Meta-Learner Stacking  ⚠️ tested, **rejected**

**Idea:** replace the global `max_conf` rule with a learned classifier (LogisticRegression /
GradientBoosting) over `[cam_conf, xgb_conf, conf_diff, conf_ratio, agree]` that decides
per-sample when to trust XGBoost.

```bash
sbatch improvements/scripts/run_meta_learner.sh  # <5 min, CPU only
```

**Result:** **worse than `max_conf`** — meta acc = 0.8331 (vs 0.8354), F1-macro = 0.6108
(vs 0.6321). The 5 confidence features carry too little signal, and `max_conf` is already
near-optimal as a decision rule. **Excluded from the final system.** Useful negative result:
any real gain must come from *better component models*, not a smarter decision layer.

---

## How each improvement was measured (isolated sweeps)

Each evaluation swaps **one box** and leaves the rest identical to the baseline, so we know
which change is responsible for any gain:

| Sweep | L2 prediction | L2 emb (for XGB) | XGBoost | Result file |
|-------|--------------|------------------|---------|-------------|
| focal | focal | — | v1 (768d) | `results/threshold_sweep_results_focal.json` |
| v2 | run2 | run2 | v2 (1536d) | `results/threshold_sweep_results_v2.json` |
| combined | **focal** | **run2** | **v2 (1536d)** | `results/threshold_sweep_results_combined.json` |

The combined sweep loads **three** CamemBERT models: focal makes the L2 prediction, run2
generates the L2 embedding XGBoost v2 was trained on, and L1 routes + supplies the L1 embedding.

---

## Status

| Step | Component | Status | Outcome |
|------|-----------|--------|---------|
| 1 | Focal CamemBERT L2 | ✅ done | `checkpoint-65316`, lifts F1-macro |
| 2 | XGBoost v2 (concat) | ✅ done | 149 models, beats baseline on all metrics |
| 3 | Meta-learner | ❌ rejected | worse than max_conf |
| — | **Combined (final)** | ✅ done | **0.8476 / 0.6722 / 0.8463** |

---

## Files

```
improvements/
├── config.json                    Parameters & final results
├── README.md                      This file
├── results/
│   ├── threshold_sweep_results_focal.json
│   ├── threshold_sweep_results_v2.json
│   └── threshold_sweep_results_combined.json
└── scripts/
    ├── train_l2_focal.py          Step 1: CamemBERT focal-loss training (--resume)
    ├── run_focal_train.sh         SLURM: 36h cap, 1 GPU
    ├── train_xgboost_v2.py        Step 2: XGBoost on L1+L2 concat embeddings
    ├── run_xgboost_v2.sh          SLURM: 36h cap, 1 GPU, 64G RAM, EMB_MODE=concat
    ├── train_meta_learner.py      Step 3: meta-learner stacking (rejected)
    └── run_meta_learner.sh        SLURM: 1h, CPU only
```

> Model weights, embeddings (`*.npz`), and `*.pkl` caches are **not** tracked in git
> (see repo `.gitignore`). Only code and configs are versioned here.
