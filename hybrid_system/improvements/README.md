# Improvements — Beating the 83.54% Hybrid Baseline

Targeted experiments to push the hybrid CamemBERT + XGBoost system past its
`max_conf` baseline. **Result: the combined system beats the baseline on every metric.**

## Final results (full 1.19M test set)

| System | Accuracy | F1-macro | F1-weighted | Best config |
|--------|----------|----------|-------------|-------------|
| **v1 baseline** (run2 + XGB v1) | 0.8354 | 0.6321 | 0.8312 | — |
| Focal L2 + XGB v1 | 0.8324 | 0.6548 | 0.8307 | gated, thr 0.8 |
| run2 + XGB v2 (concat) | 0.8467 | 0.6608 | 0.8440 | max_conf |
| Focal L2 + XGB v2 | 0.8476 | 0.6722 | 0.8463 | max_conf |
| Focal L2 + XGB v3 (engineered feats) | 0.8479 | 0.6728 | 0.8467 | max_conf |
| Focal L2 + XGB v3_balanced | 0.8490 | 0.6814 | 0.8480 | max_conf |
| **+ kNN/NCM tail expert** ⭐ **(standard)** | **0.8487** | **0.6912** | **≈0.848** | + margin τ=0.15 |

**Δ of current standard vs v1 baseline: +1.33% accuracy, +5.91% F1-macro, +1.68% F1-weighted.**

The gains come almost entirely from **better component models**, not the decision rule:
- **XGBoost v2** (1536d L1+L2 concat embeddings) drove the accuracy gain.
- **Focal loss** added F1-macro on rare classes.
- **XGBoost v3** (engineered text features) added almost nothing (+0.06pp f1m) — the
  embeddings already capture lengths/ratios.
- **XGBoost v3_balanced** (inverse-frequency class weighting): **+0.9pp F1-macro** over v2 —
  the trees' analogue of focal loss, directly targeting the long tail.
- **kNN/NCM tail expert** (third voter for the <100-sample bands): **+1.0pp F1-macro** at
  ~flat accuracy — a flat nearest-class-mean model that handles the ultra-rare band where
  gradient models can't.
- **Meta-learner** and the **decision-layer study** (calibration / learned gate) both
  confirmed `max_conf` is already near-optimal — no decision-rule tuning helps.

`max_conf` is the optimal decision mode at every threshold (no threshold tuning needed).

---

## ⭐ Reproduce the best result (optimal path)

Best result = **0.8487 acc / 0.6912 F1-macro** = Focal CamemBERT L2 + XGBoost v3_balanced
combined by `max_conf`, with the kNN/NCM tail voter on top. Run **in order** — every step
after the first reuses cached embeddings, so only step 1 is GPU-heavy. All scripts live in
`improvements/scripts/`; run them from the project root (they `cd ~/Rakuten_Project`
internally and expect the cached embeddings + base checkpoints there).

| # | Step | Script | Compute | Marginal gain |
|---|------|--------|---------|---------------|
| 0 | Base checkpoints `l2_camembert_run2`, `l1_classifier` | *(from baseline work — not retrained)* | — | prerequisite |
| 1 | **Focal CamemBERT L2** (the L2 predictor) | `run_focal_train.sh` | 1 GPU, ~hours · **only heavy step** | F1-macro on rare L2 |
| 2 | **XGBoost v3_balanced** (per-L1, rare-weighted) | `run_xgboost_v3_balanced.sh` | ~2h, reuses cached embeddings | +0.9 pp F1-macro |
| 3 | Evaluate `max_conf` base | `eval_threshold_sweep_combined.py` (v3_balanced env) | CPU | → 0.8490 / 0.6814 |
| 4 | **kNN/NCM tail voter** (final) | `run_knn_tail.sh` | CPU, ~15–25 min · **no training** | +1.0 pp F1-macro → **0.6912** |

**Results vs effort — pick your stopping point:**
- **Cheapest meaningful system:** Focal L2 + XGB **v2** by `max_conf` → **0.8476 / 0.6722**
  (skip steps 2 & 4). Already +5 pp F1-macro over the v1 baseline.
- Steps 2 and 4 are both **cheap** (no GPU retrain — they reuse cached embeddings) and each
  adds ~+1 pp F1-macro, so the full path is worth it. The only expensive part is step 1.
- `max_conf` needs **no threshold tuning** — it wins at every threshold.

**Do NOT run these for best results** (documented dead-ends, kept only as negatives —
simpler is better and none beat the path above): `run_meta_learner.sh` (Step 3),
`eval_decision_layer.py` (3b), `run_xgboost_v3.sh` (Step 4, +0.06 pp), and the embedding-lever
experiments `run_setfit_flat.sh`, `run_learned_detector.sh`, `run_proj_head.sh` +
`run_proj_eval.sh` (Step 7). The tail is **information-limited**, so no decision-rule or
embedding tuning helps.

---

## Where each improvement plugs in

```
                 text
                  │
   ┌──────────────┴───────────────┐
   ▼                              ▼
[CamemBERT L2 flat]        [L1 router → XGBoost]
 (6067-way)                 (per-L1 specialist)
   │  ◄── Step 1: Focal      │  ◄── Step 2: richer embeddings (1536d)
   │      loss retrain       │      Step 4: + engineered text features (1546d)
   │                         │      Step 5: + inverse-freq class weights ⭐
   └──────────────┬───────────────┘
                  ▼
         max_conf decision     ◄── Step 3: meta-learner (tested, rejected)
                  ▼
       kNN/NCM tail override    ◄── Step 6: flat tail expert, fires when margin>τ ⭐
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
sbatch improvements/scripts/run_focal_train.sh   # 1 GPU, --resume supported
```

Starts from `checkpoints/l2_camembert_run2/best`, outputs `checkpoints/l2_camembert_focal/checkpoint-65316`.

**Result:** standalone the focal model is weaker on accuracy (0.8100) but stronger on rare
classes. In the hybrid it lifts F1-macro by ~+1.1pp over XGB v2 alone (0.6608 → 0.6722).

---

## Step 2 — XGBoost with Better Embeddings (v2)  ⭐ biggest single win

**Problem:** XGBoost v1 uses only the L1 encoder's embeddings (768d, trained for the 157-class L1 task), which miss fine L2 distinctions.

**Solution:** feed XGBoost the **concatenation of L1 + L2 CLS embeddings (1536d)**. The BERT models stay frozen — only the 149 per-L1 XGBoost trees are trained on the richer features.

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
near-optimal. **Excluded from the final system.** Useful negative result: real gains come
from *better component models*, not a smarter decision layer.

A follow-up **decision-layer study** (`eval_decision_layer.py`, fit on VAL / eval on TEST)
re-tested four rules — `max_conf`, calibrated `max_conf` (isotonic), per-L1 gate, and a
learned gate. **`max_conf` won or tied on every metric**; calibration and the learned gate
both *hurt*. This confirms the decision rule is saturated — the only remaining headroom is
in the component models.

---

## Step 4 — XGBoost v3: engineered text features  ⚠️ negligible gain

**Idea:** append 10 cheap, CPU-only text features (title/desc lengths, digit/upper/punct
ratios, has-desc, unique-word ratio) to the 1536d v2 embeddings → **1546d**. Trees often
benefit from explicit non-linear signals that are hard to read off a CLS embedding.

```bash
sbatch improvements/scripts/run_xgboost_v3.sh   # reuses cached concat embeddings, ~2h
```

Outputs `checkpoints/l2_xgboost_v3/` (149 models, strict superset of v2 features).

**Result:** **+0.06pp F1-macro only** (0.6722 → 0.6728). The embeddings already capture these
signals — engineered features are essentially redundant. Kept as the feature base for Step 5.

---

## Step 5 — XGBoost v3_balanced: rare-class weighting  ⭐ new standard

**Problem:** F1-macro still lags accuracy because the per-L1 XGBoost trees are dominated by
frequent subcategories.

**Solution:** fit each per-L1 model with **inverse-frequency sample weights** —
`w = (N / n_subcat) ** α`, normalized to mean 1, capped at 10× (`α=0.5`). This is the trees'
analogue of focal loss: it amplifies rare L2 subcategories, directly targeting F1-macro.

```bash
sbatch improvements/scripts/run_xgboost_v3_balanced.sh   # BALANCE=1, ~2h
```

Outputs `checkpoints/l2_xgboost_v3_balanced/` (149 models).

**Result:** **0.8490 acc / 0.6814 f1m / 0.8480 f1w.** The weighting added **+0.9pp F1-macro**
over v2 while accuracy held/edged up. The standalone XGBoost confirms the source: v3_balanced
XGB-only f1m = 0.6735 vs v3 = 0.6641.

---

## Step 6 — kNN / NCM tail expert  ⭐ current standard

**Diagnostic first (`diagnose_tail.py`):** F1-macro is dragged down entirely by the tail —
ultra-rare classes sit at F1 0.137, rare at 0.496, while head is 0.861. The tail is 1,742
classes but only ~1.1% of test samples, so fixing it is a near-pure F1-macro play.
**77–99% of tail errors are "both base experts wrong"** (need a new model), and the oracle
ceiling is **+15.8pp macro at flat accuracy**. → worth a dedicated expert.

**Approach:** a **flat Nearest-Class-Mean classifier over all 6067 classes** (L2-normalized
centroids from the cached `run2` L2 embeddings). It is *flat*, so it **bypasses L1 routing**
(only the XGBoost branch is L1-capped) and works at 1–5 samples/class where gradient models
can't. It is folded into `max_conf` as a **tail-only third voter**:

> override the base prediction with the NCM tail prediction **only when**
> `margin = sim(nearest tail centroid) − sim(nearest non-tail centroid) > τ`.

The margin is essential: an absolute cosine threshold has **no precision** (BERT embeddings
are anisotropic — everything is moderately similar, so the tail-only kNN fired on all 1.19M
samples). The margin vs non-tail centroids removes that baseline — a head item's nearest
centroid is its own head class → negative margin → it won't fire.

```bash
sbatch run_knn_tail.sh    # CPU only, ~15-25 min, reuses cached embeddings, no training
```

**Result (τ=0.15):** **0.8487 acc / 0.6912 f1m** — **+1.0pp F1-macro at −0.03pp accuracy.**
VAL recall@1 on the tail is 0.704 (embeddings *do* separate rare classes), but the gain caps
near +1pp because of the precision/recall tension: raising τ protects accuracy but drops
tail-recall. Operating points: τ=0.125 (max f1m, 0.6914 @ −0.06pp acc); τ=0.20 (acc-neutral,
0.6875 @ ±0pp). Far below the +15.8pp oracle — the detector boundary is the limiter, not the
embeddings (confirmed by Step 7 below).

---

## Step 7 — SupCon projection head on frozen run2  ⚠️ tested, **rejected**

**Idea:** Step 6's tail voter is capped by run2's recall@1 (0.704). Directly test whether the
tail is *representation-limited* by learning a non-linear metric **on the frozen run2 space** —
a small MLP (768→1024→768) trained with supervised-contrastive (SupCon) loss to pull
same-tail-class embeddings together and push classes apart. The encoder is frozen (cached
vectors only), so it **cannot degrade the base** — it can only add a learned transform.

```bash
sbatch run_proj_head.sh    # train on cached run2 vectors, ~30 min (1 GPU)
sbatch run_proj_eval.sh    # NCM + margin voter on projected emb, CPU + memmap, ~30-60 min
```

Outputs `embeddings_proj/{train,val,test}_emb_proj.npz` (768d projected).

**Result:** recall@1 **rose** 0.704 → **0.717** (+1.3pp) — the learned metric *does* separate
the tail better. But the net tail-voter gain **fell**: best **Δf1m = +0.80pp** at flat accuracy
(τ=0.15) → **0.6894 f1m**, vs the run2 voter's **+1.0pp → 0.6912**. Higher retrieval didn't
convert: at the firing threshold the override precision is no better, so the extra neighbours
land on classes that don't lift macro-F1. **Excluded — run2 + simple margin stays the
standard.** Strong negative: optimizing the embedding geometry *specifically for the tail*
still doesn't beat the simple voter → **the tail is information-limited, not
representation-limited.**

---

## How each improvement was measured (isolated sweeps)

Each evaluation swaps **one box** and leaves the rest identical, so we know which change is
responsible for any gain. All sweeps use focal as the L2 predictor + run2 for the L2 embedding:

| Sweep | L2 prediction | L2 emb (for XGB) | XGBoost | Result file |
|-------|--------------|------------------|---------|-------------|
| focal | focal | — | v1 (768d) | `results/threshold_sweep_results_focal.json` |
| v2 | run2 | run2 | v2 (1536d) | `results/threshold_sweep_results_v2.json` |
| combined | **focal** | **run2** | **v2 (1536d)** | `results/threshold_sweep_results_combined.json` |
| v3 | **focal** | **run2** | **v3 (1546d)** | `results/threshold_sweep_results_combined_v3.json` |
| **v3_balanced** | **focal** | **run2** | **v3_balanced** | `results/threshold_sweep_results_combined_v3_balanced.json` |

The combined sweeps load **three** CamemBERT models: focal makes the L2 prediction, run2
generates the L2 embedding XGBoost was trained on, and L1 routes + supplies the L1 embedding.
The eval harness appends the same 10 engineered features at inference for the v3 variants
(`XGB_DIR` / `XGB_ENG` / `XGB_VARIANT` env knobs in `eval_threshold_sweep_combined.py`).

---

## Status

| Step | Component | Status | Outcome |
|------|-----------|--------|---------|
| 1 | Focal CamemBERT L2 | ✅ done | `checkpoint-65316`, lifts F1-macro |
| 2 | XGBoost v2 (concat) | ✅ done | 149 models, beats baseline on all metrics |
| 3 | Meta-learner | ❌ rejected | worse than max_conf |
| 3b | Decision-layer study | ❌ rejected | calibration / learned gate hurt; max_conf saturated |
| 4 | XGBoost v3 (engineered feats) | ⚠️ negligible | +0.06pp f1m only |
| 5 | XGBoost v3_balanced | ✅ done | rare-class weighting, +0.9pp f1m |
| 6 | kNN/NCM tail expert | ✅ done | tail-only third voter, +1.0pp f1m at ~flat acc |
| 7 | SupCon projection head | ❌ rejected | recall@1 0.704→0.717 but net +0.8pp < +1.0pp — tail is information-limited |
| — | **Current standard (1+2→5+6)** | ✅ done | **0.8487 / 0.6912 / ≈0.848** |

---

## Files

```
improvements/
├── config.json                    Parameters & final results
├── README.md                      This file
├── results/
│   ├── threshold_sweep_results_focal.json
│   ├── threshold_sweep_results_v2.json
│   ├── threshold_sweep_results_combined.json
│   ├── threshold_sweep_results_combined_v3.json
│   ├── threshold_sweep_results_combined_v3_balanced.json
│   └── decision_layer_results.json
└── scripts/
    # --- Steps 1-5: the optimal-path components ---
    ├── train_l2_focal.py              Step 1: CamemBERT focal-loss L2 training
    ├── run_focal_train.sh             SLURM: 1 GPU
    ├── train_xgboost_v2.py            Step 2: XGBoost on L1+L2 concat embeddings
    ├── run_xgboost_v2.sh              SLURM: 1 GPU, EMB_MODE=concat
    ├── train_xgboost_v3.py            Steps 4 & 5: engineered feats (+ BALANCE=1 = v3_balanced)
    ├── run_xgboost_v3.sh              SLURM: Step 4 (negligible)
    ├── run_xgboost_v3_balanced.sh     SLURM: Step 5  ⭐
    ├── eval_threshold_sweep_combined.py   max_conf eval harness (focal + run2 + XGB)
    ├── eval_threshold_sweep_focal.py      focal-only sweep
    ├── eval_threshold_sweep_v2.py         v2 sweep
    # --- Step 6: tail expert (current standard) ---
    ├── diagnose_tail.py               frequency-band F1 + oracle ceiling
    ├── eval_knn_tail.py               NCM tail voter (final, --emb l2 --tail-max 100)
    ├── run_knn_tail.sh                SLURM: CPU ~20 min  ⭐
    ├── eval_knn_tail_whiten.py        whitening / projected-emb sweep
    # --- Rejected experiments (negatives - do NOT run for best results) ---
    ├── train_meta_learner.py          Step 3: meta-learner stacking (rejected)
    ├── run_meta_learner.sh
    ├── eval_decision_layer.py         Step 3b: calibration / learned gate (rejected)
    ├── train_proj_head.py             Step 7: SupCon projection head (rejected)
    ├── run_proj_head.sh
    ├── run_proj_eval.sh
    ├── train_setfit_flat.py           flat SetFit tail embedding (rejected)
    ├── run_setfit_flat.sh
    ├── run_knn_setfit_eval.sh
    ├── eval_learned_detector.py       learned binary tail detector (rejected)
    └── run_learned_detector.sh
```

