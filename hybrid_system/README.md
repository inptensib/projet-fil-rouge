# Hybrid Classification System — Architecture & Results

## Folder Structure

```
hybrid_system/
│
├── Gated+Hybrid/                              ← FINAL SYSTEM (acc=83.54%, F1m=63.21%)
│   │
│   ├── config.json                            System parameters, model registry, sweep config
│   ├── README.md                              Architecture diagram, results table, reproduction steps
│   │
│   ├── scripts/                               Evaluation & training code
│   │   ├── eval_threshold_sweep.py            Final optimized sweep (3h GPU, 21 combos)
│   │   ├── eval_hybrid_confidence.py          v1: XGBoost-first, gated at t=0.5 (acc=83.2%)
│   │   ├── eval_hybrid_confidence_v2.py       v2: LoRA/SetFit priority ordering
│   │   ├── eval_hybrid_confidence_v3.py       v3: SetFit+gates + generative probe (acc=82.3%)
│   │   ├── eval_hybrid.py                     Gated routing via routing table (acc=78.5%)
│   │   └── train_l2_xgboost.py                XGBoost training (149 per-L1-class models)
│   │
│   ├── analysis/                              Configuration & taxonomy data
│   │   ├── gating_config.json                 L1 class → specialist type routing table
│   │   ├── l1_to_l2_mapping.json              157 L1 classes → valid L2 subcategories
│   │   ├── l1_class_difficulty.json           Per-class difficulty scores
│   │   ├── l2_subcategory_counts.json         L2 label frequency distribution
│   │   └── routing_recommendations.json       Auto-generated routing suggestions
│   │
│   ├── analysis_scripts/                      Scripts to regenerate analysis/ data
│   │   ├── analyze_gating.py                  Routing decision analysis
│   │   └── analyze_taxonomy.py                L1→L2 taxonomy structure analysis
│   │
│   └── results/                               Experiment outputs (JSON)
│       ├── threshold_sweep_results.json       21-combo sweep + baselines → BEST = max_conf
│       ├── hybrid_results_confidence.json     v1 per-route breakdown
│       └── hybrid_results_confidence_v3.json  v3 with generative probe stats
│
├── Specialists/                               ← ABLATION MODELS (all excluded from final)
│   │
│   ├── config.json                            Hyperparameters, results, exclusion reasons
│   ├── README.md                              Purpose, individual findings, why each failed
│   │
│   └── scripts/                               Training & evaluation code + SLURM runners
│       ├── train_setfit.py                    SetFit training (27 L1 classes)
│       ├── train_l2_lora.py                   LoRA adapter training (59 L1 classes)
│       ├── generative_fallback.py             CroissantLLM constrained decoding evaluation
│       ├── run_setfit.sh                      SLURM job script for SetFit
│       ├── run_lora.sh                        SLURM job script for LoRA
│       └── run_generative.sh                  SLURM job script for generative
│
└── improvements/                              ← BEATS BASELINE (acc=84.76%, F1m=67.22%)
    │
    ├── config.json                            Parameters & final results
    ├── README.md                              Final results, isolated-sweep methodology
    │
    ├── results/                               Threshold-sweep outputs (JSON)
    │   ├── threshold_sweep_results_focal.json     Isolated: focal L2 + XGB v1
    │   ├── threshold_sweep_results_v2.json        Isolated: run2 + XGB v2 (concat)
    │   └── threshold_sweep_results_combined.json  FINAL: focal L2 + XGB v2
    │
    └── scripts/                               Training code + SLURM runners
        ├── train_l2_focal.py                  Step 1: focal-loss CamemBERT L2
        ├── train_xgboost_v2.py                Step 2: XGBoost on L1+L2 concat embeddings
        ├── train_meta_learner.py              Step 3: meta-learner stacking (rejected)
        ├── run_focal_train.sh                 SLURM job script for focal training
        ├── run_xgboost_v2.sh                  SLURM job script for XGBoost v2
        └── run_meta_learner.sh                SLURM job script for meta-learner
```

---

## System Architecture

```
Input (product title + description)
        │
        ▼
┌─────────────────────────────────────┐
│  CamemBERT L2 (flat ~6K classes)    │  ← ALL 1.19M samples
│  → prediction + confidence          │
└─────────────────────────────────────┘
        │
        ├── cam_conf ≥ 0.9 → ACCEPT CamemBERT (77.2% of samples, ~98% accurate)
        │
        ▼ cam_conf < 0.9 (uncertain)
┌─────────────────────────────────────┐
│  L1 Classifier (157 classes)        │  ← only uncertain samples
│  + CLS embedding extraction         │     (single forward pass via
│                                     │      output_hidden_states=True)
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  XGBoost (per-L1-class model)       │  ← batched by predicted L1 class
│  → xgb_prediction + xgb_confidence  │     149/157 classes covered
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Decision: max(cam_conf, xgb_conf)  │
│  Use whichever model is more        │
│  confident → final L2 prediction    │
└─────────────────────────────────────┘
```

---

## Results Summary

| System | Accuracy | F1-macro | F1-weighted | Notes |
|--------|----------|----------|-------------|-------|
| CamemBERT L2 alone | 0.8197 | 0.5733 | 0.8148 | Baseline |
| XGBoost alone | 0.8069 | 0.6058 | 0.8038 | All samples through XGBoost |
| max_conf hybrid (v1 baseline) | 0.8354 | 0.6321 | 0.8312 | +1.57% acc, +5.88% F1m |
| Confidence v1 (gated, t=0.5) | 0.8320 | 0.6205 | 0.8275 | Previous best |
| Confidence v3 (SetFit+gates) | 0.8230 | 0.5920 | 0.8210 | SetFit hurts |
| Gated routing v2 | 0.7850 | 0.5420 | 0.8180 | L1 errors cascade |
| **Focal L2 + XGB v2 (FINAL, `improvements/`)** | **0.8476** | **0.6722** | **0.8463** | **+1.22% acc, +4.01% F1m over v1** |

The `improvements/` folder pushes past the v1 baseline by retraining the two
component models (focal-loss CamemBERT + XGBoost on richer 1536d embeddings).
Same `max_conf` decision rule — see [improvements/README.md](improvements/README.md).

---

## Key Findings

1. **`max_conf` strategy is optimal and hyperparameter-free** — just pick whichever model (CamemBERT or XGBoost) is more confident. No threshold to tune.

2. **XGBoost is the only beneficial specialist.** SetFit (4.87% override accuracy), LoRA (1.2%), and Generative (7.3%) all degrade performance vs CamemBERT alone.

3. **Routing by L1 class is harmful for uncertain samples.** The L1 model has ~8% error overall, but on hard/uncertain samples the error rate is much higher → wrong specialist → catastrophic.

4. **The 0.9 confidence ceiling saves 70% of L1+XGBoost compute** with zero accuracy loss (CamemBERT at >0.9 confidence is ~98% correct).

---

## Reproduction

```bash
# Full sweep (GPU, ~3h on RTX 3090):
sbatch Gated+Hybrid/scripts/run_threshold_sweep.sh

# Just re-sweep from cached predictions (CPU, instant):
python Gated+Hybrid/scripts/eval_threshold_sweep.py --sweep-only

# Train specialists (for ablation analysis):
sbatch Specialists/scripts/run_setfit.sh
sbatch Specialists/scripts/run_lora.sh
sbatch Specialists/scripts/run_generative.sh

# Improvements that beat the baseline (focal L2 + XGBoost v2):
sbatch improvements/scripts/run_focal_train.sh
sbatch improvements/scripts/run_xgboost_v2.sh
```

### Requirements
- Python 3.12, CUDA GPU (RTX 3090 24GB recommended)
- transformers, torch, xgboost, scikit-learn, datasets, sentence-transformers, peft, tqdm
- Checkpoints: `l1_classifier/`, `l2_camembert_run2/`, `l2_xgboost/`
- Test data: `dataset/test_split.jsonl` (1,194,356 samples)
