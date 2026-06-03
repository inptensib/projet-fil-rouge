# Gated + Hybrid Classification System

## Overview

Hierarchical hybrid product classification for Rakuten France: **CamemBERT L2** as primary classifier with **XGBoost** confidence-based override for uncertain predictions.

**Final result:** Accuracy = **83.54%**, F1-macro = **63.21%**, F1-weighted = **83.12%**

Improvement over CamemBERT alone: **+1.57% accuracy, +5.88% F1-macro**

---

## Architecture

```
Input (product title + description)
        │
        ▼
┌─────────────────────────────────────┐
│  CamemBERT L2 (flat ~6K classes)    │  ← ALL samples
│  → prediction + confidence          │
└─────────────────────────────────────┘
        │
        ├── cam_conf ≥ 0.9 → ACCEPT (77.2% of samples)
        │
        ▼ cam_conf < 0.9
┌─────────────────────────────────────┐
│  L1 Classifier (157 classes)        │  ← only uncertain samples
│  + CLS embedding extraction         │     (single forward pass)
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  XGBoost (per-L1-class model)       │  ← batched by L1 class
│  → xgb_prediction + xgb_confidence  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Decision: max(cam_conf, xgb_conf)  │
│  Pick whichever model is more       │
│  confident → final L2 prediction    │
└─────────────────────────────────────┘
```

---

## Results Summary

### Threshold Sweep (21 combinations tested)

| Configuration | Accuracy | F1-macro | F1-weighted | % XGBoost |
|--------------|----------|----------|-------------|-----------|
| **max_conf (BEST)** | **0.8354** | **0.6321** | **0.8312** | 22.8% |
| gated, t=0.9 | 0.8354 | 0.6321 | 0.8312 | 22.8% |
| gated, t=0.5 | 0.8320 | 0.6205 | 0.8275 | 10.5% |
| CamemBERT-only | 0.8197 | 0.5733 | 0.8148 | 0% |
| XGBoost-only | 0.8069 | 0.6058 | 0.8038 | 100% |

### Specialist Comparison (from ablation experiments)

| System | Accuracy | F1-macro | Notes |
|--------|----------|----------|-------|
| CamemBERT L2 alone | 0.8197 | 0.5733 | Baseline |
| **max_conf hybrid** | **0.8354** | **0.6321** | Best overall |
| Confidence v1 (gated, t=0.5) | 0.8320 | 0.6205 | Previous best |
| Confidence v3 (SetFit+gates) | 0.8230 | 0.5920 | SetFit hurts |
| Gated routing (v2) | 0.7850 | 0.5420 | L1 errors cascade |

---

## Key Findings

1. **`max_conf` is threshold-independent** — the confidence comparison alone determines the optimal split. No hyperparameter tuning needed.

2. **XGBoost is the only beneficial specialist.** SetFit (broken calibration), LoRA (L1 cascade errors), and Generative (7.3% on ambiguous samples) all degrade performance.

3. **Gating by L1 routing table is harmful.** The L1 model has ~8% error rate, but for uncertain samples (where we need specialists most) the error rate is much higher → wrong specialist → catastrophic accuracy.

4. **The 0.9 ceiling saves 70% compute** with no accuracy loss — CamemBERT at >0.9 confidence is ~98% correct.

---

## Files

### scripts/
| File | Description |
|------|-------------|
| `eval_threshold_sweep.py` | Final optimized sweep: collects predictions in 1 pass, tests 21 combinations |
| `eval_hybrid_confidence.py` | Confidence-only v1 (XGBoost first, t=0.5 gated) |
| `eval_hybrid.py` | Gated routing v2 (uses gating_config.json routing table) |
| `eval_hybrid_confidence_v2.py` | v2: LoRA/SetFit priority ordering |
| `eval_hybrid_confidence_v3.py` | v3: SetFit+gates + generative probe |
| `train_l2_xgboost.py` | Training script for per-L1-class XGBoost models |

### analysis/
| File | Description |
|------|-------------|
| `gating_config.json` | Routing table: L1 class → specialist type |
| `l1_to_l2_mapping.json` | 157 L1 classes → list of valid L2 subcategories |
| `l1_class_difficulty.json` | Per-class difficulty metrics |
| `l2_subcategory_counts.json` | L2 label frequency distribution |
| `routing_recommendations.json` | Auto-generated routing suggestions |

### results/
| File | Description |
|------|-------------|
| `threshold_sweep_results.json` | Full 21-combination sweep output + baselines |
| `hybrid_results_confidence.json` | Detailed v1 results (per-route breakdown) |
| `hybrid_results_confidence_v3.json` | v3 results with gen probe stats |

---

## Reproduction

```bash
# On SLURM cluster with RTX 3090:
sbatch run_threshold_sweep.sh     # ~3h, produces threshold_sweep_results.json

# Or just the sweep pass if predictions already cached:
python eval_threshold_sweep.py --sweep-only
```

### Requirements
- Python 3.12
- transformers, torch, xgboost, scikit-learn, datasets, tqdm
- Checkpoints: `l1_classifier/`, `l2_camembert_run2/`, `l2_xgboost/`
- Test data: `dataset/test_split.jsonl` (1.19M samples)
