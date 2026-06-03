# Specialist Models: SetFit, LoRA, Generative

## Overview

Three specialist approaches for rare L2 subcategory classification, each targeting subcategories with insufficient training data for the main CamemBERT L2 classifier.

**Status: All three EXCLUDED from the final hybrid system** — see findings below.

---

## Models

### 1. SetFit (Few-Shot Learning)

**Approach:** One SetFit model per L1 class with rare subcategories. Uses contrastive learning on sentence-transformer embeddings + logistic regression head.

```bash
# Train
sbatch scripts/run_setfit.sh
# Or directly:
python scripts/train_setfit.py          # full (27 classes)
python scripts/train_setfit.py --debug  # test with 2 classes
```

**Coverage:** 27/157 L1 classes  
**Output:** `checkpoints/setfit_models/<l1_class>/`

**Why it failed:**
- `predict_proba()` returns high confidence (>0.8) even on wrong predictions
- Cannot be meaningfully compared to CamemBERT softmax probabilities
- Override accuracy: **4.87%** (out of 10,636 overrides attempted)

---

### 2. LoRA (Parameter-Efficient Fine-Tuning)

**Approach:** One LoRA adapter per L1 class. Adds rank-16 adapters to CamemBERT's query/value matrices. Only adapter weights saved (~2-4MB per class).

```bash
# Train
sbatch scripts/run_lora.sh
# Or directly:
python scripts/train_l2_lora.py          # full (59 classes)
python scripts/train_l2_lora.py --debug  # test with 2 classes
```

**Coverage:** 59/157 L1 classes  
**Output:** `checkpoints/lora_models/<l1_class>/`

**Why it failed:**
- Requires correct L1 prediction to select the right adapter
- L1 error rate is ~8% overall, but **much higher on uncertain samples** (exactly where specialists are needed)
- Wrong L1 → wrong adapter → **1.2% accuracy** on specialist-handled samples

---

### 3. Generative (CroissantLLM)

**Approach:** Constrained log-probability scoring. For each product, compute P(label | prompt) for every valid L2 subcategory. Pick the highest scoring label — guarantees valid output (no hallucination).

```bash
# Evaluate
sbatch scripts/run_generative.sh
# Or directly:
python scripts/generative_fallback.py              # default threshold
python scripts/generative_fallback.py --debug      # test 1 class
python scripts/generative_fallback.py --threshold 0.4
```

**Model:** `croissantllm/CroissantLLMChat-v0.1` (1.3B, French)  
**Output:** `checkpoints/generative_models/`

**Why it failed:**
- **7.3% accuracy** on uncertain samples vs CamemBERT's **27.6%** on the same set
- Oracle ceiling (either model correct): 31.9%
- Generative wins over CamemBERT in only 4,490 / 103,467 cases
- Too slow for production (sequential decoding per candidate)

---

## Key Findings

| Model | Override Accuracy | Problem |
|-------|-----------------|---------|
| SetFit | 4.87% | Uncalibrated confidence |
| LoRA | 1.2% | L1 routing errors cascade |
| Generative | 7.3% | Lacks discriminative precision |
| **XGBoost** | **40-60%** | ✅ Well-calibrated, batch-efficient |

### Root Cause Analysis

The fundamental issue is **L1 routing under uncertainty:**
- Specialists are invoked only when CamemBERT is uncertain (cam_conf < 0.5)
- These hard samples have much higher L1 error rates than average
- Wrong L1 → wrong specialist bucket → model trained on different categories
- This error cascade makes all L1-routed specialists unreliable

### Recommendation

Use the **max_conf hybrid** system (see `Gated+Hybrid/`):
- CamemBERT L2 + XGBoost only
- No routing table, no threshold tuning
- **+1.57% accuracy, +5.88% F1-macro** over CamemBERT alone

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train_setfit.py` | Train SetFit models (27 L1 classes) |
| `scripts/train_l2_lora.py` | Train LoRA adapters (59 L1 classes) |
| `scripts/generative_fallback.py` | Evaluate generative log-prob scoring |
| `scripts/run_setfit.sh` | SLURM job for SetFit training |
| `scripts/run_lora.sh` | SLURM job for LoRA training |
| `scripts/run_generative.sh` | SLURM job for generative evaluation |
| `config.json` | Full model specs + hyperparameters + results |
