# Flat Classifier HYDRA — Technical design

Specification of the `Flat` model implementation as defined in the HYDRA
paper (EMNLP 2025). This document covers the architecture, the loss, all
training parameters, and the implementation decisions.

Official reference:
[`modeling/baselines.py`](https://github.com/FKarl/HYDRA/blob/main/modeling/baselines.py)

> **Note on dimensions.** The tensor shapes in this document
> (`L2 = 6 126`, unified output `6 283 = 157 + 6 126`) correspond to the
> **original v1 dataset**, on which the reference FlatHydra model was
> trained (F1 L2 = 71.7 %). On the **de-leaked v2 dataset** (commonly delivered),
> `L2 = 5 777` and the unified output is `5 934 = 157 + 5 777`. The architecture
> is identical: the L2 head size is derived from the data at runtime
> (`len(l2_encoder.classes_)`), never hard-coded. See
> [METRICS.md](METRICS.md) for details on the v1 → v2 leak.

---

## Table of contents

- [Context and objective](#context-and-objective)
- [Model architecture](#model-architecture)
- [The loss: HYDRALoss (CrossEntropy)](#the-loss-hydraloss-crossentropy)
- [Training parameters](#training-parameters)
- [Optimizer and scheduling](#optimizer-and-scheduling)
- [Mixed precision (AMP)](#mixed-precision-amp)
- [Early stopping](#early-stopping)
- [Training pipeline](#training-pipeline)
- [Tracking metrics](#tracking-metrics)
- [Produced artifacts](#produced-artifacts)

---

## Context and objective

The HYDRA paper defines `Flat` as the **reference baseline**: a single
model that predicts L1 (157 classes) and L2 simultaneously through
one unified head, without any modeling of the hierarchy.

It is the lower bound to beat with the `local`, `global`,
and `nested` variants.

---

## Model architecture

### Overview

```
Product text (designation + description)
        │
        ▼
┌─────────────────────────────────────┐
│  Tokenizer CamemBERT                │
│  max_length = 256, fixed padding    │
│  → input_ids      (B, 256)          │
│  → attention_mask (B, 256)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CamemBERT Encoder                  │
│  12 Transformer layers, 768 dims    │
│  last_hidden_state  (B, 256, 768)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Embedding Projector                │
│  Linear(768 → 1536)                 │
│  applied to FULL last_hidden_state  │
│  → (B, 256, 1536)                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CLS Pooling                        │
│  token[:, 0, :]  →  (B, 1536)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Unified Classifier                 │
│  Linear(1536 → 1536)                │
│  LayerNorm(1536)                    │
│  ReLU                               │
│  Dropout(p=0.2)                     │
│  Linear(1536 → 6283)  ← 157 + 6126 │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
logits_L1 (B, 157)   logits_L2 (B, 6126)
  slice [:157]          slice [157:]
```

### Key point: projection BEFORE pooling

The official implementation projects the entire `last_hidden_state`
**before** extracting the CLS token, unlike `HYDRABase` which projects
afterwards:

```python
# Official Flat (baselines.py)
outputs = self.embedding_projector(outputs.last_hidden_state)  # (B, 256, 1536)
full_embedding = outputs[:, 0, :]                              # CLS → (B, 1536)

# HYDRABase (different)
full_embedding = outputs.last_hidden_state[:, 0, :]            # CLS → (B, 768)
embedding = self.projector(full_embedding)                     # → (B, 1536)
```

Consequence: the CLS token aggregates information **in the projected space**,
which gives it access to a multi-level representation as early as the pooling
step.

### Dimensions

| Step                   | Tensor                      | Dimension        |
|:-----------------------|:----------------------------|:----------------:|
| Tokens                 | `input_ids`                 | `(B, 256)`       |
| Encoder output         | `last_hidden_state`         | `(B, 256, 768)`  |
| After projection       | `projected`                 | `(B, 256, 1536)` |
| After CLS pooling      | `embedding`                 | `(B, 1536)`      |
| Unified logits         | `unified_logits`            | `(B, 6283)`      |
| L1 logits              | `unified_logits[:, :157]`   | `(B, 157)`       |
| L2 logits              | `unified_logits[:, 157:]`   | `(B, 6126)`      |

### Return interface

```python
return [], unified_logits
#  ↑ empty level_outputs (common interface with HYDRA)
#                ↑ raw unnormalized logits (B, 6283)
```

---

## The loss: HYDRALoss (CrossEntropy)

### Choice relative to the official implementation

The HYDRA paper uses `F.binary_cross_entropy_with_logits` because its benchmarks
(EURLEX, WOS) are **multi-label** — a document can belong to several
categories simultaneously.

Rakuten is **single-label**: each product belongs to exactly one L1
and exactly one L2. This context justifies a different choice:

| Criterion             | BCE (official HYDRA)              | CrossEntropy / HYDRALoss (our choice)  |
|:----------------------|:----------------------------------|:---------------------------------------|
| Paradigm              | Multi-label (N possible answers)  | **Multi-class (only 1 answer)**        |
| Normalization         | Independent sigmoid per class     | **Softmax over all classes**           |
| Labels                | Concatenated one-hot (B, 6283)    | **Separate integers L1 and L2**        |
| Competing classes     | No — each class is independent     | **Yes — classes are mutually exclusive** |
| L1/L2 rebalancing     | None (L2 dominates by volume)     | **`level_weight_l1=0.5`**             |
| Suited to Rakuten     | No                                | **Yes**                               |

With uniform BCE over 6283 positions, L2 would mechanically dominate the gradient
(6126 positions against 157). `HYDRALoss` with `level_weight_l1=0.5` explicitly
rebalances the two levels.

### How HYDRALoss works

`HYDRALoss` is instantiated with `architecture="global"` to leverage the
slicing mechanism over the unified logits:

```python
self.criterion = HYDRALoss(
    architecture="global",
    alpha=1.0,               # weight of the unified_loss (no level_outputs)
    num_classes_l1=157,
    num_classes_l2=6126,
    level_weight_l1=0.5,     # L1 weighs 0.5, L2 weighs 1.0
    use_focal=False,         # standard CrossEntropy
    focal_gamma=2.0,         # ignored if use_focal=False
)
```

Internally, `HYDRALoss` with `architecture="global"` does:

```python
# Slicing the unified logits
logits_l1 = unified_output[:, :num_classes_l1]   # (B, 157)
logits_l2 = unified_output[:, num_classes_l1:]   # (B, 6126)

# CrossEntropy per level (integer labels)
loss_l1 = CE(logits_l1, labels_l1)   # scalar
loss_l2 = CE(logits_l2, labels_l2)   # scalar

# Weighted combination
total_loss = alpha * (level_weight_l1 * loss_l1 + loss_l2)
#          = 1.0  * (0.5 * loss_l1 + loss_l2)
```

### CrossEntropy formula

For a batch of size B and C classes:

```
CE(logits, y) = -(1/B) · Σ_b log( exp(logit_{b,y_b}) / Σ_c exp(logit_{b,c}) )
             = -(1/B) · Σ_b [ logit_{b,y_b} - log(Σ_c exp(logit_{b,c})) ]
```

The softmax normalizes the logits into probabilities that **sum to 1** — which
is consistent with the fact that a product belongs to a single class.

### Prediction at inference

With CrossEntropy (softmax), the prediction is the `argmax` over the probabilities
of each level separately:

```python
# Slicing
logits_l1 = unified_logits[:, :self.num_classes_l1]   # (B, 157)
logits_l2 = unified_logits[:, self.num_classes_l1:]   # (B, 6126)

# Softmax per level (consistent with CE during training)
probs_l1 = torch.softmax(logits_l1, dim=-1)
probs_l2 = torch.softmax(logits_l2, dim=-1)

# Prediction: most probable class
confs_l1, preds_l1 = probs_l1.max(dim=-1)
confs_l2, preds_l2 = probs_l2.max(dim=-1)
```

---

## Training parameters

All canonical parameters are defined in
`configs/baselines/flat_hydra.yaml`.

### Encoder

| Parameter         | Value           | Explanation                                           |
|:------------------|:---------------:|:------------------------------------------------------|
| `encoder`         | `camembert-base`| RoBERTa trained on French (CCNet + Oscar)             |
| `max_length`      | `256`           | Max length of tokenized sequences (truncation)        |
| `freeze_encoder`  | `false`         | Full fine-tuning — the CamemBERT weights adapt        |

**Why `max_length=256`?** The Rakuten designation + description rarely exceeds
200 tokens. 256 offers full coverage without excessive memory cost
(attention complexity is O(n²)).

**Why not freeze the encoder?** In `freeze_encoder=True` mode, only
the classification head learns — this is useful for quick testing but
yields much lower performance. Full fine-tuning is consistently better on
domain-specific data (French e-commerce).

### Architecture

| Parameter           | Value  | Explanation                                             |
|:--------------------|:------:|:--------------------------------------------------------|
| `project_embedding` | `true` | Linear(768→1536) on `last_hidden_state` before pooling  |
| `pooling`           | `cls`  | `[CLS]` token as the sequence representation             |
| `dropout`           | `0.2`  | Dropout in the unified head (paper value)               |

**Why `dropout=0.2` and not `0.3`?** The HYDRA variants
(local/global/nested) use `dropout=0.3`. The official `Flat` uses
`0.2` — probably because the unified head is simpler and less prone
to overfitting than multiple heads.

**Why CLS pooling?** CamemBERT's `[CLS]` token is trained to
aggregate information from the whole sequence. `mean` pooling can be
slightly better on some tasks, but CLS remains the paper's default
choice.

### Batch and epochs

| Parameter    | Value  | Official paper  | Explanation                                    |
|:-------------|:------:|:---------------:|:-----------------------------------------------|
| `batch_size` | `64`   | `32`            | Larger batch → more stable gradient            |
| `epochs`     | `10`   | `50`            | Early stopping compensates for the difference  |

**Why `batch_size=64` and not 32?** With the cluster GPUs (P100 /
3090, 16-24 GB), 64 fits in memory and doubles the throughput. The gradient with
batch=64 is more stable than with 32 (less stochastic noise).

**Why `epochs=10` and not 50?** The paper does not use early
stopping — it trains for a fixed 50 epochs. With early stopping (patience=2),
convergence is detected automatically. On 8M Rakuten products,
3 to 5 epochs are generally enough to saturate the val F1.

---

## Optimizer and scheduling

### AdamW

```
θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε) - α · λ · θ_t
```

| Parameter      | Value    | Explanation                                             |
|:---------------|:--------:|:--------------------------------------------------------|
| `lr`           | `3.5e-5` | Base learning rate for CamemBERT fine-tuning            |
| `weight_decay` | `0.01`   | Decoupled L2 regularization (the "W" in AdamW)          |
| `max_grad_norm`| `1.0`    | Gradient clipping threshold                            |

**Why `lr=3.5e-5`?** It is the empirically validated value for the
fine-tuning of CamemBERT/RoBERTa models on classification tasks.
Higher values (>5e-5) cause instability; lower ones
(<1e-5) needlessly slow down convergence.

**Why AdamW and not Adam?** Classic Adam applies weight decay
to the gradient (not the weights), which is theoretically incorrect. AdamW
decouples L2 regularization from learning-rate adaptation — it is
the standard for fine-tuning Transformers since 2019 (Loshchilov &
Hutter).

**Gradient clipping (`max_grad_norm=1.0`)**: if the L2 norm of the gradients
exceeds 1.0, all gradients are scaled down proportionally. This
prevents gradient explosions during the first steps, which are common
with large LMs.

### Linear Warmup + Linear Decay

The learning rate follows a two-phase schedule:

```
Phase 1 — Linear warmup (0 → lr): steps 0 to warmup_steps
Phase 2 — Linear decay (lr → 0): steps warmup_steps to total_steps
```

```
lr(t) = lr × t / warmup_steps                  if t < warmup_steps
lr(t) = lr × (total_steps - t) / (total_steps - warmup_steps)  otherwise
```

| Parameter      | Value  | Explanation                                             |
|:---------------|:------:|:--------------------------------------------------------|
| `warmup_ratio` | `0.1`  | 10% of the total steps are dedicated to warmup          |

**Why a warmup?** At the start of training, the classification head's
weights are random — their gradients are very large and
unstable. Starting with a small LR and then raising it gradually lets
the encoder adapt gently before being subjected to aggressive
updates.

**Concrete example** for 8M products, batch=64, 10 epochs:
- Total steps: `(8M × 0.7) / 64 × 10 ≈ 875 000 steps`
- Warmup: `87 500 steps` (~1 epoch)

---

## Mixed precision (AMP)

| Parameter | Value  | Explanation                                        |
|:----------|:------:|:---------------------------------------------------|
| `fp16`    | `true` | Computation in float16, master weights in float32  |

**Why FP16?** float16 operations are 2-8× faster on
NVIDIA GPUs (Tensor Cores). Memory consumption is divided by ~2, which
allows larger batches.

**GradScaler**: FP16 has a reduced dynamic range — small gradients
can underflow to 0. `GradScaler` multiplies the loss by a scale
factor S before the backward pass, then divides the gradients by S before
`optimizer.step()`. If an overflow is detected, the step is skipped and S is
reduced.

```python
scaler.scale(loss).backward()      # loss × S, gradients × S
scaler.unscale_(optimizer)         # gradients / S (for clipping)
clip_grad_norm_(params, 1.0)       # clipping on the real gradients
scaler.step(optimizer)             # step skipped if overflow
scaler.update()                    # adjust S for the next step
```

---

## Early stopping

| Parameter                    | Value   | Explanation                                   |
|:-----------------------------|:-------:|:----------------------------------------------|
| `early_stopping_patience`    | `2`     | Number of epochs without improvement before stopping |
| `early_stopping_min_delta`   | `0.001` | Minimum F1 L2 improvement to count            |

**Criterion**: `val_f1_l2_macro`. If the F1 L2 does not improve by at least
0.001 for 2 consecutive epochs, training stops.

**Why `patience=2`?** With a linearly decaying LR, the improvement
slows down progressively. A patience of 2 is conservative but avoids
continuing to train for nothing after convergence. On large datasets
(8M), each epoch is expensive — we prefer to stop early.

**Why `min_delta=0.001`?** Fluctuations of ±0.001 in macro F1
are numerical noise over 6126 classes. This threshold filters out these false positives.

---

## Training pipeline

### Per-batch flow

```
batch (input_ids, attention_mask, labels_l1, labels_l2)
         │
         ▼
  FlatHydra.forward()
         │
         ▼
  unified_logits (B, 6283)
         │
         ▼
  one_hot(labels_l1) + one_hot(labels_l2) → unified_labels (B, 6283)
         │
         ▼
  F.binary_cross_entropy_with_logits(unified_logits, unified_labels)
         │
         ▼
       loss
         │
  scaler.scale(loss).backward()
  clip_grad_norm_(params, 1.0)
  optimizer.step()
  scheduler.step()
```

### Per-epoch flow

```
Epoch N
  ├─► Train loop (shuffle=True, drop_last=True)
  │     └─► accumulate train_loss
  │
  └─► evaluate(val_loader)
        ├─► sigmoid(unified_logits) → probs
        ├─► argmax(probs_l1) → preds_l1
        ├─► argmax(probs_l2) → preds_l2
        └─► compute_all_metrics() → f1_l1, f1_l2, hcr, constrained_f1
              │
              ├─► early stopping on f1_l2_macro
              ├─► save best_model.pt if improvement
              ├─► save checkpoint_latest.pt (auto-resume)
              ├─► metrics.json (local fallback)
              └─► mlflow.log_metrics()
```

### Auto-resume (SLURM job array)

The SLURM script launches up to 10 sequential slots (`--array=0-9%1`).
Each slot checks:
1. If `training_complete.txt` exists → skip
2. If `checkpoint_latest.pt` exists → resume from that checkpoint
3. Otherwise → start from scratch

```bash
sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_variant.slurm
# Smoke test:
sbatch --job-name=flat-hydra-smoke --array=0-2%1 \
  scripts/run_flat_variant.slurm configs/baselines/flat_hydra.yaml 5000
```

---

## Tracking metrics

| Metric                 | Level  | Formula                                               | Target        |
|:-----------------------|:------:|:------------------------------------------------------|:-------------:|
| `f1_l1_macro`          | L1     | Mean of per-class F1 over L1 (157 classes)            | Reference     |
| `f1_l2_macro`          | L2     | Mean of per-class F1 over L2 (6126 classes)           | **Primary**   |
| `constrained_f1_macro` | L2     | F1 L2 set to 0 if the predicted L1 is wrong           | Quality       |
| `hcr`                  | Global | % of (pred_L1, pred_L2) pairs that are hierarchically valid | Target > 95% |

**`f1_macro`**: each class carries equal weight, regardless of its
frequency. It is the metric suited to imbalanced datasets (some
Rakuten categories have 10× more examples than others).

**`constrained_f1`**: if the model predicts the correct L2 subcategory but
the wrong parent L1 category, the L2 prediction is counted as wrong.
It penalizes cross-level inconsistencies.

**`hcr` (Hierarchical Consistency Rate)**: measures whether `l2_to_l1_mapping[pred_L2] == pred_L1`.
It is an internal consistency — it does not look at the ground truth.
A model can have HCR=100% but poor F1 (if it is consistent but
systematically in the wrong direction).

---

## Produced artifacts

```
models/flat_hydra/
    tokenizer/               serialized CamemBERT tokenizer
    best_model.pt            weights of the best checkpoint (best val f1_l2)
    checkpoint_latest.pt     resume checkpoint (epoch, optimizer, scheduler, scaler)
    l1_label_encoder.pkl     sklearn LabelEncoder → decode int → L1 label
    l2_label_encoder.pkl     sklearn LabelEncoder → decode int → L2 label
    l2_to_l1_mapping.json    L2 → L1 hierarchy (for HCR and constrained F1)
    config.json              full FlatHydraConfig + num_classes
    metrics.json             per-epoch history (fallback without MLflow)
    training_complete.txt    completion flag (prevents accidental SLURM re-run)
```

### Contents of `checkpoint_latest.pt`

```python
{
    "model_state_dict":     ...,   # FlatHydra weights
    "optimizer_state_dict": ...,   # AdamW state (moments m, v)
    "scheduler_state_dict": ...,   # LR scheduler state (current step)
    "scaler_state_dict":    ...,   # GradScaler state (scale factor S)
    "epoch":                N,     # next epoch to resume from
    "best_val_f1":          0.xx,  # best F1 L2 reached
    "epochs_no_improve":    K,     # early stopping counter
    "history":              {...}, # full curves
}
```

---

## Comparison with our HYDRA variants

| Criterion                | Flat (baseline)                      | HYDRA local/global/nested       |
|:-------------------------|:-------------------------------------|:--------------------------------|
| Classification heads     | 1 unified head                       | 2 separate heads (L1 + L2)      |
| Loss                     | HYDRALoss — CE L1 + CE L2 (slicing) | HYDRALoss — CE L1 + CE L2       |
| Labels format            | Separate integers (l1, l2)           | Separate integers (l1, l2)      |
| Inference                | Per-level softmax + argmax           | Per-level softmax + argmax      |
| Explicit hierarchy       | No                                   | Yes (mask / conditioning)       |
| `dropout`                | `0.2`                                | `0.3`                           |
| `level_weight_l1`        | `0.5`                                | `0.5`                           |
| Paper reference          | Official baseline (to reproduce)     | Variants to beat                |
