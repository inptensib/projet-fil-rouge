# HYDRA v2 (hierarchy) — ONNX inference

Standalone ONNX inference for the **3 HYDRA hierarchy models** trained on dataset
v2 (local / nested / global), whose defining property is a **hierarchical mask at
inference** that forces L2 to be a valid child of the predicted L1 — giving the
maximal hierarchical consistency (HCR).

## Performance (test set, dataset v2, mode hier)

| Variant | F1 L1 | F1 L2 | HCR | HCR multi-parent | Constr. F1 |
|---------|------:|------:|----:|-----------------:|-----------:|
| local   | 92.60 | 77.38 | 87.04 | **100.0** | 79.31 |
| nested  | 92.59 | 76.41 | 87.02 | **100.0** | 78.56 |
| global  | 92.91 | 77.29 | 86.99 | **100.0** | 79.17 |

Source: `results/hydra_hierarchy_v2_metrics.json`. The retained product model is
the simpler **CamemBERT Flat v2** (F1 L2 78.93 %, see `../flat_inference/`); these
HYDRA models are provided for their consistency guarantee.

## How it works

The 3 variants all predict from their **local heads** `(logits_l1, logits_l2)`
(the global/nested extra head is only used at training). The ONNX graph exposes
exactly these two outputs. The hierarchical mask is applied in **post-processing**
(numpy), reproducing the PyTorch `enforce_hierarchy`:

```
pred_l1 = argmax(logits_l1)
logits_l2[ parent_child_matrix[pred_l1] == 0 ] = -inf      # keep only valid children
pred_l2 = argmax(logits_l2)
```

`parent_child_matrix_v2.npy` (157 × 5 775, **multi-parent**: 545 L2 have >1 valid
L1 parent) is built from the v2 data by `build_hierarchy_matrix.py`.

## Files

| File | Role |
|------|------|
| `export_onnx.py` | (dev) export a HYDRA checkpoint → ONNX `(logits_l1, logits_l2)` + parity check |
| `build_hierarchy_matrix.py` | (dev) build `parent_child_matrix_v2.npy` from the v2 splits |
| `predict_onnx.py` | inference: onnxruntime + numpy masking — **no torch** |
| `parent_child_matrix_v2.npy` | the multi-parent mask matrix (shipped, ~900 KB) |

The `.onnx` weights (~454 MB each) are not in git — fetch from the model hub
(`hydra-{local,nested,global}-hierarchy-camembert-v2/`). Tokenizer + label
encoders are **reused from the Flat v2 bundle** (`../flat_inference/`, same v2
label space 157/5 775 — the HYDRA folders ship no encoders).

## Usage

```bash
# export (once, dev side) — needs torch + the rakuten package
python3 export_onnx.py \
    --ckpt models/v2/HYDRA_local_CamemBERT/best_model.pt \
    --out  models/v2/HYDRA_local_CamemBERT/hydra_local_v2.onnx

# inference (no torch) — masking ON by default (the whole point)
python3 predict_onnx.py \
    --onnx models/v2/HYDRA_local_CamemBERT/hydra_local_v2.onnx \
    --encoders models/v2/flat_hydra \
    --matrix scripts/hydra_inference/parent_child_matrix_v2.npy \
    --designation "40 Pcs Automatique Poulet Buveur D'eau Volaille Abreuvoir Potable Mamelon Élevage Accessoires"
```

`--no-hierarchy` disables the mask (raw local-head predictions). `--top-k N`,
`--input/--output` (JSONL batch) as in `flat_inference/predict_onnx.py`.

## Validation

ONNX vs PyTorch (with masking): **identical predictions** (12/12 real test rows
per variant); logits parity < 4e-5; HCR = 100 % by construction of the mask.
`predict_onnx.py` needs `onnxruntime` + `transformers` (tokenizer) +
`scikit-learn` (encoders) + `numpy` — **not** torch.
