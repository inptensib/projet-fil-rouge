# Rakuten product classifier — CamemBERT Flat (v2)

Standalone inference package for the **retained model** of the Télécom Paris ×
Rakuten project: hierarchical classification of French product listings into
**157 L1 categories** (`typelabel`) and **5 775 L2 categories** (`producttype`).

> **Note on class counts:** the model has **5 775** L2 heads — 2 of the dataset's
> 5 777 L2 classes have no training sample after de-duplication, so the head count
> is derived from the data. See [`docs/METRICS.md`](../../docs/METRICS.md) for the
> v1 → v2 dataset history.

## Performance (test set, dataset v2)

| Metric | Value |
|--------|------:|
| F1 L1 macro | **92.47 %** |
| F1 L2 macro | **78.93 %** |
| Constrained F1 (L2 valid only if L1 parent correct) | 79.90 % |
| Hierarchical consistency rate (HCR) | 85.17 % |

Source: `results/flat_hydra_v2/test_metrics.json`. Macro = every class weighs
equally (the hard target on a long-tailed 5 775-class problem).

## Architecture

A single `camembert-base` encoder feeds **one unified head** that predicts L1 and
L2 jointly:

```
text -> CamemBERT-base -> Linear(768 -> 1536)   (projection BEFORE pooling)
     -> CLS pooling -> MLP[1536 -> 1536 -> 5 932]
     -> logits[:157] = L1   |   logits[157:] = L2
```

`max_length = 256`. Input text = `designation + ". " + description`.

## Install

```bash
pip install -r requirements.txt
```

## Get the weights

The weights (~497 MB) are not in git. Download them from the model hub
(`TahianaAndriambahoaka/rakuten-models`, private — set a token):

```bash
export HF_TOKEN=hf_...
python download_weights.py .        # -> ./flat-hydra-camembert-v2
```

The model directory contains: `best_model.pt`, `tokenizer/`,
`l1_label_encoder.pkl`, `l2_label_encoder.pkl`, `l2_to_l1_mapping.json`,
`config.json`.

> On first run, `camembert-base` is fetched from the Hugging Face Hub (its
> weights are then overwritten by the trained checkpoint). Set
> `HF_HOME` / `TRANSFORMERS_OFFLINE=1` to use a local cache.

## Usage

### Single product

```bash
python predict.py --model-dir flat-hydra-camembert-v2 \
    --designation "Cage à hamster 3 étages tubes" \
    --description "Cage en métal pour rongeurs, 60x40 cm, roue et abreuvoir inclus"
```

```json
{
  "text": "Cage à hamster 3 étages tubes. Cage en métal ...",
  "level1": {"label": "Accessoires animalerie", "confidence": 0.98},
  "level2": {"label": "Cage à rongeurs", "confidence": 0.74},
  "hierarchy_enforced": false
}
```

### Batch (JSONL in → JSONL out)

```bash
python predict.py --model-dir flat-hydra-camembert-v2 \
    --input products.jsonl --output predictions.jsonl
```

Each input line: `{"designation": "...", "description": "..."}`. Each output line
is the JSON above.

### Python API

```python
from predict import FlatClassifier

clf = FlatClassifier("flat-hydra-camembert-v2")
print(clf.classify("Ballon de football taille 5", "Cuir synthétique, cousu main"))
```

## Options

| Flag | Effect |
|------|--------|
| `--top-k N` | Return the N most likely L2 categories (`level2_topk`). |
| `--enforce-hierarchy` | Constrain L2 to children of the predicted L1 (single primary parent in `l2_to_l1_mapping.json`). Off by default — the reported metrics are the raw, unconstrained predictions. |
| `--device cpu\|cuda` | Force device (default: auto). |

## Notes

- Self-contained: `predict.py` embeds a copy of the model definition (kept in
  sync with `rakuten/models/baselines/flat_model.py`), so it runs without the
  training package.
- French input. Designation alone works; the description improves accuracy.
- CPU inference is fine for low volume; use a GPU for large batches.
