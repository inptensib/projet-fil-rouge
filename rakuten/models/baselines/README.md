# Flat Classification (`rakuten.models.baselines`)

Classification « à plat » (non-hiérarchique) L1+L2. Ce package contient
**FlatHydra**, le modèle flat **retenu** (CamemBERT, tête unique L1+L2 — meilleur
F1 L2 macro du projet).

Les baselines classiques de comparaison (TF-IDF, Word2Vec, fastText) vivent dans
[`notebooks/baselines/`](../../../notebooks/baselines/) ; leurs chiffres
canoniques sont dans `results/baselines/baseline_results.json` (cf.
[docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)).

## FlatHydra

- `flat_model.py` — projection 768→1536 **avant** le pooling CLS, puis une tête
  unique produisant les logits L1+L2 (slicés `[:157]` / `[157:]`).
- `flat_trainer.py` — `FlatHydraConfig` + `FlatHydraTrainer` (entraînement,
  évaluation, reprise sur checkpoint).

## Configurations FlatHydra

| Config (`configs/baselines/`) | Encodeur | GPU | Batch | Description |
|---|---|---|:--:|---|
| `flat_hydra.yaml` | `camembert-base` | CPU / debug | 64 | Référence, source de vérité pour `FlatHydraConfig` |
| `flat_hydra_full.yaml` | `camembert-base` | 2× P100 | 256 | Full training dataset complet |
| `flat_hydra_3090.yaml` | `camembert-base` | 2× RTX 3090 | 256 | Full training dataset complet |
| `flat_hydra_mbert.yaml` | `bert-base-multilingual-cased` | 2× P100 | 128 | mBERT (plus lourd que CamemBERT) |
| `flat_hydra_title_3090.yaml` | `camembert-base` | 2× RTX 3090 | 512 | Title-only (`max_length: 128`) |

> Les configs ciblant le **dataset v2** dé-fuité (suffixe `_v2` :
> `flat_hydra_camembert_v2.yaml`, `flat_hydra_v2.yaml`, `flat_hydra_mbert_v2.yaml`)
> activent le kit de rééquilibrage v2 — cf. [docs/PIPELINE.md](../../../docs/PIPELINE.md).

## Structure

```
rakuten/models/baselines/     # code (package installable)
  flat_model.py               # FlatHydra : tête unique L1+L2, projection avant pooling
  flat_trainer.py             # FlatHydraConfig + FlatHydraTrainer (train, eval, resume)
configs/baselines/            # configurations FlatHydra (flat_hydra*.yaml, hors package)
```
