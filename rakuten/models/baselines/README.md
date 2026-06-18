# Baselines — Flat Classification (`rakuten.models.baselines`)

Classification "à plat" (non-hiérarchique) sur les **157 catégories L1** du catalogue Rakuten. Sert de référence pour évaluer les gains de l'approche hiérarchique HYDRA sur le niveau 1.

## Approches

| # | Méthode | Description |
|---|---------|-------------|
| 1 | **TF-IDF + Linear SVM** | Bag-of-words (unigrams+bigrams, 50k features) + LinearSVC |
| 2 | **Word2Vec + Linear SVM** | Skip-gram 200-d, moyenne des embeddings + LinearSVC |
| 3 | **BERT Flat** | CamemBERT / XLM-R / mBERT + tête linéaire (feature extraction ou fine-tuning) |

## Résultats L1 (50k samples, 80/20 split, 157 catégories)

| Méthode | F1 Macro | F1 Weighted | Temps |
|---------|:--------:|:-----------:|:-----:|
| TF-IDF + LinearSVC | **86.2%** | **88.6%** | ~37s |
| TF-IDF + LogReg | 65.9% | 75.2% | ~53s |
| Word2Vec + LinearSVC | 69.8% | 76.6% | ~169s |
| BERT CamemBERT (fine-tuned) | ~83% | ~87% | ~min |
| BERT XLM-RoBERTa (fine-tuned) | ~84% | ~88% | ~min |
| BERT feature extraction | ~5-10% | ~5-10% | ~min |

> TF-IDF + LinearSVC est une baseline étonnamment forte. Le feature extraction BERT (encodeur gelé) échoue car les représentations pré-entraînées ne sont pas adaptées au domaine produit.

## Usage

Commandes à lancer depuis la racine du repo :

```bash
python -m rakuten.models.baselines.tfidf_pipeline --data_path <chemin>/products.jsonl --n_samples 50000
python -m rakuten.models.baselines.word2vec_pipeline --data_path <chemin>/products.jsonl --n_samples 50000
python -m rakuten.models.baselines.bert_flat_classifier --data_path <chemin>/products.jsonl --backbone camembert --mode fine_tuning
```

## Configurations FlatHydra

| Config (`configs/`)          | Encodeur                         | GPU         | Batch | Description |
|------------------------------|----------------------------------|-------------|:-----:|-------------|
| `flat_hydra.yaml`            | `camembert-base`                 | CPU / debug | 64    | Référence, source de vérité pour `FlatHydraConfig` |
| `flat_hydra_full.yaml`       | `camembert-base`                 | 2× P100     | 256   | Full training dataset complet (128 samples/GPU) |
| `flat_hydra_3090.yaml`       | `camembert-base`                 | 2× RTX 3090 | 256   | Full training dataset complet (~56 % VRAM/GPU) |
| `flat_hydra_mbert.yaml`      | `bert-base-multilingual-cased`   | 2× P100     | 128   | mBERT (179M params, plus lourd que CamemBERT) |
| `flat_hydra_title_3090.yaml` | `camembert-base`                 | 2× RTX 3090 | 512   | Title-only (`max_length: 128`, `lr: 4.9e-5` scaling √k) |

> Les configs ciblant le **dataset v2** dé-fuité (suffixe `_v2`, p.ex.
> `flat_hydra_v2.yaml`, `flat_hydra_camembert_v2.yaml`, `flat_hydra_mbert_v2.yaml`)
> activent le kit de rééquilibrage v2 — cf. [docs/PIPELINE.md](../../../docs/PIPELINE.md).

## Structure

```
rakuten/models/baselines/     # code (package installable)
  tfidf_pipeline.py           # TF-IDF + classifieurs linéaires
  word2vec_pipeline.py        # Word2Vec + classifieurs linéaires
  bert_flat_classifier.py     # BERT feature extraction / fine-tuning
  flat_model.py               # FlatHydra : tête unique L1+L2, projection avant pooling
  flat_trainer.py             # FlatHydraConfig + FlatHydraTrainer (train, eval, resume)
  common.py                   # Utilitaires partagés (stopwords, filtrage classes rares)
  evaluate.py                 # Comparaison cross-méthodes
configs/baselines/            # Configurations FlatHydra (flat_hydra*.yaml, hors package)
```
