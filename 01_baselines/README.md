# 01_baselines -- Flat Classification Baselines

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

```bash
python src/tfidf_pipeline.py --data_path shared/dataset/products.jsonl --n_samples 50000
python src/word2vec_pipeline.py --data_path shared/dataset/products.jsonl --n_samples 50000
python src/bert_flat_classifier.py --data_path shared/dataset/products.jsonl --backbone camembert --mode fine_tuning
```

## Structure

```
01_baselines/
  src/
    tfidf_pipeline.py        # TF-IDF + classifieurs linéaires
    word2vec_pipeline.py      # Word2Vec + classifieurs linéaires
    bert_flat_classifier.py   # BERT feature extraction / fine-tuning
    evaluate.py               # Comparaison cross-méthodes
```
