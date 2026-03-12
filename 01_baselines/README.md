# 01_baselines -- Flat Classification Baselines

Classification "a plat" (non-hierarchique) sur les 157 categories du catalogue Rakuten. Sert de reference pour evaluer les gains de l'approche hierarchique HYDRA.

## Approches

| # | Methode | Description |
|---|---------|-------------|
| 1 | **TF-IDF + Linear SVM** | Bag-of-words (unigrams+bigrams, 50k features) + LinearSVC |
| 2 | **Word2Vec + Linear SVM** | Skip-gram 200-d, moyenne des embeddings + LinearSVC |
| 3 | **BERT Flat** | CamemBERT / XLM-R / mBERT + tete lineaire (feature extraction ou fine-tuning) |

## Resultats (50k samples, 80/20 split)

| Methode | F1 Macro | F1 Weighted | Temps |
|---------|:--------:|:-----------:|:-----:|
| TF-IDF + LinearSVC | **86.2%** | **88.6%** | ~37s |
| TF-IDF + LogReg | 65.9% | 75.2% | ~53s |
| Word2Vec + LinearSVC | 69.8% | 76.6% | ~169s |
| BERT CamemBERT (fine-tuned) | ~83% | ~87% | ~min |
| BERT XLM-RoBERTa (fine-tuned) | ~84% | ~88% | ~min |
| BERT feature extraction | ~5-10% | ~5-10% | ~min |

> TF-IDF + LinearSVC est une baseline etonnamment forte. Le feature extraction BERT (encodeur gele) echoue car les representations pre-entrainees ne sont pas adaptees au domaine produit.

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
    tfidf_pipeline.py        # TF-IDF + classifieurs lineaires
    word2vec_pipeline.py      # Word2Vec + classifieurs lineaires
    bert_flat_classifier.py   # BERT feature extraction / fine-tuning
    evaluate.py               # Comparaison cross-methodes
```
