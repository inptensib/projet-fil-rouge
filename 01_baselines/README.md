# 01_baselines -- Flat Classification Baselines

Baseline approaches for Rakuten product classification using flat (non-hierarchical)
methods. Each approach treats the problem as a single multi-class classification task
over the `typelabel` column (~165 classes).

## Approaches

### 1. TF-IDF + Linear SVM (`tfidf_pipeline.py`)

Bag-of-words baseline using TF-IDF vectorisation (unigrams + bigrams, 50k features)
followed by a LinearSVC classifier. Fast to train, strong baseline.

```bash
python src/tfidf_pipeline.py --data_path ../../dataset/train.jsonl --n_samples 50000
```

### 2. Word2Vec + Linear SVM (`word2vec_pipeline.py`)

Trains Skip-gram Word2Vec embeddings (200-d), computes document vectors by
averaging word embeddings, then classifies with LinearSVC.

```bash
python src/word2vec_pipeline.py --data_path ../../dataset/train.jsonl --n_samples 50000
```

### 3. BERT Flat Classifier (`bert_flat_classifier.py`)

Transformer-based classification using the [CLS] token + linear head.
Supports CamemBERT, XLM-RoBERTa, and mBERT backbones in two modes:

- **Feature extraction** -- BERT encoder frozen, only the classification head is trained.
- **Fine-tuning** -- All BERT and head parameters are trained end-to-end.

```bash
# Feature extraction
python src/bert_flat_classifier.py --data_path ../../dataset/train.jsonl \
    --backbone camembert --mode feature_extraction

# Full fine-tuning
python src/bert_flat_classifier.py --data_path ../../dataset/train.jsonl \
    --backbone camembert --mode fine_tuning --epochs 5
```

## Evaluation

Compare all baselines with a single command:

```bash
python src/evaluate.py --results_dirs models_tfidf models_word2vec \
    models_bert_flat_camembert_fine_tuning --detailed
```

## Results (50k samples, 80/20 split)

| Method                           | F1-Macro | F1-Weighted | Train Time |
|----------------------------------|----------|-------------|------------|
| TF-IDF + LinearSVC               | 86.2%    | 88.6%       | ~37s       |
| TF-IDF + Logistic Regression     | 65.9%    | 75.2%       | ~53s       |
| TF-IDF + SGDClassifier           | 65.5%    | 74.4%       | ~3s        |
| Word2Vec + LinearSVC             | 69.8%    | 76.6%       | ~169s      |
| Word2Vec + Logistic Regression   | 68.5%    | 75.8%       | ~399s      |
| Word2Vec + SGDClassifier         | 62.9%    | 71.2%       | ~29s       |
| BERT CamemBERT (fine-tuned)      | ~83%     | ~87%        | ~minutes   |
| BERT XLM-RoBERTa (fine-tuned)    | ~84%     | ~88%        | ~minutes   |
| BERT mBERT (fine-tuned)          | ~82%     | ~86%        | ~minutes   |
| BERT feature extraction          | ~5-10%   | ~5-10%      | ~minutes   |

Notes:
- TF-IDF + LinearSVC is a surprisingly strong baseline for this dataset.
- BERT feature extraction (frozen encoder) performs poorly since the pre-trained
  representations are not adapted to the product classification domain.
- BERT fine-tuning reaches competitive performance but at much higher compute cost.

## File Structure

```
01_baselines/
  src/
    __init__.py
    tfidf_pipeline.py        # TF-IDF + linear classifiers
    word2vec_pipeline.py      # Word2Vec + linear classifiers
    bert_flat_classifier.py   # BERT feature extraction / fine-tuning
    evaluate.py               # Cross-method comparison
  README.md
```
