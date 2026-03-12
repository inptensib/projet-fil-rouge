# Rakuten Hierarchical Product Classification

Projet academique **Telecom Paris** (IP Paris) x **Rakuten Institute of Technology**.

Classification hierarchique de produits e-commerce : **8M produits**, **157 categories (L1)**, **6 100+ sous-categories (L2)**.

## Resultats cles

| Modele | F1 L1 Macro | F1 L2 Macro | F1 L1 Weighted | F1 L2 Weighted |
|--------|:-----------:|:-----------:|:--------------:|:--------------:|
| TF-IDF + SVM (baseline) | 86.2% | -- | 88.6% | -- |
| **HYDRA Local** (8M, 3 epochs) | **92.5%** | **71.4%** | **93.9%** | **84.8%** |

## Structure

```
rakuten-classification/
├── 01_baselines/       # TF-IDF, Word2Vec, BERT flat
├── 02_hydra/           # HYDRA : multi-tete hierarchique (EMNLP 2025)
├── 02b_hlpd/           # HLPD : decodeur non-autoregressif (ECIR 2025, a venir)
├── 03_production/      # Training distribue DDP sur cluster SLURM
├── shared/             # data_loader, metrics, utils
├── scripts/            # run_train.py, setup_cluster.sh
├── tests/              # 61 tests pytest
├── notebooks/          # EDA, visualisations
├── Dockerfile          # Environnement reproductible
├── docker-compose.yml  # MLflow, tests, training, baselines
└── blog.md             # Blog technique detaille
```

## Quick Start

```bash
git clone <repo-url> && cd rakuten-classification

# Option 1 : venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v

# Option 2 : Docker (recommande)
docker compose run --rm --profile test test     # 61 tests
docker compose up mlflow -d                     # MLflow UI :5000
docker compose run --rm --profile gpu train     # HYDRA training (GPU)
docker compose run --rm --profile baselines baseline-tfidf  # TF-IDF baseline
```

## Dataset

Le dataset Rakuten est proprietaire et **non inclus** dans ce repository.

1. Obtenir `products.jsonl` aupres du Rakuten Institute of Technology
2. Le placer dans `shared/dataset/` :
   ```bash
   cp /chemin/vers/products.jsonl shared/dataset/
   ```

Sans ce fichier, seuls les tests et MLflow fonctionnent. Le training et les baselines necessitent le dataset.

## Stack

| | |
|---|---|
| **Encodeurs** | CamemBERT, XLM-RoBERTa, mBERT |
| **Framework** | PyTorch, HuggingFace Transformers |
| **Baselines** | scikit-learn (SVM, SGD, LogReg) |
| **Distribue** | PyTorch DDP, NCCL, SLURM (8x P100) |
| **MLOps** | MLflow, Docker Compose |

## References

- **HYDRA** -- Karl et al., *Hierarchical Multi-label Classification using Attention-based Architecture*, EMNLP 2025.
- **HLPD** -- Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder for Multi-label Text Classification*, ECIR 2025.

## Auteurs

Tahiana H. Andriambahoaka, Mehdi Aithamma, Younes Boutrouft, Amir H. Kashani Rahimi

*Remerciements au Rakuten Institute of Technology pour le dataset et la collaboration.*
