# Rakuten Hierarchical Product Classification

Projet académique **Télécom Paris** (IP Paris) x **Rakuten Institute of Technology**.

Classification hiérarchique de produits e-commerce : **8M produits**, **157 catégories (L1)**, **6 100+ sous-catégories (L2)**.

## Structure

```
rakuten-classification/
├── 01_baselines/       # TF-IDF, Word2Vec, BERT flat
├── 02_hydra/           # HYDRA : multi-tête hiérarchique (EMNLP 2025)
├── 02b_hlpd/           # HLPD : décodeur non-autorégressif (ECIR 2025, à venir)
├── 03_production/      # Training distribué DDP sur cluster SLURM
├── shared/             # data_loader, metrics, utils
├── scripts/            # run_train.py, setup_cluster.sh
├── tests/              # 61 tests pytest
├── notebooks/          # EDA, visualisations
├── Dockerfile          # Environnement reproductible
├── docker-compose.yml  # MLflow, tests, training, baselines
└── blog.md             # Blog technique détaillé
```

## Quick Start

```bash
git clone <repo-url> && cd rakuten-classification

# Option 1 : venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v

# Option 2 : Docker (recommandé)
docker compose run --rm --profile test test     # 61 tests
docker compose up mlflow -d                     # MLflow UI :5000
docker compose run --rm --profile gpu train     # HYDRA training (GPU)
docker compose run --rm --profile baselines baseline-tfidf  # TF-IDF baseline
```

## Dataset

Le dataset Rakuten est propriétaire et **non inclus** dans ce repository.

1. Obtenir `products.jsonl` auprès du Rakuten Institute of Technology
2. Le placer dans `shared/dataset/` :
   ```bash
   cp /chemin/vers/products.jsonl shared/dataset/
   ```

**Format JSONL** — champs : `designation` (titre), `description` (texte complet), `typelabel` (catégorie L1), `producttype` (sous-catégorie L2).

**Objectif** : maximiser `f1_l2_macro` (chaque sous-catégorie pèse également) tout en maintenant un HCR > 95% (cohérence hiérarchique L1-L2).

Sans ce fichier, seuls les tests et MLflow fonctionnent. Le training et les baselines nécessitent le dataset.

## Stack

| | |
|---|---|
| **Encodeurs** | CamemBERT, XLM-RoBERTa, mBERT |
| **Framework** | PyTorch, HuggingFace Transformers |
| **Baselines** | scikit-learn (SVM, SGD, LogReg) |
| **Distribué** | PyTorch DDP, NCCL, SLURM (8x P100) |
| **MLOps** | MLflow, Docker Compose |

## Références

- **HYDRA** -- Karl et al., *Hierarchical Multi-label Classification using Attention-based Architecture*, EMNLP 2025.
- **HLPD** -- Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder for Multi-label Text Classification*, ECIR 2025.

## Auteurs

Tahiana H. Andriambahoaka, Mehdi Aithamma, Younes Boutrouft, Amir H. Kashani Rahimi

*Remerciements au Rakuten Institute of Technology pour le dataset et la collaboration.*
