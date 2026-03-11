# Rakuten Hierarchical Product Classification

Projet academique de **Telecom Paris** (Institut Polytechnique de Paris) en partenariat avec le **Rakuten Institute of Technology (RIT)**.

Classification hierarchique de produits e-commerce sur le catalogue Rakuten France : **8 millions de produits**, organises en **157 categories (L1)** et **6 100+ sous-categories (L2)**.

---

## Structure du projet

```
projet-fil-rouge/
├── 01_baselines/          # Baselines : TF-IDF, Word2Vec, BERT flat
│   ├── src/               #   tfidf_pipeline.py, word2vec_pipeline.py, bert_flat_classifier.py
│   └── configs/
├── 02_hydra/              # HYDRA : classification hierarchique multi-tete (EMNLP 2025)
│   ├── src/               #   models.py, losses.py, trainer.py, predict.py
│   └── configs/
├── 02b_hlpd/              # HLPD : decodeur hierarchique non-autoregressif (ECIR 2025)
│   └── README.md          #   Architecture detaillee (code a venir)
├── 03_production/         # Entrainement distribue (DDP) + MLOps
│   ├── src/               #   trainer_ddp.py, data_sharding.py, ddp_utils.py, error_analysis.py
│   └── scripts/           #   launch_distributed.sh, launch_node.sh
├── shared/                # Code partage (data_loader, metrics, utils)
├── tests/                 # Suite de tests pytest (83 tests)
├── notebooks/             # Notebooks exploratoires (EDA, visualisations)
├── blog.md                # Blog technique detaille du projet
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# Cloner le repository
git clone https://github.com/inptensib/projet-fil-rouge.git
cd projet-fil-rouge

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dependances
pip install -r requirements.txt

# Lancer les tests
python -m pytest tests/ -v
```

---

## Tests

Tests unitaires couvrant les modules implementes :

```bash
$ python -m pytest tests/ -v
```

| Fichier de test | Module teste | Tests |
|-----------------|-------------|:-----:|
| `test_hydra_models.py` | Modeles HYDRA (3 variantes + pooling) | 11 |
| `test_hydra_losses.py` | FocalLoss, HYDRALoss | 10 |
| `test_shared_metrics.py` | F1, Constrained F1, HCR | 11 |
| `test_shared_data_loader.py` | Chargement et preprocessing | 10 |
| `test_shared_utils.py` | Utilitaires (seed, device, params) | 6 |
| `test_production.py` | DDP utils | 3 |
| `test_imports.py` | Verification des imports | 5 |

---

## Stack technique

| Couche | Technologies |
|--------|-------------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| Encodeurs NLP | CamemBERT, XLM-RoBERTa, mBERT |
| ML Classique | scikit-learn (SVM, SGD, Logistic Regression) |
| Entrainement distribue | PyTorch DDP, NCCL, SLURM |
| MLOps | MLflow |
| Hardware | 8x Tesla P100 (cluster HPC) |

---

## References

- **HYDRA** — Karl et al., *HYDRA: Hierarchical Multi-label Classification using Attention-based Architecture*, EMNLP 2025.
- **HLPD** — Torba et al., *Hierarchical Label Pre-populated Decoder for Multi-label Text Classification*, ECIR 2025.
- **These** — Fatos Torba, *Classification hierarchique multi-label de textes*, 2025.

---

## Auteurs

- **Tahiana Hajanirina ANDRIAMBAHOAKA**
- **Mehdi Aithamma**
- **Younes Boutrouft**
- **Amir Hossein Kashani Rahimi**

---

## Remerciements

Nous remercions le **Rakuten Institute of Technology** pour la mise a disposition du dataset et pour leur collaboration avec la communaute academique.
