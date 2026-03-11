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
│   ├── src/               #   model.py, hierarchical_attention.py, losses.py
│   └── configs/
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

## Resultats

Metriques evaluees sur le jeu de test (split 80/10/10).

| Modele | F1 Macro L1 | F1 Macro L2 | F1 Weighted L1 | F1 Weighted L2 |
|--------|:-----------:|:-----------:|:--------------:|:--------------:|
| TF-IDF + SVM lineaire | 0.870 | 0.620 | — | — |
| Word2Vec + SVM | 0.855 | 0.600 | — | — |
| CamemBERT (flat) | 0.910 | 0.680 | — | — |
| **HYDRA Local (ours)** | **0.925** | **0.714** | **0.939** | **0.848** |

---

## Architecture HYDRA

```
Texte (titre + description)
        |
  [Tokenizer : max_length=128]
        |
  [Encodeur partage : CamemBERT / XLM-RoBERTa / mBERT]
        |
  [Projection d'embedding]
        |
   +--------+--------+
   |                  |
[Head L1]         [Head L2]
(157 classes)     (6100+ classes)
   |                  |
 Softmax           Softmax
```

3 variantes implementees :
- **HYDRALocal** : tetes L1 et L2 independantes
- **HYDRAGlobal** : + tete globale (vue unifiee L1+L2)
- **HYDRANested** : + contraintes d'imbrication hierarchique

---

## Tests

83 tests unitaires couvrant tous les modules :

```bash
$ python -m pytest tests/ -v
# 83 passed in 15s
```

| Fichier de test | Module teste | Tests |
|-----------------|-------------|:-----:|
| `test_hydra_models.py` | Modeles HYDRA (3 variantes + pooling) | 11 |
| `test_hydra_losses.py` | FocalLoss, HYDRALoss | 10 |
| `test_hlpd.py` | Hierarchical Self-Attention, ZLPR Loss | 20 |
| `test_shared_metrics.py` | F1, Constrained F1, HCR | 11 |
| `test_shared_data_loader.py` | Chargement et preprocessing | 10 |
| `test_shared_utils.py` | Utilitaires (seed, device, params) | 6 |
| `test_production.py` | DDP utils | 3 |
| `test_imports.py` | Verification des imports | 7 |

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

Encadrante : **Charlotte Laclau** (Telecom Paris)

---

## Remerciements

Nous remercions le **Rakuten Institute of Technology** pour la mise a disposition du dataset et pour leur collaboration avec la communaute academique.
