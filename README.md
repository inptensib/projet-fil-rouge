<div align="center">

# 🛒 Rakuten Hierarchical Product Classification

**Classification hiérarchique de 8M+ produits e-commerce en 157 catégories et 6 059 sous-catégories**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-77%20passed-brightgreen?logo=pytest&logoColor=white)

> Projet académique **Télécom Paris** (IP Paris) × **Rakuten Institute of Technology**

</div>

---

## 📌 À propos

Ce projet implémente **HYDRA** (EMNLP 2025), une architecture multi-tête encoder-only pour la classification hiérarchique de texte, appliquée au catalogue produits de Rakuten France.

À partir du titre et de la description d'un produit en français, le modèle prédit simultanément :
- La **catégorie L1** (157 classes) — ex: *Électronique*
- La **sous-catégorie L2** (6 059 classes) — ex: *Coque téléphone*

**Objectif** : maximiser le `f1_l2_macro` tout en maintenant un taux de cohérence hiérarchique (HCR) > 95%.

---

## 🏗️ Architecture HYDRA

```
┌──────────────────────────────────────────────────────────────┐
│  Texte produit                                                │
│  "Coque iPhone 12 silicone transparente..."                   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  CamemBERT (12 couches Transformer, 110M params)             │
│  Encodeur français — toutes couches fine-tunées               │
└──────────────────────┬───────────────────────────────────────┘
                       │ embedding 768d
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Projection (768 → 1536)                                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
          ┌───────────┐ ┌───────────┐
          │  Head L1   │ │  Head L2   │
          │  (→ 157)   │ │  (→ 6059)  │
          └─────┬─────┘ └─────┬─────┘
                │             │
                ▼             ▼
           Catégorie    Sous-catégorie
```

| Variante | Description | Loss |
|----------|-------------|------|
| **Local** | Têtes L1/L2 indépendantes | 0.5 × CE(L1) + CE(L2) |
| **Global** | + tête unifiée sur l'embedding | + α × unified_loss |
| **Nested** | + MLP sur concat(logits L1, logits L2) | + α × unified_loss |

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---------------|-------------|
| **3 variantes HYDRA** | Local, Global, Nested — comparaison sur le même dataset |
| **Fine-tuning complet** | Toutes les 12 couches CamemBERT fine-tunées |
| **DataParallel** | Multi-GPU automatique (détection des GPUs disponibles) |
| **DDP distribué** | Training multi-nœud sur cluster SLURM (P100 / RTX 3090) |
| **Checkpoint resume** | Reprise automatique après interruption SLURM (36h) |
| **MLflow enrichi** | 30+ params, confusion matrices, per-class F1, training curves |
| **Splits pré-générés** | 70/15/15 stratifié, hébergé sur HuggingFace Hub |
| **77 tests pytest** | Couverture modèles, métriques, data loading, hiérarchie |

---

## 🛠️ Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Encodeur | CamemBERT, XLM-RoBERTa, mBERT | Représentation textuelle française |
| Framework | PyTorch, HuggingFace Transformers | Modèle + tokenization |
| Baselines | scikit-learn (TF-IDF + SVM/SGD/LogReg) | Référence non-neuronale |
| Distribué | PyTorch DDP, NCCL, RoCE, SLURM | Training multi-GPU/multi-nœud |
| Tracking | MLflow | Params, métriques, artefacts visuels |
| Infra | Docker Compose, HuggingFace Hub | Reproductibilité, partage dataset |
| Tests | pytest | 77 tests automatisés |

---

## 🚀 Quick Start

### Prérequis

- Python 3.10+
- GPU CUDA (recommandé : RTX 3090 24GB ou P100 16GB)

### Installation

```bash
git clone <repo-url> && cd rakuten-classification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Tests

```bash
python -m pytest tests/ -v    # 77 tests
```

### Docker (alternatif)

```bash
docker compose run --rm --profile test test       # Tests
docker compose up mlflow -d                       # MLflow UI → http://localhost:5000
docker compose run --rm --profile gpu train       # Training (GPU)
```

---

## 📊 Dataset

Le dataset Rakuten est propriétaire et **non inclus** dans ce repository.

### Obtenir le dataset

**Option 1 — HuggingFace Hub** (splits pré-générés, recommandé) :

```bash
export HF_TOKEN="hf_..."   # Token read-only (demander à Tahiana)
python scripts/download_dataset.py
```

**Option 2 — Fichier brut** :

1. Obtenir `products.jsonl` auprès du Rakuten Institute of Technology
2. Le placer dans `../dataset/`
3. Générer les splits : `sbatch scripts/run_split.slurm`

### Format

| Champ | Type | Description |
|-------|------|-------------|
| `designation` | str | Titre du produit |
| `description` | str | Description détaillée |
| `typelabel` | str | Catégorie L1 (157 classes) |
| `producttype` | list[str] | Sous-catégorie L2 (6 059 classes) |

### Splits

| Split | Échantillons | % |
|-------|-------------|---|
| Train | 5 573 659 | 70% |
| Validation | 1 194 355 | 15% |
| Test | 1 194 356 | 15% |

---

## 🔬 Training

### Single-GPU (prototypage)

```bash
python scripts/run_train.py --split-dir ../dataset/ --variant local
```

### Multi-GPU DataParallel (RTX 3090)

```bash
sbatch scripts/run_train_3090.slurm           # HYDRA Local
sbatch scripts/run_train_3090_nested.slurm     # HYDRA Nested
```

### Multi-nœud DDP (cluster P100)

```bash
./03_production/scripts/launch_distributed.sh local CamemBERT 4
```

### Reprendre après interruption

Le checkpoint est sauvegardé à chaque epoch. Il suffit de resoumettre le même job SLURM — la reprise est automatique.

### Configuration

Source de vérité : `02_hydra/configs/default.yaml`

| Paramètre | Valeur |
|-----------|--------|
| Encodeur | `camembert-base` |
| `max_length` | 256 tokens |
| `batch_size` | 64 (×2 GPUs = 128 effectif) |
| `lr` | 3.5e-5 |
| Warmup | Linéaire, 10% des steps |
| Weight decay | 0.01 (sauf bias/LayerNorm) |
| Early stopping | Patience 2, sur `val_f1_l2_macro` |
| Epochs max | 10 |

---

## 📈 Métriques

| Métrique | Description | Cible |
|----------|-------------|-------|
| `f1_l1_macro` | F1 macro sur les 157 catégories L1 | — |
| `f1_l2_macro` | F1 macro sur les 6 059 sous-catégories L2 | **Maximiser** |
| `constrained_f1` | F1 L2 valide uniquement si L1 parent correct | — |
| `HCR` | % de prédictions où L2 est enfant valide de L1 | **> 95%** |

### Dashboard MLflow

Chaque run génère automatiquement :

| Artefact | Description |
|----------|-------------|
| `confusion_matrix_l1.png` | Heatmap 157×157 classes L1 |
| `confusion_matrix_l2_top50.png` | Heatmap top 50 classes L2 |
| `f1_per_class_l1.png` | Barplot F1 trié par classe L1 |
| `f1_per_class_l2.png` | Top/bottom 30 classes L2 |
| `training_curves.png` | Loss + F1 + HCR par epoch |
| `hard_val_step_*.json` | Exemples difficiles (erreurs + low confidence) |

---

## 📁 Structure du projet

```
rakuten-classification/
├── 01_baselines/src/          TF-IDF, Word2Vec, BERT flat (baselines)
├── 02_hydra/
│   ├── configs/default.yaml   Hyperparamètres (source de vérité)
│   └── src/                   models.py, losses.py, trainer.py, predict.py
├── 03_production/
│   ├── src/                   main.py, trainer_ddp.py, data_sharding.py
│   └── scripts/               launch_distributed.sh, launch_node.sh
├── shared/
│   ├── data_loader.py         Chargement, encoding, splits
│   ├── metrics.py             F1, HCR, confusion matrices, plots
│   └── utils.py               ENCODERS, enforce_hierarchy, seeds
├── scripts/
│   ├── split_dataset.py       Split 70/15/15 stratifié (2 passes)
│   ├── download_dataset.py    Téléchargement depuis HF Hub
│   ├── run_train.py           CLI single-GPU
│   └── run_train_3090.slurm   Job SLURM RTX 3090
├── tests/                     77 tests pytest
├── models/                    Checkpoints (best_model.pt, checkpoint_latest.pt)
├── Dockerfile                 Environnement reproductible
├── docker-compose.yml         MLflow, tests, training
└── blog.md                    Article technique détaillé
```

---

## 📚 Références

- **HYDRA** — Karl et al., *Hierarchical Multi-label Classification using Attention-based Architecture*, EMNLP 2025. [[GitHub]](https://github.com/FKarl/HYDRA)
- **HLPD** — Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder for Multi-label Text Classification*, ECIR 2025.

---

## 👥 Auteurs

| Nom | Rôle |
|-----|------|
| **Tahiana H. Andriambahoaka** | Architecture HYDRA, training distribué, MLOps |
| **Mehdi Aithamma** | Baselines, évaluation |
| **Younes Boutrouft** | Infrastructure, déploiement |
| **Amir H. Kashani Rahimi** | Data engineering, analyse |

> *Remerciements au Rakuten Institute of Technology pour le dataset et la collaboration.*
