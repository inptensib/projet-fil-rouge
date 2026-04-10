<div align="center">

<img src="docs/screenshots/hero.jpg" alt="HieraCat" width="80%" style="border-radius:12px; margin-bottom:16px"/>

# 🛒 Rakuten Hierarchical Product Classification

**8M+ produits · 157 catégories · 6 126 sous-catégories · CamemBERT**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-77%20passed-brightgreen?logo=pytest&logoColor=white)
![SLURM](https://img.shields.io/badge/SLURM-Multi--GPU-orange)

> **Télécom Paris** (IP Paris) × **Rakuten France**

</div>

---

## 📌 Le projet en 30 secondes

Un produit arrive avec un titre et une description en français. Le modèle prédit **simultanément** sa catégorie (L1) et sa sous-catégorie (L2) :

```
"Coque iPhone 12 silicone transparente"
          │
          ▼
    ┌─ HYDRA ─┐
    │CamemBERT│
    │  12 × 🔄 │
    └────┬────┘
    ┌────┴────┐
    ▼         ▼
 L1: 157   L2: 6 126
Électronique  Coque téléphone
```

Deux approches comparées :
- **HYDRA** (EMNLP 2025) — Multi-tête encoder-only ✅
- **HLPD** (ECIR 2025) — Décodeur non-autorégressif avec attention hiérarchique 📋 *(à venir)*

---

## 🏗️ Architecture HYDRA

```mermaid
graph TD
    A["📝 Texte produit<br/><i>designation + description</i>"] --> B
    B["🔤 CamemBERT Tokenizer<br/><i>max_length=256</i>"] --> C
    C["🧠 CamemBERT<br/><i>12 couches Transformer<br/>110M params, fine-tuné</i>"] --> D
    D["📐 Projection 768 → 1536"] --> E1
    D --> E2

    E1["🏷️ Head L1<br/>Linear→LN→ReLU→Drop→Linear<br/><b>→ 157 classes</b>"]
    E2["🏷️ Head L2<br/>Linear→LN→ReLU→Drop→Linear<br/><b>→ 6 126 classes</b>"]

    E1 --> F["🎯 Prédiction"]
    E2 --> F

    style C fill:#4A90D9,color:white
    style E1 fill:#27AE60,color:white
    style E2 fill:#E67E22,color:white
    style F fill:#8E44AD,color:white
```

**3 variantes testées :**

| | Local | Global | Nested |
|---|---|---|---|
| **Idée** | Têtes indépendantes | + tête unifiée L1+L2 | + MLP de raffinement |
| **Loss** | `0.5·CE(L1) + CE(L2)` | + α · unified | + α · unified |
| **Cohérence L1↔L2** | Post-hoc (masque) | Implicite | Par construction |

---

## 📏 Baselines

| Modèle | F1 L1 | F1 L2 | HCR |
|--------|-------|-------|-----|
| TF-IDF + LinearSVC | 82.50% | 42.23% | 57.29% |
| FastText Supervised | 83.72% | 9.48% | 28.60% |
| Word2Vec + LinearSVC | 60.24% | 22.27% | 25.06% |

---

## ⚡ Fonctionnalités clés

<table>
<tr>
<td width="50%">

**🔬 Training**
- Fine-tuning complet (12 couches)
- DataParallel multi-GPU automatique
- DDP multi-nœud sur cluster SLURM
- Checkpoint tous les 10K steps
- Auto-resume après interruption SLURM
- Early stopping sur `val_f1_l2_macro`

</td>
<td width="50%">

**📊 Audit & MLflow**
- 30+ hyperparamètres loggés
- Confusion matrices L1/L2 à chaque epoch
- Per-class F1 (barplots triés)
- Training curves (loss + F1 + HCR)
- Explicabilité Gradient × Input (HTML)
- 77 tests pytest

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# Installation
git clone <repo-url> && cd rakuten-classification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Dataset (splits pré-générés sur HuggingFace Hub)
export HF_TOKEN="hf_..."
python scripts/download_dataset.py

# Training
python scripts/run_train.py --split-dir /path/to/splits --variant local

# SLURM — job array par variante (10 tentatives, auto-resume)
sbatch --job-name=hydra-local --array=0-9%1 scripts/run_train_variant.slurm local

# SLURM — 3 variantes en parallèle
sbatch scripts/run_train_array.slurm

# SLURM — chaîner N arrays
./scripts/chain_submit_array.sh 3

# Tests
python -m pytest tests/ -v
```

---

## 📊 Dataset

> Propriétaire — non inclus dans le repo. Hébergé sur HuggingFace Hub (privé).

| Split | Échantillons | % | Champs |
|-------|-------------|---|--------|
| Train | 5 573 659 | 70% | `designation`, `description`, `typelabel` (L1), `producttype` (L2) |
| Val | 1 194 355 | 15% | Split stratifié par L2, seed=42 |
| Test | 1 194 356 | 15% | |

---

## 🎯 Métriques & Audit

```mermaid
graph LR
    A["Fin d'epoch"] --> B["📈 Métriques MLflow"]
    A --> C["🖼️ Artefacts visuels"]
    A --> D["🔍 Explicabilité"]

    B --> B1["f1_l1_macro<br/>f1_l2_macro<br/>HCR<br/>constrained_f1"]
    C --> C1["confusion_matrix_l1.png<br/>confusion_matrix_l2_top50.png<br/>f1_per_class_l1.png<br/>training_curves.png"]
    D --> D1["explainability_report.html<br/><i>Gradient × Input<br/>10 erreurs + 10 correct<br/>tokens colorés L1 & L2</i>"]

    style A fill:#E74C3C,color:white
    style B fill:#3498DB,color:white
    style C fill:#27AE60,color:white
    style D fill:#9B59B6,color:white
```

**Objectif** : maximiser `f1_l2_macro` avec HCR > 95%

---

## 🔍 Explicabilité (Gradient × Input)

Le rapport HTML montre quels mots ont contribué à chaque prédiction :

```
L1 — Vrai: Électronique | Prédit: Électronique (98%) ✔
  Coque  iPhone  12  silicone  transparente
   ░░    ██████  ░    ██        █
         ↑ drive la prédiction L1

L2 — Vrai: Coque téléphone | Prédit: Coque téléphone (91%) ✔
  Coque  iPhone  12  silicone  transparente
  █████  ████    ░    █████     █████████
  ↑ drive la prédiction L2         ↑ aussi
```

---

## ⚙️ Configuration

Source de vérité : [`02_hydra/configs/default.yaml`](02_hydra/configs/default.yaml)

| Paramètre | Valeur | | Paramètre | Valeur |
|-----------|--------|-|-----------|--------|
| Encodeur | `camembert-base` | | `batch_size` | 64 |
| `max_length` | 256 | | `lr` | 3.5e-5 |
| Warmup | 10% (linéaire) | | Weight decay | 0.01 |
| Gradient clip | 1.0 | | Early stop | patience 2 |
| `dropout` | 0.3 | | `hidden_dim` | 256 |

Variante hiérarchique : [`02_hydra/configs/hierarchy.yaml`](02_hydra/configs/hierarchy.yaml) — active `condition_l2_on_l1` (teacher forcing) et `check_hierarchy` (masque à l'inférence)

---

## 📁 Structure

```
rakuten-classification/
├── 00_eda/                    Analyse exploratoire (EDA)
├── 01_baselines/              FastText, TF-IDF+LinearSVC, Word2Vec+LinearSVC
├── 02_hydra/
│   ├── configs/               default.yaml, hierarchy.yaml
│   └── src/                   models, losses, trainer, predict
├── 02b_hlpd/                  HLPD (à venir — Mehdi Aithamma)
├── 03_production/             DDP multi-nœud, SLURM
├── shared/                    data_loader, metrics, utils
├── scripts/                   train, SLURM arrays, évaluation, comparaison
├── results/                   Rapports HTML, métriques par modèle
├── tests/                     77 tests pytest
└── docker-compose.yml         MLflow, tests, training
```

---

## 📚 Références

- **HYDRA** — Karl et al., *Hierarchical Multi-label Classification*, EMNLP 2025 [[GitHub]](https://github.com/FKarl/HYDRA)
- **HLPD** — Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder*, ECIR 2025

---

<div align="center">

**[Tahiana H. Andriambahoaka](https://github.com/tahianahajanirina) · [Mehdi Aithamma](https://github.com/MehdiiAH) · [Younes Boutrouft](https://github.com/inptensib) · Amir H. Kashani Rahimi**

</div>
