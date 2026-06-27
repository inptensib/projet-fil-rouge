<div align="center">

<img src="docs/screenshots/hero.jpg" alt="HieraCat" width="80%" style="border-radius:12px; margin-bottom:16px"/>

# Rakuten Hierarchical Product Classification

**8M+ produits · 157 catégories · 5 777 sous-catégories · CamemBERT**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-312%20passed-brightgreen?logo=pytest&logoColor=white)
![SLURM](https://img.shields.io/badge/SLURM-Multi--GPU-orange)

> **Télécom Paris** (IP Paris) × **Rakuten France**

</div>

---

## Le projet en 30 secondes

Un produit arrive avec un titre et une description en français. Le modèle prédit
**simultanément** sa catégorie L1 parmi 157 et sa sous-catégorie L2 parmi 5 777 :

```
"Coque iPhone 12 silicone transparente"
          │
          ▼
    ┌─ HYDRA ─┐
    │CamemBERT│
    │  12 couches │
    └────┬────┘
    ┌────┴────┐
    ▼         ▼
 L1: 157   L2: 5 777
Électronique  Coque téléphone
```

Trois familles de modèles comparées :

| Famille | Approche | Meilleur F1 L2 (v1, test) |
|:--------|:---------|:-------------------:|
| Baselines | TF-IDF, Word2Vec, fastText + classifieurs | 42.9 % |
| **FlatHydra** | CamemBERT + tête unique L1+L2 | **71.7 %** |
| HYDRA | CamemBERT multi-têtes (EMNLP 2025) | 66.8 % |
| HLPD | Décodeur hiérarchique (ECIR 2025) | 61.5 % |

> **Note v1 → v2 (mesuré)** : ces chiffres sont sur le dataset **v1** (fuite
> train/test non purgée). Sur le dataset **v2** dé-fuité, FlatHydra CamemBERT
> ré-évalué donne **78,9 %** de F1 L2 — non directement comparable à v1 (les
> classes L2 passent de 6 126 à 5 777). L'effet propre de la fuite, mesuré au
> niveau L1 (mêmes 157 classes), est de **~1 pt** (v1 légèrement gonflé).
> Détail : [docs/METRICS.md](docs/METRICS.md).

---

## Architecture HYDRA

```mermaid
graph TD
    A["Texte produit (designation + description)"] --> B
    B["CamemBERT Tokenizer (max_length=256)"] --> C
    C["CamemBERT (12 couches, 110M params, fine-tuné)"] --> D
    D["Projection 768 → 1536"] --> E1
    D --> E2

    E1["Head L1 : Linear → LN → ReLU → Drop → Linear → 157 classes"]
    E2["Head L2 : Linear → LN → ReLU → Drop → Linear → 5 777 classes"]

    E1 --> F["Prédiction"]
    E2 --> F

    style C fill:#4A90D9,color:white
    style E1 fill:#27AE60,color:white
    style E2 fill:#E67E22,color:white
    style F fill:#8E44AD,color:white
```

3 variantes testées :

| Variante | Idée | Cohérence L1↔L2 |
|:---------|:-----|:----------------|
| `local` | Têtes L1 et L2 indépendantes | Post-hoc (masque hiérarchique) |
| `global` | + tête unifiée L1+L2 | Implicite |
| `nested` | + MLP de raffinement L2 conditionné sur L1 | Par construction |

> **Constat clé** : la tête plate *FlatHydra* (CamemBERT + une seule tête L1+L2) obtient
> le meilleur F1 L2 du projet, au-dessus des têtes multi-niveaux HYDRA. Malgré son nom,
> FlatHydra n'est **pas** une variante de HYDRA : c'est une baseline forte (rangée dans
> `rakuten/models/baselines/`).

---

## Résultats clés (v1, test)

> F1 L1 non rapporté pour les modèles profonds — métrique cible = F1 L2.

| Modèle | F1 L1 | F1 L2 | HCR |
|:-------|:-----:|:-----:|:---:|
| TF-IDF + LinearSVC | 83.26 % | 42.75 % | 63.41 % |
| FastText Supervised | 82.60 % | 10.01 % | 29.29 % |
| Word2Vec + LinearSVC | 60.24 % | 22.27 % | 25.06 % |
| **FlatHydra CamemBERT** | **—** | **71.7 %** | — |
| HYDRA local CamemBERT | — | 66.8 % | > 95 % |
| HLPD CamemBERT | — | 61.5 % | — |

---

## Quick Start

> **Nouveau sur le projet ?** Suivre [ONBOARDING.md](ONBOARDING.md) — installation,
> accès aux données et modèles, premier entraînement, SLURM.
> Référence détaillée : [docs/PIPELINE.md](docs/PIPELINE.md) (quel script pour quel modèle)
> et [docs/METRICS.md](docs/METRICS.md) (métriques et provenance des résultats).

```bash
# 1. Cloner et installer
git clone <repo-url> && cd rakuten-classification
python -m venv venv && source venv/bin/activate
# CPU ou autre version CUDA ? Installer torch d'abord via l'index PyTorch adapté
# (cf. commentaire en tête de requirements.txt), puis :
pip install -e .

# 2. Vérifier l'installation — aucune donnée ni GPU requis
python -m pytest tests/ -q          # 312 tests doivent passer

# 3. Télécharger les données (token HF requis — voir section suivante)
export HF_TOKEN="hf_..."
python scripts/download_dataset.py

# 4. Smoke test local — nécessite le dataset (étape 3 ci-dessus)
# Durée : quelques minutes à dizaines de minutes selon la machine
# Sans token HF : s'arrêter après les 312 tests, c'est la voie de repli normale.
python scripts/run_train.py --split-dir dataset/ --n-samples 5000 \
    --config configs/hydra/default.yaml

# 5. Entraînement complet sur cluster SLURM
sbatch --job-name=hydra-local --array=0-9%1 scripts/run_train_variant.slurm local
```

---

## Accès aux données et modèles

> **Repos HuggingFace privés** — un token HF en lecture est requis.
> Contacter : `TahianaAndriambahoaka` (propriétaire des repos).

### Sans accès (voie de repli)

Il est possible d'installer le package et de lancer les 312 tests **sans données ni GPU** :

```bash
pip install -e .
python -m pytest tests/ -q   # 312 passed
```

Ce que cela couvre : imports, architecture des modèles, métriques, configs, utilitaires.
Ce que cela ne couvre pas : entraînements et évaluations réelles.

### Données (dataset v2)

Repo HF privé : [`TahianaAndriambahoaka/rakuten-dataset`][hf-dataset]
Sous-dossier : `datasets_finaux/` — splits pré-générés (70/15/15, stratifiés par L2, seed=42).

| Fichier | Lignes |
|:--------|-------:|
| `train_split.jsonl` | 5 259 035 |
| `val_split.jsonl` | 1 194 355 |
| `test_split.jsonl` | 1 194 356 |

Chaque ligne : `{"designation": "...", "description": "...", "typelabel": "L1", "producttype": "L2"}`

```bash
export HF_TOKEN="hf_..."           # token read fourni par le propriétaire
python scripts/download_dataset.py  # télécharge dans <repo>/dataset/
```

> **Dataset v2** : fuite de doublons purgée (−314 624 lignes de train nettes).
> Effet mesuré de la fuite (FlatHydra) : ~1 pt au L1 ; au L2 les chiffres v1↔v2
> ne se comparent pas terme à terme (6 126 → 5 777 classes). Voir
> [docs/METRICS.md](docs/METRICS.md).

### Modèles entraînés (~63 Go, 22 modèles)

Repo HF privé : [`TahianaAndriambahoaka/rakuten-models`][hf-models]

Le plus simple — via le script dédié :

```bash
export HF_TOKEN="hf_..."
python scripts/download_models.py                      # tous les modèles (~63 Go)
python scripts/download_models.py flat-hydra-camembert-v2   # un seul modèle
```

Ou directement en Python pour un sous-dossier précis :

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-models",
    allow_patterns="flat-hydra/*",   # remplacer selon le modèle voulu
    token="hf_...",
)
```

Correspondance sous-dossiers HF → chemins locaux attendus par les scripts d'évaluation :
voir [ONBOARDING.md](ONBOARDING.md) section « Récupérer les modèles entraînés ».

### Cluster SLURM

Les scripts `scripts/*.slurm` supposent les partitions `P100` (2×16 Go)
et `3090` (2-4×24 Go) du cluster Télécom Paris.
Sur une autre infrastructure, adapter les directives `#SBATCH` et les variables :

```bash
export PROJECT_ROOT="/chemin/vers/rakuten-classification"
export VENV_ACTIVATE="/chemin/vers/venv/bin/activate"
export DATA_DIR="/chemin/vers/dataset"
```

[hf-dataset]: https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset
[hf-models]: https://huggingface.co/TahianaAndriambahoaka/rakuten-models

---

## Métriques

Toutes définies dans `rakuten/common/metrics.py`, documentées dans [docs/METRICS.md](docs/METRICS.md).

| Métrique | Description |
|:---------|:------------|
| `f1_l1_macro` | F1 macro sur les 157 catégories L1 |
| `f1_l2_macro` | F1 macro sur les 5 777 sous-catégories L2 — **métrique principale** |
| `HCR` | Taux de cohérence hiérarchique : % de prédictions où L2 est enfant valide de L1 |
| `constrained_f1` | F1 L2 compté seulement si L1 parent correct |

Objectif : maximiser `f1_l2_macro` avec HCR > 95 %.

Chaque epoch logue également : matrices de confusion L1/L2, F1 par classe (barplots triés),
et un rapport d'explicabilité HTML (Gradient × Input) disponible dans MLflow
(`docker compose up mlflow -d`).

---

## Configuration HYDRA

Source de vérité : [`configs/hydra/default.yaml`](configs/hydra/default.yaml)

| Paramètre | Valeur | | Paramètre | Valeur |
|-----------|--------|-|-----------|--------|
| Encodeur | `camembert-base` | | `batch_size` | 64 |
| `max_length` | 256 | | `lr` | 3.5e-5 |
| Warmup | 10% (linéaire) | | Weight decay | 0.01 |
| Gradient clip | 1.0 | | Early stop | patience 2 |
| `dropout` | 0.3 | | `hidden_dim` | 256 |

Variante hiérarchique : [`configs/hydra/hierarchy.yaml`](configs/hydra/hierarchy.yaml) — active `condition_l2_on_l1` (teacher forcing) et `check_hierarchy` (masque à l'inférence)

---

## Structure

```
rakuten-classification/
├── rakuten/                   Package Python installable (pip install -e .)
│   ├── common/                data_loader, metrics, training_utils, utils
│   └── models/
│       ├── baselines/         TF-IDF, Word2Vec, fastText, FlatHydra
│       ├── hydra/             HYDRA (EMNLP 2025) : models, losses, trainer, config
│       └── hlpd/              HLPD (ECIR 2025) — Mehdi Aithamma
├── configs/                   YAML hors package (source de vérité)
│   ├── hydra/                 default.yaml, hierarchy.yaml, title_only*.yaml
│   ├── baselines/             flat_hydra*.yaml
│   └── hlpd/                  hlpd_3090.yaml
├── scripts/                   CLI Python, scripts SLURM, évaluation, comparaison
├── notebooks/
│   ├── eda/                   Analyse exploratoire
│   └── baselines/             fastText, TF-IDF, Word2Vec (notebooks)
├── docs/                      ARCHITECTURE.md, PIPELINE.md, METRICS.md
├── results/                   Rapports HTML, métriques par modèle
├── tests/                     312 tests pytest (sans données ni GPU)
├── hybrid_system/             Contribution séparée — hors package (voir README dédié)
├── ONBOARDING.md              Guide de prise en main complet
└── docker-compose.yml         MLflow, tests, training
```

Architecture détaillée et table de correspondance ancien → nouveau :
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) — branchement, pipeline qualité (tests + lint), format de commit.

---

## Références

- **HYDRA** — Karl et al., *Hierarchical Multi-label Classification*, EMNLP 2025 [[GitHub]](https://github.com/FKarl/HYDRA)
- **HLPD** — Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder*, ECIR 2025

---

<div align="center">

**[Tahiana H. Andriambahoaka](https://github.com/tahianahajanirina) · [Mehdi Aithamma](https://github.com/MehdiiAH) · [Younes Boutrouft](https://github.com/inptensib) · Amir H. Kashani Rahimi**

</div>
