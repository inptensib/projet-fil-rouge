# Architecture du projet

Ce document décrit la narration du projet, la structure du package Python
`rakuten` et la table de correspondance entre l'ancienne organisation et la
nouvelle.

## Table des Matières

1. [Narration du projet](#narration-du-projet)
1. [Structure du package](#structure-du-package)
1. [Table de correspondance ancien → nouveau](#table-de-correspondance-ancien--nouveau)
1. [Imports cibles](#imports-cibles)
1. [Note sur le pipeline distribué](#note-sur-le-pipeline-distribué)
1. [Note sur `hybrid_system/`](#note-sur-hybrid_system)

---

## Narration du Projet

Le projet suit une progression classique en machine learning supervisé :
exploration des données, établissement de baselines, puis montée en complexité
vers des architectures plus expressives.

### Étape 1 — Analyse Exploratoire (EDA)

Avant tout modèle, les notebooks `notebooks/eda/` ont caractérisé le dataset
Rakuten France : distribution des 157 catégories L1, longue traîne des 5 777
sous-catégories L2, longueur des textes (`designation` + `description`), et
déséquilibre de classes. L'EDA a motivé plusieurs choix techniques : le
macro-averaging des métriques (pour donner autant de poids aux classes rares),
le `max_length=256` (couvre 95 % des produits), et la stratification par L2
lors des splits.

### Étape 2 — Baselines

Les notebooks `notebooks/baselines/` et le code `rakuten/models/baselines/`
implémentent trois familles de baselines classiques :

- **TF-IDF + LinearSVC** : sac de mots TF-IDF, SVM linéaire. Référence la plus
  forte (F1 L2 = 42.9 %).
- **Word2Vec + LinearSVC** : embeddings Word2Vec pré-entraînés, SVM.
- **fastText** : classification supervisée (mode supervised de fastText).
- **FlatHydra** : CamemBERT avec une unique tête plate L1+L2 (5 934 sorties en
  v2 = 157 + 5 777 ; 6 283 sur le dataset v1 d'origine). C'est la meilleure
  baseline (F1 L2 = 71.7 % en v1, 78,9 % en v2) et le point de départ naturel pour
  HYDRA. **Constat clé** : FlatHydra obtient le meilleur F1 L2 du projet, au-dessus
  des têtes multi-niveaux HYDRA. Malgré son nom, FlatHydra n'est **pas** une variante
  de HYDRA : c'est une baseline forte (rangée dans `rakuten/models/baselines/`).

### Étape 3 — HYDRA (EMNLP 2025)

`rakuten/models/hydra/` implémente **HYDRA** (Hierarchical YDRA), un encodeur
CamemBERT à deux têtes de classification indépendantes : une pour les 157
classes L1, une pour les 5 777 classes L2. Trois variantes ont été comparées :

| Variante | Idée | Cohérence L1↔L2 |
|:---------|:-----|:----------------|
| `local`  | Têtes L1 et L2 indépendantes | Post-hoc (masque hiérarchique à l'inférence) |
| `global` | + tête unifiée L1+L2 | Implicite (représentation partagée) |
| `nested` | + MLP de raffinement L2 conditionné sur L1 | Par construction |

La variante hiérarchique (`configs/hydra/hierarchy.yaml`) ajoute le
*teacher forcing* (`condition_l2_on_l1`) pendant l'entraînement et le masque
hiérarchique à l'inférence (`check_hierarchy`).

### Étape 4 — HLPD (ECIR 2025)

`rakuten/models/hlpd/` implémente **HLPD** (Hierarchical Label Pre-populated
Decoder), une architecture encode–décode non-autorégressive développée par
Mehdi Aithamma. Le décodeur reçoit en entrée les embeddings des étiquettes L2
et les raffine via une attention hiérarchique biaisée par la structure
parent-enfant. HLPD atteint F1 L2 = 61.5 % en utilisant les mêmes données.

---

## Structure du Package

```
rakuten-classification/
├── pyproject.toml              # packaging setuptools, dépendances, ruff
├── README.md  ONBOARDING.md  CLAUDE.md
├── rakuten/                    # package installable (pip install -e .)
│   ├── __init__.py
│   ├── common/                 # briques partagées par tous les modèles
│   │   ├── data_loader.py      # chargement JSONL + splits
│   │   ├── metrics.py          # f1_l1_macro, f1_l2_macro, HCR, constrained_f1
│   │   ├── training_utils.py   # EarlyStopping, compute_class_weights
│   │   ├── text_normalization.py
│   │   └── utils.py
│   └── models/
│       ├── baselines/          # TF-IDF, Word2Vec, fastText, FlatHydra
│       ├── hydra/              # HYDRA (EMNLP 2025) : models, losses, trainer, config
│       └── hlpd/               # HLPD (ECIR 2025) : models, losses, attention_bias, trainer
├── configs/                    # YAML hors package (non importés en Python)
│   ├── hydra/                  # default.yaml, hierarchy.yaml, title_only*.yaml, mbert_default.yaml
│   ├── baselines/              # flat_hydra.yaml, flat_hydra_3090.yaml, …
│   └── hlpd/                   # hlpd_3090.yaml
├── scripts/                    # CLI Python + scripts SLURM
├── notebooks/                  # exploration et baselines (hors package)
│   ├── eda/                    # analyse exploratoire
│   └── baselines/              # notebooks fastText, TF-IDF, Word2Vec
├── tests/                      # 305 tests pytest (sans données ni GPU)
├── docs/                       # documentation technique
│   ├── ARCHITECTURE.md         # ce fichier
│   ├── PIPELINE.md             # matrice script × étape
│   └── METRICS.md              # définitions métriques, provenance chiffres
├── hybrid_system/              # contribution séparée (cf. note ci-dessous)
└── results/                    # métriques JSON, rapports HTML
```

---

## Table de Correspondance Ancien → Nouveau

| Ancien chemin | Nouveau chemin | Remarque |
|:--------------|:---------------|:---------|
| `shared/` | `rakuten/common/` | Briques partagées |
| `02_hydra/src/` | `rakuten/models/hydra/` | Imports relatifs internes préservés |
| `01_baselines/src/` | `rakuten/models/baselines/` | FlatHydra inclus |
| `02b_hlpd/src/` | `rakuten/models/hlpd/` | Contribution Mehdi Aithamma |
| `03_distributed/src/` | **exclu du livrable** | Voir note ci-dessous |
| `02_hydra/configs/` | `configs/hydra/` | YAML hors package |
| `01_baselines/configs/` | `configs/baselines/` | |
| `02b_hlpd/configs/` | `configs/hlpd/` | |
| `00_eda/` | `notebooks/eda/` | Hors package |
| `01_baselines/notebooks/` | `notebooks/baselines/` | Hors package |

---

## Imports Cibles

Après `pip install -e .` (ou installation dans le venv), les imports sont
propres depuis n'importe quel répertoire de travail :

```python
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.common.metrics import compute_all_metrics

from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.models.hydra.models import HYDRALocal, create_hydra_model
from rakuten.models.hydra.losses import HYDRALoss
from rakuten.models.hydra.config import map_config

from rakuten.models.baselines.flat_trainer import FlatHydraTrainer
from rakuten.models.baselines.flat_model import FlatHydra

from rakuten.models.hlpd import HLPDConfig, HLPDTrainer
from rakuten.models.hlpd.losses import ZLPRLoss
```

---

## Ajouter une Variante HYDRA

Pour intégrer une nouvelle variante (ex. `bilinear`) :

- Créer la classe dans `rakuten/models/hydra/models.py` en héritant de `HYDRABase`,
  puis l'enregistrer dans le dictionnaire `HYDRA_VARIANTS` (ligne ~249).
- Si la variante produit une sortie unifiée (comme `global`/`nested`), gérer le cas
  dans `HYDRALoss.__init__` et `forward` (`rakuten/models/hydra/losses.py`) via le
  paramètre `architecture`.
- Ajouter un test dans `tests/test_hydra_models.py` (forward pass, dimensions de sortie).
- Déclarer une config YAML dans `configs/hydra/` si les hyperparamètres divergent de `default.yaml`.

---

## Note sur le Pipeline Distribué

L'entraînement distribué (DDP, DistributedDataParallel) a été exploré pendant
le projet (`03_distributed/`) mais **écarté du livrable** au profit du
DataParallel single-node, qui s'est avéré suffisant sur les nœuds GPU
disponibles (P100 2×16 Go, 3090 2-4×24 Go) et bien plus simple à maintenir.

Le code DDP (~2 000 lignes : `trainer_ddp.py`, `ddp_utils.py`,
`data_sharding.py`, `error_analysis.py`, `predict.py`, `main.py`) est
**disponible au tag git `archive/distributed-pipeline`** et peut être
récupéré à tout moment via :

```bash
git checkout archive/distributed-pipeline -- 03_distributed/
```

---

## Note sur `hybrid_system/`

Le dossier `hybrid_system/` est la **contribution séparée d'un coéquipier**
(CamemBERT + XGBoost), développée en parallèle du package `rakuten`. Il n'est
**pas intégré au package** (`pip install -e .` ne l'expose pas) et ne dépend
pas du code de `rakuten/`. Son évaluation et ses résultats sont documentés
dans ses propres fichiers.
