# Guide d'Onboarding

Ce guide permet à un nouveau développeur de passer de zéro à un premier
entraînement reproductible. Suivre les sections dans l'ordre.

## Table des Matières

1. [Le projet en 5 minutes](#le-projet-en-5-minutes)
1. [Prérequis](#prérequis)
1. [Installation](#installation)
1. [Récupérer le dataset](#récupérer-le-dataset)
1. [Récupérer les modèles entraînés](#récupérer-les-modèles-entraînés)
1. [Premier entraînement](#premier-entraînement)
1. [Travailler sur le cluster SLURM](#travailler-sur-le-cluster-slurm)
1. [Carte du dépôt](#carte-du-dépôt)
1. [Pièges connus](#pièges-connus)
1. [Pour aller plus loin](#pour-aller-plus-loin)

## Le Projet en 5 Minutes

Classification hiérarchique de **8M+ produits Rakuten France** à partir de
leur texte (`designation` + `description`, en français) :

- **Niveau 1 (L1)** : 157 catégories parentes
- **Niveau 2 (L2)** : 6 126 catégories feuilles, chacune fille d'une L1

Trois familles de modèles sont comparées :

| Famille    | Approche                                  | Meilleur F1 L2 |
|:-----------|:------------------------------------------|:--------------:|
| Baselines  | TF-IDF, Word2Vec, fastText + classifieurs | 42.9 %         |
| FlatHydra  | CamemBERT + tête unique L1+L2             | **71.7 %**     |
| HYDRA      | CamemBERT multi-têtes (EMNLP 2025)        | 66.8 %         |
| HLPD       | Décodeur hiérarchique (ECIR 2025)         | 61.5 %         |

Les métriques principales sont `f1_l1_macro`, `f1_l2_macro` et le **HCR**
(taux de cohérence hiérarchique) — définies dans `shared/metrics.py` et
documentées dans [docs/METRICS.md](docs/METRICS.md). Le livrable de
comparaison est `results/comparison.html`.

## Prérequis

- Python 3.10+ (le projet tourne en 3.13)
- GPU optionnel en local (obligatoire pour le full training — cluster SLURM)
- **Deux tokens HuggingFace** (repos privés) à demander au propriétaire
  (`TahianaAndriambahoaka`) : un token *read* suffit pour dataset et modèles

## Installation

```bash
git clone <url-du-repo> rakuten-classification
cd rakuten-classification

python -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt

# Smoke test : la suite complète passe sans données ni GPU
python -m pytest tests/ -q
```

Les 192 tests doivent passer. Alternative Docker :

```bash
docker compose run --rm --profile test test
```

Pour entraîner localement sur GPU : `docker compose run --rm --profile gpu train`

## Récupérer le Dataset

Les splits pré-générés (70/15/15, stratifiés par classe L2) sont hébergés
sur le repo privé HuggingFace [`TahianaAndriambahoaka/rakuten-dataset`][hf-dataset] :

| Fichier             | Lignes    |
|:--------------------|:----------|
| `train_split.jsonl` | 5 573 659 |
| `val_split.jsonl`   | 1 194 355 |
| `test_split.jsonl`  | 1 194 356 |

```bash
export HF_TOKEN="hf_..."          # token read fourni par le propriétaire
python scripts/download_dataset.py # télécharge dans <repo>/dataset/
```

Chaque ligne est un objet JSON avec les champs `designation`,
`description`, `typelabel` (label L1) et `producttype` (label L2).

> **Note** : les scripts SLURM cherchent les données dans
> `$HOME/fil_rouge/dataset` par défaut. Si vous les téléchargez ailleurs,
> exportez `DATA_DIR=/chemin/vers/dataset` avant tout `sbatch`.

> **Attention** : vérifier `ls -lh "$DATA_DIR/train_split.jsonl"` avant
> tout `sbatch` — un chemin erroné produit "ERROR: Split file not found".

## Récupérer les Modèles Entraînés

Tous les poids entraînés (~50 Go) sont archivés sur le repo privé
[`TahianaAndriambahoaka/rakuten-models`][hf-models] — ils ne sont **pas**
dans ce dépôt git. Télécharger uniquement le modèle voulu :

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-models",
    allow_patterns="hydra-local-camembert/*",   # un seul sous-dossier
    token="hf_...",
)
```

Correspondance entre les sous-dossiers HF et les chemins locaux attendus
par les scripts d'évaluation :

| Sous-dossier HF                     | Chemin local attendu                          |
|:------------------------------------|:----------------------------------------------|
| `hydra-{variante}-camembert/`        | `models/HYDRA_{variante}_CamemBERT/`           |
| `hydra-local-mbert/`                 | `models/HYDRA_local_mBERT/`                    |
| `hydra-{v}-hierarchy-camembert/`     | `models_hierarchy/HYDRA_{v}_CamemBERT/`        |
| `hydra-{v}-title-camembert/`         | `models_title/HYDRA_{v}_CamemBERT_title/`      |
| `hydra-{v}-title-hierarchy-...`      | `models_title_hierarchy/HYDRA_{v}_..._title_hierarchy/` |
| `flat-hydra/`, `flat-hydra-title/`   | `models/flat_hydra/`, `models/flat_hydra_title/` |
| `hlpd/`                              | `models/hlpd/`                                 |
| `baselines/{tfidf,word2vec,...}/`    | `models/baselines/{...}/`                      |

Chaque dossier contient `best_model.pt` (poids d'inférence, ~440 Mo) et
`checkpoint_latest.pt` (état complet avec optimiseur, pour reprendre un
entraînement). La carte du repo HF (scores, structure) est dans la
constante `MODEL_CARD` de `scripts/upload_models_hf.py`.

## Premier Entraînement

Smoke test local sur un sous-échantillon (CPU ou GPU, ~10 min) :

```bash
python scripts/run_train.py \
    --split-dir dataset/ \
    --n-samples 5000 \
    --config 02_hydra/configs/default.yaml
```

Les hyperparamètres viennent du YAML (source de vérité :
`02_hydra/configs/default.yaml`). Le suivi des runs est dans MLflow
(`mlflow.db` à la racine, interface via `docker compose up mlflow -d`).

## Travailler sur le Cluster SLURM

Tous les scripts `scripts/*.slurm` utilisent quatre variables
d'environnement surchargeables (valeurs par défaut basées sur `$HOME`) :

```bash
export PROJECT_ROOT="$HOME/fil_rouge/rakuten-classification"
export VENV_ACTIVATE="$HOME/fil_rouge/venv/bin/activate"
export DATA_DIR="$HOME/fil_rouge/dataset"
# HF_HOME : cache HuggingFace (défaut ~/.cache/huggingface)
```

Lancement type d'un entraînement HYDRA (job array avec auto-reprise) :

```bash
sbatch --job-name=hydra-local --array=0-9%1 \
    scripts/run_train_variant.slurm local 02_hydra/configs/default.yaml
```

Le motif `--array=0-9%1` enchaîne jusqu'à 10 tâches de 36 h maximum,
une seule à la fois : chaque tâche reprend `checkpoint_latest.pt` et
s'arrête proprement quand `training_complete.txt` apparaît.

> **Warning** : règles du cluster (héritées d'un avertissement admin) —
> toujours des job arrays `%1`, jamais de soumission en boucle ni de
> script de surveillance qui resoumets automatiquement. Partitions GPU :
> `P100` (2×16 Go) et `3090` (2-4×24 Go), limite 36 h, QOS 10 CPUs/job.

La matrice complète « quel script pour quel modèle » est dans
[docs/PIPELINE.md](docs/PIPELINE.md).

## Carte du Dépôt

```text
00_eda/                 Analyse exploratoire (notebook + rapport HTML)
01_baselines/           TF-IDF, Word2Vec, fastText, FlatHydra (src + configs)
02_hydra/               HYDRA : modèles, pertes, trainer, configs YAML
02b_hlpd/               HLPD : décodeur hiérarchique (Mehdi Aithamma)
03_distributed/         Entraînement DDP multi-nœuds (SLURM)
shared/                 data_loader, métriques, training_utils, utilitaires communs
scripts/                CLI Python + scripts SLURM (voir docs/PIPELINE.md)
tests/                  192 tests pytest (sans données ni GPU)
results/                Métriques JSON, rapports HTML, comparison.html
docs/                   PIPELINE.md, METRICS.md, conception FlatHydra
```

## Pièges Connus

- **Deux modes d'évaluation non comparables** : avec ou sans masque
  hiérarchique. `scripts/eval_hierarchy.py --output-mode auto` choisit le
  bon mode selon la config — lire [docs/METRICS.md](docs/METRICS.md)
  avant de publier des chiffres.
- **`TRANSFORMERS_OFFLINE=1`** dans les scripts SLURM : les nœuds GPU
  n'ont pas d'accès internet. Pré-télécharger les poids des encodeurs en
  local (`AutoModel.from_pretrained(...)` une fois) pour peupler le cache.
- **Packages `src` homonymes** : `01_baselines/src`, `02_hydra/src`,
  `02b_hlpd/src` et `03_distributed/src` entrent en collision. Utiliser
  `tests/helpers.py::import_from_*` dans les tests et
  `03_distributed/src/hydra_imports.py` côté DDP.
- **`_ensure_l2_coverage()`** duplique ~923 échantillons (valeur
  indicative, recalculée à chaque chargement) pour garantir chaque classe
  L2 dans chaque split — comportement normal.
- **Configs title-only** : `batch_size`/`lr` plus élevés que `default.yaml`
  (scaling en racine carrée, séquences plus courtes) — justifié dans
  [docs/PIPELINE.md](docs/PIPELINE.md).

## Pour Aller Plus Loin

- [docs/PIPELINE.md](docs/PIPELINE.md) — matrice modèle × étape, configs,
  scripts SLURM, dettes techniques connues
- [docs/METRICS.md](docs/METRICS.md) — définitions, modes d'évaluation,
  provenance de chaque chiffre publié
- [docs/flat_hydra_conception.md](docs/flat_hydra_conception.md) —
  conception de la baseline FlatHydra
- READMEs de modules : `01_baselines/`, `02_hydra/`, `02b_hlpd/`,
  `03_distributed/`

[hf-dataset]: https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset
[hf-models]: https://huggingface.co/TahianaAndriambahoaka/rakuten-models
