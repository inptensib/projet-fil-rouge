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
- **Niveau 2 (L2)** : 5 777 catégories feuilles, chacune fille d'une L1

Trois familles de modèles sont comparées :

| Famille    | Approche                                  | Meilleur F1 L2 |
|:-----------|:------------------------------------------|:--------------:|
| Baselines  | TF-IDF, Word2Vec, fastText + classifieurs | 42.9 %         |
| FlatHydra  | CamemBERT + tête unique L1+L2             | **71.7 %**     |
| HYDRA      | CamemBERT multi-têtes (EMNLP 2025)        | 66.8 %         |
| HLPD       | Décodeur hiérarchique (ECIR 2025)         | 61.5 %         |

Les métriques principales sont `f1_l1_macro`, `f1_l2_macro` et le **HCR**
(taux de cohérence hiérarchique) — définies dans `rakuten/common/metrics.py`
et documentées dans [docs/METRICS.md](docs/METRICS.md). Le livrable de
comparaison est `results/comparison.html`.

## Prérequis

- Python 3.11+ (environnement validé : 3.13) — le plancher 3.11 est imposé par
  les versions épinglées de `numpy` et `scikit-learn`
- GPU optionnel en local (obligatoire pour le full training — cluster SLURM)
- **Token HuggingFace en lecture** pour accéder aux repos privés
  (`TahianaAndriambahoaka/rakuten-dataset` et `TahianaAndriambahoaka/rakuten-models`) :
  à demander au propriétaire (`TahianaAndriambahoaka`)

> **Sans ce token** : le package s'installe et les 305 tests passent, mais il est
> impossible de reproduire les entraînements ou évaluations (pas de données locales).
> C'est la voie de repli recommandée pour une prise en main initiale du code.

## Installation

```bash
git clone <url-du-repo> rakuten-classification
cd rakuten-classification

python -m venv ../venv
source ../venv/bin/activate
# CPU ou autre version CUDA ? Installer torch d'abord via l'index PyTorch adapté
# (cf. commentaire en tête de requirements.txt), puis :
pip install -e .

# Smoke test : la suite complète passe sans données ni GPU
python -m pytest tests/ -q
```

Les 305 tests doivent passer. Alternative Docker :

```bash
docker compose run --rm --profile test test
```

Pour entraîner localement sur GPU : `docker compose run --rm --profile gpu train`

## Récupérer le Dataset

Les splits pré-générés (70/15/15, stratifiés par classe L2) sont hébergés
sur le repo privé HuggingFace [`TahianaAndriambahoaka/rakuten-dataset`][hf-dataset] :

| Fichier             | Lignes    |
|:--------------------|:----------|
| `train_split.jsonl` | 5 259 035 |
| `val_split.jsonl`   | 1 194 355 |
| `test_split.jsonl`  | 1 194 356 |

> **Dataset v2** : fuite de doublons purgée (−314 624 lignes de train nettes).
> Voir [docs/METRICS.md](docs/METRICS.md) pour l'historique v1 → v2 et l'effet mesuré.

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

Smoke test local sur un sous-échantillon — **nécessite le dataset** (section précédente).
Durée : quelques minutes à dizaines de minutes selon la machine.

> **Sans token HF** : s'arrêter après les 305 tests, c'est la voie de repli normale.

```bash
python scripts/run_train.py \
    --split-dir dataset/ \
    --n-samples 5000 \
    --config configs/hydra/default.yaml
```

Les hyperparamètres viennent du YAML (source de vérité :
`configs/hydra/default.yaml`). Le suivi des runs est dans MLflow
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
    scripts/run_train_variant.slurm local configs/hydra/default.yaml
```

Le motif `--array=0-9%1` enchaîne jusqu'à 10 tâches de 36 h maximum,
une seule à la fois : chaque tâche reprend `checkpoint_latest.pt` et
s'arrête proprement quand `training_complete.txt` apparaît.

> **Warning** : règles du cluster (héritées d'un avertissement admin) —
> toujours des job arrays `%1`, jamais de soumission en boucle ni de
> script de surveillance qui resoumets automatiquement. Partitions GPU :
> `P100` (2×16 Go) et `3090` (2-4×24 Go), limite 36 h, QOS 10 CPUs/job.

> **Autre infrastructure ?** Les directives `#SBATCH` (partition, mémoire, temps)
> sont en tête de chaque fichier `.slurm` et s'adaptent facilement.
> Les variables `PROJECT_ROOT`, `VENV_ACTIVATE` et `DATA_DIR` sont paramétrables
> via `export` avant chaque `sbatch`.

La matrice complète « quel script pour quel modèle » est dans
[docs/PIPELINE.md](docs/PIPELINE.md).

## Carte du Dépôt

```text
rakuten/                Package Python installable (pip install -e .)
  common/               data_loader, métriques, training_utils, utilitaires
  models/baselines/     TF-IDF, Word2Vec, fastText, FlatHydra
  models/hydra/         HYDRA (EMNLP 2025) : models, losses, trainer, config
  models/hlpd/          HLPD (ECIR 2025) — Mehdi Aithamma
configs/hydra/          default.yaml, hierarchy.yaml, title_only*.yaml, mbert_default.yaml
configs/baselines/      flat_hydra*.yaml
configs/hlpd/           hlpd_3090.yaml
scripts/                CLI Python + scripts SLURM (voir docs/PIPELINE.md)
notebooks/eda/          Analyse exploratoire
notebooks/baselines/    fastText, TF-IDF, Word2Vec (notebooks)
tests/                  305 tests pytest (sans données ni GPU)
results/                Métriques JSON, rapports HTML, comparison.html
docs/                   ARCHITECTURE.md, PIPELINE.md, METRICS.md
hybrid_system/          Contribution séparée (CamemBERT + XGBoost)
```

Architecture détaillée et table de correspondance : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Pièges Connus

- **Deux modes d'évaluation non comparables** : avec ou sans masque
  hiérarchique. `scripts/eval_hierarchy.py --output-mode auto` choisit le
  bon mode selon la config — lire [docs/METRICS.md](docs/METRICS.md)
  avant de publier des chiffres.
- **`TRANSFORMERS_OFFLINE=1`** dans les scripts SLURM : les nœuds GPU
  n'ont pas d'accès internet. Pré-télécharger les poids des encodeurs en
  local (`AutoModel.from_pretrained(...)` une fois) pour peupler le cache.
- **`_ensure_l2_coverage()`** duplique ~923 échantillons (valeur
  indicative, recalculée à chaque chargement) pour garantir chaque classe
  L2 dans chaque split — comportement normal.
- **Configs title-only** : `batch_size`/`lr` plus élevés que `default.yaml`
  (scaling en racine carrée, séquences plus courtes) — justifié dans
  [docs/PIPELINE.md](docs/PIPELINE.md).
- **Dataset v1 vs v2** : les chiffres publiés sont sur v1 (fuité). Effet mesuré
  de la fuite : ~1 pt au L1 ; au L2 v1↔v2 ne se comparent pas (6 126 → 5 777
  classes) — voir [docs/METRICS.md](docs/METRICS.md).

## Pour Aller Plus Loin

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — narration du projet,
  table de correspondance ancien → nouveau, imports cibles
- [docs/PIPELINE.md](docs/PIPELINE.md) — matrice modèle × étape, configs,
  scripts SLURM, dettes techniques connues
- [docs/METRICS.md](docs/METRICS.md) — définitions, modes d'évaluation,
  provenance de chaque chiffre publié
- [docs/flat_hydra_conception.md](docs/flat_hydra_conception.md) —
  conception de la baseline FlatHydra
- [CONTRIBUTING.md](CONTRIBUTING.md) — branchement, pipeline qualité (tests + lint),
  format de commit Conventional Commits

[hf-dataset]: https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset
[hf-models]: https://huggingface.co/TahianaAndriambahoaka/rakuten-models
