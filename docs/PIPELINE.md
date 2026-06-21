# Pipeline d'Expérimentation

Carte de référence : quel script et quelle config pour entraîner, évaluer
et publier chaque modèle du projet.

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
1. [Matrice modèle × étape](#matrice-modèle--étape)
1. [Configs YAML](#configs-yaml)
1. [Scripts SLURM](#scripts-slurm)
1. [Génération des rapports](#génération-des-rapports)
1. [Utilitaires](#utilitaires)
1. [Dettes techniques connues](#dettes-techniques-connues)

## Vue d'Ensemble

```text
dataset HF (splits JSONL)
        |
        v
   entraînement  ----->  models*/...                (best_model.pt + checkpoint)
        |                   |
        v                   v
   MLflow (mlflow.db)   évaluation  ----->  results/*.json   (métriques test)
                            |
                            v
                  F1 par classe  ----->  results/<modele>/f1_per_class_l*.json
                            |
                            v
            generate_comparison.py  ----->  results/comparison.html
```

## Matrice Modèle × Étape

Commandes depuis la racine du repo.

> **Prérequis** : exporter `PROJECT_ROOT`, `VENV_ACTIVATE` et `DATA_DIR`
> avant tout `sbatch` (voir [ONBOARDING.md](../ONBOARDING.md)) ;
> `--array=0-9%1` = reprise automatique sur checkpoint, une tâche à la fois.

### HYDRA CamemBERT standard (local, nested, global)

```bash
# Entraîner (une variante, job array auto-reprise)
sbatch --job-name=hydra-local --array=0-9%1 \
    scripts/run_train_variant.slurm local configs/hydra/default.yaml

# Ou les 3 variantes en parallèle (1 GPU chacune)
sbatch scripts/run_train_array.slurm configs/hydra/default.yaml

# Évaluer (3 variantes, avec/sans masque)
sbatch scripts/run_eval.slurm

# F1 par classe
sbatch scripts/run_f1_perclass.slurm
```

### HYDRA hierarchy (teacher forcing + masque)

```bash
# Entraîner — chaînage de N arrays si > 36 h
./scripts/chain_submit_array.sh 3 configs/hydra/hierarchy.yaml

# Évaluer (les modèles sont dans models_hierarchy/)
sbatch scripts/run_eval.slurm models_hierarchy

# F1 par classe
sbatch scripts/run_f1_perclass_hierarchy.slurm
```

### HYDRA mBERT

```bash
sbatch --job-name=hydra-local-mbert --array=0-9%1 \
    scripts/run_train_variant.slurm local configs/hydra/mbert_default.yaml
sbatch scripts/run_eval_hydra_mbert.slurm
sbatch scripts/run_f1_perclass_mbert.slurm
```

### HYDRA title-only (designation seule)

```bash
sbatch scripts/run_hydra_title_3090.slurm             # standard
sbatch scripts/run_hydra_title_hierarchy_3090.slurm    # + hiérarchie
sbatch scripts/run_eval_hydra_title_hierarchy.slurm     # évaluation
```

### FlatHydra (tête unique L1+L2)

```bash
sbatch --job-name=flat --array=0-9%1 scripts/run_flat_3090.slurm     # 2×3090
sbatch --job-name=flat --array=0-9%1 scripts/run_flat_p100.slurm     # 2×P100
sbatch --job-name=flat-mbert --array=0-9%1 scripts/run_flat_mbert.slurm
sbatch scripts/run_flat_hydra_title_3090.slurm                       # title-only

sbatch scripts/run_eval_flat_hydra.slurm          # éval full-text
sbatch scripts/run_eval_flat_hydra_title.slurm     # éval title-only
```

### HLPD

```bash
sbatch --job-name=hlpd --array=0-9%1 scripts/run_hlpd_3090.slurm
# Config : configs/hlpd/hlpd_3090.yaml
```

### Baselines

```bash
sbatch scripts/run_baseline_tfidf.slurm                    # TF-IDF
sbatch scripts/run_baseline_tfidf_title.slurm              # TF-IDF title
sbatch scripts/run_baseline_word2vec.slurm                 # Word2Vec
sbatch scripts/run_baseline_word2vec_title.slurm           # Word2Vec title
sbatch scripts/run_baseline_fasttext_softmax_title.slurm   # fastText title

# fastText full-text : machine directe (lame10, 80 CPUs), PAS SLURM
bash scripts/run_fasttext_softmax.sh
bash scripts/run_fasttext_eval.sh
```

## Configs YAML

`configs/hydra/default.yaml` est la **source de vérité**. Divergences
des autres configs :

| Config                       | Batch | LR        | Divergences clés                          |
|:-----------------------------|:-----:|:----------|:-------------------------------------------|
| `default.yaml`               | 64    | 3.5e-5    | Référence (dropout 0.3, hidden 256)         |
| `hierarchy.yaml`             | 128   | 5.0e-5    | `use_focal`, `condition_l2_on_l1`, masque   |
| `title_only.yaml`            | 320   | 7.8e-5    | Title seul, `save_dir: models_title`        |
| `title_only_hierarchy.yaml`  | 640   | 1.12e-4   | Title + hiérarchie                          |
| `mbert_default.yaml`         | 64    | 3.5e-5    | Encodeur mBERT, reste identique             |

> **Note — règle de scaling** : quand le batch augmente, le LR suit la
> racine carrée du facteur : `lr = 3.5e-5 × √(batch/64)`. Vérification :
> batch 128 → ×1.41 (5e-5), batch 320 → ×2.24 (7.8e-5), batch 640 → ×3.2
> (1.12e-4). Les configs title-only supportent de gros batches car les
> séquences (titre seul) sont bien plus courtes.

Configs FlatHydra (`configs/baselines/`) : `dropout: 0.2` (fidélité au
papier, ≠ 0.3 HYDRA) ; le batch varie selon le GPU — `flat_hydra.yaml`
(64, CPU/debug), `flat_hydra_3090.yaml` (256), `flat_hydra_title_3090.yaml`
(512), `flat_hydra_mbert.yaml` (128, mBERT plus lourd).

> **Configs dataset v2** : `configs/hydra/v2_default.yaml` et
> `configs/baselines/flat_hydra_v2.yaml` ciblent le **dataset v2 dé-fuité**
> (5 777 L2) et activent le kit de rééquilibrage (`normalize_text`,
> `synthetic_policy: drop_oversampled`, `class_weighting: inv_sqrt`,
> `logit_adjustment_tau`). Elles s'invoquent comme les autres
> (`--config configs/hydra/v2_default.yaml`) ; ce sont les configs de
> référence pour reproduire les résultats v2. Les configs ci-dessus
> (sans suffixe `v2`) correspondent aux entraînements v1 historiques.

## Scripts SLURM

Conventions communes : job arrays `--array=0-9%1` (reprise automatique
sur `checkpoint_latest.pt`, skip si `training_complete.txt`), limite 36 h
par tâche, partitions `P100` (2×16 Go) et `3090` (2-4×24 Go).

| Script                         | Rôle                            | GPU       |
|:-------------------------------|:--------------------------------|:----------|
| `run_train_variant.slurm`      | 1 variante HYDRA, 1 config      | 1-2 GPU   |
| `run_train_array.slurm`        | 3 variantes en parallèle        | 1 GPU/var |
| `chain_submit_array.sh`        | N arrays chaînés (`afterany`)   | —         |
| `run_flat_*.slurm`             | FlatHydra par GPU/encodeur      | 2 GPU     |
| `run_hlpd_3090.slurm`          | HLPD                            | 2×3090    |
| `run_eval*.slurm`              | Évaluations sur test            | 1-2 GPU   |
| `run_f1_perclass*.slurm`       | F1 par classe                   | 1 GPU     |
| `run_baseline_*.slurm`         | Baselines CPU                   | —         |
| `cluster_status.sh`            | État GPU/CPU/RAM des nœuds      | —         |

> **Warning** : ne jamais soumettre de jobs en boucle ni recréer de
> script de surveillance auto-resoumission (incident des 110 jobs,
> avertissement admin). Les job arrays `%1` couvrent le besoin.

## Génération des Rapports

```bash
# Page de comparaison globale (lit mlflow.db + tous les results/*.json)
python scripts/generate_comparison.py        # → results/comparison.html

# F1 par classe d'un modèle déjà entraîné
python scripts/generate_f1_per_class.py \
    --split-dir "$DATA_DIR" --config configs/hydra/default.yaml \
    --variant local --results-dir hydra-local-camembert

# Rapports HTML par modèle (confusion, courbes)
python scripts/generate_report.py

# Métadonnées de classes (support test, parent L1)
python scripts/build_class_metadata.py --dataset-dir "$DATA_DIR"
```

La provenance de chaque JSON consommé est tracée dans
[METRICS.md](METRICS.md).

## Utilitaires

| Script                    | Rôle                                          |
|:--------------------------|:----------------------------------------------|
| `download_dataset.py`     | Télécharge les splits depuis HuggingFace       |
| `split_dataset.py`        | Régénère des splits 70/15/15 stratifiés L2     |
| `upload_models_hf.py`     | Archive les poids vers HF (`--dry-run` dispo)  |
| `eval_hierarchy.py`       | Éval avec/sans masque (`--output-mode auto`)   |

## Dettes Techniques Connues

- **Duplication des trainers** : `rakuten/models/hydra/trainer.py`,
  `rakuten/models/baselines/flat_trainer.py` et le trainer HLPD partagent
  beaucoup de logique train/eval sans factorisation commune. La
  `rakuten/common/training_utils.py` fournit les briques partagées
  (EarlyStopping, compute_class_weights).
- **Structures JSON hétérogènes dans `results/`** : trois formats
  coexistent selon la famille de modèle (voir [METRICS.md](METRICS.md)) ;
  harmoniser exigerait de régénérer toutes les évaluations.
- **Pipeline DDP** : exploré pendant le projet mais écarté du livrable au
  profit du DataParallel single-node. Disponible au tag git
  `archive/distributed-pipeline` — voir [docs/ARCHITECTURE.md](ARCHITECTURE.md).
