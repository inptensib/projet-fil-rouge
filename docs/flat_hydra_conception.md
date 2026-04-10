# Flat Classifier HYDRA — Conception

Document de conception pour l'implémentation du modèle `Flat` tel que
défini dans le papier HYDRA (EMNLP 2025), avec le plan de lancement
via les scripts existants.

---

## Table des matières

- [Contexte et objectif](#contexte-et-objectif)
- [Architecture du modèle Flat](#architecture-du-modèle-flat)
- [Schéma du forward pass](#schéma-du-forward-pass)
- [Comparaison avec les baselines existantes](#comparaison-avec-les-baselines-existantes)
- [Conception de la loss](#conception-de-la-loss)
- [Conception du pipeline d'entraînement](#conception-du-pipeline-dentraînement)
- [Suivi des métriques](#suivi-des-métriques)
- [Fichiers à créer ou modifier](#fichiers-à-créer-ou-modifier)
- [Configuration YAML](#configuration-yaml)
- [Lancer l'entraînement](#lancer-lentraînement)
- [Artefacts produits](#artefacts-produits)

---

## Contexte et objectif

Le papier HYDRA définit `Flat` comme la **baseline de référence** : un seul
modèle qui prédit L1 et L2 simultanément via une unique tête unifiée, sans
aucune modélisation de la hiérarchie.

L'implémentation officielle est dans
[`modeling/baselines.py`](https://github.com/FKarl/HYDRA/blob/main/modeling/baselines.py).

**Objectif de ce document** : concevoir la réutilisation de l'architecture
HYDRA (encoder + projection + tête) pour reproduire fidèlement ce `Flat`,
et définir comment le lancer via les scripts existants dans `scripts/`.

---

## Architecture du modèle Flat

### Vue d'ensemble

```
Texte produit (designation + description)
        │
        ▼
┌─────────────────────────────────────┐
│  Tokenizer CamemBERT                │
│  max_length = 256                   │
│  → input_ids (B, 256)               │
│  → attention_mask (B, 256)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CamemBERT Encoder                  │
│  last_hidden_state (B, 256, 768)    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Embedding Projector  [optionnel]   │
│  Linear(768 → 1536)                 │
│  last_hidden_state (B, 256, 1536)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Pooling  (stratégie : CLS)         │
│  token[:, 0, :]  →  (B, 1536)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Unified Classifier                 │
│  Linear(1536 → 1536)                │
│  LayerNorm(1536)                    │
│  ReLU                               │
│  Dropout(0.2)                       │
│  Linear(1536 → 6257)  ← 157 + 6100 │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
logits_L1 (B, 157)   logits_L2 (B, 6100)
  slice [:157]          slice [157:]
```

### Détail des dimensions

| Étape                  | Tenseur                   | Dimension     |
|:-----------------------|:--------------------------|:-------------:|
| Tokens                 | `input_ids`               | `(B, 256)`    |
| Encoder output         | `last_hidden_state`       | `(B, 256, 768)` |
| Après projection       | `projected`               | `(B, 256, 1536)` |
| Après CLS pooling      | `embedding`               | `(B, 1536)`   |
| Logits unifiés         | `unified_logits`          | `(B, 6257)`   |
| Logits L1              | `unified_logits[:, :157]` | `(B, 157)`    |
| Logits L2              | `unified_logits[:, 157:]` | `(B, 6100)`   |

### Point clé : la projection est appliquée avant le pooling

```python
# Officiel HYDRA baselines.py
outputs = self.embedding_projector(outputs.last_hidden_state)  # (B, 256, 1536)
full_embedding = outputs[:, 0, :]                              # CLS → (B, 1536)
```

Contrairement à `HYDRABase` qui projette **après** pooling, le `Flat`
officiel projette le tenseur entier puis extrait le CLS. Cela signifie que
le token CLS agrège l'information **dans l'espace projeté** — une différence
subtile mais fidèle au papier.

### Interface de retour

```python
return [], unified_logits
# ↑ level_outputs vides   ↑ output unifié (B, 6257)
```

Cette interface est identique à celle de `HYDRAGlobal` / `HYDRANested`,
ce qui permet de réutiliser `HYDRALoss` sans modification.

---

## Schéma du forward pass

```
input_ids, attention_mask
         │
         ▼
   [ CamemBERT ]
  last_hidden_state
    (B, 256, 768)
         │
         │  project_embedding=True
         ▼
   Linear(768→1536)           ← projection AVANT pooling
    (B, 256, 1536)
         │
         │  pooling='cls'
         ▼
    token CLS [:, 0, :]
      (B, 1536)
         │
         ▼
  ┌──────────────────┐
  │ Linear(1536→1536)│
  │ LayerNorm        │
  │ ReLU             │
  │ Dropout(0.2)     │
  │ Linear(1536→6257)│
  └────────┬─────────┘
           │
     unified_logits
       (B, 6257)
           │
    ┌──────┴───────┐
    ▼              ▼
 [:, :157]     [:, 157:]
 logits L1     logits L2
  (B, 157)    (B, 6100)
```

---

## Comparaison avec les baselines existantes

| Critère                  | FlatCascade (actuel)      | **Flat HYDRA (à implémenter)** | HYDRALocal           |
|:-------------------------|:--------------------------|:-------------------------------|:---------------------|
| Nombre de modèles        | 2 indépendants            | **1**                          | 1                    |
| Têtes de classification  | 1 par modèle              | **1 tête unifiée**             | 2 têtes séparées     |
| Gradients L1/L2          | Séparés (0 partage)       | **Partagés (tête commune)**    | Partagés             |
| Projection               | Aucune                    | **768 → 1536 (avant pooling)** | 768 → 1536 (après)  |
| Loss                     | CE × 2 (indépendante)     | **CE L1 + CE L2 (slicing)**    | α·CE_L1 + CE_L2      |
| Hiérarchie explicite     | Non                       | **Non**                        | Non                  |
| Fidélité au papier       | Non                       | **Oui (reproduction exacte)**  | Oui                  |
| Référence dans le papier | Non                       | **Baseline officielle**        | Variante locale       |

> La `FlatCascade` reste utile comme **borne inférieure absolue**.
> Le `Flat HYDRA` est la **baseline du papier**, à battre avec les variantes
> `local`, `global`, `nested`.

---

## Conception de la loss

La loss réutilise `HYDRALoss` avec `architecture="global"` pour exploiter
le mécanisme de slicing existant.

```python
# Dans HYDRALoss.forward() avec architecture="global" :
unified_l1_logits = unified_output[:, :num_classes_l1]   # (B, 157)
unified_l2_logits = unified_output[:, num_classes_l1:]   # (B, 6100)
unified_loss_l1   = CE(unified_l1_logits, labels_l1)
unified_loss_l2   = CE(unified_l2_logits, labels_l2)
total_loss        = level_weight_l1 * unified_loss_l1 + unified_loss_l2
```

Paramètres retenus pour la baseline :

| Paramètre         | Valeur     | Justification                             |
|:------------------|:----------:|:------------------------------------------|
| `use_focal`       | `false`    | Fidélité au papier (CE standard)          |
| `level_weight_l1` | `0.5`      | L1 compte moins (157 classes vs 6100)     |
| `alpha`           | `1.0`      | Pas de tête locale → unified_loss pèse 1 |
| `architecture`    | `"global"` | Réutilise le slicing de `HYDRALoss`       |

---

## Conception du pipeline d'entraînement

### Nouveaux fichiers à créer

```
01_baselines/src/
    flat_model.py         ← classe FlatHydra (le modèle)

scripts/
    run_flat_hydra.py     ← CLI entry point (miroir de run_train.py)
    run_flat_3090.slurm   ← job SLURM (miroir de run_train_3090.slurm)
    run_flat_p100.slurm   ← job SLURM partition P100
```

### Réutilisation maximale

Le pipeline réutilise sans modification :

| Composant               | Source                           | Rôle                       |
|:------------------------|:---------------------------------|:---------------------------|
| `HierarchicalProductDataset` | `shared/data_loader.py`     | Dataset L1+L2              |
| `prepare_hierarchical_data`  | `shared/data_loader.py`     | Chargement + split         |
| `HYDRALoss`             | `02_hydra/src/losses.py`         | Loss unifiée               |
| `compute_all_metrics`   | `shared/metrics.py`              | F1 macro, HCR, constrained |
| `AdamW` + warmup        | `torch` + `transformers`         | Optimiseur                 |
| `GradScaler` AMP        | `torch.amp`                      | Mixed precision            |

### Flux de données

```
products.jsonl
      │
      ▼  prepare_hierarchical_data()
      │
      ├── X_train  (textes)          ──┐
      ├── y_l1_train (labels L1)     ──┤─► HierarchicalProductDataset ─► DataLoader
      └── y_l2_train (labels L2)     ──┘          (train / val)

DataLoader ──► FlatHydra.forward() ──► unified_logits (B, 6257)
                                               │
                                    HYDRALoss(architecture="global")
                                               │
                                          total_loss
                                               │
                                        loss.backward()
                                         optimizer.step()
```

---

## Suivi des métriques

### Flux global par epoch

```
Fin de chaque epoch
        │
        ├─► evaluate(val_loader)           ← boucle d'inférence sur le val set
        │       │
        │       └─► compute_all_metrics()  ← shared/metrics.py
        │               │
        │               ├── f1_l1_macro
        │               ├── f1_l2_macro         ← critère d'early stopping
        │               ├── constrained_f1_macro
        │               └── hcr
        │
        ├─► history{}                      ← embarqué dans checkpoint_latest.pt
        │
        └─► mlflow.log_metrics()           ← scalaires + artefacts visuels
```

### Métriques calculées — `shared/metrics.py`

`compute_all_metrics()` est appelé après chaque epoch de validation.
Il retourne 4 métriques dérivées des prédictions L1 et L2 :

| Métrique               | Ce qu'elle mesure                                         | Objectif      |
|:-----------------------|:----------------------------------------------------------|:-------------:|
| `f1_l1_macro`          | F1 macro sur les 157 classes L1                           | Référence     |
| `f1_l2_macro`          | F1 macro sur les 6100+ classes L2                         | **Principal** |
| `constrained_f1_macro` | F1 L2 compté seulement si L1 parent est correct           | Qualité       |
| `hcr`                  | % de prédictions où L2 est enfant valide de L1 prédit     | Cible > 95%  |

**`constrained_f1`** : si `pred_L1 ≠ true_L1`, la prédiction L2 est
remplacée par `-1` (forcément fausse), même si elle était correcte.

**`hcr`** : vérifie que `l2_to_l1_mapping[pred_L2] == pred_L1`. C'est
une cohérence interne entre les deux prédictions, sans regarder la vérité terrain.

### Adaptation nécessaire pour le `Flat`

Le `Flat` retourne `[], unified_logits`. La méthode `evaluate()` de
`HYDRATrainer` attend `level_outputs, _ = model(...)` puis
`logits_l1, logits_l2 = level_outputs` — ce qui plante avec une liste vide.

Le trainer `FlatHydra` devra extraire les logits par slicing avant
d'appeler `compute_all_metrics()` :

```python
# Dans FlatHydraTrainer.evaluate()
_, unified_logits = self.model(input_ids, attention_mask)
logits_l1 = unified_logits[:, :self.num_classes_l1]   # (B, 157)
logits_l2 = unified_logits[:, self.num_classes_l1:]   # (B, 6100)

preds_l1 = logits_l1.argmax(dim=-1)
preds_l2 = logits_l2.argmax(dim=-1)

metrics = compute_all_metrics(
    labels_l1, preds_l1, labels_l2, preds_l2, self.l2_to_l1_mapping
)
```

Tout le reste — `history`, early stopping, MLflow — est réutilisable
sans modification.

### Historique en mémoire

```python
history = {
    "epoch":           [],   # [1, 2, 3, ...]
    "train_loss":      [],   # loss moyenne par epoch
    "val_f1_l1_macro": [],
    "val_f1_l2_macro": [],   # critère d'early stopping
    "val_hcr":         [],
}
```

L'historique est sauvegardé dans `checkpoint_latest.pt` et restauré
automatiquement lors d'une reprise (`--resume`).

### Early stopping

Basé exclusivement sur `f1_l2_macro` (métrique principale) :

```
Si val_f1_l2 > best_val_f1 + min_delta (0.001) :
    → sauvegarder best_model.pt
    → réinitialiser le compteur de patience

Sinon :
    → incrémenter le compteur
    → si compteur ≥ patience (2) : arrêt anticipé
```

### Fallback local — `metrics.json`

Si MLflow n'est pas disponible, les métriques sont persistées dans un
fichier JSON dans le répertoire du modèle. Ce fichier est écrit à la fin
de **chaque epoch**, ce qui le rend récupérable même en cas d'interruption :

```python
# Dans FlatHydraTrainer.train(), après history[...].append(...)
import json
with open(model_dir / "metrics.json", "w") as f:
    json.dump(history, f, indent=2)
```

Contenu du fichier après 3 epochs :

```json
{
  "epoch":            [1, 2, 3],
  "train_loss":       [1.823, 1.241, 0.987],
  "val_f1_l1_macro":  [0.612, 0.701, 0.743],
  "val_f1_l2_macro":  [0.381, 0.452, 0.489],
  "val_hcr":          [0.921, 0.948, 0.961]
}
```

Pour plotter les courbes après l'entraînement :

```python
import json
from shared.metrics import plot_training_curves

with open("models/flat_hydra/metrics.json") as f:
    history = json.load(f)

fig = plot_training_curves(history)
fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")
```

`plot_training_curves()` est déjà dans `shared/metrics.py` et génère les
3 subplots (Loss / F1 L1+L2 / HCR) à partir du même dict `history`.

> **Note** : l'`history` est aussi embarqué dans `checkpoint_latest.pt`,
> mais charger un `.pt` de plusieurs Go juste pour plotter des courbes
> n'est pas pratique. Le `metrics.json` est la solution légère.

### Logging MLflow — par epoch

Scalaires loggés à chaque epoch si MLflow est actif :

```python
mlflow.log_metrics({
    "train_loss":       avg_train_loss,
    "val_f1_l1_macro":  val_metrics["f1_l1_macro"],
    "val_f1_l2_macro":  val_f1_l2,
    "val_hcr":          val_metrics.get("hcr", 0.0),
}, step=epoch + 1)
```

Artefacts visuels regénérés à chaque epoch (écrasés) :

| Artefact MLflow             | Source                   | Contenu                              |
|:----------------------------|:-------------------------|:-------------------------------------|
| `confusion_matrix_l1.png`   | `plot_confusion_matrix`  | Matrice normalisée, 157 classes L1   |
| `confusion_matrix_l2.png`   | `plot_confusion_matrix`  | Top 50 classes L2 les plus fréquentes|
| `per_class_f1_l1.png`       | `plot_per_class_f1`      | 30 pires + 30 meilleures classes L1  |
| `training_curves.png`       | `plot_training_curves`   | Loss + F1 L1/L2 + HCR par epoch     |

---

## Fichiers à créer ou modifier

### `01_baselines/src/flat_model.py` — nouveau

```python
class FlatHydra(nn.Module):
    """Flat classifier fidèle à HYDRA baselines.py officiel.

    Une seule tête unifiée prédit L1 + L2 simultanément.
    La projection est appliquée sur last_hidden_state AVANT le pooling.

    Retourne : ([], unified_logits) pour compatibilité HYDRALoss.
    """

    def __init__(
        self,
        encoder_name: str,          # 'camembert-base'
        num_classes_l1: int,        # 157
        num_classes_l2: int,        # 6100+
        pooling: str = "cls",
        project_embedding: bool = True,
        dropout: float = 0.2,       # 0.2 selon le papier (≠ 0.3 HYDRA)
        freeze_encoder: bool = False,
    ): ...

    def forward(self, input_ids, attention_mask):
        # 1. Encoder → last_hidden_state (B, seq, 768)
        # 2. Projection AVANT pooling → (B, seq, 1536)
        # 3. CLS pooling → (B, 1536)
        # 4. Unified classifier → (B, 6257)
        # 5. return [], unified_logits
        ...
```

### `scripts/run_flat_hydra.py` — nouveau

Miroir de `run_train.py` avec :

- `--variant flat` comme seul choix de variant
- Import de `FlatHydra` au lieu de `create_hydra_model`
- Même interface `--data`, `--split-dir`, `--config`, `--resume`
- Config lue depuis `01_baselines/configs/flat_hydra.yaml`

### `01_baselines/configs/flat_hydra.yaml` — nouveau

```yaml
# Flat HYDRA — Configuration
encoder: camembert-base
max_length: 256
project_embedding: true
pooling: cls
dropout: 0.2          # valeur papier

batch_size: 64
epochs: 10
lr: 3.5e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0

use_focal: false
level_weight_l1: 0.5

seed: 42
fp16: true
save_dir: models/flat_hydra
experiment_name: flat_hydra
```

### `scripts/run_flat_3090.slurm` — nouveau

```bash
#!/bin/bash
#SBATCH --job-name=flat-hydra
#SBATCH --partition=3090
#SBATCH --nodelist=node40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-flat-hydra-%j.out
#SBATCH --error=slurm-flat-hydra-%j.err

source /home/infres/tandriam-25/fil_rouge/venv/bin/activate
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/infres/tandriam-25/.cache/huggingface

CKPT="models/flat_hydra/checkpoint_latest.pt"
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "Checkpoint trouvé : $CKPT — reprise du training"
    RESUME_FLAG="--resume $CKPT"
fi

python scripts/run_flat_hydra.py \
    --split-dir /home/infres/tandriam-25/fil_rouge/dataset/ \
    --config 01_baselines/configs/flat_hydra.yaml \
    $RESUME_FLAG
```

---

## Configuration YAML

La config suit exactement la même structure que `02_hydra/configs/default.yaml`
pour permettre la comparaison directe.

| Paramètre           | Flat HYDRA | HYDRA local | Justification               |
|:--------------------|:----------:|:-----------:|:----------------------------|
| `dropout`           | `0.2`      | `0.3`       | Valeur officielle du papier |
| `level_weight_l1`   | `0.5`      | `0.5`       | Identique                   |
| `use_focal`         | `false`    | `false`     | CE standard = reproduction  |
| `project_embedding` | `true`     | `true`      | Identique                   |
| `pooling`           | `cls`      | `cls`       | Identique                   |
| `batch_size`        | `64`       | `64`        | Identique                   |
| `lr`                | `3.5e-5`   | `3.5e-5`    | Identique                   |

---

## Lancer l'entraînement

### Développement local (CPU / 1 GPU)

```bash
python scripts/run_flat_hydra.py \
    --data shared/dataset/products.jsonl \
    --n-samples 10000
```

### Depuis des splits pré-générés (même que HYDRA)

```bash
python scripts/run_flat_hydra.py \
    --split-dir /chemin/vers/dataset/ \
    --config 01_baselines/configs/flat_hydra.yaml
```

### Sur le cluster — partition 3090

```bash
sbatch scripts/run_flat_3090.slurm
```

### Sur le cluster — partition P100

```bash
sbatch scripts/run_flat_p100.slurm
```

### Reprendre un training interrompu

```bash
python scripts/run_flat_hydra.py \
    --split-dir /chemin/vers/dataset/ \
    --resume models/flat_hydra/checkpoint_latest.pt
```

> **Workflow recommandé** : vérifier les GPUs avec `/gpu-status` avant
> de soumettre un job SLURM.

---

## Artefacts produits

À l'issue de l'entraînement, le répertoire `models/flat_hydra/` contient :

```
models/flat_hydra/
    tokenizer/              tokenizer CamemBERT sauvegardé
    best_model.pt           poids du meilleur checkpoint (best val f1_l2)
    checkpoint_latest.pt    dernier checkpoint (auto-resume)
    l1_label_encoder.pkl    LabelEncoder sklearn pour L1
    l2_label_encoder.pkl    LabelEncoder sklearn pour L2
    l2_to_l1_mapping.json   hiérarchie L2 → L1 (pour calcul HCR)
    config.json             config complète + num_classes
    training_complete.txt   flag de fin (évite re-run accidentel)
```

### Métriques suivies

| Métrique              | Niveau | Objectif               |
|:----------------------|:------:|:-----------------------|
| `f1_l1_macro`         | L1     | Référence              |
| `f1_l2_macro`         | L2     | **Métrique principale** |
| `constrained_f1`      | L2     | F1 L2 avec L1 correct  |
| `hcr`                 | Global | Cible > 95%            |

---

## Prochaines étapes

1. Implémenter `01_baselines/src/flat_model.py` (`FlatHydra`)
1. Créer `scripts/run_flat_hydra.py` (CLI entry point)
1. Créer `01_baselines/configs/flat_hydra.yaml`
1. Créer `scripts/run_flat_3090.slurm` et `run_flat_p100.slurm`
1. Ajouter les tests dans `tests/test_flat_hydra.py`
1. Lancer `/gpu-status` puis `sbatch scripts/run_flat_3090.slurm`
1. Comparer avec HYDRA local via MLflow (`compare_runs`)
