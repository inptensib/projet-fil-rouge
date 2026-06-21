# Flat Classifier HYDRA — Conception technique

Spécification de l'implémentation du modèle `Flat` tel que défini dans le
papier HYDRA (EMNLP 2025). Ce document couvre l'architecture, la loss, tous
les paramètres d'apprentissage et les décisions d'implémentation.

Référence officielle :
[`modeling/baselines.py`](https://github.com/FKarl/HYDRA/blob/main/modeling/baselines.py)

> **Note sur les dimensions.** Les shapes de tenseurs de ce document
> (`L2 = 6 126`, sortie unifiée `6 283 = 157 + 6 126`) correspondent au
> **dataset v1 d'origine**, sur lequel le modèle FlatHydra de référence a été
> entraîné (F1 L2 = 71.7 %). Sur le **dataset v2 dé-fuité** (couramment livré),
> `L2 = 5 777` et la sortie unifiée vaut `5 934 = 157 + 5 777`. L'architecture
> est identique : la taille de la tête L2 est dérivée des données à l'exécution
> (`len(l2_encoder.classes_)`), jamais codée en dur. Cf.
> [METRICS.md](METRICS.md) pour le détail de la fuite v1 → v2.

---

## Table des matières

- [Contexte et objectif](#contexte-et-objectif)
- [Architecture du modèle](#architecture-du-modèle)
- [La loss : HYDRALoss (CrossEntropy)](#la-loss--hydraloss-crossentropy)
- [Paramètres d'apprentissage](#paramètres-dapprentissage)
- [Optimiseur et scheduling](#optimiseur-et-scheduling)
- [Mixed precision (AMP)](#mixed-precision-amp)
- [Early stopping](#early-stopping)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Métriques de suivi](#métriques-de-suivi)
- [Artefacts produits](#artefacts-produits)

---

## Contexte et objectif

Le papier HYDRA définit `Flat` comme la **baseline de référence** : un seul
modèle qui prédit L1 (157 classes) et L2 simultanément via
une unique tête unifiée, sans aucune modélisation de la hiérarchie.

C'est la borne inférieure à battre avec les variantes `local`, `global`,
`nested`.

---

## Architecture du modèle

### Vue d'ensemble

```
Texte produit (designation + description)
        │
        ▼
┌─────────────────────────────────────┐
│  Tokenizer CamemBERT                │
│  max_length = 256, padding fixe     │
│  → input_ids      (B, 256)          │
│  → attention_mask (B, 256)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CamemBERT Encoder                  │
│  12 couches Transformer, 768 dims   │
│  last_hidden_state  (B, 256, 768)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Embedding Projector                │
│  Linear(768 → 1536)                 │
│  appliqué sur TOUT last_hidden_state│
│  → (B, 256, 1536)                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CLS Pooling                        │
│  token[:, 0, :]  →  (B, 1536)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Unified Classifier                 │
│  Linear(1536 → 1536)                │
│  LayerNorm(1536)                    │
│  ReLU                               │
│  Dropout(p=0.2)                     │
│  Linear(1536 → 6283)  ← 157 + 6126 │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
logits_L1 (B, 157)   logits_L2 (B, 6126)
  slice [:157]          slice [157:]
```

### Point clé : projection AVANT pooling

L'implémentation officielle projette `last_hidden_state` entier **avant**
d'extraire le token CLS, contrairement à `HYDRABase` qui projette après :

```python
# Officiel Flat (baselines.py)
outputs = self.embedding_projector(outputs.last_hidden_state)  # (B, 256, 1536)
full_embedding = outputs[:, 0, :]                              # CLS → (B, 1536)

# HYDRABase (différent)
full_embedding = outputs.last_hidden_state[:, 0, :]            # CLS → (B, 768)
embedding = self.projector(full_embedding)                     # → (B, 1536)
```

Conséquence : le token CLS agrège l'information **dans l'espace projeté**,
ce qui lui donne accès à une représentation multi-niveau dès le pooling.

### Dimensions

| Étape                  | Tenseur                     | Dimension        |
|:-----------------------|:----------------------------|:----------------:|
| Tokens                 | `input_ids`                 | `(B, 256)`       |
| Encoder output         | `last_hidden_state`         | `(B, 256, 768)`  |
| Après projection       | `projected`                 | `(B, 256, 1536)` |
| Après CLS pooling      | `embedding`                 | `(B, 1536)`      |
| Logits unifiés         | `unified_logits`            | `(B, 6283)`      |
| Logits L1              | `unified_logits[:, :157]`   | `(B, 157)`       |
| Logits L2              | `unified_logits[:, 157:]`   | `(B, 6126)`      |

### Interface de retour

```python
return [], unified_logits
#  ↑ level_outputs vides (interface commune avec HYDRA)
#                ↑ logits bruts non normalisés (B, 6283)
```

---

## La loss : HYDRALoss (CrossEntropy)

### Choix par rapport à l'implémentation officielle

Le papier HYDRA utilise `F.binary_cross_entropy_with_logits` car ses benchmarks
(EURLEX, WOS) sont **multi-label** — un document peut appartenir à plusieurs
catégories simultanément.

Rakuten est **single-label** : chaque produit appartient à exactement une L1
et exactement une L2. Ce contexte justifie un choix différent :

| Critère               | BCE (officiel HYDRA)              | CrossEntropy / HYDRALoss (notre choix) |
|:----------------------|:----------------------------------|:---------------------------------------|
| Paradigme             | Multi-label (N réponses possibles)| **Multi-classe (1 seule réponse)**     |
| Normalisation         | Sigmoid indépendant par classe    | **Softmax sur toutes les classes**     |
| Labels                | One-hot concaténé (B, 6283)       | **Entiers séparés L1 et L2**           |
| Classes compétitives  | Non — chaque classe est seule     | **Oui — les classes s'excluent**       |
| Rééquilibrage L1/L2   | Aucun (L2 domine par volume)      | **`level_weight_l1=0.5`**             |
| Adapté à Rakuten      | Non                               | **Oui**                               |

Avec BCE uniforme sur 6283 positions, L2 dominerait mécaniquement le gradient
(6126 positions contre 157). `HYDRALoss` avec `level_weight_l1=0.5` rééquilibre
explicitement les deux niveaux.

### Fonctionnement de HYDRALoss

`HYDRALoss` est instanciée avec `architecture="global"` pour exploiter le
mécanisme de slicing sur les logits unifiés :

```python
self.criterion = HYDRALoss(
    architecture="global",
    alpha=1.0,               # poids de la unified_loss (pas de level_outputs)
    num_classes_l1=157,
    num_classes_l2=6126,
    level_weight_l1=0.5,     # L1 pèse 0.5, L2 pèse 1.0
    use_focal=False,         # CrossEntropy standard
    focal_gamma=2.0,         # ignoré si use_focal=False
)
```

En interne, `HYDRALoss` avec `architecture="global"` fait :

```python
# Slicing des logits unifiés
logits_l1 = unified_output[:, :num_classes_l1]   # (B, 157)
logits_l2 = unified_output[:, num_classes_l1:]   # (B, 6126)

# CrossEntropy par niveau (labels entiers)
loss_l1 = CE(logits_l1, labels_l1)   # scalaire
loss_l2 = CE(logits_l2, labels_l2)   # scalaire

# Combinaison pondérée
total_loss = alpha * (level_weight_l1 * loss_l1 + loss_l2)
#          = 1.0  * (0.5 * loss_l1 + loss_l2)
```

### Formule CrossEntropy

Pour un batch de taille B et C classes :

```
CE(logits, y) = -(1/B) · Σ_b log( exp(logit_{b,y_b}) / Σ_c exp(logit_{b,c}) )
             = -(1/B) · Σ_b [ logit_{b,y_b} - log(Σ_c exp(logit_{b,c})) ]
```

Le softmax normalise les logits en probabilités qui **somment à 1** — ce qui
est cohérent avec le fait qu'un produit appartient à une seule classe.

### Prédiction à l'inférence

Avec CrossEntropy (softmax), la prédiction est l'`argmax` sur les probabilités
de chaque niveau séparément :

```python
# Slicing
logits_l1 = unified_logits[:, :self.num_classes_l1]   # (B, 157)
logits_l2 = unified_logits[:, self.num_classes_l1:]   # (B, 6126)

# Softmax par niveau (cohérent avec CE à l'entraînement)
probs_l1 = torch.softmax(logits_l1, dim=-1)
probs_l2 = torch.softmax(logits_l2, dim=-1)

# Prédiction : classe la plus probable
confs_l1, preds_l1 = probs_l1.max(dim=-1)
confs_l2, preds_l2 = probs_l2.max(dim=-1)
```

---

## Paramètres d'apprentissage

Tous les paramètres canoniques sont définis dans
`configs/baselines/flat_hydra.yaml`.

### Encodeur

| Paramètre         | Valeur          | Explication                                           |
|:------------------|:---------------:|:------------------------------------------------------|
| `encoder`         | `camembert-base`| RoBERTa entraîné sur du français (CCNet + Oscar)      |
| `max_length`      | `256`           | Longueur max des séquences tokenisées (truncation)    |
| `freeze_encoder`  | `false`         | Fine-tuning complet — les poids CamemBERT s'adaptent  |

**Pourquoi `max_length=256` ?** La désignation + description Rakuten dépasse
rarement 200 tokens. 256 offre une couverture complète sans coût mémoire
excessif (la complexité attention est O(n²)).

**Pourquoi ne pas geler l'encodeur ?** En mode `freeze_encoder=True`, seule
la tête de classification apprend — c'est utile pour tester rapidement mais
donne des performances bien inférieures. Le fine-tuning complet est
systématiquement meilleur sur des données domaine-spécifiques (e-commerce
français).

### Architecture

| Paramètre           | Valeur | Explication                                             |
|:--------------------|:------:|:--------------------------------------------------------|
| `project_embedding` | `true` | Linear(768→1536) sur `last_hidden_state` avant pooling  |
| `pooling`           | `cls`  | Token `[CLS]` comme représentation de la séquence       |
| `dropout`           | `0.2`  | Dropout dans la tête unifiée (valeur papier)            |

**Pourquoi `dropout=0.2` et non `0.3` ?** Les variantes HYDRA
(local/global/nested) utilisent `dropout=0.3`. Le `Flat` officiel utilise
`0.2` — probablement car la tête unifiée est plus simple et moins sujette
à l'overfitting que les têtes multiples.

**Pourquoi CLS pooling ?** Le token `[CLS]` de CamemBERT est entraîné pour
agréger l'information de toute la séquence. `mean` pooling peut être
légèrement meilleur sur certaines tâches mais CLS reste le choix par défaut
du papier.

### Batch et epochs

| Paramètre    | Valeur | Papier officiel | Explication                                    |
|:-------------|:------:|:---------------:|:-----------------------------------------------|
| `batch_size` | `64`   | `32`            | Plus grand batch → gradient plus stable        |
| `epochs`     | `10`   | `50`            | Early stopping compense la différence          |

**Pourquoi `batch_size=64` et non 32 ?** Avec les GPUs du cluster (P100 /
3090, 16-24 GB), 64 tient en mémoire et double le débit. Le gradient avec
batch=64 est plus stable qu'avec 32 (moins de bruit stochastique).

**Pourquoi `epochs=10` et non 50 ?** Le papier n'utilise pas d'early
stopping — il entraîne 50 epochs fixes. Avec early stopping (patience=2),
la convergence est détectée automatiquement. Sur 8M de produits Rakuten,
3 à 5 epochs suffisent généralement à saturer le val F1.

---

## Optimiseur et scheduling

### AdamW

```
θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε) - α · λ · θ_t
```

| Paramètre      | Valeur   | Explication                                              |
|:---------------|:--------:|:---------------------------------------------------------|
| `lr`           | `3.5e-5` | Learning rate de base pour le fine-tuning CamemBERT      |
| `weight_decay` | `0.01`   | Régularisation L2 découplée (le "W" dans AdamW)          |
| `max_grad_norm`| `1.0`    | Seuil de gradient clipping                              |

**Pourquoi `lr=3.5e-5` ?** C'est la valeur validée empiriquement pour le
fine-tuning de modèles CamemBERT/RoBERTa sur des tâches de classification.
Des valeurs plus élevées (>5e-5) causent de l'instabilité ; plus basses
(<1e-5) ralentissent inutilement la convergence.

**Pourquoi AdamW et non Adam ?** Adam classique applique le weight decay
sur le gradient (pas les poids), ce qui est incorrect théoriquement. AdamW
découple la régularisation L2 de l'adaptation du learning rate — c'est
le standard pour le fine-tuning de Transformers depuis 2019 (Loshchilov &
Hutter).

**Gradient clipping (`max_grad_norm=1.0`)** : si la norme L2 des gradients
dépasse 1.0, tous les gradients sont réduits proportionnellement. Cela
prévient les explosions de gradient lors des premières étapes, fréquentes
avec les grands LM.

### Linear Warmup + Linear Decay

Le learning rate suit un schedule en deux phases :

```
Phase 1 — Warmup linéaire (0 → lr) : steps 0 à warmup_steps
Phase 2 — Decay linéaire (lr → 0)  : steps warmup_steps à total_steps
```

```
lr(t) = lr × t / warmup_steps                  si t < warmup_steps
lr(t) = lr × (total_steps - t) / (total_steps - warmup_steps)  sinon
```

| Paramètre      | Valeur | Explication                                              |
|:---------------|:------:|:---------------------------------------------------------|
| `warmup_ratio` | `0.1`  | 10% des steps totaux sont dédiés au warmup               |

**Pourquoi un warmup ?** Au début de l'entraînement, les poids de la tête
de classification sont aléatoires — leurs gradients sont très grands et
instables. Commencer avec un petit LR puis le monter progressivement laisse
l'encodeur s'adapter doucement avant d'être soumis à des mises à jour
agressives.

**Exemple concret** pour 8M produits, batch=64, 10 epochs :
- Steps totaux : `(8M × 0.7) / 64 × 10 ≈ 875 000 steps`
- Warmup : `87 500 steps` (~1 epoch)

---

## Mixed precision (AMP)

| Paramètre | Valeur | Explication                                        |
|:----------|:------:|:---------------------------------------------------|
| `fp16`    | `true` | Calculs en float16, poids maîtres en float32       |

**Pourquoi FP16 ?** Les opérations en float16 sont 2-8× plus rapides sur
les GPUs NVIDIA (Tensor Cores). La mémoire consommée est divisée par ~2, ce
qui permet des batchs plus grands.

**GradScaler** : FP16 a une plage dynamique réduite — les petits gradients
peuvent underflow à 0. `GradScaler` multiplie la loss par un facteur
d'échelle S avant le backward, puis divise les gradients par S avant
`optimizer.step()`. Si un overflow est détecté, le step est sauté et S est
réduit.

```python
scaler.scale(loss).backward()      # loss × S, gradients × S
scaler.unscale_(optimizer)         # gradients / S (pour le clipping)
clip_grad_norm_(params, 1.0)       # clipping sur les vrais gradients
scaler.step(optimizer)             # step sauté si overflow
scaler.update()                    # ajuste S pour le prochain step
```

---

## Early stopping

| Paramètre                    | Valeur  | Explication                                   |
|:-----------------------------|:-------:|:----------------------------------------------|
| `early_stopping_patience`    | `2`     | Nombre d'epochs sans amélioration avant arrêt |
| `early_stopping_min_delta`   | `0.001` | Amélioration minimale du F1 L2 pour compter   |

**Critère** : `val_f1_l2_macro`. Si le F1 L2 n'améliore pas d'au moins
0.001 pendant 2 epochs consécutives, l'entraînement s'arrête.

**Pourquoi `patience=2` ?** Avec un LR decayant linéairement, l'amélioration
ralentit progressivement. Une patience de 2 est conservative mais évite
de continuer à entraîner pour rien après convergence. Sur des datasets
larges (8M), chaque epoch est coûteuse — on préfère arrêter tôt.

**Pourquoi `min_delta=0.001` ?** Des fluctuations de ±0.001 en F1 macro
sont du bruit numérique sur 6126 classes. Ce seuil filtre ces faux positifs.

---

## Pipeline d'entraînement

### Flux par batch

```
batch (input_ids, attention_mask, labels_l1, labels_l2)
         │
         ▼
  FlatHydra.forward()
         │
         ▼
  unified_logits (B, 6283)
         │
         ▼
  one_hot(labels_l1) + one_hot(labels_l2) → unified_labels (B, 6283)
         │
         ▼
  F.binary_cross_entropy_with_logits(unified_logits, unified_labels)
         │
         ▼
       loss
         │
  scaler.scale(loss).backward()
  clip_grad_norm_(params, 1.0)
  optimizer.step()
  scheduler.step()
```

### Flux par epoch

```
Epoch N
  ├─► Boucle train (shuffle=True, drop_last=True)
  │     └─► accumule train_loss
  │
  └─► evaluate(val_loader)
        ├─► sigmoid(unified_logits) → probs
        ├─► argmax(probs_l1) → preds_l1
        ├─► argmax(probs_l2) → preds_l2
        └─► compute_all_metrics() → f1_l1, f1_l2, hcr, constrained_f1
              │
              ├─► early stopping sur f1_l2_macro
              ├─► sauvegarde best_model.pt si amélioration
              ├─► sauvegarde checkpoint_latest.pt (auto-resume)
              ├─► metrics.json (fallback local)
              └─► mlflow.log_metrics()
```

### Auto-resume (job array SLURM)

Le script SLURM lance jusqu'à 10 slots séquentiels (`--array=0-9%1`).
Chaque slot vérifie :
1. Si `training_complete.txt` existe → skip
2. Si `checkpoint_latest.pt` existe → reprise depuis ce checkpoint
3. Sinon → démarrage from scratch

```bash
sbatch --job-name=flat-hydra --array=0-9%1 scripts/run_flat_variant.slurm
# Smoke test :
sbatch --job-name=flat-hydra-smoke --array=0-2%1 \
  scripts/run_flat_variant.slurm configs/baselines/flat_hydra.yaml 5000
```

---

## Métriques de suivi

| Métrique               | Niveau | Formule                                               | Objectif      |
|:-----------------------|:------:|:------------------------------------------------------|:-------------:|
| `f1_l1_macro`          | L1     | Moyenne des F1 par classe L1 (157 classes)            | Référence     |
| `f1_l2_macro`          | L2     | Moyenne des F1 par classe L2 (6126 classes)           | **Principal** |
| `constrained_f1_macro` | L2     | F1 L2 mis à 0 si L1 prédit est faux                   | Qualité       |
| `hcr`                  | Global | % de paires (pred_L1, pred_L2) hiérarchiquement valides | Cible > 95% |

**`f1_macro`** : chaque classe pèse également, indépendamment de sa
fréquence. C'est la métrique adaptée aux datasets déséquilibrés (certaines
catégories Rakuten ont 10× plus d'exemples que d'autres).

**`constrained_f1`** : si le modèle prédit la bonne sous-catégorie L2 mais
la mauvaise catégorie L1 parente, la prédiction L2 est comptée comme fausse.
Pénalise les incohérences inter-niveaux.

**`hcr` (Hierarchical Consistency Rate)** : mesure si `l2_to_l1_mapping[pred_L2] == pred_L1`.
C'est une cohérence interne — elle ne regarde pas la vérité terrain.
Un modèle peut avoir HCR=100% mais de mauvais F1 (s'il est cohérent mais
systématiquement dans la mauvaise direction).

---

## Artefacts produits

```
models/flat_hydra/
    tokenizer/               tokenizer CamemBERT sérialisé
    best_model.pt            poids du meilleur checkpoint (best val f1_l2)
    checkpoint_latest.pt     checkpoint de reprise (epoch, optimizer, scheduler, scaler)
    l1_label_encoder.pkl     LabelEncoder sklearn → decode int → label L1
    l2_label_encoder.pkl     LabelEncoder sklearn → decode int → label L2
    l2_to_l1_mapping.json    hiérarchie L2 → L1 (pour HCR et constrained F1)
    config.json              FlatHydraConfig complet + num_classes
    metrics.json             historique par epoch (fallback sans MLflow)
    training_complete.txt    flag de fin (évite re-run SLURM accidentel)
```

### Contenu de `checkpoint_latest.pt`

```python
{
    "model_state_dict":     ...,   # poids FlatHydra
    "optimizer_state_dict": ...,   # état AdamW (moments m, v)
    "scheduler_state_dict": ...,   # état du LR scheduler (step courant)
    "scaler_state_dict":    ...,   # état GradScaler (facteur d'échelle S)
    "epoch":                N,     # epoch suivante à reprendre
    "best_val_f1":          0.xx,  # meilleur F1 L2 atteint
    "epochs_no_improve":    K,     # compteur early stopping
    "history":              {...}, # courbes complètes
}
```

---

## Comparaison avec nos variantes HYDRA

| Critère                  | Flat (baseline)                      | HYDRA local/global/nested       |
|:-------------------------|:-------------------------------------|:--------------------------------|
| Têtes de classification  | 1 tête unifiée                       | 2 têtes séparées (L1 + L2)      |
| Loss                     | HYDRALoss — CE L1 + CE L2 (slicing) | HYDRALoss — CE L1 + CE L2       |
| Labels format            | Entiers séparés (l1, l2)             | Entiers séparés (l1, l2)        |
| Inférence                | Softmax par niveau + argmax          | Softmax par niveau + argmax     |
| Hiérarchie explicite     | Non                                  | Oui (masque / conditioning)     |
| `dropout`                | `0.2`                                | `0.3`                           |
| `level_weight_l1`        | `0.5`                                | `0.5`                           |
| Référence papier         | Baseline officielle (à reproduire)   | Variantes à battre              |
