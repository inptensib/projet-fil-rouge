# Spécification technique — Amélioration de l’architecture Flat HYDRA par un paradigme Dual-Expert LLM

**Version :** v1.0  
**Date :** 2026-06-19  
**Objet :** améliorer la classification hiérarchique e-commerce `L1 > L2` à partir de `designation/title + description`, en particulier la performance `L2` sur classes rares.  
**Contexte actuel :** modèle `Flat HYDRA` basé sur CamemBERT, avec prédiction simultanée de `L1` et `L2` via une tête unifiée.  
**Inspiration principale :** architecture Dual-Expert pour la catégorisation e-commerce : un expert domaine génère les `top-K` catégories candidates, puis un expert général LLM choisit la meilleure catégorie parmi ces candidates avec des définitions de catégories générées offline.

---

## 1. Résumé exécutif

L’architecture actuelle `Flat HYDRA` doit être conservée comme **Domain Expert** principal. L’amélioration proposée ne consiste pas à remplacer CamemBERT, mais à transformer son rôle :

1. **CamemBERT / Flat HYDRA** devient un générateur de candidats `L2` à haute couverture (`top-K recall`).
2. Un scoring hiérarchique utilise simultanément `P(L1)` et `P(L2)` pour classer les candidats.
3. Les classes rares `L2` sont mieux apprises via **focal loss / class-balanced loss** et **label-path augmentation**.
4. Des **définitions françaises de catégories L2** sont générées offline par LLM à partir d’exemples produits.
5. Un **LLM General Expert** n’est appelé que sur les cas difficiles pour reranker les `top-K` catégories candidates.
6. Les cas faciles restent servis uniquement par CamemBERT afin de contrôler le coût et la latence.

Architecture cible :

```text
Produit: designation + description
        │
        ▼
Flat HYDRA CamemBERT = Domain Expert
        │
        ├── logits L1
        ├── logits L2
        └── top-K L2 candidates
                │
                ▼
Scoring hiérarchique L1 > L2
                │
                ▼
Router de difficulté
        ├── Fast path: accepter prédiction Domain Expert
        └── Hard path: LLM General Expert
                        │
                        ├── produit
                        ├── top-K chemins L1 > L2
                        ├── définitions L2 générées offline
                        └── signaux positifs/négatifs
                                │
                                ▼
                        prédiction finale L1 > L2 + raison courte
```

---

## 2. Objectifs produit et ML

### 2.1 Objectif principal

Améliorer la performance `L2`, surtout sur les classes rares, sans dégrader significativement la performance `L1` ni exploser les coûts d’inférence.

### 2.2 Objectifs mesurables

| Objectif | Métrique | Cible initiale |
|---|---:|---:|
| Améliorer la classification fine | `L2 macro F1` | +2 à +5 points relatif selon baseline |
| Améliorer classes rares | `rare L2 macro F1` | +5 à +15 points relatif |
| Garder une forte cohérence hiérarchique | `HCR` | > 98% si scoring contraint activé |
| Assurer la faisabilité du reranking | `L2 top-10 recall` | > 90% global, > 75–85% rare classes |
| Contrôler les appels LLM | `LLM pass-through rate` | 10–25% du trafic |
| Limiter la latence | `p95 latency fast path` | proche de l’existant |
| Limiter la latence LLM | `p95 latency hard path` | dépend infra, à mesurer |

### 2.3 Hypothèses

- Chaque produit appartient à exactement **une L1** et exactement **une L2**.
- `L2` a un parent `L1` unique.
- Les textes sont majoritairement en français.
- Le modèle actuel utilise CamemBERT avec une tête unifiée `157 + 6126` logits.
- Les classes rares `L2` sont une cause majeure de faible `L2 accuracy` / `L2 macro F1`.
- Les descriptions peuvent être courtes, bruitées, ou incomplètes.

---

## 3. Architecture actuelle à conserver

### 3.1 Rappel du Flat HYDRA existant

```text
Texte produit (designation + description)
        │
        ▼
Tokenizer CamemBERT
max_length = 256
        │
        ▼
CamemBERT Encoder
last_hidden_state: (B, 256, 768)
        │
        ▼
Embedding Projector
Linear(768 → 1536) appliqué sur tout last_hidden_state
projected: (B, 256, 1536)
        │
        ▼
CLS pooling
embedding: (B, 1536)
        │
        ▼
Unified Classifier
Linear(1536 → 1536)
LayerNorm
ReLU
Dropout(0.2)
Linear(1536 → 6283)
        │
        ├── logits_L1: (B, 157)
        └── logits_L2: (B, 6126)
```

### 3.2 Décision de design

Le modèle existant reste la base. Il sera renommé fonctionnellement :

```text
FlatHydraClassifier → DomainExpertClassifier
```

Le code peut conserver les classes existantes, mais les modules d’inférence et de training doivent ajouter :

- export `top-K` L2 ;
- scoring hiérarchique ;
- métriques top-K ;
- focal/class-balanced loss ;
- label-path augmentation ;
- router hard/easy ;
- LLM reranker optionnel.

---

## 4. Architecture cible détaillée

## 4.1 Composants

| Composant | Responsabilité | Online/offline |
|---|---|---|
| `DomainExpertClassifier` | Encoder CamemBERT + logits L1/L2 | Online |
| `HierarchyAwareScorer` | Combine `P(L2)` et `P(parent L1)` | Online |
| `TopKCandidateGenerator` | Retourne les `K` meilleurs L2 | Online |
| `DifficultyRouter` | Décide fast path vs hard path | Online |
| `CategoryDefinitionGenerator` | Génère définitions L2 par LLM | Offline |
| `CategoryDefinitionStore` | Stocke définitions, signaux, version | Online lookup |
| `LLMGeneralExpertReranker` | Choisit la meilleure L2 parmi top-K | Online hard path |
| `AuditLogger` | Journalise décisions, scores, prompts, réponses | Online |
| `EvaluationSuite` | Mesure accuracy/F1/top-K/rare buckets/cost | Offline |

## 4.2 Flux complet d’inférence

```text
Input product
  ├── product_id
  ├── designation
  └── description
        │
        ▼
preprocess_text()
        │
        ▼
DomainExpertClassifier.forward()
        ├── logits_l1
        └── logits_l2
        │
        ▼
softmax(logits_l1), softmax(logits_l2)
        │
        ▼
HierarchyAwareScorer
        │
        ▼
TopKCandidateGenerator
        ├── top_k_l2_ids
        ├── top_k_scores
        ├── top_k_probs_l2
        └── parent_l1 for each l2
        │
        ▼
DifficultyRouter
        ├── easy → final = top1 hierarchy-aware L2
        └── hard → fetch definitions → LLMGeneralExpertReranker → final L2
```

## 4.3 Fast path

Utilisé pour produits faciles et fréquents.

Conditions typiques :

```text
- confiance L2 élevée
- marge top1-top2 élevée
- L1 prédite cohérente avec parent L2
- classe L2 non rare
- texte suffisamment informatif
```

Sortie :

```json
{
  "product_id": "...",
  "pred_l1_id": 12,
  "pred_l2_id": 583,
  "decision_source": "domain_expert_fast_path",
  "confidence": 0.93,
  "llm_used": false
}
```

## 4.4 Hard path

Utilisé pour produits ambigus, classes rares, faible confiance ou incohérence hiérarchique.

Conditions typiques :

```text
- top1 L2 confidence < threshold
- marge top1-top2 < threshold
- predicted L2 rare
- parent(pred_L2) != pred_L1
- description courte ou absente
- plusieurs candidats top-K sont des siblings proches
```

Sortie :

```json
{
  "product_id": "...",
  "pred_l1_id": 12,
  "pred_l2_id": 583,
  "decision_source": "llm_general_expert",
  "domain_top1_l2_id": 761,
  "llm_selected_l2_id": 583,
  "llm_confidence": "medium",
  "llm_reason": "Le produit est une housse de couette et non un plaid décoratif...",
  "llm_used": true
}
```

---

# 5. Domain Expert — modifications d’entraînement

## 5.1 Correction de la loss single-label

Pour le contexte Rakuten/e-commerce single-label, conserver `CrossEntropy` par niveau.

```python
logits_l1 = unified_logits[:, :num_classes_l1]
logits_l2 = unified_logits[:, num_classes_l1:]

loss_l1 = F.cross_entropy(logits_l1, labels_l1)
loss_l2 = F.cross_entropy(logits_l2, labels_l2)

loss = level_weight_l1 * loss_l1 + loss_l2
```

Configuration actuelle recommandée :

```yaml
loss:
  type: hierarchical_ce
  level_weight_l1: 0.5
  level_weight_l2: 1.0
```

**Important :** supprimer toute utilisation de `one_hot(labels)` + `binary_cross_entropy_with_logits` dans le pipeline single-label. Cette logique est adaptée au multi-label, pas au cas `1 produit = 1 L1 + 1 L2`.

---

## 5.2 Ajout de focal loss pour L2

### 5.2.1 Motivation

Les classes `L2` rares reçoivent peu de signal. La focal loss réduit l’impact des exemples faciles et augmente l’attention sur les exemples difficiles.

### 5.2.2 Formule

Pour un exemple de label vrai `y` :

```text
CE = -log(p_y)
FL = alpha_y * (1 - p_y)^gamma * CE
```

Où :

- `p_y` = probabilité prédite pour la vraie classe ;
- `gamma` = force de focalisation ;
- `alpha_y` = poids de classe, plus élevé pour classes rares.

### 5.2.3 Implémentation PyTorch

```python
import torch
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss multi-classe.

    Args:
        logits: (B, C)
        targets: (B,)
        class_weights: (C,), optional
        gamma: focal gamma
        reduction: mean, sum, none
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    focal_factor = (1.0 - target_probs).clamp(min=1e-6).pow(gamma)
    loss = -focal_factor * target_log_probs

    if class_weights is not None:
        weights = class_weights.to(logits.device).gather(0, targets)
        loss = loss * weights

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
```

### 5.2.4 Loss totale recommandée

```python
loss_l1 = F.cross_entropy(logits_l1, labels_l1)
loss_l2 = focal_cross_entropy(
    logits_l2,
    labels_l2,
    class_weights=l2_class_weights,
    gamma=config.loss.focal_gamma_l2,
)

loss = config.loss.level_weight_l1 * loss_l1 + loss_l2
```

### 5.2.5 Configuration

```yaml
loss:
  type: hierarchical_ce_focal_l2
  level_weight_l1: 0.5
  level_weight_l2: 1.0
  use_focal_l1: false
  use_focal_l2: true
  focal_gamma_l2: 2.0
  class_weighting_l2: effective_number
  effective_number_beta: 0.9999
  max_class_weight: 10.0
```

---

## 5.3 Class-balanced weights pour L2

### 5.3.1 Formule Effective Number

```text
effective_num_c = 1 - beta ^ n_c
weight_c = (1 - beta) / effective_num_c
```

Puis normalisation :

```text
weight_c = weight_c / mean(weight)
weight_c = clip(weight_c, min=0, max=max_class_weight)
```

### 5.3.2 Script de génération

Fichier : `scripts/build_l2_class_weights.py`

Entrées :

```text
train_labels.parquet
l2_label_encoder.pkl
```

Sorties :

```text
models/flat_hydra/l2_class_counts.json
models/flat_hydra/l2_class_weights.pt
models/flat_hydra/l2_frequency_buckets.json
```

Pseudo-code :

```python
from collections import Counter
import torch


def build_effective_number_weights(labels_l2, num_classes, beta=0.9999, max_weight=10.0):
    counts = Counter(labels_l2)
    weights = []

    for c in range(num_classes):
        n = max(counts.get(c, 0), 1)
        effective_num = 1.0 - beta ** n
        w = (1.0 - beta) / effective_num
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max_weight)
    return weights
```

---

## 5.4 Deux phases d’entraînement

L’approche recommandée s’inspire du papier :

- **Phase 1 — Exploration long-tail :** focal loss + class weights.
- **Phase 2 — Stabilisation :** CrossEntropy ou bootstrap CE pour réduire l’effet du bruit label.

### 5.4.1 Phase 1 — Long-tail exploration

But : améliorer le rappel top-K sur classes rares.

Configuration :

```yaml
training_phase_1:
  enabled: true
  epochs: 2-4
  loss: hierarchical_ce_focal_l2
  focal_gamma_l2: 2.0
  class_weighting_l2: effective_number
  learning_rate: 3.5e-5
```

### 5.4.2 Phase 2 — Stabilisation

But : éviter l’overfitting sur labels rares bruités et améliorer top-1.

Option A, simple et recommandée pour commencer :

```yaml
training_phase_2:
  enabled: true
  epochs: 2-6
  loss: hierarchical_ce
  learning_rate: 1.0e-5
  load_from_phase_1_checkpoint: true
```

Option B, avancée : bootstrapped CE.

```text
q = softmax(logits)
y_onehot = one_hot(label)
y_bootstrap = beta * y_onehot + (1 - beta) * stop_gradient(q)
loss = CE_soft_targets(logits, y_bootstrap)
```

Configuration :

```yaml
training_phase_2:
  loss: bootstrapped_ce_l2
  bootstrap_beta: 0.9
  bootstrap_start_step_ratio: 0.2
```

Implémentation soft-target CE :

```python
def soft_target_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

---

## 5.5 Label-path augmentation

## 5.5.1 Motivation

Le modèle discriminatif ne connaît pas explicitement la sémantique des labels. Il faut exposer les chemins `L1 > L2` au modèle pendant l’entraînement.

## 5.5.2 Stratégies d’augmentation

### Variante A — préfixe label path

```text
[CAT] {l1_name} > {l2_name} [TEXT] {designation}. {description}
```

### Variante B — masquage L1

```text
[CAT] [MASK] > {l2_name} [TEXT] {designation}. {description}
```

### Variante C — masquage L2

```text
[CAT] {l1_name} > [MASK] [TEXT] {designation}. {description}
```

### Variante D — remplacement partiel du texte par label path

```text
{designation}. [CAT] {l1_name} > {l2_name}
```

## 5.5.3 Politique recommandée

Augmenter surtout les classes rares.

```yaml
label_path_augmentation:
  enabled: true
  apply_to_train_only: true
  rare_l2_threshold: 100
  augmentation_factor:
    l2_count_lt_10: 5
    l2_count_10_50: 3
    l2_count_50_100: 2
    l2_count_ge_100: 1
  mask_probability: 0.3
  variants:
    - prefix_full_path
    - mask_l1
    - mask_l2
```

## 5.5.4 Implémentation Dataset

```python
class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, label_path_aug_config=None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_path_aug_config = label_path_aug_config

    def build_text(self, row):
        designation = str(row.get("designation", "") or "")
        description = str(row.get("description", "") or "")
        base_text = f"{designation}. {description}".strip()

        if not self.label_path_aug_config or not self.label_path_aug_config.enabled:
            return base_text

        if not row.get("use_augmented_text", False):
            return base_text

        l1_name = row["l1_name"]
        l2_name = row["l2_name"]

        if random.random() < self.label_path_aug_config.mask_probability:
            if random.random() < 0.5:
                l1_name = "<mask>"
            else:
                l2_name = "<mask>"

        return f"[CAT] {l1_name} > {l2_name} [TEXT] {base_text}"
```

**Note CamemBERT :** utiliser le token `<mask>` natif de CamemBERT plutôt que `[MASK]`.

---

# 6. Candidate generation et scoring hiérarchique

## 6.1 Objectif

Le Domain Expert doit produire une liste `top-K` de candidats `L2` qui maximise la probabilité que la vraie `L2` soit présente dans la liste.

## 6.2 Métriques top-K obligatoires

À ajouter dans l’évaluation :

```text
l2_top1_accuracy
l2_top3_recall
l2_top5_recall
l2_top10_recall
l2_top20_recall
rare_l2_top10_recall
rare_l2_top20_recall
```

Formule :

```text
topK_recall = count(y_true_l2 ∈ topK_pred_l2) / N
```

## 6.3 Scoring hiérarchique

### 6.3.1 Softmax par niveau

```python
probs_l1 = torch.softmax(logits_l1, dim=-1)
probs_l2 = torch.softmax(logits_l2, dim=-1)
```

### 6.3.2 Score combiné recommandé

```text
score(l2) = log(P(l2 | text) + eps) + alpha * log(P(parent_l1(l2) | text) + eps)
```

Paramètre :

```yaml
hierarchy_scoring:
  enabled: true
  alpha: 0.5
  eps: 1.0e-12
```

### 6.3.3 Pseudo-code

```python
def hierarchy_aware_scores(probs_l1, probs_l2, l2_to_l1, alpha=0.5, eps=1e-12):
    """Compute hierarchy-aware score for every L2.

    Args:
        probs_l1: Tensor (B, num_l1)
        probs_l2: Tensor (B, num_l2)
        l2_to_l1: LongTensor (num_l2,) mapping each L2 id to parent L1 id
    Returns:
        scores_l2: Tensor (B, num_l2)
    """
    parent_probs = probs_l1[:, l2_to_l1]  # (B, num_l2)
    return torch.log(probs_l2 + eps) + alpha * torch.log(parent_probs + eps)
```

## 6.4 Top-K candidate generation

```python
def generate_topk_candidates(logits_l1, logits_l2, l2_to_l1, k=10, alpha=0.5):
    probs_l1 = torch.softmax(logits_l1, dim=-1)
    probs_l2 = torch.softmax(logits_l2, dim=-1)

    scores = hierarchy_aware_scores(probs_l1, probs_l2, l2_to_l1, alpha=alpha)
    top_scores, top_l2_ids = torch.topk(scores, k=k, dim=-1)

    top_l1_ids = l2_to_l1[top_l2_ids]
    top_l2_probs = probs_l2.gather(1, top_l2_ids)
    top_l1_probs = probs_l1.gather(1, top_l1_ids)

    return {
        "top_l2_ids": top_l2_ids,
        "top_l1_ids": top_l1_ids,
        "top_scores": top_scores,
        "top_l2_probs": top_l2_probs,
        "top_l1_probs": top_l1_probs,
    }
```

## 6.5 Valeurs de K à tester

```yaml
candidate_generation:
  k_values_for_eval: [3, 5, 10, 20]
  default_k_for_llm: 10
```

`K=10` est recommandé pour commencer. `K=20` peut améliorer le rappel mais augmente le coût prompt LLM.

---

# 7. Difficulty Router

## 7.1 Rôle

Le router décide si un produit est traité par le `fast path` ou par le `LLM hard path`.

## 7.2 Signaux utilisés

| Signal | Description |
|---|---|
| `p_l2_top1` | Probabilité softmax du meilleur L2 |
| `margin_l2` | `p_l2_top1 - p_l2_top2` |
| `p_l1_parent` | Probabilité du L1 parent du L2 top1 |
| `hierarchy_inconsistent` | `argmax L1 != parent(argmax L2)` |
| `is_rare_l2` | top1 L2 a peu d’exemples train |
| `description_length` | texte très court = plus d’incertitude |
| `sibling_confusion` | plusieurs top-K ont même L1 parent ou noms proches |

## 7.3 Politique initiale

```yaml
llm_routing:
  enabled: true
  default_top_k: 10

  thresholds:
    min_l2_confidence: 0.75
    min_l2_margin: 0.15
    min_parent_l1_confidence: 0.50
    rare_class_threshold: 100
    rare_class_min_confidence: 0.90
    min_text_chars: 30

  rules:
    send_if_hierarchy_inconsistent: true
    send_if_low_l2_confidence: true
    send_if_low_l2_margin: true
    send_if_rare_and_not_very_confident: true
    send_if_short_text: false
    send_if_sibling_confusion: true

  target_pass_through_rate:
    min: 0.10
    max: 0.25
```

## 7.4 Pseudo-code

```python
def should_route_to_llm(
    probs_l1,
    probs_l2,
    top_l2_ids,
    l2_to_l1,
    l2_counts,
    text,
    config,
):
    pred_l1 = int(torch.argmax(probs_l1))
    pred_l2 = int(top_l2_ids[0])
    parent_l1 = int(l2_to_l1[pred_l2])

    top2_probs, _ = torch.topk(probs_l2, k=2)
    p1 = float(top2_probs[0])
    p2 = float(top2_probs[1])
    margin = p1 - p2

    hierarchy_inconsistent = pred_l1 != parent_l1
    rare_l2 = l2_counts[pred_l2] < config.thresholds.rare_class_threshold
    short_text = len(text.strip()) < config.thresholds.min_text_chars
    parent_l1_conf = float(probs_l1[parent_l1])

    reasons = []

    if config.rules.send_if_hierarchy_inconsistent and hierarchy_inconsistent:
        reasons.append("hierarchy_inconsistent")

    if config.rules.send_if_low_l2_confidence and p1 < config.thresholds.min_l2_confidence:
        reasons.append("low_l2_confidence")

    if config.rules.send_if_low_l2_margin and margin < config.thresholds.min_l2_margin:
        reasons.append("low_l2_margin")

    if config.rules.send_if_rare_and_not_very_confident and rare_l2 and p1 < config.thresholds.rare_class_min_confidence:
        reasons.append("rare_l2_not_confident")

    if parent_l1_conf < config.thresholds.min_parent_l1_confidence:
        reasons.append("low_parent_l1_confidence")

    if config.rules.send_if_short_text and short_text:
        reasons.append("short_text")

    return len(reasons) > 0, reasons
```

---

# 8. Génération offline des définitions de catégories L2

## 8.1 Objectif

Créer une définition textuelle française, stable et versionnée pour chaque `L2`, utilisée par le LLM au moment du reranking.

Le LLM ne doit pas recevoir des exemples bruts non structurés à l’inférence. Il doit recevoir une définition compacte et claire.

## 8.2 Données sources

Pour chaque `L2` :

```text
- l2_id
- l2_name
- parent l1_id
- l1_name
- chemin: L1 > L2
- exemples produits: designation + description
- nombre d’exemples train
- siblings L2 sous le même L1
- description métier / taxonomy metadata si disponible
```

## 8.3 Sampling des exemples

### 8.3.1 Objectif de sampling

Chaque définition doit résumer les caractéristiques centrales de la catégorie, sans surreprésenter les produits populaires ni le bruit seller.

### 8.3.2 Stratégie recommandée

Pour chaque catégorie `L2` :

1. Sélectionner `N_popular` produits populaires ou fréquents si score disponible.
2. Sélectionner `N_representative` produits représentatifs via embeddings + clustering.
3. Ajouter `N_recent` produits récents si la taxonomie évolue rapidement.
4. Dédupliquer par normalisation titre.
5. Filtrer les exemples trop courts ou manifestement bruités.

Valeurs initiales :

```yaml
category_definition_generation:
  examples_per_category:
    target_total: 20
    popular: 8
    representative: 10
    recent: 2
  rare_category_min_examples: 3
  max_description_chars_per_example: 300
```

### 8.3.3 Embedding clustering

Utiliser le même CamemBERT ou un sentence-transformer français.

Pseudo-code :

```python
def select_representative_examples(products, embedder, n_clusters=10):
    texts = [f"{p.designation}. {p.description}" for p in products]
    embeddings = embedder.encode(texts, batch_size=128)

    k = min(n_clusters, len(products))
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)

    selected = []
    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        centroid = embeddings[idxs].mean(axis=0)
        best_idx = idxs[np.argmin(np.linalg.norm(embeddings[idxs] - centroid, axis=1))]
        selected.append(products[best_idx])

    return selected
```

## 8.4 Prompt de génération offline

Fichier : `prompts/generate_category_definition_fr.txt`

```text
Tu es un expert en taxonomie e-commerce française.

Objectif : générer une définition claire, courte et opérationnelle pour une catégorie L2.
Cette définition sera utilisée par un autre modèle pour choisir entre plusieurs catégories candidates.

Contraintes :
- Écris en français.
- Ne copie pas les exemples produit mot à mot.
- Décris la fonction principale et l’usage typique des produits.
- Mentionne les produits inclus.
- Mentionne les produits à exclure, surtout les catégories voisines.
- Liste des signaux lexicaux positifs utiles.
- Liste des signaux lexicaux négatifs / pièges fréquents.
- Si les exemples sont insuffisants ou bruités, indique une définition prudente basée sur le chemin taxonomique.

Catégorie :
- L1: {l1_name}
- L2: {l2_name}
- Chemin: {l1_name} > {l2_name}

Catégories voisines sous le même L1 :
{siblings_l2_names}

Exemples produits associés à cette L2 :
{sampled_products}

Réponds uniquement en JSON valide avec ce schéma :
{
  "definition_fr": "...",
  "included_products": ["..."],
  "excluded_products": ["..."],
  "positive_signals": ["..."],
  "negative_signals": ["..."],
  "ambiguities": ["..."],
  "confidence": "high|medium|low"
}
```

## 8.5 Schéma de stockage

Fichier : `data/category_definitions/category_definitions_v1.jsonl`

```json
{
  "l2_id": 1842,
  "l1_id": 31,
  "l1_name": "Maison",
  "l2_name": "Linge de lit",
  "path_text": "Maison > Linge de lit",
  "definition_fr": "Produits textiles destinés à équiper un lit, incluant les housses de couette, draps, taies et parures. Cette catégorie exclut les plaids décoratifs, couvertures d’appoint et coussins de canapé.",
  "included_products": ["housse de couette", "drap", "taie", "parure de lit"],
  "excluded_products": ["plaid", "couverture décorative", "coussin", "rideau"],
  "positive_signals": ["housse de couette", "drap", "taie", "parure", "220x240"],
  "negative_signals": ["plaid", "canapé", "décoratif", "rideau"],
  "ambiguities": ["Les parures peuvent contenir plusieurs pièces ; vérifier l’usage literie."],
  "confidence": "high",
  "num_examples_used": 20,
  "generated_by_model": "mixtral-8x7b-instruct-q4",
  "generation_prompt_version": "v1",
  "definition_version": "v1",
  "created_at": "2026-06-19"
}
```

## 8.6 Validation qualité des définitions

### 8.6.1 Contrôles automatiques

Rejeter ou marquer `needs_review` si :

```text
- JSON invalide
- definition_fr vide
- longueur definition_fr < 80 caractères
- longueur definition_fr > 800 caractères
- positive_signals vide
- définition mentionne une autre L2 comme catégorie principale
- confidence = low
```

### 8.6.2 Contrôles humains

Échantillonner :

```text
- 100 catégories fréquentes
- 100 catégories rares
- 100 catégories à forte confusion
- toutes les catégories avec confidence=low
```

Créer un fichier QA :

```text
data/category_definitions/category_definitions_v1_review.csv
```

Colonnes :

```text
l1_name,l2_name,path_text,definition_fr,positive_signals,negative_signals,confidence,review_status,review_comment
```

---

# 9. LLM General Expert Reranker

## 9.1 Rôle

Le LLM choisit exactement une catégorie `L2` parmi les candidats fournis par le Domain Expert. Il ne doit jamais inventer une catégorie hors liste.

## 9.2 Entrées

```json
{
  "product_id": "...",
  "designation": "...",
  "description": "...",
  "candidates": [
    {
      "rank": 1,
      "l1_id": 31,
      "l1_name": "Maison",
      "l2_id": 1842,
      "l2_name": "Linge de lit",
      "path_text": "Maison > Linge de lit",
      "domain_score": -0.42,
      "domain_p_l2": 0.61,
      "domain_p_parent_l1": 0.87,
      "definition_fr": "...",
      "positive_signals": ["..."],
      "negative_signals": ["..."]
    }
  ]
}
```

## 9.3 Sortie attendue

```json
{
  "selected_l2_id": 1842,
  "selected_path": "Maison > Linge de lit",
  "confidence": "high",
  "reason": "Le titre mentionne une housse de couette 220x240, ce qui correspond à la literie plutôt qu’à un plaid décoratif."
}
```

## 9.4 Prompt d’inférence

Fichier : `prompts/llm_rerank_l2_fr_v1.txt`

```text
Tu es un expert en classification de produits e-commerce.

Ta tâche est de choisir exactement UNE catégorie L2 parmi les catégories candidates.
Tu ne dois jamais proposer une catégorie absente de la liste.
Tu dois classer selon la fonction principale et l’usage prévu du produit.
La désignation est prioritaire si la description est bruitée ou marketing.
Ne choisis pas une catégorie seulement parce qu’un mot est présent : compare le sens global.

Produit :
- Désignation : {designation}
- Description : {description}

Catégories candidates :
{candidate_blocks}

Pour choisir :
1. Identifie les indices importants dans la désignation et la description.
2. Compare les catégories proches.
3. Applique les exclusions indiquées dans les définitions.
4. Sélectionne uniquement une catégorie candidate.

Réponds uniquement en JSON valide :
{
  "selected_l2_id": <int>,
  "selected_path": "<L1 > L2>",
  "confidence": "high|medium|low",
  "reason": "<raison courte en français>"
}
```

### 9.4.1 Format candidate block

```text
[{rank}] ID L2: {l2_id}
Chemin: {path_text}
Définition: {definition_fr}
Signaux positifs: {positive_signals}
Exclusions / signaux négatifs: {negative_signals}
Score Domain Expert: {domain_score}
```

## 9.5 Paramètres LLM

```yaml
llm_general_expert:
  enabled: true
  provider: local
  model: mixtral-8x7b-instruct
  quantization: q4_k_m
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 256
  timeout_seconds: 10
  retry_count: 1
  require_json: true
  fallback_to_domain_expert_on_error: true
```

## 9.6 Modèles LLM à benchmarker

| Modèle | Rôle | Notes |
|---|---|---|
| `mixtral-8x7b-instruct` quantized | baseline inspirée papier | bon compromis qualité/coût, lourd |
| `mistral-7b-instruct` | baseline léger | moins cher, à tester |
| `qwen2.5-7b-instruct` / équivalent | baseline multilingue | bon raisonnement, à tester |
| API propriétaire | upper bound qualité | coût + données à valider juridiquement |

## 9.7 Parsing robuste

```python
import json
import re


def parse_llm_json(response_text: str) -> dict:
    text = response_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(match.group(0))
```

## 9.8 Validation de sortie LLM

Après parsing :

```python
def validate_llm_output(output, candidate_l2_ids):
    selected = int(output["selected_l2_id"])
    if selected not in candidate_l2_ids:
        raise ValueError("LLM selected a category outside candidate list")

    if output.get("confidence") not in {"high", "medium", "low"}:
        output["confidence"] = "medium"

    return output
```

Fallback : si validation échoue, utiliser `top1` du Domain Expert avec `decision_source = domain_expert_fallback_after_llm_error`.

---

# 10. API et interfaces logicielles

## 10.1 Interface DomainExpertClassifier

```python
class DomainExpertClassifier(nn.Module):
    def forward(self, input_ids, attention_mask):
        """Returns unified logits.

        Returns:
            unified_logits: Tensor (B, num_l1 + num_l2)
        """
        ...

    def predict_proba(self, batch):
        """Returns probs_l1 and probs_l2."""
        ...

    def predict_topk(self, batch, k: int, hierarchy_scoring_config):
        """Returns hierarchy-aware top-K candidates."""
        ...
```

## 10.2 Interface CategoryDefinitionStore

```python
class CategoryDefinitionStore:
    def __init__(self, path: str):
        self.definitions = self._load_jsonl(path)

    def get(self, l2_id: int) -> dict:
        if l2_id not in self.definitions:
            return self._fallback_definition(l2_id)
        return self.definitions[l2_id]

    def get_many(self, l2_ids: list[int]) -> list[dict]:
        return [self.get(l2_id) for l2_id in l2_ids]
```

Fallback definition :

```json
{
  "definition_fr": "Catégorie définie par le chemin taxonomique fourni. Utiliser le nom L1 et L2 pour déterminer l’usage principal.",
  "positive_signals": [],
  "negative_signals": [],
  "confidence": "low"
}
```

## 10.3 Interface Router

```python
class DifficultyRouter:
    def should_route(self, prediction_context: dict) -> tuple[bool, list[str]]:
        ...
```

## 10.4 Interface LLMGeneralExpertReranker

```python
class LLMGeneralExpertReranker:
    def rerank(self, product: dict, candidates: list[dict]) -> dict:
        """Returns selected L2 id and reason."""
        prompt = self.prompt_builder.build(product, candidates)
        raw_response = self.llm_client.generate(prompt)
        parsed = parse_llm_json(raw_response)
        validated = validate_llm_output(parsed, [c["l2_id"] for c in candidates])
        return validated
```

## 10.5 Interface complète de prédiction

```python
class DualExpertProductClassifier:
    def predict(self, product: dict) -> dict:
        text = preprocess(product)
        domain_output = self.domain_expert.predict_topk(text, k=self.config.top_k)
        route_to_llm, reasons = self.router.should_route(domain_output)

        if not route_to_llm:
            return self.build_fast_path_result(product, domain_output)

        definitions = self.definition_store.get_many(domain_output.top_l2_ids)
        candidates = self.build_llm_candidates(domain_output, definitions)

        try:
            llm_output = self.llm_reranker.rerank(product, candidates)
            return self.build_llm_result(product, domain_output, llm_output, reasons)
        except Exception as exc:
            return self.build_fallback_result(product, domain_output, exc, reasons)
```

---

# 11. Configuration globale

Fichier : `configs/dual_expert_flat_hydra.yaml`

```yaml
project:
  name: dual_expert_flat_hydra
  version: v1
  language: fr

model:
  architecture: flat_hydra_domain_expert
  encoder: camembert-base
  max_length: 256
  num_classes_l1: 157
  num_classes_l2: 6126
  projection_dim: 1536
  dropout: 0.2

loss:
  type: hierarchical_ce_focal_l2
  level_weight_l1: 0.5
  level_weight_l2: 1.0
  use_focal_l2: true
  focal_gamma_l2: 2.0
  class_weighting_l2: effective_number
  effective_number_beta: 0.9999
  max_class_weight: 10.0

training:
  batch_size: 64
  epochs: 10
  lr: 3.5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  fp16: true
  early_stopping_metric: val_f1_l2_macro
  early_stopping_patience: 2
  early_stopping_min_delta: 0.001

training_phases:
  enabled: true
  phase_1:
    name: long_tail_exploration
    epochs: 3
    loss: hierarchical_ce_focal_l2
    lr: 3.5e-5
  phase_2:
    name: stabilization
    epochs: 5
    loss: hierarchical_ce
    lr: 1.0e-5

label_path_augmentation:
  enabled: true
  apply_to_train_only: true
  rare_l2_threshold: 100
  mask_probability: 0.3
  augmentation_factor:
    l2_count_lt_10: 5
    l2_count_10_50: 3
    l2_count_50_100: 2
    l2_count_ge_100: 1

hierarchy_scoring:
  enabled: true
  alpha: 0.5
  eps: 1.0e-12

candidate_generation:
  default_top_k: 10
  eval_top_k_values: [1, 3, 5, 10, 20]

category_definitions:
  enabled: true
  path: data/category_definitions/category_definitions_v1.jsonl
  version: v1

llm_routing:
  enabled: true
  thresholds:
    min_l2_confidence: 0.75
    min_l2_margin: 0.15
    min_parent_l1_confidence: 0.50
    rare_class_threshold: 100
    rare_class_min_confidence: 0.90
    min_text_chars: 30
  rules:
    send_if_hierarchy_inconsistent: true
    send_if_low_l2_confidence: true
    send_if_low_l2_margin: true
    send_if_rare_and_not_very_confident: true
    send_if_short_text: false
    send_if_sibling_confusion: true
  target_pass_through_rate:
    min: 0.10
    max: 0.25

llm_general_expert:
  enabled: true
  provider: local
  model: mixtral-8x7b-instruct
  quantization: q4_k_m
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 256
  timeout_seconds: 10
  retry_count: 1
  require_json: true
  fallback_to_domain_expert_on_error: true

audit:
  enabled: true
  log_fast_path_sample_rate: 0.01
  log_hard_path_sample_rate: 1.0
  output_path: logs/dual_expert_predictions.jsonl
```

---

# 12. Données et artefacts

## 12.1 Nouveaux artefacts à produire

```text
models/flat_hydra/
    tokenizer/
    best_model.pt
    checkpoint_latest.pt
    l1_label_encoder.pkl
    l2_label_encoder.pkl
    l2_to_l1_mapping.json
    config.json
    metrics.json
    training_complete.txt

    # Nouveaux artefacts
    l2_class_counts.json
    l2_class_weights.pt
    l2_frequency_buckets.json
    topk_recall_report.json
    hierarchy_scoring_report.json
    routing_policy.yaml
    llm_reranker_config.yaml
    llm_reranker_prompt.txt
    hard_case_predictions.jsonl
    llm_audit_samples.csv

data/category_definitions/
    category_definitions_v1.jsonl
    category_definitions_v1_review.csv
    generation_inputs_v1.jsonl
    generation_outputs_raw_v1.jsonl
    generation_errors_v1.jsonl

prompts/
    generate_category_definition_fr_v1.txt
    llm_rerank_l2_fr_v1.txt
```

## 12.2 Logs d’audit

Chaque prédiction hard path doit être journalisée.

```json
{
  "timestamp": "2026-06-19T12:00:00Z",
  "product_id": "abc123",
  "designation": "Housse de couette coton 220x240",
  "description_hash": "sha256:...",
  "domain_pred_l1": 31,
  "domain_pred_l2": 900,
  "domain_top_k": [900, 1842, 3021, 88],
  "domain_scores": [-0.3, -0.5, -1.2, -2.0],
  "routing_reasons": ["low_l2_margin", "sibling_confusion"],
  "llm_selected_l2": 1842,
  "final_l2": 1842,
  "decision_source": "llm_general_expert",
  "llm_confidence": "high",
  "llm_reason": "...",
  "prompt_version": "llm_rerank_l2_fr_v1",
  "definition_version": "v1",
  "latency_ms": 1830
}
```

Pour raisons de confidentialité, éviter de logger les descriptions complètes en production si non nécessaire. Stocker un hash ou une version tronquée.

---

# 13. Évaluation

## 13.1 Splits

Utiliser les splits existants :

```text
train / validation / test
```

Aucune définition de catégorie ne doit être générée à partir du `test`.

Recommandation : les définitions peuvent être générées à partir du `train` uniquement, ou du catalogue complet hors labels test si contexte production. Pour une évaluation scientifique stricte, utiliser `train` uniquement.

## 13.2 Métriques globales

```text
l1_accuracy
l1_macro_f1
l2_accuracy
l2_macro_f1
l2_weighted_f1
constrained_l2_macro_f1
hcr
```

## 13.3 Métriques top-K

```text
l2_top1_accuracy
l2_top3_recall
l2_top5_recall
l2_top10_recall
l2_top20_recall
```

## 13.4 Métriques par rareté

Créer des buckets `L2` selon le nombre d’exemples train :

```text
bucket_0: n < 10
bucket_1: 10 <= n < 50
bucket_2: 50 <= n < 100
bucket_3: 100 <= n < 500
bucket_4: n >= 500
```

Pour chaque bucket :

```text
count_classes
count_examples
l2_accuracy
l2_macro_f1
l2_top10_recall
llm_pass_through_rate
llm_improvement_vs_domain
```

## 13.5 Métriques LLM

```text
hard_case_accuracy_domain_top1
hard_case_accuracy_llm
llm_delta_accuracy
llm_valid_json_rate
llm_out_of_candidate_rate
llm_fallback_rate
average_prompt_tokens
average_completion_tokens
p50_latency_ms
p95_latency_ms
cost_per_10k_products
```

## 13.6 Métriques de routing

```text
pass_through_rate
fast_path_accuracy
hard_path_domain_accuracy
hard_path_llm_accuracy
false_easy_rate
false_hard_rate
```

Définitions :

```text
false_easy_rate = produits routés fast path mais Domain Expert incorrect
false_hard_rate = produits routés hard path mais Domain Expert correct et LLM non nécessaire
```

---

# 14. Plan expérimental

## 14.1 Expérience A — Baseline actuelle

```text
Flat HYDRA actuel
CE L1 + CE L2
argmax L1 + argmax L2
```

Objectif : établir la baseline.

Métriques : toutes les métriques globales + buckets rares.

## 14.2 Expérience B — Top-K sans changement training

```text
Flat HYDRA actuel
Export top-K L2
```

Objectif : savoir si le vrai L2 est présent dans les candidats.

Critère de succès :

```text
L2 top-10 recall suffisamment élevé pour justifier LLM reranking.
```

## 14.3 Expérience C — Scoring hiérarchique

Tester :

```text
alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

`alpha = 0.0` correspond à score L2 seul.

Objectif : améliorer cohérence hiérarchique et top-K recall.

## 14.4 Expérience D — Focal loss L2

```text
CE L1 + focal CE L2
```

Tester :

```text
gamma ∈ {1.0, 2.0}
beta ∈ {0.999, 0.9999}
max_class_weight ∈ {5, 10}
```

Objectif : améliorer rare-class recall et macro F1.

## 14.5 Expérience E — Label-path augmentation

Comparer :

```text
E0: sans augmentation
E1: augmentation toutes classes
E2: augmentation classes rares uniquement
E3: augmentation classes rares + mask L1/L2
```

Objectif : améliorer top-K recall et rare macro F1.

## 14.6 Expérience F — LLM reranking path-only

```text
Domain Expert top-10 + LLM avec chemins L1 > L2 uniquement
```

Objectif : baseline LLM sans définitions.

## 14.7 Expérience G — LLM reranking avec définitions

```text
Domain Expert top-10 + LLM avec chemins + definitions_fr + signaux
```

Objectif : mesurer l’apport des définitions L2.

## 14.8 Expérience H — Routing production-like

```text
Fast path + hard path uniquement selon router
```

Objectif : mesurer compromis qualité/coût.

Résultats à reporter :

```text
- L2 macro F1 global
- rare L2 macro F1
- pass-through rate LLM
- coût pour 10k produits
- latence p95
```

---

# 15. Critères d’acceptation

## 15.1 Critères ML

Le projet est acceptable si au moins 3 critères sur 4 sont satisfaits :

```text
1. L2 macro F1 augmente significativement vs baseline.
2. Rare L2 macro F1 augmente significativement.
3. L2 weighted F1 ne baisse pas de plus de 0.5 point.
4. HCR reste supérieur à 98%.
```

## 15.2 Critères LLM

```text
- valid_json_rate >= 99%
- out_of_candidate_rate = 0 après validation/fallback
- fallback_rate <= 2%
- pass_through_rate entre 10% et 25%
```

## 15.3 Critères production

```text
- Fast path compatible avec latence actuelle.
- Hard path timeout contrôlé.
- Fallback systématique au Domain Expert en cas d’erreur LLM.
- Tous les prompts/réponses hard path sont auditables.
- Les définitions de catégories sont versionnées.
```

---

# 16. Roadmap d’implémentation

## Phase 0 — Nettoyage et instrumentation

Durée estimée : 2–4 jours.

Tâches :

```text
- Corriger le pipeline training pour supprimer one-hot + BCE.
- Ajouter métriques top-K.
- Ajouter buckets de rareté L2.
- Ajouter export top-K candidates.
- Ajouter rapports par L1 et par bucket L2.
```

Livrables :

```text
metrics_topk.py
topk_recall_report.json
l2_frequency_buckets.json
```

## Phase 1 — Scoring hiérarchique

Durée estimée : 2–3 jours.

Tâches :

```text
- Implémenter HierarchyAwareScorer.
- Tester alpha values.
- Intégrer HCR/constrained F1.
```

Livrables :

```text
hierarchy_scorer.py
hierarchy_scoring_report.json
```

## Phase 2 — Rare-class training

Durée estimée : 1–2 semaines.

Tâches :

```text
- Générer l2_class_weights.pt.
- Ajouter focal CE L2.
- Ajouter training phases.
- Comparer CE vs focal vs two-phase.
```

Livrables :

```text
losses/focal_loss.py
scripts/build_l2_class_weights.py
configs/flat_hydra_focal_l2.yaml
```

## Phase 3 — Label-path augmentation

Durée estimée : 1 semaine.

Tâches :

```text
- Implémenter dataset augmentation.
- Tester rare-only vs all classes.
- Mesurer impact top-K recall.
```

Livrables :

```text
data/augmentation.py
configs/flat_hydra_label_path_aug.yaml
```

## Phase 4 — Génération définitions catégories

Durée estimée : 1–2 semaines selon volume.

Tâches :

```text
- Construire inputs de génération par L2.
- Implémenter sampling popular + representative.
- Générer category_definitions_v1.jsonl.
- Valider automatiquement JSON et qualité.
- Préparer review CSV.
```

Livrables :

```text
scripts/build_category_definition_inputs.py
scripts/generate_category_definitions.py
scripts/validate_category_definitions.py
data/category_definitions/category_definitions_v1.jsonl
```

## Phase 5 — LLM reranker offline evaluation

Durée estimée : 1 semaine.

Tâches :

```text
- Implémenter PromptBuilder.
- Implémenter LLM client local/API.
- Tester sur validation hard cases.
- Comparer path-only vs definitions.
```

Livrables :

```text
llm/reranker.py
llm/prompt_builder.py
prompts/llm_rerank_l2_fr_v1.txt
reports/llm_reranking_eval.json
```

## Phase 6 — Router + production-like evaluation

Durée estimée : 1 semaine.

Tâches :

```text
- Implémenter DifficultyRouter.
- Tuner thresholds pour 10–25% pass-through.
- Mesurer qualité/coût/latence.
```

Livrables :

```text
routing/difficulty_router.py
configs/routing_policy.yaml
reports/routing_eval.json
```

## Phase 7 — Déploiement progressif

Durée estimée : dépend infra.

Étapes :

```text
1. Shadow mode : LLM produit une prédiction mais non utilisée.
2. Offline audit : comparer LLM vs vérité terrain / human review.
3. Canary : activer sur faible trafic ou segment de catégories.
4. Full hard-path deployment : activer uniquement les cas routés hard.
5. Monitoring continu : drift, pass-through, erreurs JSON, latence.
```

---

# 17. Risques et mitigations

## 17.1 Le vrai L2 n’est pas dans le top-K

Risque : le LLM ne peut pas corriger si la bonne catégorie est absente.

Mitigations :

```text
- Optimiser top-K recall avant LLM.
- Tester K=20 pour hard cases.
- Améliorer focal loss / augmentation.
- Utiliser top-M L1 parents au lieu d’un seul L1.
```

## 17.2 Définitions de catégories bruitées

Risque : le LLM reranker suit une mauvaise définition.

Mitigations :

```text
- Validation automatique.
- Review humaine sur catégories rares/critiques.
- Versioning des définitions.
- Ajouter confidence et fallback definition.
```

## 17.3 LLM invente une catégorie

Risque : sélection hors liste.

Mitigations :

```text
- Prompt strict.
- JSON schema validation.
- Rejet si selected_l2_id hors candidates.
- Fallback Domain Expert.
```

## 17.4 Coût et latence LLM

Risque : hard path trop fréquent ou trop lent.

Mitigations :

```text
- Router avec target pass-through 10–25%.
- Cache par hash produit + top-K candidates.
- Quantization locale.
- Timeout + fallback.
- Batch LLM offline pour bulk catalog.
```

## 17.5 LLM sensible aux prompts

Risque : instabilité entre versions de prompt/modèle.

Mitigations :

```text
- temperature=0.
- prompt versionné.
- jeu de régression hard cases.
- comparaison avant/après toute modification.
```

## 17.6 Données confidentielles

Risque : fuite de descriptions produit vers API externe.

Mitigations :

```text
- Préférer LLM local pour données sensibles.
- Masquer données personnelles si présentes.
- Hacher/tronquer logs.
- Validation juridique avant API externe.
```

---

# 18. Structure de code recommandée

```text
src/
  models/
    flat_hydra.py
    domain_expert.py

  training/
    train_domain_expert.py
    losses.py
    class_weights.py
    label_path_augmentation.py

  inference/
    dual_expert_classifier.py
    hierarchy_scorer.py
    candidate_generator.py
    difficulty_router.py

  category_definitions/
    build_inputs.py
    sampler.py
    generate.py
    validate.py
    store.py

  llm/
    client.py
    prompt_builder.py
    reranker.py
    output_parser.py

  evaluation/
    metrics.py
    topk_metrics.py
    rare_bucket_metrics.py
    llm_eval.py
    routing_eval.py

  logging/
    audit_logger.py

configs/
  flat_hydra_baseline.yaml
  flat_hydra_focal_l2.yaml
  flat_hydra_label_path_aug.yaml
  dual_expert_flat_hydra.yaml
  routing_policy.yaml

prompts/
  generate_category_definition_fr_v1.txt
  llm_rerank_l2_fr_v1.txt

scripts/
  build_l2_class_weights.py
  build_category_definition_inputs.py
  generate_category_definitions.py
  validate_category_definitions.py
  evaluate_topk.py
  evaluate_llm_reranker.py
  evaluate_routing.py
```

---

# 19. Tests unitaires minimaux

## 19.1 Loss

```text
- focal_cross_entropy retourne scalaire positif.
- class_weights sont appliqués correctement.
- loss backward fonctionne sans NaN.
```

## 19.2 Hierarchy scorer

```text
- scores shape = (B, num_l2).
- parent_probs correctement indexé via l2_to_l1.
- alpha=0 équivaut à log P(L2).
```

## 19.3 Candidate generator

```text
- retourne exactement K candidats.
- top_l1_ids = parent(top_l2_ids).
- scores triés décroissants.
```

## 19.4 Router

```text
- low confidence déclenche hard path.
- high confidence + margin élevée retourne fast path.
- incohérence hiérarchique déclenche hard path.
```

## 19.5 LLM parser

```text
- JSON valide parsé.
- texte avec JSON encapsulé parsé.
- selected_l2_id hors liste rejeté.
- fallback appelé en cas d’erreur.
```

## 19.6 Definition store

```text
- get(l2_id) retourne définition.
- missing l2_id retourne fallback definition.
- version chargée correctement.
```

---

# 20. Décisions de design finales

1. **Conserver CamemBERT Flat HYDRA comme Domain Expert.**  
   Il est déjà adapté au français et au domaine.

2. **Optimiser d’abord le top-K recall.**  
   Le LLM ne peut améliorer que si la bonne catégorie est présente dans les candidats.

3. **Utiliser le LLM uniquement sur cas difficiles.**  
   Le coût doit rester contrôlé.

4. **Générer les définitions de catégories offline.**  
   Elles sont réutilisées à l’inférence, versionnées, auditables et moins coûteuses que des exemples few-shot bruts.

5. **Ne jamais laisser le LLM choisir hors top-K.**  
   Le LLM est un reranker, pas un classifieur libre sur 6126 catégories.

6. **Prioriser `L2 macro F1`, `rare L2 F1` et `top-K recall`.**  
   L’accuracy globale peut masquer l’échec sur classes rares.

7. **Déployer progressivement en shadow mode.**  
   Le hard path LLM doit être audité avant d’impacter la production.

---

# 21. Checklist d’implémentation

```text
[ ] Supprimer incohérence BCE/one-hot dans pipeline single-label.
[ ] Ajouter top-K candidate export.
[ ] Ajouter metrics top-K globales et par bucket rareté.
[ ] Ajouter hierarchy-aware scoring.
[ ] Générer l2_class_counts.json et l2_class_weights.pt.
[ ] Ajouter focal CE L2.
[ ] Ajouter training phase 1 / phase 2.
[ ] Ajouter label-path augmentation.
[ ] Construire inputs pour génération définitions L2.
[ ] Générer category_definitions_v1.jsonl.
[ ] Valider automatiquement les définitions.
[ ] Implémenter CategoryDefinitionStore.
[ ] Implémenter LLM prompt builder.
[ ] Implémenter LLM reranker + JSON parser + validation.
[ ] Implémenter DifficultyRouter.
[ ] Évaluer path-only vs definitions.
[ ] Évaluer pass-through/cost/latency.
[ ] Mettre en place audit logging.
[ ] Lancer shadow mode.
```
