# 02_hydra -- Hierarchical Multi-task Classification

Implementation de l'architecture **HYDRA** pour la classification hierarchique de produits sur le catalogue Rakuten, basee sur :

> Karl et al., *"HYDRA: Hierarchical Multi-label Classification using Attention-based Architecture"*, EMNLP 2025.

---

## Architecture

```
Texte (titre + description du produit)
        |
  [Tokenizer : max_length=128]
        |
  [Encodeur partage : CamemBERT / XLM-RoBERTa / mBERT]
        |
  [Pooling : CLS | mean | max | last]
        |
  [Projection d'embedding : Linear(d_model, hidden_dim)]
        |
   +-----------+-----------+
   |                       |
[Head L1]              [Head L2]
Linear(hidden_dim,     Linear(hidden_dim,
       157)                   6100+)
   |                       |
 Softmax                Softmax
```

### 3 variantes

| Variante | Description | Specifite |
|----------|-------------|-----------|
| **HYDRALocal** | Tetes L1 et L2 independantes | Baseline multi-tete, chaque head classifie son niveau |
| **HYDRAGlobal** | Local + tete globale | Ajoute une tete unifiee qui opere sur la concatenation L1+L2 |
| **HYDRANested** | Local + tete nested | Ajoute des contraintes d'imbrication hierarchique (L2 conditionne par L1) |

---

## Fonctions de perte

### FocalLoss

Variante de la CrossEntropy qui reduit le poids des exemples bien classifies pour se concentrer sur les cas difficiles (Lin et al., 2017) :

```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```
- `gamma=2.0` : intensite du focusing (0 = CE standard)
- `alpha` : poids par classe (optionnel)

### HYDRALoss

Combine les pertes par niveau avec des poids configurables :

```
L_total = w_L1 * L(head_L1) + w_L2 * L(head_L2) [+ w_global * L(head_global)]
```

Supporte CrossEntropy ou FocalLoss via la configuration.

---

## Configuration

Fichier : `configs/default.yaml`

| Parametre | Default | Description |
|-----------|---------|-------------|
| `encoder` | `camembert-base` | Backbone pre-entraine |
| `variant` | `local` | `local`, `global`, ou `hybrid` |
| `hidden_dim` | `256` | Dimension de la projection |
| `dropout` | `0.3` | Dropout avant les heads |
| `lr` | `3.5e-5` | Learning rate |
| `batch_size` | `128` | Taille de batch |
| `epochs` | `20` | Nombre d'epoques |
| `loss` | `cross_entropy` | `cross_entropy` ou `focal` |
| `level_weights` | `[0.4, 0.6]` | Poids L1/L2 dans la perte |

---

## Structure

```
02_hydra/
  __init__.py
  configs/
    default.yaml          # Hyperparametres
  src/
    __init__.py
    models.py             # HYDRABase, HYDRALocal, HYDRAGlobal, HYDRANested
    losses.py             # FocalLoss, HYDRALoss
    trainer.py            # Boucle d'entrainement avec MLflow
    predict.py            # Inference et evaluation
```

---

## Usage

```python
from src.models import create_hydra_model
from src.losses import HYDRALoss

# Creer un modele HYDRA Local
model = create_hydra_model(
    variant="local",
    encoder_name="camembert-base",
    num_l1=157,
    num_l2=6100,
    hidden_dim=256,
)

# Perte multi-tete
criterion = HYDRALoss(
    variant="local",
    loss_type="focal",
    level_weights=[0.4, 0.6],
    num_l1=157,
    num_l2=6100,
)

# Forward pass
logits_l1, logits_l2 = model(input_ids, attention_mask)
loss = criterion(logits_l1, logits_l2, labels_l1, labels_l2)
```

---

## References

- **HYDRA** -- Karl et al., *HYDRA: Hierarchical Multi-label Classification using Attention-based Architecture*, EMNLP 2025.
- **FocalLoss** -- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
