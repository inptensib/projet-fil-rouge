# 02_hydra -- HYDRA : Classification Hierarchique Multi-tete

Implementation de **HYDRA** (Karl et al., EMNLP 2025) pour la classification hierarchique de produits Rakuten.

## Architecture

```
Texte (titre + description)
      |
  [Tokenizer : max_length=128]
      |
  [Encodeur : CamemBERT / XLM-RoBERTa / mBERT]
      |
  [Pooling : CLS | mean | max | last]
      |
  [Projection : Linear(d_model, hidden_dim)]
      |
  +--------+--------+
  |                  |
[Head L1]        [Head L2]
 (157)           (6100+)
```

### 3 variantes

| Variante | Mecanisme |
|----------|-----------|
| **HYDRALocal** | Tetes L1/L2 independantes |
| **HYDRAGlobal** | + tete globale (concatenation L1+L2) |
| **HYDRANested** | + contraintes d'imbrication (L2 conditionne par L1) |

## Pertes

- **FocalLoss** : `FL(p_t) = -(1 - p_t)^gamma * log(p_t)` -- concentre l'apprentissage sur les cas difficiles (gamma=0 revient a la CE standard)
- **HYDRALoss** : `L = w_L1 * L1 + w_L2 * L2 [+ w_global * L_global]` -- combine CrossEntropy ou FocalLoss par niveau

## Configuration (`configs/default.yaml`)

| Parametre | Default | Description |
|-----------|---------|-------------|
| `encoder` | `camembert-base` | Backbone |
| `variant` | `local` | `local` / `global` / `nested` |
| `hidden_dim` | `256` | Dimension projection |
| `dropout` | `0.3` | Dropout avant heads |
| `lr` | `3.5e-5` | Learning rate |
| `batch_size` | `128` | Taille de batch |
| `epochs` | `20` | Nombre d'epoques |
| `loss` | `cross_entropy` | `cross_entropy` / `focal` |
| `level_weights` | `[0.4, 0.6]` | Poids L1/L2 |

## Usage

```python
from src.models import create_hydra_model
from src.losses import HYDRALoss

model = create_hydra_model(
    variant="local", encoder_name="camembert-base",
    num_classes_l1=157, num_classes_l2=6100,
)

criterion = HYDRALoss(
    architecture="local", num_classes_l1=157, num_classes_l2=6100,
    use_focal=True, level_weight_l1=0.4,
)

level_outputs, unified_output = model(input_ids, attention_mask)
logits_l1, logits_l2 = level_outputs
loss, loss_l1, loss_l2 = criterion(level_outputs, unified_output, labels_l1, labels_l2)
```

## Structure

```
02_hydra/
  configs/default.yaml    # Hyperparametres
  src/
    models.py             # HYDRABase, HYDRALocal, HYDRAGlobal, HYDRANested
    losses.py             # FocalLoss, HYDRALoss
    trainer.py            # Boucle d'entrainement single-GPU
    predict.py            # Inference et evaluation
```

## References

- Karl et al., *HYDRA*, EMNLP 2025.
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
