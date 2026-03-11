# HLPD -- Hierarchical Label Pre-populated Decoder

Implementation of the **HLPD** architecture for hierarchical multi-label text classification on the Rakuten product catalogue, based on:

> Fatos Torba, Charlotte Laclau et al., *"HLPD: Hierarchical Label Pre-populated Decoder for Multi-label Text Classification"*, ECIR 2025.

## Key ideas

1. **Label Pre-population** -- The decoder is initialised with embeddings for *all* label nodes in the taxonomy (root + 157 L1 + 6100+ L2 classes). Labels are encoded once using textual descriptions following the template *"Child is a sub-topic of Parent"*.

2. **Hierarchical Self-Attention** -- Standard self-attention in the decoder is augmented with a learnable positional bias derived from shortest-path distances in the taxonomy tree:
   ```
   Attn_hiera(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + W . M_hiera) V
   ```

3. **Non-autoregressive decoding** -- All labels are classified in a single forward pass.

4. **ZLPR loss** -- Zero-bounded Log-sum-exp & Pairwise Rank-based loss for multi-label optimisation.

## Structure

```
02b_hlpd/
  configs/
    default.yaml        # Hyperparameters
  src/
    __init__.py
    model.py             # HLPDModel, LabelEncoder, HLPDecoderLayer
    hierarchical_attention.py  # HierarchicalSelfAttention, distance matrix
    losses.py            # ZLPRLoss, HierarchicalBCELoss, multi-hot targets
```

## Usage

```python
from src.model import HLPDModel
from src.losses import ZLPRLoss, build_multi_hot_targets

model = HLPDModel(
    encoder_name="camembert-base",
    num_l1=157,
    num_l2=6100,
    l2_to_l1=l2_to_l1_mapping,
)
criterion = ZLPRLoss()

logits = model(input_ids, attention_mask)           # (B, 1+157+6100)
targets = build_multi_hot_targets(labels_l1, labels_l2, 157, 6100)
loss = criterion(logits, targets)
```
