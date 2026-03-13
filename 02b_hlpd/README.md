# 02b_hlpd -- HLPD : Décodeur Hiérarchique Non-Autorégressif

Implémentation de **HLPD** (Torba, Laclau et al., ECIR 2025) pour la classification hiérarchique multi-label.

> **Note** : Code d'implémentation à venir (Mehdi Aithamma). Ce README décrit l'architecture cible.

## Idée clé

Contrairement à HYDRA (têtes parallèles), HLPD utilise un **décodeur Transformer non-autorégressif** qui prédit tous les labels en un seul forward pass, avec la hiérarchie encodée directement dans le mécanisme d'attention.

## Architecture

```
Texte --> [Encodeur CamemBERT] --> H_enc
                                      |
Labels taxonomie --> [Embeddings pré-populated] --> E_labels
                                      |
                    [Cross-attention H_enc x E_labels]
                                      |
                    [Self-attention hiérarchique]  x N couches
                                      |
                    [Classification linéaire] --> Logits (1+157+6100)
```

**Self-attention hiérarchique** : un biais positionnel appris, dérivé des distances dans l'arbre taxonomique, force les labels proches (parent-enfant) à avoir une attention plus forte.

```
Attn_hiera(Q,K,V) = Softmax((QK^T / sqrt(d_k)) + W * M_hiera) * V
```

**ZLPR Loss** : ranking pairwise entre labels positifs/négatifs, plus stable que la BCE pour les problèmes avec beaucoup de classes négatives.

## HYDRA vs HLPD

| | HYDRA | HLPD |
|---|---|---|
| Paradigme | Multi-tête discriminatif | Encoder-Decoder |
| Cohérence hiérarchique | Post-hoc (Nested head) | Par construction (masque d'attention) |
| Perte | Focal / CE | ZLPR (pairwise ranking) |
| Complexité | O(n_classes) | O(n_labels^2) |

## Références

- Torba et al., *HLPD*, ECIR 2025.
- Su et al., *ZLPR Loss*, 2022.
