# 02b_hlpd -- HLPD : Decodeur Hierarchique Non-Autoregressif

Implementation de **HLPD** (Torba, Laclau et al., ECIR 2025) pour la classification hierarchique multi-label.

> **Note** : Code d'implementation a venir (Mehdi Aithamma). Ce README decrit l'architecture cible.

## Idee cle

Contrairement a HYDRA (tetes paralleles), HLPD utilise un **decodeur Transformer non-autoregressif** qui predit tous les labels en un seul forward pass, avec la hierarchie encodee directement dans le mecanisme d'attention.

## Architecture

```
Texte --> [Encodeur CamemBERT] --> H_enc
                                      |
Labels taxonomie --> [Embeddings pre-populated] --> E_labels
                                      |
                    [Cross-attention H_enc x E_labels]
                                      |
                    [Self-attention hierarchique]  x N couches
                                      |
                    [Classification lineaire] --> Logits (1+157+6100)
```

**Self-attention hierarchique** : un biais positionnel appris, derive des distances dans l'arbre taxonomique, force les labels proches (parent-enfant) a avoir une attention plus forte.

```
Attn_hiera(Q,K,V) = Softmax((QK^T / sqrt(d_k)) + W * M_hiera) * V
```

**ZLPR Loss** : ranking pairwise entre labels positifs/negatifs, plus stable que la BCE pour les problemes avec beaucoup de classes negatives.

## HYDRA vs HLPD

| | HYDRA | HLPD |
|---|---|---|
| Paradigme | Multi-tete discriminatif | Encoder-Decoder |
| Coherence hierarchique | Post-hoc (Nested head) | Par construction (masque d'attention) |
| Perte | Focal / CE | ZLPR (pairwise ranking) |
| Complexite | O(n_classes) | O(n_labels^2) |

## References

- Torba et al., *HLPD*, ECIR 2025.
- Su et al., *ZLPR Loss*, 2022.
