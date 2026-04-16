# 02b_hlpd — HLPD : Décodeur Hiérarchique Non-Autorégressif

Implémentation de **HLPD** (Torba, Laclau et al., ECIR 2025) pour la classification
hiérarchique multi-label sur le dataset Rakuten (157 L1 × 6 126 L2).

## Idée clé

Contrairement à HYDRA (têtes parallèles), HLPD utilise un **décodeur Transformer
non-autorégressif** qui prédit tous les labels en un seul forward pass, avec la
hiérarchie encodée directement dans le mécanisme d'attention.

## Architecture

```
Texte ──→ [CamemBERT] ──→ H_enc (B, 128, 768)
                                  │
Labels (L=6283) ──→ [LabelEmbeddings] ──→ E_labels (B, L, 768)
                                  │
              [Cross-attention  Q=E_labels, K=V=H_enc]
                                  │
              [Self-attention hiérarchique  W⊙M_hiera + b]  × num_layers
                                  │
              [Logits = E_labels · LabelEmbeddings.Tᵀ]  (tied weights)
                                  │
                           (B, L=6283)
```

**Self-attention hiérarchique** : biais additif appris `W ⊙ M_hiera + b` sur les
scores QKᵀ avant softmax, où `M_hiera` est la matrice binaire encodant les relations
parent-enfant et siblings dans la taxonomie.

**Tied weights** : la matrice de projection finale partage ses poids avec
`LabelEmbeddings` — réduit les paramètres de ~4.8M et améliore la cohérence des
représentations.

**ZLPRLoss** : ranking pairwise zero-bounded (Su et al., 2022), plus stable que la
BCE multiclass pour L=6283 classes.

## Configuration matérielle

> **Adaptation par rapport au papier original.**
>
> Le papier Torba et al. (ECIR 2025) utilise `num_decoder_layers=2` et `nhead=4`.
> L'attention self-attention est en O(L²) avec L=6 283, ce qui donne une matrice
> `(B, nhead, 6283, 6283)` — très coûteuse en VRAM et en calcul.
>
> Sur 2× RTX 3090 (24 GB/GPU), la config papier atteignait **213 h/epoch**
> (benchmark mesuré), ce qui rendrait 10 epochs impossibles dans le budget SLURM.
>
> **Config retenue** (`hlpd_3090.yaml`) :
>
> | Paramètre | Papier | Notre config | Impact |
> |---|---|---|---|
> | `num_decoder_layers` | 2 | **1** | ×2 sur le décodeur |
> | `nhead` | 2 | **2** | ×2 sur l'attention O(L²) |
> | `gradient_checkpointing` | — | **false** | VRAM suffisante |
> | `batch_size` | — | **16** (8/GPU) | Throughput maximal |
>
> L'architecture essentielle est préservée : biais hiérarchique, tied weights, ZLPR,
> même encodeur (CamemBERT), même espace de labels (L=6 283).
> La comparaison avec HYDRA reste valide dans ce cadre.

## Structure

```
02b_hlpd/
├── src/
│   ├── __init__.py          # Exports publics
│   ├── attention_bias.py    # build_M_hiera() — matrice binaire L×L
│   ├── losses.py            # ZLPRLoss
│   ├── models.py            # HLPD, HLPDDecoderLayer
│   └── trainer.py           # HLPDConfig, HLPDTrainer
├── configs/
│   └── hlpd_3090.yaml       # Config 2× RTX 3090
└── README.md
```

## Comparaison HYDRA vs HLPD

| | HYDRA | HLPD |
|---|---|---|
| Paradigme | Multi-tête discriminatif | Encoder-Decoder |
| Cohérence hiérarchique | Post-hoc (variante Nested) | Par construction (masque d'attention) |
| Perte | FocalLoss / CrossEntropy | ZLPR (pairwise ranking) |
| Complexité | O(L) | O(L²) — coûteux pour L=6 283 |
| Tied weights | Non | Oui |

## Références

- Torba, Laclau et al., *HLPD: Hierarchical Label Prediction Decoder*, ECIR 2025.
- Su et al., *ZLPR: A Novel Loss for Multi-label Classification*, 2022.
- Repo officiel : https://github.com/FatosTorba/HLPD (Apache 2.0)
