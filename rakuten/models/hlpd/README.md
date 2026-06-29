# HLPD : Décodeur Hiérarchique Non-Autorégressif

Implémentation de **HLPD** (Torba, Laclau et al., ECIR 2025) pour la classification
hiérarchique multi-label sur le dataset Rakuten (157 L1 × ~6 000 L2), avec un
encodeur **CamemBERT**.

## Idée clé

Plutôt que de filtrer les prédictions a posteriori, HLPD encode la hiérarchie
**directement dans le mécanisme d'attention**. Le décodeur est non-autorégressif :
tous les labels sont prédits en une seule passe. La cohérence hiérarchique est ainsi
apprise par construction (HCR ≈ 98 % sans contrainte à l'inférence).

## Architecture

```
Texte ───────────────→ [CamemBERT] ──→ H_text (B, 128, 768)

Descriptions labels ──→ [CamemBERT CLS] ──→ label embeddings (L, 768)   [calculé une fois, mis en cache]
                                  │
                          [LabelPooler]            projection non-linéaire par label
                                  │
              [Hierarchical Self-Attention]        scores += alpha · M_hiera + beta   (sans dim. batch)
                                  │
                          expand → (B, L, 768)
                                  │
              [Cross-Attention (Q=labels, K=V=texte) + FFN] × num_decoder_layers
                                  │
                          [LabelWisePooler]         un logit par label
                                  │
                              (B, L)
```

**Self-attention hiérarchique** : biais additif `alpha · M_hiera + beta` ajouté aux
scores QKᵀ avant softmax. `M_hiera` encode la distance dans l'arbre (plus court chemin
ROOT → L1 → L2). Le biais est paramétré par **deux scalaires** (`alpha`, `beta`), ce
qui rend le coût indépendant du nombre de labels.

**Embeddings de labels mis en cache** : chaque label est décrit par un court texte
(L1 = son nom ; L2 = « {sous-type} est un sous-type de {parent}. »). Ces descriptions
sont encodées une seule fois via le token `[CLS]` de CamemBERT, puis figées. On évite
ainsi de ré-encoder L descriptions à chaque pas.

**Self-attention sans dimension batch** : les labels sont partagés entre tous les
exemples, donc l'auto-attention tourne en `O(L²)` (et non `O(B · L²)`).

**ZLPRLoss** : ranking pairwise zero-bounded (Su et al., 2022), adapté au
déséquilibre extrême (2 positifs sur ~6 000 labels).

## Configuration

Les valeurs canoniques sont dans `configs/hlpd/default.yaml` :

| Paramètre | Valeur |
|---|---|
| `num_decoder_layers` | 3 |
| `nhead` | 8 |
| `ffn_dim` | 3072 |
| `max_length` | 128 |
| `label_max_len` | 64 |
| `lr` | 2e-5 |
| `batch_size` × `grad_accum` | 16 × 2 (effectif 32) |
| `precision` | bfloat16 |

## Résultats (test, 1.17M produits)

| Métrique | Valeur |
|---|---|
| F1 L1 (macro) | 0.807 |
| F1 L2 (macro) | 0.615 |
| HCR | 0.978 |
| Hierarchical accuracy | 0.820 |

## Structure

```
rakuten/models/hlpd/          # code (package installable)
├── __init__.py               # exports publics
├── attention_bias.py         # build_M_hiera() — biais de distance L×L
├── losses.py                 # ZLPRLoss
├── models.py                 # HLPD + briques (LabelPooler, self/cross-attn, LabelWisePooler)
├── trainer.py                # HLPDConfig, HLPDTrainer
└── README.md
configs/hlpd/default.yaml     # configuration (hors package)
```

Entraînement : `python scripts/run_hlpd.py --split-dir <dataset> --config configs/hlpd/default.yaml`

## Comparaison HYDRA vs HLPD

| | HYDRA | HLPD |
|---|---|---|
| Paradigme | Multi-tête discriminatif | Encoder-Decoder non-autorégressif |
| Cohérence hiérarchique | Post-hoc (variante Nested) | Par construction (biais d'attention) |
| Perte | FocalLoss / CrossEntropy | ZLPR (pairwise ranking) |
| Self-attention labels | — | O(L²), biais hiérarchique scalaire |
| Embeddings labels | — | [CLS] CamemBERT mis en cache |

## Références

- Torba, Laclau et al., *Hierarchical Label Pre-populated Decoder*, ECIR 2025.
- Su et al., *ZLPR: A Novel Loss for Multi-label Classification*, 2022.
- Dépôt officiel : https://github.com/FatosTorba/HLPD (Apache 2.0)
