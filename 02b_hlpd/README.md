# 02b_hlpd -- Hierarchical Label Pre-populated Decoder

Implementation de l'architecture **HLPD** pour la classification hierarchique multi-label de produits sur le catalogue Rakuten, basee sur :

> Fatos Torba, Charlotte Laclau et al., *"HLPD: Hierarchical Label Pre-populated Decoder for Multi-label Text Classification"*, ECIR 2025.

> **Note** : Le code d'implementation sera ajoute prochainement par Mehdi Aithamma. Ce README decrit l'architecture cible.

---

## Idee generale

Les approches classiques de classification hierarchique traitent chaque niveau de la taxonomie de maniere sequentielle (d'abord L1, puis L2 conditionnellement). HLPD propose une approche **non-autoregressive** : un decodeur de type Transformer qui predit **tous les labels en un seul forward pass**, en exploitant la structure hierarchique de la taxonomie.

---

## Architecture

Le pipeline complet se decompose en 4 blocs :

### 1. Encodeur (Encoder)

Un modele pre-entraine (CamemBERT, XLM-RoBERTa ou mBERT) encode le texte d'entree (titre + description du produit) en une sequence de representations contextuelles.

```
Input : "Chaussures de sport Nike Air Max"
        |
  [Tokenizer : max_length=128]
        |
  [Encodeur pre-entraine]
        |
  H_enc = (h_1, h_2, ..., h_n)   dim: (B, seq_len, d_model)
```

### 2. Pre-population des labels (Label Pre-population)

Le decodeur est initialise avec des embeddings pre-calcules pour **chaque noeud de la taxonomie** : la racine, les 157 categories L1, et les 6 100+ sous-categories L2.

Chaque label est encode textuellement a partir de sa description hierarchique :
```
Template : "[Child] is a sub-topic of [Parent]"
Exemple : "Chaussures de sport is a sub-topic of Chaussures"
```

Ces descriptions sont passees dans le meme encodeur pre-entraine pour obtenir un embedding par label :
```
Labels : [Root, L1_1, L1_2, ..., L1_157, L2_1, L2_2, ..., L2_6100]
         |
  [Encodeur (poids partages)]
         |
  E_labels = (e_root, e_L1_1, ..., e_L2_6100)   dim: (1+157+6100, d_model)
```

### 3. Decodeur avec Hierarchical Self-Attention

Le decodeur est un Transformer standard, mais avec une modification cle : la **self-attention hierarchique**. Au lieu d'utiliser une self-attention classique, on ajoute un biais positionnel appris, derive des distances dans l'arbre taxonomique :

```
Attention classique :
  Attn(Q, K, V) = Softmax(QK^T / sqrt(d_k)) . V

Attention hierarchique (HLPD) :
  Attn_hiera(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + W . M_hiera) . V
```

Ou :
- **M_hiera** est la matrice de distances (shortest-path) dans l'arbre taxonomique, de dimension `(N_labels x N_labels)`
- **W** est un poids scalaire appris qui controle l'intensite du biais hierarchique

**Intuition** : les labels proches dans la taxonomie (ex: "Chaussures de sport" et "Chaussures") auront un biais d'attention plus fort, ce qui encourage le modele a faire des predictions coherentes avec la hierarchie.

### 4. Classification non-autoregressive

Apres les couches de decodeur, chaque embedding de label passe par une couche de classification lineaire pour produire un score (logit). Le modele predit **tous les labels en un seul forward pass** :

```
E_labels (pre-populated)
     |
  [Cross-attention avec H_enc]
     |
  [Self-attention hierarchique]   x N couches
     |
  [Classification lineaire]
     |
  Logits : (B, 1+157+6100)
```

---

## Fonction de perte : ZLPR

La **ZLPR Loss** (Zero-bounded Log-sum-exp Pairwise Rank-based) est une fonction de perte multi-label qui :

1. Separe les logits en positifs (labels corrects) et negatifs (labels incorrects)
2. Penalise les cas ou un label negatif a un score superieur a un label positif
3. Ajoute un terme de regularisation pour que les scores positifs soient > 0 et negatifs < 0

```
L_ZLPR = log(1 + sum_neg exp(s_neg)) + log(1 + sum_pos exp(-s_pos))
```

Une perte hierarchique (HierarchicalBCELoss) peut aussi etre utilisee, ponderant les erreurs selon leur distance dans la taxonomie.

---

## Taxonomie Rakuten

```
Root
 |
 +-- Chaussures (L1)
 |    +-- Chaussures de sport (L2)
 |    +-- Chaussures de ville (L2)
 |    +-- Bottes (L2)
 |    +-- ...
 |
 +-- Electronique (L1)
 |    +-- Smartphones (L2)
 |    +-- Tablettes (L2)
 |    +-- ...
 |
 +-- ... (157 categories L1 au total)
      +-- ... (6 100+ sous-categories L2 au total)
```

---

## Differences avec HYDRA

| Aspect | HYDRA | HLPD |
|--------|-------|------|
| Approche | Multi-tete (heads paralleles) | Encoder-Decoder |
| Decodage | Independant par niveau | Non-autoregressif (tous les labels en une passe) |
| Structure hierarchique | Via les tetes (Local/Global/Nested) | Via le biais d'attention hierarchique |
| Perte | CrossEntropy / FocalLoss | ZLPR (pairwise ranking) |
| Complexite | O(n_classes) | O(n_labels^2) pour l'attention |

---

## References

- **HLPD** -- Torba et al., *Hierarchical Label Pre-populated Decoder for Multi-label Text Classification*, ECIR 2025.
- **ZLPR** -- Su et al., *ZLPR: A Zero-bounded Log-sum-exp Pairwise Rank-based Loss*, 2022.
- **These** -- Fatos Torba, *Classification hierarchique multi-label de textes*, 2025.
