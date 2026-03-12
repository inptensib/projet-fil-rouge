# Classifier 8 Millions de Produits : Architecture Hierarchique sur Cluster GPU

Comment classer **8 millions de produits** dans **157 categories** et **6 100+ sous-categories** ? La classification "a plat" atteint vite ses limites. Ce post retrace notre parcours : des baselines statistiques jusqu'a l'implementation de **HYDRA** et **HLPD** sur un cluster de 8 GPUs.

---

## Phase 1 : Baselines "Plates"

Ignorer la hierarchie et predire directement les 6 100 classes.

**TF-IDF + Linear SVM** sur le texte (Titre + Description) :

| Approche | F1 Macro | F1 Weighted |
|----------|:--------:|:-----------:|
| TF-IDF + Linear SVM | 0.679 | 0.774 |

Rapide a entrainer, mais incapable de saisir les nuances semantiques. Il nous fallait une architecture qui *modelise* la hierarchie.

---

## Phase 2 : HYDRA -- Preuve de Concept (50K)

Nous avons implemente **HYDRA** (Karl et al., EMNLP 2025) : un encodeur partage (CamemBERT) avec des tetes de classification paralleles.

- **Local** : tetes L1 et L2 independantes
- **Global** : concatene les logits L1 avec l'embedding pour conditionner L2
- **Nested** : masque les logits L2 selon les probabilites L1 (coherence hierarchique)

Resultats sur un echantillon de 50K produits (2x P100) :

| Variante | F1 L1 Macro | F1 L1 Weighted | F1 L2 Macro | F1 L2 Weighted |
|----------|:-----------:|:--------------:|:-----------:|:--------------:|
| Local | 0.578 | 0.739 | 0.106 | 0.166 |
| Global | 0.530 | 0.724 | 0.081 | 0.125 |
| Nested | 0.416 | 0.685 | 0.020 | 0.035 |

> Sur 50K echantillons pour 6 100 classes L2, les tetes Global et Nested manquent cruellement d'exemples par sous-classe pour generaliser.

---

## Etat de l'Art : Classification Hierarchique de Texte (HTC)

### Methodes discriminatives

- **Locales** : un classifieur par niveau (notre HYDRA Local). Simple mais risque d'incoherences hierarchiques.
- **Globales** : modelisent la structure taxonomique explicitement :
  - **GCN** sur l'arbre de labels
  - **HiAGM** : attention entre encodeur textuel et hierarchie
  - **HGCLR / HJCL** : apprentissage contrastif hierarchique

### Methodes generatives

Reformulent la HTC en generation de sequence (chemin dans la taxonomie) :
- **BFS** : labels generes niveau par niveau (PAAM-HiA-T5)
- **DFS** : chemin complet racine-feuille par branche (Seq2Tree)

### Metriques cles

| Metrique | Mesure |
|----------|--------|
| **Micro-F1** | Performance globale (favorise classes frequentes) |
| **Macro-F1** | Moyenne non-ponderee (penalise erreurs sur classes rares) |
| **Constrained F1** | Correct ssi *tous* les niveaux sont corrects |
| **HCR** | Taux de coherence hierarchique (L2 enfant de L1 ?) |

### Papiers fondateurs

**HYDRA** (EMNLP 2025) : tetes multiples (Local + Global + Nested) partageant un encodeur pre-entraine. Montre que la combinaison capture des signaux complementaires.

**HLPD** (ECIR 2025, Torba & Laclau) : decodeur non-autoregressif avec self-attention hierarchique. Predit tous les labels en une passe, avec coherence *par construction*. Utilise la **ZLPR loss** (pairwise ranking), plus stable que la BCE pour les espaces de labels larges. Co-ecrit par notre encadrante Charlotte Laclau.

---

## Phase 3 : Passage a l'Echelle (8M)

### Optimisations cles

1. **Titre + Description** : l'ajout de la description complete a donne un boost significatif a l'encodeur.
2. **Focal Loss** (gamma=2.0) : reduit le poids des exemples bien classifies, force le modele sur les classes rares.

### Resultats (HYDRA Local, 8M produits)

| Epoque | Loss | F1 L1 Macro | F1 L2 Macro | F1 L1 Weighted | F1 L2 Weighted |
|:------:|:----:|:-----------:|:-----------:|:--------------:|:--------------:|
| 1 | 6.22 | 0.865 | 0.443 | 0.898 | 0.702 |
| 2 | 2.70 | 0.912 | 0.649 | 0.927 | 0.816 |
| 3 | 1.81 | **0.925** | **0.714** | **0.939** | **0.848** |

---

## Phase 4 : Infrastructure DDP

Pour entrainer sur 8M textes sans exploser la RAM : **8 Tesla P100** via SLURM.

1. **DDP** (Distributed Data Parallel) avec synchronisation NCCL
2. **Sharding** : dataset decoupe en 8 fragments sur disque, chaque GPU charge sa part
3. **RoCE** (RDMA over Converged Ethernet) pour minimiser la latence inter-noeuds

---

## Phase 5 : Error Tracking et Audit

### Suivi des "Hard Examples" (regle 80/10/10)

- **Train (80%)** : capture des produits a loss elevee a chaque batch → detection du label noise
- **Validation (10%)** : erreurs de generalisation, categories trop proches (ex: Smartphone ↔ Tablette)
- **Test (10%)** : sanctuaire blinde, aucune analyse detaillee pour garantir l'impartialite du score final

### Tracking inter-epoques

Les produits systematiquement mal classes a *chaque epoque* revelent du bruit d'etiquetage. Cette persistance fournit une liste de nettoyage precise pour les 8M de donnees.

---

## Phase 6 : MLflow

- **Tracking temps reel** : loss et learning rate logues toutes les 50 steps
- **Hard Examples** : rapports d'erreurs en tableaux interactifs (`mlflow.log_table`)
- **Artefacts** : modeles PyTorch et graphiques sauvegardes par run
- **Comparaison** : historique protege, comparaison Local vs Global vs Nested en un clic
- Serveur SQLite local sur le cluster (donnees securisees)

---

## Phase 7 : Pilotage "Dual-Speed" et Flexibilite Materielle

### Scheduler a deux vitesses

- **Sniper (~15 min / 400 steps)** : surveille la training loss, `ReduceLROnPlateau` intervient (factor=0.85) si stagnation
- **Audit (~1h / 1600 steps)** : calcul des F1 sur validation + extraction des hard examples

### Agnosticisme materiel

```bash
./scripts/launch_distributed.sh local CamemBERT 4 p100      # 8 GPUs via DDP (reseau)
./scripts/launch_distributed.sh local CamemBERT 1 rtx3090   # 2 GPUs via DataParallel (local)
```

### Ponderation asymetrique

Le L1 converge vite (F1 > 90%). Ponderation **1:2** en faveur du L2 (`level_weight_l1=0.5`) pour forcer l'apprentissage des 6 100 sous-categories.

---

## Phase 8 : HLPD (ECIR 2025)

Approche complementaire a HYDRA : un **decodeur non-autoregressif** avec masque d'attention hierarchique.

| | HYDRA | HLPD |
|---|---|---|
| Paradigme | Multi-tete discriminatif | Encoder-Decoder |
| Coherence | Post-hoc (Nested) | Par construction (masque) |
| Perte | Focal / CE | ZLPR (pairwise ranking) |
| Inference | Parallele | Parallele |

Le masque encode les distances parent-enfant dans l'arbre : un noeud L2 ne "voit" que son parent L1 et ses siblings. La **ZLPR loss** compare directement les scores positifs/negatifs, plus stable que la BCE avec 6 100+ classes.

> Co-ecrit par notre encadrante Charlotte Laclau, ce qui nous a donne un acces direct aux intuitions architecturales.

---

## Phase 9 : Restructuration Industrielle

Passage d'un code experimental a une architecture modulaire :

```
rakuten-classification/
├── 01_baselines/    # Methodes de reference
├── 02_hydra/        # HYDRA (Local, Global, Nested)
├── 02b_hlpd/        # HLPD (ECIR 2025)
├── 03_production/   # DDP + MLOps
├── shared/          # data_loader, metrics, utils
└── Dockerfile       # Reproductibilite
```

Principes : separation recherche/production, mutualisation via `shared/`, reproductibilite via Docker et `requirements.txt`.

---

## Validation : Comparaison avec le Code Officiel HYDRA

Verification ligne par ligne contre le repo officiel (FKarl). Correspondances exactes pour les 3 variantes.

### Divergences intentionnelles

| Composant | Officiel | Notre version | Justification |
|-----------|----------|---------------|---------------|
| Perte | BCE | **Focal Loss** (gamma=2.0) | Dataset desequilibre (6 100 classes) |
| Scheduler | Linear Warmup+Decay | **ReduceLROnPlateau** | Adaptatif aux zigzags sur 8M |
| Tache | Multi-label (Sigmoide) | **Single-label** (Softmax) | Un produit = une categorie |
| Audit | Aucun | **Error tracking** | Detection du label noise |
| Memoire | Tout en RAM | **Sharding disque** | Necessaire pour 8M produits |

Le **gradient clipping** (`max_norm=1.0`) du code officiel a ete integre apres notre audit initial.

---

## References

1. Karl et al. (2025). **HYDRA: Hierarchical Multi-label Classification using Attention-based Architecture.** EMNLP 2025.
2. Torba, Laclau et al. (2025). **HLPD: Hierarchical Label Pre-populated Decoder.** ECIR 2025.
3. Torba, F. (2025). **These : Classification hierarchique multi-label de textes.**
4. Lin et al. (2017). **Focal Loss for Dense Object Detection.** ICCV 2017.
5. Su et al. (2022). **ZLPR: Zero-bounded Log-sum-exp Pairwise Ranking Loss.**
