# Classifier 8 Millions de Produits : L'Evolution d'une Architecture Hierarchique sur Cluster GPU

La classification de texte est un probleme classique en NLP. Mais que se passe-t-il lorsque vous devez classer **8 millions de fiches produits** dans une taxonomie complexe comprenant **157 categories principales (Niveau 1)** et plus de **6100 sous-categories (Niveau 2)** ?

Traiter cela comme un simple probleme de classification "a plat" (flat classification) mene inevitablement a un mur de performances. Ce post retrace notre voyage architectural : des baselines statistiques basiques jusqu'a l'implementation poussee du papier de recherche **HYDRA**, puis de **HLPD**, en detaillant notre demarche empirique face aux limites materielles et mathematiques rencontrees en cours de route.

---

## Phase 1 : L'Echec Previsible des Approches "Plates"

Avant de courir, il faut savoir marcher. Notre objectif premier etait d'etablir des "baselines" solides. L'idee de ces approches est d'ignorer la hierarchie en arbre des categories et de predire directement le label final, comme si les 6100 classes etaient parfaitement independantes.

**L'ere statistique (TF-IDF) :** Nous avons commence par extraire les caracteristiques textuelles (Titre + Description) via un `TfidfVectorizer` couple a un classifieur lineaire `Linear SVM`.

| Approche | Input | F1 Macro | F1 Pondere (Weighted) |
| :--- | :--- | :--- | :--- |
| **TF-IDF + Linear SVM** | Titre + Description | 0.6788 | 0.7735 |

Bien qu'extremement rapide a entrainer, l'incapacite de TF-IDF a saisir les nuances semantiques complexes a produit des scores "plafonds" impossibles a depasser. Il nous fallait une architecture qui *modelise* la hierarchie et comprenne le contexte.

---

## Phase 2 : La Revelation HYDRA (Preuve de Concept sur 50K)

Plutot que d'aplatir le probleme, nous avons decide d'implementer l'architecture decrite dans le papier de recherche **HYDRA (Hierarchical Classification with Deep Neural Networks)**.

HYDRA partage un unique encodeur surpuissant (`CamemBERT`), mais utilise des tetes de classification paralleles :
*   **HYDRA Local :** Une tete pour le Niveau 1, une autre pour le Niveau 2.
*   **HYDRA Global & Nested :** Des tetes supplementaires qui concatenent les logits pour forcer le modele a utiliser sa certitude sur le Niveau 1 pour corriger ses doutes sur le Niveau 2.

Pour tester la viabilite de ce code de maniere agile, nous avons cree un sous-projet `hydra/50K/`. Nous avons echantillonne aleatoirement **50 000 produits** et lance l'entrainement sur 2 modestes GPUs Tesla P100.

| Variante HYDRA (sur 50K) | F1 L1 (Macro) | F1 L1 (Weighted) | F1 L2 (Macro) | F1 L2 (Weighted) | Temps d'entrainement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Heads** | 0.5780 | 0.7388 | 0.1064 | 0.1655 | ~1.8 h |
| **Local + Global** | 0.5295 | 0.7235 | 0.0806 | 0.1251 | ~1.8 h |
| **Local + Nested** | 0.4157 | 0.6848 | 0.0195 | 0.0353 | ~1.8 h |

*Note: Sur un sous-echantillon aussi petit (50K) pour 6100 classes L2, les tetes Global et Nested peinent a generaliser a cause du manque cruel d'exemples par sous-classe.*

---

## Etat de l'Art : Classification Hierarchique de Texte

Avant de plonger plus loin dans nos experimentations, il est essentiel de situer notre travail dans le paysage de la recherche. La **Hierarchical Text Classification (HTC)** est un sous-domaine actif du NLP, regulierement explore dans les grandes conferences telles que **EMNLP**, **ECIR**, **SIGIR**, **WWW** et **AAAI**. Les approches se divisent en deux grandes familles.

### Methodes Discriminatives

Les methodes discriminatives traitent la HTC comme un probleme de prediction directe des labels a chaque niveau de la hierarchie.

**Approches Locales :** On entraine un classifieur distinct par noeud (*per-node*) ou par niveau (*per-level*) de la taxonomie. C'est conceptuellement simple et c'est exactement ce que fait notre HYDRA Local. L'inconvenient majeur : les classifieurs locaux n'echangent pas d'information entre eux, ce qui peut mener a des predictions hierarchiquement inconsistantes (par exemple, predire "Accessoire Telephone" au Niveau 2 alors que le Niveau 1 predit "Livres").

**Approches Globales :** Pour resoudre ce probleme d'incoherence, les approches globales modelisent explicitement la structure de la taxonomie. On retrouve ici :
*   **GCN (Graph Convolutional Networks)** appliquees a l'arbre de labels, ou chaque noeud "herite" d'information de ses voisins dans le graphe.
*   **HiAGM** (Hierarchy-Aware Global Model) : un mecanisme d'attention entre l'encodeur textuel et la structure hierarchique, permettant au modele de "consulter" la taxonomie pendant la prediction.
*   **Apprentissage contrastif hierarchique** : des travaux comme **HGCLR** et **HJCL** forcent les representations des classes parent-enfant a etre proches dans l'espace latent, tout en eloignant les classes non reliees.

### Methodes Generatives

Plus recentes, les methodes generatives reformulent la HTC comme un probleme de generation de sequence, ou le modele "ecrit" le chemin dans la taxonomie.

*   **Approche BFS (Breadth-First)** : Le modele genere les labels niveau par niveau, de la racine vers les feuilles. C'est l'approche de **PAAM-HiA-T5**, qui utilise un decodeur T5 avec un mecanisme de prompt adaptatif.
*   **Approche DFS (Depth-First)** : Le modele genere un chemin complet de la racine a la feuille pour chaque branche pertinente, comme dans **Seq2Tree**.

### Baselines Cles de la Litterature

Le tableau suivant resume les travaux de reference qui jalonnent ce domaine, chacun apportant une innovation specifique :

| Modele | Innovation Principale |
| :--- | :--- |
| **SGM** | Sequence Generation Model pour la classification multi-label |
| **HiAGM** | Attention hierarchique sur le graphe de labels |
| **HTCInfoMax** | Maximisation de l'information mutuelle entre niveaux |
| **HiMatch** | Matching semantique entre texte et labels dans la hierarchie |
| **HGCLR** | Apprentissage contrastif hierarchique avec augmentation de graphe |
| **PAAM-HiA** | Prompt-Aware Attention pour la generation hierarchique |
| **HPT** | Hierarchical Prompt Tuning avec des soft prompts par niveau |
| **HBGL** | Hierarchical BERT with Graph-based Learning |
| **K-HTC** | Injection de connaissances externes dans la taxonomie |
| **HJCL** | Joint Contrastive Learning hierarchique |

### Metriques d'Evaluation

L'evaluation en HTC ne se limite pas au F1 classique. Quatre metriques cles emergent de la litterature :
*   **Micro-F1** : Performance globale, ponderee par le nombre d'exemples. Favorise les classes frequentes.
*   **Macro-F1** : Moyenne non-ponderee sur toutes les classes. Penalise severement les erreurs sur les classes rares.
*   **Constrained F1** : Ne compte une prediction comme correcte que si *tous* les niveaux hierarchiques sont corrects simultanement.
*   **HCR (Hierarchical Consistency Rate)** : Mesure la proportion de predictions ou le label L2 predit est bien un enfant du label L1 predit. Un HCR < 100% signale des incoherences structurelles.

### Le Papier HYDRA (EMNLP 2025, Karl et al.)

C'est le papier fondateur de notre architecture. **HYDRA** propose une approche discriminative a tetes multiples partageant un meme encodeur pre-entraine. Son apport principal est de montrer qu'en combinant trois strategies de classification (Local, Global, Nested), on capture des signaux complementaires :
*   La **tete Local** donne des predictions independantes par niveau.
*   La **tete Global** concatene l'embedding avec les logits du Niveau 1 pour conditionner la prediction du Niveau 2.
*   La **tete Nested** utilise les probabilites du Niveau 1 comme masque attentionnel sur les sous-classes du Niveau 2.

Le papier demontre sur plusieurs benchmarks que cette combinaison surpasse les approches a tete unique, tout en restant simple a implementer et a entrainer.

### Le Papier HLPD (ECIR 2025, Torba, Laclau et al.)

**HLPD (Hierarchical Label Prediction with Decoding)** prend le contrepied de l'approche generative auto-regressive classique. Au lieu de generer les labels mot par mot (comme un modele de langue), HLPD utilise un **decodeur non-autoregressif** dote d'un **masque d'auto-attention hierarchique**. Ce masque force le decodeur a respecter la structure parent-enfant de la taxonomie pendant l'inference, sans avoir besoin de generer sequentiellement.

Les contributions cles :
*   Un mecanisme de **hierarchical self-attention** qui encode directement les contraintes structurelles dans les poids d'attention du decodeur.
*   L'utilisation de la **ZLPR loss** (Zero-bounded Log-sum-exp Pairwise Ranking loss), plus stable que la BCE pour les problemes multi-labels avec un grand nombre de classes negatives.
*   Des resultats competitifs avec les methodes auto-regressives, mais avec une **inference significativement plus rapide** grace au decodage parallele.

Ce papier est co-ecrit par **Charlotte Laclau**, notre encadrante de projet, ce qui nous a donne un acces privilegie aux intuitions derriere l'architecture.

### These de Fatos Torba sur la HMTC

Mentionnons egalement la **these de doctorat de Fatos Torba** (2025), qui propose un cadre unifie pour la **Hierarchical Multi-label Text Classification** appliquee a l'analyse d'appels d'offres. Cette these formalise les problemes de coherence hierarchique et explore en profondeur les interactions entre les niveaux de la taxonomie, fournissant un socle theorique solide pour des travaux comme HLPD.

---

## Phase 3 : L'Ascension vers les 8 Millions et la Lutte contre le Plafond de Verre

Nous avons bascule dans le dossier `hydra/8M/` pour attaquer l'integralite du dataset. Cette phase n'a pas ete un long fleuve tranquille, mais une suite de diagnostics et de corrections.

### 1. De "Titre seul" a "Titre + Description"
Initialement, nous ne passions que le **Titre** de l'article. Les scores plafonnaient. L'apport de la description complete a donne un net coup de boost a l'encodeur CamemBERT en fournissant un contexte semantique riche.

### 2. Le remede de la Focal Loss
Avec 6000 classes, les micro-gradients des classes negatives "ecrasent" le signal d'apprentissage. Nous avons remplace la Cross-Entropy par une **Focal Loss** ($\gamma=2.0$). Cette fonction reduit le poids des exemples deja bien appris pour forcer le modele a se concentrer sur les classes rares et les cas difficiles.

### Les Resultats sur 8 Millions (HYDRA Local)

| Epoque | Loss Globale | Val F1 L1 (Macro) | Val F1 L2 (Macro) | Val F1 L1 (Weighted) | Val F1 L2 (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epoch 1** | 6.22 | 0.865 | 0.443 | 0.898 | 0.702 |
| **Epoch 2** | 2.70 | 0.912 | 0.649 | 0.927 | 0.816 |
| **Epoch 3** | 1.81 | **0.925** | **0.714** | **0.939** | **0.848** |

---

## Phase 4 : Le Defi MLOps (DDP et Sharding)

Pour entrainer sur 8 millions de textes sans exploser la RAM, nous avons deploye une infrastructure sur **8 Tesla P100** via :
1.  **Distributed Data Parallel (DDP)** avec synchronisation NFS.
2.  **Sharding avance** : Le dataset est decoupe en 8 fragments sur disque, chaque GPU ne chargeant que sa part.
3.  **Reseau RoCE** : Utilisation de l'Infiniband pour minimiser la latence entre les noeuds.

---

## Phase 5 : Refactorisation et Tracking des "Hard Examples"

Pour transformer ce projet de recherche en plateforme industrielle, nous avons opere deux changements majeurs :

### 1. Restructuration Modulaire (`src/`)
Le script monolithique a ete eclate en un module Python propre (`data.py`, `models.py`, `trainer.py`). Cela permet une maintenance aisee et l'ajout de nouvelles tetes de classification sans risque de regression.

### 2. Le Systeme d'Audit Intelligent (Tracking)
Nous avons integre un monitoring de pointe pour identifier les produits qui "resistent" au modele. Nous suivons la regle d'or **80/10/10** :

*   **Train Set (80%) -- Detection du "Label Noise"** : A chaque batch d'entrainement, le Rank 0 capture les produits ayant la **Loss la plus elevee**. Si un produit est systematiquement mal classe par le modele alors qu'il fait partie de ses donnees d'apprentissage, c'est la preuve d'un mauvais etiquetage dans le dataset Rakuten.
*   **Validation Set (10%) -- Audit Semantique** : A la fin de chaque epoque, on isole les erreurs de generalisation (Donnees Dev). Cela nous permet de voir quelles categories sont trop proches (ex: *Smartphone* -> *Tablette*).
*   **Test Set (10%) -- Sanctuaire Blinde** : Ce jeu de donnees reste une "boite noire" totale. Aucune analyse d'erreur detaillee n'y est faite pour garantir l'impartialite absolue du score final.

### Meta-Insights : La Persistance des Erreurs
Le point fort de notre approche est le **Tracking Inter-Epoques**. Grace a un script de synthese final (`global_error_insight.py`), nous identifions les produits qui apparaissent comme des erreurs a **chaque epoque**. Cette persistance mathematique nous donne une liste de nettoyage ultra-precise pour affiner nos 8 millions de donnees.

---

## Validation de l'Architecture : Comparaison avec le Code Officiel

Une etape cruciale de notre demarche a ete de confronter notre implementation a celle du **repository officiel de HYDRA** (FKarl). Cette validation nous a permis de confirmer notre rigueur architecturale tout en assumant pleinement nos divergences intentionnelles.

### Correspondances Exactes

Apres analyse ligne par ligne du code officiel, nous avons confirme que nos trois variantes architecturales (Local, Global, Nested) reproduisent fidlement les mecanismes decrits dans le papier :
*   La **tete Local** utilise deux classifieurs independants sur l'embedding CLS, exactement comme dans le repo officiel.
*   La **tete Global** concatene les logits L1 avec l'embedding projete avant de predire L2, conformement a l'implementation de reference.
*   La **tete Nested** masque les logits L2 en fonction des probabilites L1 via la matrice parent-enfant, identique au code source.

### Divergences Intentionnelles

| Composant | Implementation Officielle | Notre Version | Justification |
| :--- | :--- | :--- | :--- |
| **Fonction de perte** | BCE (Binary Cross-Entropy) | **Focal Loss** ($\gamma=2.0$) | Dataset fortement desequilibre (6100 classes) |
| **Scheduler** | Linear Warmup + Decay | **ReduceLROnPlateau** | Adaptatif aux zigzags de la Loss sur 8M |
| **Nature de la tache** | Multi-labels (Sigmoide) | **Multi-classes Single-label** (Softmax) | Chaque produit n'a qu'une seule categorie |
| **Audit des erreurs** | Aucun | **Tracking persistant** | Identification du label noise a grande echelle |
| **Gestion memoire** | Tout en RAM | **Sharding sur disque** | Necessaire pour 8M de produits |

### Correction Ajoutee : Gradient Clipping

Lors de notre audit du code officiel, nous avons identifie l'utilisation d'un **gradient clipping** (`max_norm=1.0`) que nous n'avions pas implemente initialement. Cette omission pouvait causer des instabilites de la Loss, notamment lors des premieres epoques ou les gradients sont grands. Le gradient clipping a ete ajoute a notre `trainer.py`, alignant notre pipeline sur les bonnes pratiques du papier.

### Schema d'architecture adapte

<pre>
                                     +-------------------------+
                                     |  Titre + Description    |
                                     +-----------+-------------+
                                                 |
                                                 v
                                     +-------------------------+
                                     |   CamemBERT / XLM-R    |
                                     +-----------+-------------+
                                                 |
                                                 v
                                     +-------------------------+
                                     |  Embedding Projete      |
                                     +-----+-----------+-------+
                                           |           |
                     +---------------------v-+       +-v---------------------+
                     |    Tete Niveau 1      |       |    Tete Niveau 2      |
                     +-------------+---------+       +---------+-------------+
                                   |                           |
                                   v                           v
                     +---------------------+         +---------------------+
                     |      Softmax        |         |      Softmax        |
                     +-------------+---------+       +---------+-------------+
                                   |                           |
                                   v                           v
                     +---------------------+         +---------------------+
                     | Dist. Probabilites  |         | Dist. Probabilites  |
                     |      L1 (157)       |         |    L2 (6100+)       |
                     +-------------+---------+       +---------+-------------+
                                   |                           |
                                   v                           v
                     +---------------------+         +---------------------+
                     |  Focal Loss L1      |         |  Focal Loss L2      |
                     +-------------+---------+       +---------+-------------+
                                   |                           |
                                   +------------+--------------+
                                                v
                                     +-------------------------+
                                     | Loss Totale Combinee    |
                                     +-------------------------+
</pre>

---

## Phase 6 : Integration MLOps avec MLflow

Pour une gestion professionnelle de nos experiences a grande echelle, nous avons integre **MLflow** a notre pipeline d'entrainement. Cette etape marque notre passage d'un script de recherche a une veritable plateforme industrielle.

### Fonctionnalites implementees :
*   **Tracking en temps reel** : La `Loss` d'entrainement et le `Learning Rate` sont logues toutes les 50 etapes, permettant de visualiser la convergence en direct sans attendre la fin de l'epoque.
*   **Visualisation des Hard Examples** : Grace a `mlflow.log_table`, nos rapports d'erreurs (Train et Validation) sont desormais consultables sous forme de tableaux interactifs directement dans l'interface MLflow.
*   **Gestion des Artefacts** : Les modeles finaux (`PyTorch`) et les graphiques de performance sont automatiquement sauvegardes et rattaches a chaque run.
*   **Historisation Securisee** : Chaque nouveau lancement cree une nouvelle `Run` dans MLflow. L'historique est protege : aucun run n'ecrase les statistiques d'un run precedent, ce qui permet de comparer nos experiences (ex: *Local* vs *Global*) en un clic.
*   **Confidentialite Totale** : Le serveur MLflow (base SQLite) tourne localement sur le cluster de l'ecole, garantissant que nos donnees de produits restent securisees au sein de l'infrastructure de l'etablissement.


## Phase 7 : Flexibilite Materielle et Pilotage "Dual-Speed"

Face a l'immensite de 8 millions de produits, nous avons transforme notre pipeline pour le rendre agnostique au materiel et ultra-reactif aux variations de la Loss.

### 1. Strategie de Pilotage "Dual-Speed" (Sniper LR)
L'entrainement sur cluster est sujet a des "zigzags" de la Loss lorsque le Learning Rate devient trop grand pour la finesse du terrain. Pour une convergence acceleree, nous avons implemente un systeme de surveillance a deux vitesses :
*   **Le "Sniper" (Toutes les 15 min / ~400 steps)** : Le code surveille la **Training Loss**. Si elle stagne, le scheduler `ReduceLROnPlateau` intervient immediatement en divisant le LR par petites touches (factor=0.85). Cela tue les zigzags sans attendre une validation longue.
*   **L'Audit Qualite (Toutes les heures / ~1600 steps)** : Le modele suspend l'entrainement pour calculer les scores F1 reels sur le set de Validation et extraire les nouveaux "Hard Examples".

### 2. Agnosticisme Materiel : Parallelisme Hybride
Nous avons rendu notre lanceur (`lancer.sh`) capable de switcher entre deux architectures radicalement differentes selon la disponibilite des ressources :

*   **Mode Distribue (Tesla P100)** : Distribution sur 8 GPUs via **DDP (Distributed Data Parallel)**. Chaque GPU gere un processus independant, synchronisant les gradients via le reseau RoCE.
*   **Mode Dense (RTX 3090)** : Utilisation du `node40` comme une machine unique massive. Pour optimiser la RAM et la simplicite, nous utilisons **un seul processus Python pour 2 GPUs** via **DataParallel**. Cela permet de traiter des batchs massifs de 256 produits sans la latence du reseau inter-noeuds.

```bash
# Lancement sur 4 noeuds P100 (Force brute reseau)
./lancer.sh local CamemBERT 4 p100

# Lancement sur 1 noeud RTX 3090 (Velocite locale)
./lancer.sh local CamemBERT 1 rtx3090
```

### 3. Focus sur le Niveau 2 (Ponderation asymetrique)
Lors des premieres epoques, le modele assimile tres rapidement les 157 classes du **Niveau 1** (F1 > 90%). Cependant, la veritable complexite reside dans les 6126 sous-categories du **Niveau 2**. Pour l'aider a franchir ce mur, nous avons bascule d'une ponderation "Fair Play" (1:1) a une **ponderation asymetrique (1:2)**.
En fixant `level_weight_l1=0.5`, la Loss globale accorde mathematiquement deux fois plus d'importance aux erreurs commises sur le Niveau 2, forcant le reseau a affiner sa comprehension des details subtils sans pour autant desapprendre l'architecture principale.

---

## Phase 8 : Implementation de HLPD (ECIR 2025)

Apres avoir pousse HYDRA dans ses retranchements, nous avons decide d'explorer une approche fondamentalement differente : **HLPD** (Hierarchical Label Prediction with Decoding), presente a ECIR 2025 par Torba, Laclau et al.

### Pourquoi HLPD ?

Ce choix n'est pas anodin. HLPD est co-ecrit par notre encadrante **Charlotte Laclau**, ce qui nous a donne un acces direct aux motivations et aux subtilites de l'architecture. Lors de nos echanges, elle nous a oriente vers cette methode comme complement naturel a HYDRA : la ou HYDRA empile des tetes de classification independantes, HLPD integre la contrainte hierarchique directement dans le mecanisme d'attention du decodeur.

### Architecture HLPD

L'architecture se decompose en trois blocs :

1.  **Encodeur** : Un modele pre-entraine (CamemBERT dans notre cas) encode le texte en une sequence de representations contextuelles.
2.  **Decodeur non-autoregressif** : Contrairement aux approches generatives classiques (GPT-like), HLPD predit *tous les labels simultanement* en une seule passe du decodeur. Chaque position du decodeur correspond a un noeud potentiel dans la taxonomie.
3.  **Masque d'auto-attention hierarchique** : C'est le coeur de l'innovation. Un masque binaire encode la structure parent-enfant de la taxonomie dans les couches d'attention du decodeur. Concretement, un noeud L2 ne peut "voir" (via l'attention) que son parent L1 et ses siblings, jamais les noeuds des autres branches. Cela force le decodeur a respecter la coherence hierarchique par construction.

### Differences Cles avec HYDRA

| Aspect | HYDRA | HLPD |
| :--- | :--- | :--- |
| **Paradigme** | Discriminatif a tetes multiples | Decodeur non-autoregressif |
| **Coherence hierarchique** | Post-hoc (via Nested head) | Par construction (masque d'attention) |
| **Fonction de perte** | Focal Loss / BCE | **ZLPR Loss** |
| **Inference** | Parallele (une passe) | Parallele (une passe) |
| **Complexite** | Simple a implementer | Plus complexe (decodeur Transformer) |

La **ZLPR Loss** (Zero-bounded Log-sum-exp Pairwise Ranking) merite une mention particuliere. Contrairement a la BCE qui traite chaque classe independamment, la ZLPR compare directement les scores des classes positives et negatives via un ranking pairwise. Elle est plus stable numeriquement pour les problemes avec un tres grand nombre de classes negatives (notre cas avec 6100+ classes L2).

---

## Phase 9 : Restructuration et Industrialisation

A ce stade du projet, nous avions accumule un volume considerable de code experimental reparti dans plusieurs repertoires ad hoc. Pour preparer la mise en production et faciliter la collaboration, nous avons opere une **restructuration complete** du projet dans un nouveau repertoire : `rakuten-classification/`.

### Nouvelle Arborescence

```
rakuten-classification/
|-- 01_baselines/          # TF-IDF + SVM, FastText, modeles de reference
|-- 02_hydra/              # Implementation complete de HYDRA (Local, Global, Nested)
|-- 02b_hlpd/              # Implementation de HLPD (ECIR 2025)
|-- 03_production/         # Pipeline d'inference, API, deploiement
|-- shared/                # Code mutualise (data loading, metrics, utils)
|-- requirements.txt       # Dependances figees
|-- README.md              # Documentation du projet
```

### Principes de Code Propre

Cette restructuration n'est pas qu'un simple deplacement de fichiers. Nous avons applique plusieurs principes d'ingenierie logicielle :

*   **Separation des responsabilites** : Chaque repertoire correspond a une etape logique du projet. Le code de recherche (`02_hydra`, `02b_hlpd`) est isole du code de production (`03_production`).
*   **Mutualisation via `shared/`** : Les fonctions de chargement de donnees, les metriques hierarchiques et les utilitaires de logging sont centralises dans un module partage. Cela elimine la duplication de code entre HYDRA et HLPD.
*   **Reproductibilite** : Un fichier `requirements.txt` unique fige toutes les dependances. Chaque experience est reproductible a partir d'un `git clone` et d'un `pip install`.
*   **Documentation integree** : Chaque module contient ses docstrings et ses scripts de lancement documentes.

### Pret pour GitHub

Le projet est desormais structure pour etre heberge sur un repository Git public ou prive, avec une organisation qui permet a tout nouveau contributeur de comprendre la logique du projet en quelques minutes. La progression naturelle (baselines -> HYDRA -> HLPD -> production) est inscrite dans la numerotation meme des repertoires.

---

## References

1. Karl, F. et al. (2025). **HYDRA: Hierarchical Classification with Deep Neural Networks.** *Proceedings of EMNLP 2025.*

2. Torba, F., Laclau, C. et al. (2025). **Decoding the Hierarchy: A Hybrid Approach to Hierarchical Multi-label Text Classification.** *Proceedings of ECIR 2025.*

3. Torba, F. (2025). **PhD Thesis: Hierarchical Multi-label Text Classification for Tender Analysis.**

4. Lin, T.-Y. et al. (2017). **Focal Loss for Dense Object Detection.** *Proceedings of ICCV 2017.*

5. Su, J. et al. (2022). **ZLPR: Zero-bounded Log-sum-exp Pairwise Ranking Loss for Multi-label Classification.**
