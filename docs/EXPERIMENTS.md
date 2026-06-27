# Expériences — récapitulatif consolidé

Vue d'ensemble de **toutes** les expériences du projet (baselines, HYDRA, HLPD,
FlatHydra), avec les chiffres de **test** tracés à leur fichier source. Pour les
définitions de métriques, les deux modes d'évaluation et la provenance détaillée,
voir [`METRICS.md`](METRICS.md). Pour la conception du modèle retenu, voir
[`flat_hydra_conception.md`](flat_hydra_conception.md). Pour reproduire, voir
[`PIPELINE.md`](PIPELINE.md).

> **Tâche** : classer un produit Rakuten (texte français) en **157 classes L1**
> (`typelabel`) et **5 777 classes L2** (`producttype`). Métrique cible :
> `f1_l2_macro` (chaque classe pèse 1 — la longue traîne est l'enjeu).

## Modèle retenu

**FlatHydra — CamemBERT, dataset v2** : une seule tête unifiée, **F1 L2 macro =
78,9 %** sur le test. C'est le meilleur F1 L2 mesuré, avec l'architecture la plus
simple (pas de têtes multiples ni de masque hiérarchique). Source :
`results/flat_hydra_v2/test_metrics.json`.

| Métrique | Valeur |
|:---------|-------:|
| F1 L1 macro | 92,47 % |
| **F1 L2 macro** | **78,93 %** |
| Constrained F1 | 79,90 % |
| HCR | 85,17 % |

## Lire ce document (comparabilité)

Trois axes rendent certaines lignes **non comparables terme à terme** :

- **Dataset `v1` vs `v2`** : `v1` contenait une fuite (11,2 % de val/test
  dupliqués dans le train) ; `v2` est purgé et rééquilibré (synthétique).
  L'inflation v1 réelle est **~1 pt** (mesurée au L1) ; au L2 v1 et v2 ne se
  comparent pas directement (voir le warning de [`METRICS.md`](METRICS.md)).
- **Mode `base` vs `hier`** : `hier` applique un masque hiérarchique → HCR ≈ 99,7 %
  *par construction* et modifie le F1 L2. Ne pas comparer un `base` à un `hier`.
- **Plein texte vs titre seul** : `titre` = `designation` uniquement ;
  `plein texte` = `designation + ". " + description`.

Tous les chiffres sont en **% sur le test**. Tableaux groupés par comparabilité.

## 1. Baselines classiques (dataset v1, mode base, plein texte)

| Modèle | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:-------|------:|------:|----:|-----------:|
| TF-IDF (50K, 1-2grammes) + LinearSVC | 83,26 | 42,75 | 63,41 | 42,23 |
| fastText (L1 softmax, L2 softmax) | 82,61 | 42,87 | 67,41 | 42,59 |
| fastText (L1 softmax, L2 OVA) | 82,60 | 10,01 | 29,29 | 9,79 |
| Word2Vec (200d, skip-gram) + LinearSVC | 60,24 | 22,27 | 25,06 | — |

Source : `results/baselines/baseline_results.json`. Le L2 plafonne à **~43 %** :
les méthodes sac-de-mots ne tiennent pas la longue traîne.

## 2. HYDRA — CamemBERT (dataset v1)

Multi-têtes encoder-only (3 variantes). Mode `base` = têtes indépendantes ;
mode `hier` = masque hiérarchique.

**Standard, mode base** — `results/hydra_eval_test_metrics.json` (job 772505) :

| Variante | F1 L1 | F1 L2 | HCR |
|:---------|------:|------:|----:|
| local | 93,50 | 66,65 | 85,82 |
| nested | 93,62 | 64,26 | 85,83 |
| global | 93,56 | 66,20 | 85,87 |

**Hierarchy, mode hier** — `results/hydra_eval_hierarchy_metrics.json` :

| Variante | F1 L1 | F1 L2 | HCR |
|:---------|------:|------:|----:|
| local | 91,33 | 61,18 | 99,74 |
| nested | 92,06 | 61,14 | 99,74 |
| global | 92,59 | 62,70 | 99,74 |

Le masque pousse le HCR à ~99,7 % mais **fait baisser le F1 L2** (~−4 pts vs
standard) : il interdit des L2 hors de la L1 prédite, y compris des bonnes.

## 3. HYDRA — dataset v2 (mode hier)

Ré-entraîné sur v2 avec le correctif du **masque multi-parents** (une L2 à
plusieurs parents L1 n'est plus masquée à tort). Source :
`results/hydra_hierarchy_v2_metrics.json` (2026-06-25).

| Variante | F1 L1 | F1 L2 | HCR | HCR multi-parents | Constr. F1 |
|:---------|------:|------:|----:|------------------:|-----------:|
| local | 92,60 | 77,38 | 87,04 | 100,0 | 79,31 |
| nested | 92,59 | 76,41 | 87,02 | 100,0 | 78,56 |
| global | 92,91 | 77,29 | 86,99 | 100,0 | 79,17 |

Sur v2, HYDRA hierarchy monte à ~77 % de F1 L2 (vs ~61 % en v1) et atteint un
**HCR multi-parents de 100 %**, mais reste **sous le FlatHydra v2 (78,9 %)** en
F1 L2 pour une architecture bien plus lourde.

## 4. FlatHydra (tête unifiée, mode base)

| Modèle | Dataset | Texte | F1 L1 | F1 L2 | HCR | Constr. F1 | Source |
|:-------|:-------:|:-----:|------:|------:|----:|-----------:|:-------|
| CamemBERT | v1 | plein | 93,24 | 71,70 | 85,49 | 72,19 | `flat_hydra/` |
| CamemBERT | v1 | titre | 92,13 | 69,53 | 85,00 | 70,00 | `flat_hydra_title/` |
| **CamemBERT** | **v2** | **plein** | **92,47** | **78,93** | **85,17** | **79,90** | `flat_hydra_v2/` ★ |
| mBERT | v1 | plein | 93,38 | 71,78 | 85,61 | 72,43 | `flat_hydra_mbert/` |
| mBERT | v2 | plein | 92,09 | 77,40 | 85,20 | 78,34 | `flat_hydra_mbert_v2/` |

★ = modèle retenu. Tous les fichiers : `results/<dir>/test_metrics.json`.

## 5. Comparaison des encodeurs (CamemBERT vs mBERT)

À méthode identique (FlatHydra), les deux encodeurs sont **quasi équivalents** :

| Dataset | CamemBERT (F1 L2) | mBERT (F1 L2) | Écart |
|:-------:|------------------:|--------------:|------:|
| v1 | 71,70 | 71,78 | −0,08 |
| v2 | 78,93 | 77,40 | +1,53 |

Le multilingue (mBERT) n'apporte pas d'avantage décisif sur un corpus français ;
**CamemBERT** est retenu (léger gain sur v2, monolingue français natif).

## 6. HLPD (décodeur hiérarchique)

`results/hlpd/test_metrics.json` (dataset v1, mode natif) : F1 L1 = 80,70,
F1 L2 = 61,50, **HCR = 97,80**. Le décodeur garantit une forte cohérence
hiérarchique mais reste en retrait sur le F1 (cible du projet).

## Enseignements

1. **Les modèles profonds écrasent les baselines** sur le L2 : 78,9 %
   (FlatHydra v2) contre ~43 % (meilleure baseline TF-IDF/fastText) — +36 pts.
2. **La description compte** : plein texte > titre seul (FlatHydra CamemBERT v1
   71,7 vs 69,5 ; écart bien plus marqué sur les baselines).
3. **Le masque hiérarchique est un compromis** : HCR ~99,7 % *par construction*
   mais F1 L2 plafonné. Le FlatHydra (sans masque) garde un HCR ~85 % « appris »
   tout en maximisant le F1 L2.
4. **Simple gagne** : une tête unifiée (FlatHydra) égale ou dépasse HYDRA
   (3 têtes + masque) en F1 L2, pour une complexité moindre → modèle retenu.
5. **CamemBERT ≈ mBERT** ; CamemBERT légèrement devant sur v2.
6. **v2 (purgé + rééquilibré)** porte le F1 L2 du FlatHydra à 78,9 % — le
   rééquilibrage de la longue traîne (données synthétiques, voir
   [`../scripts/synthetic_pipeline/`](../scripts/synthetic_pipeline/)) tire le
   macro vers le haut.

## Annexe — variantes « titre seul » (designation uniquement, dataset v1)

Baselines (mode base) — `results/baselines/baseline_results_title.json` :

| Modèle | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:-------|------:|------:|----:|-----------:|
| TF-IDF + LinearSVC | 82,55 | 41,10 | 61,38 | 40,69 |
| fastText (softmax) | 72,94 | 34,32 | 64,29 | 32,17 |
| Word2Vec + LinearSVC | 64,02 | 28,16 | 27,55 | 21,83 |

HYDRA hierarchy (mode hier) — `results/hydra_eval_title_hierarchy_metrics.json` :

| Variante | F1 L1 | F1 L2 | HCR | Constr. F1 |
|:---------|------:|------:|----:|-----------:|
| local | 89,72 | 58,94 | 99,75 | 60,79 |
| nested | 90,79 | 58,67 | 99,75 | 60,41 |
| global | 91,63 | 60,49 | 99,74 | 62,16 |

Le FlatHydra CamemBERT titre seul (92,13 / 69,53 / 85,00) figure au §4. Partout,
le titre seul reste sous le plein texte — la description apporte du signal.

## Reproduire

- Évaluations HYDRA / FlatHydra : voir [`PIPELINE.md`](PIPELINE.md) (matrice
  modèle × commande) et [`METRICS.md`](METRICS.md) (modes, provenance).
- Rapport HTML comparatif : `results/comparison.html`
  (`scripts/generate_comparison.py`).
- Historique complet des runs : `mlflow.db` (`docker compose up mlflow -d`).
