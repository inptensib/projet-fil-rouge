# Métriques et Provenance des Résultats

Définitions des métriques du projet, explication des deux modes
d'évaluation (le piège classique du repo) et traçabilité de chaque chiffre
publié dans `results/`.

> **Warning — fuite de données dans les benchmarks v1 (12/06/2026)** :
> un audit du dataset a révélé que **11,2 % des lignes de val et de test
> avaient un jumeau exact dans le train** (même fiche produit scrapée
> plusieurs fois, 98 % avec le même label). Tous les résultats de `results/`
> et `comparison.html` ont été entraînés et évalués sur ce dataset « v1 »
> (fuite non purgée). Les comparaisons *entre* modèles restent valides
> (même fuite pour tous).
>
> Le dataset redressé (« v2 », fuite purgée : **−314 624 lignes de train
> nettes**, 6 126 → 5 777 classes L2) est sur HuggingFace dans
> `datasets_finaux/`.
>
> **Effet réel mesuré (≠ l'estimation initiale de +1,5–3 pts).** FlatHydra a
> été ré-entraîné + ré-évalué sur v2 (CamemBERT **et** mBERT, méthode naïve
> identique à v1, seule la donnée change) :
> - **L1 (mêmes 157 classes — comparaison propre)** : v2 est **0,8 à 1,3 pt
>   plus bas** que v1 → c'est le vrai effet de la fuite, qui gonflait v1
>   d'**~1 pt**.
> - **L2** : v2 est *plus haut* (CamemBERT 71,7 → 78,9 ; mBERT 71,8 → 77,4),
>   mais **pas directement comparable** : ~3 pts viennent mécaniquement des 351
>   classes droppées (F1 moyen 0,20), et le reste du fait que dé-fuiter a
>   *amélioré* l'entraînement (doublons retirés), pas gonflé le test.
>
> Conclusion : l'inflation v1 réelle est **~1 pt** (mesurée au L1), bien en
> deçà de l'estimation initiale ; au L2 les chiffres v1 et v2 ne se comparent
> pas terme à terme. (Mesuré sur FlatHydra / 2 encodeurs ; HYDRA et HLPD n'ont
> pas été ré-entraînés sur v2.) Détail des 4 évals : `results/flat_hydra*/`.
>
> *Note : les modèles v2 ont **5 775** têtes L2 (et non 5 777) — 2 classes L2 du
> dataset n'ont aucun échantillon en train après purge, le nombre de têtes étant
> dérivé des données à l'exécution. D'où `num_classes_l2: 5775` dans
> `results/flat_hydra_v2/test_metrics.json`.*

## Table des Matières

1. [Définitions](#définitions)
1. [Les deux modes d'évaluation](#les-deux-modes-dévaluation)
1. [L'incident mBERT et le mode auto](#lincident-mbert-et-le-mode-auto)
1. [Provenance des chiffres publiés](#provenance-des-chiffres-publiés)
1. [Reproduire une évaluation](#reproduire-une-évaluation)
1. [Évaluer un nouveau modèle](#évaluer-un-nouveau-modèle)

## Définitions

Toutes les métriques sont implémentées dans `rakuten/common/metrics.py` :

| Métrique               | Définition                                            |
|:-----------------------|:-------------------------------------------------------|
| `f1_l1_macro`          | F1 macro sur les 157 classes L1 (chaque classe pèse 1) |
| `f1_l2_macro`          | F1 macro sur les 5 777 classes L2 — **métrique cible** |
| `constrained_f1_macro` | F1 L2 compté correct seulement si le parent L1 l'est   |
| `hcr`                  | % de prédictions où L2 est un enfant valide de L1      |

Le **macro**-averaging donne le même poids aux classes rares qu'aux
classes fréquentes — c'est voulu : la longue traîne des 5 777 classes L2
est l'enjeu du projet. Objectif global : maximiser `f1_l2_macro` en
maintenant un HCR > 95 % pour les variantes hiérarchiques.

## Les Deux Modes d'Évaluation

Un même modèle HYDRA peut être évalué de deux façons :

| Mode   | Mécanisme                                        | HCR typique |
|:-------|:--------------------------------------------------|:-----------:|
| `base` | Têtes L1 et L2 indépendantes, aucune contrainte    | ~0.86       |
| `hier` | Masque hiérarchique : L2 restreint aux enfants du L1 prédit | ~0.997 |

> **Warning** : les chiffres des deux modes ne sont **pas comparables**.
> Le mode `hier` donne un HCR ≈ 0.997 *par construction* et change le
> F1 L2. Comparer un modèle évalué en `base` avec un autre en `hier`
> n'a aucun sens.

Règle de publication du projet :

- Modèles **standards** (entraînés sans hiérarchie) → publier le mode `base`
- Modèles **hierarchy** (`condition_l2_on_l1` + `check_hierarchy`) →
  publier le mode `hier` (c'est leur régime de fonctionnement nominal)

## L'Incident mBERT et le Mode Auto

Historique : `scripts/eval_hierarchy.py` sérialisait systématiquement le
mode `hier` dans son JSON de sortie. Pour HYDRA Local mBERT (modèle
standard), cela a publié F1 L2 = 60.7 % / HCR = 99.7 % au lieu des vraies
valeurs 66.8 % / 85.8 % — corrigées ensuite à la main dans
`results/hydra_eval_test_metrics_mbert.json` (champs `_note`).

Depuis, le script choisit le bon mode automatiquement :

```bash
python scripts/eval_hierarchy.py --split-dir "$DATA_DIR" \
    --config configs/hydra/mbert_default.yaml \
    --output-json results/eval.json
# → mode "base" car la config n'a ni check_hierarchy ni condition_l2_on_l1
```

Le flag `--output-mode {auto,base,hier}` permet l'override explicite, et
le JSON produit contient les métadonnées `_mode`, `_config` et `_date`.

## Provenance des Chiffres Publiés

Chaque fichier de `results/` consommé par `generate_comparison.py` :

| Fichier                              | Modèles               | Source                      | Mode  |
|:--------------------------------------|:----------------------|:-----------------------------|:------|
| `hydra_eval_test_metrics.json`         | HYDRA std CamemBERT ×3 | Job SLURM 772505             | base  |
| `hydra_eval_hierarchy_metrics.json`    | HYDRA hierarchy ×3     | Jobs 772228/229/230          | hier  |
| `hydra_eval_test_metrics_mbert.json`   | HYDRA Local mBERT      | Log training job 813697_5    | base  |
| `flat_hydra/test_metrics.json`         | FlatHydra              | `run_eval_flat_hydra.slurm`  | base  |
| `flat_hydra_title/test_metrics.json`   | FlatHydra title        | `run_eval_flat_hydra_title.slurm` | base |
| `hlpd/test_metrics.json`               | HLPD                   | Rapporté (best @ step 345k)  | natif |
| `baselines/baseline_results.json`      | TF-IDF, W2V, fastText  | Scripts/notebooks baselines  | base  |
| `baselines/baseline_results_title.json`| Idem, title-only       | Idem                         | base  |
| `<modele>/f1_per_class_l{1,2}.json`    | F1 par classe          | `generate_f1_per_class.py`   | base  |
| `class_metadata.json`                  | Support par classe     | `build_class_metadata.py`    | —     |

Particularités à connaître :

- Les JSON marqués d'un champ `_source` ou `_note` ont été **curés à la
  main** après coup (notamment le mBERT) — ne pas les régénérer sans
  relire leur en-tête.
- Les métriques des HYDRA **title-only standard** proviennent du logging
  MLflow en fin d'entraînement (pas de JSON d'évaluation dédié) ;
  `generate_comparison.py` les lit directement dans `mlflow.db`.
- `mlflow.db` (racine, ignoré par git) contient l'historique complet des
  runs — interface via `docker compose up mlflow -d`.

## Reproduire une Évaluation

```bash
# HYDRA standard (3 variantes, les deux modes affichés en console)
sbatch scripts/run_eval.slurm

# HYDRA hierarchy (modèles dans models_hierarchy/)
sbatch scripts/run_eval.slurm models_hierarchy

# Une seule variante, en local avec GPU
python scripts/eval_hierarchy.py --split-dir "$DATA_DIR" \
    --variant local --eval-batch-size 2048 \
    --output-json results/hydra_eval_test_metrics.json
```

L'affichage console montre toujours les deux modes avec leurs deltas —
seul le JSON suit `--output-mode`.

## Évaluer un Nouveau Modèle

1. **Consulter MLflow avant d'entraîner** : `search_best_runs("f1_l2_macro")`
   pour connaître la référence à battre (pas de run en double)
1. **Comparer systématiquement** au baseline TF-IDF (F1 L2 = 42.8 %) et au
   meilleur modèle (FlatHydra, F1 L2 = 71.7 %)
1. **Vérifier le HCR** en plus du F1 — un modèle hiérarchique doit tenir
   HCR > 95 %
1. **Publier le bon mode** (`--output-mode auto` fait le bon choix) et
   regénérer `results/comparison.html` via `generate_comparison.py`
