# hybrid_system/ — Contribution séparée (Younes Boutrouft)

## Statut

Ce dossier contient la contribution **séparée et indépendante** d'un coéquipier :
une approche hybride CamemBERT + XGBoost pour la classification L2.

**Points importants pour un repreneur :**

- Ce code est **hors du package `rakuten/`** : `pip install -e .` ne l'expose pas
  et il n'est pas accessible via `import rakuten.*`.
- Il n'a **pas été audité** dans le cadre de la restructuration du dépôt
  (migration vers le package `rakuten/`, révision des imports, tests).
- Il n'est **pas couvert par la suite de tests** (`tests/` ne contient aucun test
  pour `hybrid_system/`).
- Il a ses **propres dépendances** (xgboost, sentence-transformers, peft)
  non déclarées dans `pyproject.toml`.
- Les **checkpoints** nécessaires (`l1_classifier/`, `l2_camembert_run2/`,
  `l2_xgboost/`) ne sont pas hébergés avec les modèles principaux sur
  `TahianaAndriambahoaka/rakuten-models`.

## Résultats déclarés

Le meilleur système (dossier `improvements/`) atteint **F1-macro L2 = 67.22 %**
(accuracy = 84.76 %) — évalué sur `dataset/test_split.jsonl` (1 194 356 échantillons).
Ces chiffres proviennent des fichiers `results/` internes et n'ont pas été vérifiés
de manière indépendante.

## Documentation interne

L'architecture complète, les résultats par variante et les instructions de reproduction
sont dans les sous-dossiers :

- `Gated+Hybrid/README.md` — système final (F1-macro L2 = 63.21 %)
- `improvements/README.md` — meilleure variante (F1-macro L2 = 67.22 %)
- `Specialists/README.md` — ablations (SetFit, LoRA, génératif — exclus du final)

Pour toute question sur cette contribution, contacter l'auteur original.
