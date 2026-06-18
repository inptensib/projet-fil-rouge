---
name: shard-check
description: Valide l'integrite des shards DDP (coverage, overlap, distributions L1/L2, corruption). Utiliser avant tout lancement de training distribue.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [shard_dir]
---

# Shard Check

> **⚠️ Concerne le pipeline DDP archivé (hors livrable).** Ce skill valide les
> shards du pipeline distribue, archive via le tag git
> `archive/distributed-pipeline`. Le code source (`03_distributed/`) n'est pas
> present dans le livrable courant. N'invoquer ce skill que si vous travaillez
> explicitement avec l'archive DDP.

Valide l'integrite des shards de donnees pour le training DDP.

## Arguments

- `$0` : chemin du dossier de shards (default: `shards/`)

## Taches

1. **Existence** : verifier que tous les fichiers shard attendus existent dans `$0` (ou `shards/`)
2. **Coverage** : compter le nombre total de samples across all shards, comparer au dataset complet
3. **Overlap** : verifier qu'aucun produit (par ID) n'apparait dans plusieurs shards
4. **Distribution L1** : pour chaque shard, compter les samples par categorie L1 (157 classes). Alerter si un shard a 0 samples pour une classe L1
5. **Distribution L2** : meme verification pour les 5 777 sous-categories L2 (dataset v2)
6. **Equilibrage** : verifier que les shards ont des tailles similaires (ecart max 5% par rapport a la moyenne)
7. **Integrite** : tenter de charger chaque shard, verifier que les colonnes attendues existent (product_id, title, description, label_l1, label_l2)

## Output

Rapport markdown avec :
- Tableau : shard_name | nb_samples | nb_l1_classes | nb_l2_classes | taille_fichier
- Alertes si overlap, classes manquantes, ou desequilibre > 5%
- Verdict final : PASS ou FAIL avec details
