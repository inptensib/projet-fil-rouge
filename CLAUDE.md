# CLAUDE.md -- Instructions pour Claude Code

## Contexte projet

Classification hierarchique de **8M+ produits Rakuten** en **157 categories L1** et **6100+ categories L2**.
Architecture principale : **HYDRA** (multi-tete encoder-only, EMNLP 2025), 3 variantes : `local`, `global`, `nested`.
Encodeur : `camembert-base`. Donnees : texte (designation + description) en francais.

**Metriques cles** (toutes dans `shared/metrics.py`) :
- `f1_l1_macro`, `f1_l2_macro` — metriques principales (macro = chaque classe pese egalement)
- `constrained_f1` — F1 L2 valide uniquement si L1 parent correct
- `hierarchical_consistency_rate` (HCR) — % de predictions ou L2 est enfant valide de L1

Objectif : maximiser `f1_l2_macro` tout en maintenant un HCR > 95%.

## Workflow data science

Ne pas sauter d'etape. Chaque etape a un skill dedie.

```
1. COMPRENDRE     Lire le code, les resultats existants, les papiers de reference
                  → /content-report, /pdf-read, MLflow MCP (list_runs, search_best_runs)

2. PLANIFIER      Definir l'hypothese, les metriques cibles, le plan d'experience
                  → Mode Plan (ultrathink), comparer aux baselines et runs precedents

3. VERIFIER       Avant de lancer : donnees OK, GPU libres, shards valides
                  → /shard-check, /gpu-status

4. EXPERIMENTER   Lancer le training, monitorer
                  → /slurm submit, /loop 5m /slurm status, MLflow MCP (get_metric_history)

5. EVALUER        Analyser les resultats, comparer aux baselines, error analysis
                  → MLflow MCP (compare_runs), error_analysis.py, /content-report

6. DOCUMENTER     Mettre a jour READMEs, blog.md, diagrammes
                  → /sync-check, /markdown, /diagram

7. COMMITTER      Pipeline qualite puis commit
                  → /pre-commit (inclut lint + tests + sync-check + securite)
```

**Regles critiques :**
- **TOUJOURS consulter MLflow** avant de lancer un nouveau training (pas de run en double)
- **TOUJOURS comparer au baseline** TF-IDF et au meilleur run existant (via MLflow)
- **TOUJOURS verifier la coherence hierarchique** (HCR) en plus du F1
- **Tester** apres chaque modification de code (le hook auto-test le fait, sinon manuellement) :
  ```bash
  /home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -v --tb=short
  ```

## Conventions techniques

- Variantes HYDRA : `local`, `global`, `nested` (jamais `hybrid`)
- Constantes : **157** classes L1, **6100+** classes L2, `max_length=256`
- Encodeur : `camembert-base`
- FocalLoss sans alpha : `-(1-p_t)^gamma * log(p_t)`, `gamma=2.0`
- Source de verite pour les hyperparametres : `02_hydra/configs/default.yaml`
  - `dropout=0.3`, `lr=3.5e-5`, `batch_size=64`, `hidden_dim=256`, `epochs=10`, `level_weight_l1=0.5`
- Tout default dans le code Python DOIT matcher ce fichier YAML

## Structure du projet

```
01_baselines/src/      tfidf_pipeline.py, word2vec_pipeline.py, bert_flat_classifier.py, evaluate.py
02_hydra/src/          models.py, losses.py, trainer.py, predict.py
02_hydra/configs/      default.yaml (source de verite hyperparametres)
03_production/src/     main.py, trainer_ddp.py, ddp_utils.py, data_sharding.py, error_analysis.py, predict.py
shared/                data_loader.py, metrics.py, utils.py
shared/dataset/        products.jsonl (non versionne — obtenir aupres de Rakuten)
tests/                 77 tests pytest
notebooks/             explorations, visualisations
blog.md                article technique du projet
docker-compose.yml     services : test, training, baselines, mlflow
```

## Qualite code

- Pas de code mort, pas de commentaires inutiles
- Imports et signatures cross-modules verifies (`/sync-check`)
- Docker reproductible : `docker compose run --rm --profile test test`
- Dependances alignees : imports reels = requirements.txt (`/dep-audit`)

## Donnees

- **Format** : JSONL (`products.jsonl`), champs : `designation`, `description`, `typelabel` (L1), `producttype` (L2)
- **Chargement** : `shared/data_loader.py` → `prepare_hierarchical_data()` (split 80/20, validation 12.5%)
- **Sharding DDP** : `03_production/src/data_sharding.py` → valider avec `/shard-check` avant tout training distribue
- **Pas de data dans le repo** : le fichier `shared/dataset/products.jsonl` doit etre place manuellement

## Experimentation & MLflow

Le MCP MLflow est disponible quand le conteneur tourne (`docker compose up mlflow -d`).

**Outils MCP disponibles :**

| Outil MCP | Quand l'utiliser |
|:----------|:-----------------|
| `list_experiments()` | Lister toutes les experiences MLflow (decouvrir les experiments existants) |
| `list_runs(experiment_name)` | Voir les runs recents avant de lancer un nouveau training |
| `search_best_runs(metric)` | Trouver le meilleur modele sur une metrique donnee |
| `compare_runs(run_ids, metrics)` | Comparer variantes ou hyperparametres cote a cote |
| `get_metric_history(run_id, metric)` | Suivre l'evolution d'une metrique epoch par epoch |
| `get_run_details(run_id)` | Inspecter tous les parametres et metriques d'un run |

**Bonnes pratiques :**
- Avant un nouveau training → `search_best_runs("f1_l2_macro")` pour connaitre la reference
- Apres un training → `compare_runs([new_id, best_id])` pour valider l'amelioration
- Debug convergence → `get_metric_history(run_id, "train_loss")` + `"val_loss"`

---

## Skills -- TOUJOURS les invoquer quand le trigger est rempli

Les skills sont des pipelines automatises. Ne JAMAIS faire manuellement ce qu'un skill peut faire.
Invoquer le skill correspondant est **OBLIGATOIRE** dans chaque situation listee.

### Automatiques (rien a invoquer)

- **Hook auto-test** : pytest tourne apres chaque Edit/Write
- **MCP MLflow** : outils disponibles automatiquement si le conteneur tourne

### Qualite & Coherence — a chaque modification

| Quand | Skill | Pourquoi |
|:------|:------|:---------|
| Code Python, docs (.md), ou configs (.yaml) modifies | `/sync-check` | Detecte TOUT drift entre code, configs, READMEs et blog.md. Sans lui, des incoherences passent silencieusement. |
| Refactoring, nouvelle feature, ou >3 fichiers touches | `/simplify` | Detecte code duplique, abstractions inutiles, imports morts. Garantit un code minimal. |
| Avant tout commit (BLOQUANT) | `/pre-commit` | Pipeline complet : lint + 77 tests + sync-check + scan securite. Ne JAMAIS committer sans. Inclut deja `/sync-check`. |
| requirements.txt modifie, import ajoute/supprime, nouveau fichier .py | `/dep-audit` | Croise imports reels vs requirements.txt. Package manquant = crash en prod. |

### Experimentation & Training — cycle complet

| Quand | Skill | Pourquoi |
|:------|:------|:---------|
| Avant tout training : verifier les ressources GPU | `/gpu-status` | Scan du cluster : nodes, GPUs libres, memoire, partitions. Recommande batch_size selon GPU (P100→128, 3090→256). **TOUJOURS avant** `/slurm submit`. |
| Avant tout training DDP : valider les shards | `/shard-check [shard_dir]` | Verifie integrite des shards : pas d'overlap, couverture 157 L1, equilibrage entre ranks. Shard corrompu = resultats faux silencieusement. |
| Lancer un training distribue | `/slurm submit [variant] [encoder] [nb_nodes] [gpu_type]` | Soumet le job DDP via SLURM avec la bonne config. Retourne le job_id. |
| Monitorer un training en cours | `/slurm status` | Vue synthetique : jobs actifs, progression, nodes. |
| Training long (>30min) en cours | `/loop 5m /slurm status` | Monitoring continu toutes les 5 minutes, alerte si un job plante. |
| Debug d'un job en erreur | `/slurm logs <job_id>` | Logs SLURM + logs par rank. Indispensable pour les erreurs DDP (NCCL, OOM, etc.). |
| Arreter un job | `/slurm cancel <job_id>` | Annulation propre du job SLURM. |
| Optimiser avant un gros run | `/profile-train [script]` | Profile GPU/memoire/throughput, detecte bottlenecks. Top 3 optimisations par impact. |

### Recherche & Analyse — comprendre les sources et resultats

| Quand | Skill | Pourquoi |
|:------|:------|:---------|
| Lire un fichier PDF (papier, rapport, these) local ou en ligne | `/pdf-read <path_or_url> [pages]` | Extraction complete : texte, images/diagrammes (rendus en PNG), tableaux, metadonnees, TOC. Bien plus fiable que Read natif pour les figures et tableaux. |
| Comprendre le contenu de n'importe quelle source | `/content-report <path_or_url>` | Rapport structure depuis PDF, web, Python, image, notebook, YAML, dossier. Auto-detecte le type, produit summary + figures + tables + structure + references. |

### Redaction & Documentation — contenu professionnel

| Quand | Skill | Pourquoi |
|:------|:------|:---------|
| Creer ou ameliorer un document Markdown | `/markdown write <path>` ou `/markdown improve <path>` | Markdown technique impeccable : headings, tableaux alignes, code avec langage, diagrammes texte, listes paralleles. Google Markdown Style Guide. |
| Auditer la qualite d'un fichier Markdown | `/markdown review <path>` | Score /100 sur 15 criteres (hierarchie, tableaux, code, liens, images, style). Recommandations prioritaires. |
| Besoin d'un squelette de document | `/markdown template <type>` | Templates prets : `readme`, `spec`, `report`, `blog`, `api`, `changelog`. |
| Creer un diagramme (architecture, flux, classes, sequences) | `/diagram mermaid <type>` | Mermaid pro : flow, arch, C4, class, sequence, state, ER, gantt, block. Palette coherente, rendu natif GitHub. |
| Diagramme editable pour presentations/PDF | `/diagram drawio <type> <path.drawio>` | Fichier draw.io XML, ouvrable dans app.diagrams.net. Export PNG/SVG. |
