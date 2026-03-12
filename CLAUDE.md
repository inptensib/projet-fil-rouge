# CLAUDE.md -- Instructions pour Claude Code

## Workflow obligatoire

1. **Planifier** avant de coder : explorer le code, comprendre le contexte, proposer un plan
2. **Tester** apres chaque modification (le hook auto-test le fait, sinon lancer manuellement) :
   ```bash
   /home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -v --tb=short
   ```
3. **Boucler** si des tests echouent : diagnostiquer, corriger, re-tester (0 failures)
4. **`/sync-check`** apres tout changement de code, docs ou configs
5. **`/simplify`** apres tout gros refactoring ou ajout de feature

## Conventions

- Variantes HYDRA : `local`, `global`, `nested` (jamais `hybrid`)
- Constantes : **157** classes L1, **6100+** classes L2, `max_length=128`
- Encodeur principal : `camembert-base`
- FocalLoss sans alpha : `-(1-p_t)^gamma * log(p_t)`, `gamma=2.0`
- Les defaults dans le code doivent matcher `02_hydra/configs/default.yaml` :
  - `dropout=0.3`, `lr=3.5e-5`, `batch_size=128`, `hidden_dim=256`, `epochs=10`, `level_weight_l1=0.5`

## Structure critique

```
02_hydra/src/          models.py, losses.py, trainer.py, predict.py
02_hydra/configs/      default.yaml (config de reference)
03_production/src/     main.py, trainer_ddp.py, ddp_utils.py, data_sharding.py, error_analysis.py, predict.py
shared/                data_loader.py, metrics.py, utils.py
tests/                 61 tests pytest
```

## Qualite

- Pas de code mort, pas de commentaires inutiles
- Imports et signatures cross-modules verifies
- Docker reproductible : `docker compose run --rm --profile test test`

## Skills -- TOUJOURS les utiliser des que l'occasion se presente

Les skills sont des pipelines automatises puissants. Ne JAMAIS faire manuellement ce qu'un skill peut faire. Invoquer le skill correspondant est OBLIGATOIRE dans chaque situation listee ci-dessous.

### Automatiques (deja configures)
- **Hook auto-test** : pytest tourne apres chaque Edit/Write — rien a faire
- **MCP MLflow** : disponible quand le conteneur MLflow tourne (`docker compose up mlflow -d`)

### Qualite code — a chaque modification

| Trigger | Skill | Ce qu'il fait |
|---------|-------|---------------|
| Tu as modifie du code Python, des docs (.md), ou des configs (.yaml) | `/sync-check` | Detecte TOUT drift entre code, configs YAML, READMEs et blog.md. Sans lui, des incoherences passent en prod silencieusement. |
| Tu as fait un refactoring, ajoute une feature, ou touche a >3 fichiers | `/simplify` | Revue de qualite : detecte code duplique, abstractions inutiles, imports morts. Garantit un code minimal et propre. |
| L'utilisateur demande un commit ou tu t'appretes a en faire un | `/pre-commit` | Pipeline complet : lint + 61 tests + sync-check + scan securite. BLOQUANT : ne jamais committer sans l'avoir lance. Supersede `/sync-check` (il l'inclut deja). |
| Tu as modifie requirements.txt, ajoute/supprime un import, ou cree un nouveau fichier Python | `/dep-audit` | Croise imports reels vs requirements.txt. Detecte packages manquants (crash en prod) ou inutiles (bloat image Docker). |

### GPU & Training — avant et pendant les runs

| Trigger | Skill | Ce qu'il fait |
|---------|-------|---------------|
| L'utilisateur veut lancer un training ou demande les ressources | `/gpu-status` | Scan complet du cluster : nodes, GPUs libres, memoire, jobs en cours. Recommande la meilleure partition et le batch_size optimal. TOUJOURS lancer avant `/slurm submit`. |
| L'utilisateur veut soumettre un job DDP | `/slurm submit [variant] [encoder] [nb_nodes] [gpu_type]` | Lance le training distribue via SLURM. Gere les scripts, la config DDP, et retourne le job_id. |
| Un training DDP tourne ou l'utilisateur demande le statut | `/slurm status` | Vue synthetique : jobs actifs, progression, nodes utilises. |
| L'utilisateur veut voir les logs ou debug un job | `/slurm logs <job_id>` | Affiche les logs SLURM et les logs par rank. Indispensable pour diagnostiquer les erreurs DDP. |
| L'utilisateur veut arreter un job | `/slurm cancel <job_id>` | Annule proprement le job SLURM. |
| Un training long tourne (>30min) | `/loop 5m /slurm status` | Monitoring continu : verifie le statut toutes les 5 minutes et alerte si un job plante. |
| Avant un training DDP, des shards ont ete generes | `/shard-check [shard_dir]` | Valide l'integrite des shards : pas d'overlap, couverture des 157 classes L1, equilibrage entre ranks. Un shard corrompu = training silencieusement faux. |
| L'utilisateur veut optimiser les performances avant un gros run | `/profile-train [script]` | Profile GPU/memoire/throughput, detecte les bottlenecks (data loading, mixed precision, num_workers). Donne le top 3 des optimisations par impact. |
