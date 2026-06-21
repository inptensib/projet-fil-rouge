---
name: slurm
description: Gere les jobs SLURM (soumettre, monitorer, annuler, lire les logs). Utiliser pour tout training DDP sur le cluster GPU.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [status|submit|cancel|logs] [job_id]
---

# SLURM Manager

Gestion des jobs SLURM pour le training HYDRA DDP.

## Commandes

- `/slurm status` — Etat de la queue et des nodes GPU
- `/slurm submit [variant] [encoder]` — Lancer un training DDP
- `/slurm cancel [job_id]` — Annuler un job
- `/slurm logs [job_id]` — Lire les logs d'un job

## Actions par commande

### `status` (default si pas d'argument)

```bash
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %.4C %.12R" 2>/dev/null
sinfo -p P100,3090 -N --Format="NodeList:12,Partition:10,Gres:8,GresUsed:16,StateLong:12" --noheader 2>/dev/null
```

Presenter un tableau synthetique : jobs actifs, nodes disponibles, GPUs libres par partition.

### `submit`

> **Note** : le pipeline DDP distribue est archive via le tag git
> `archive/distributed-pipeline` et hors livrable. Le training standard
> (single-node, DataParallel) s'effectue via les scripts `.slurm` du projet.

Utiliser, depuis la racine du projet, le script SLURM adapte :
```bash
# HYDRA par variante (job array %1 = reprise auto sur checkpoint)
sbatch scripts/run_train_variant.slurm <variant> <config>
#   ex : sbatch scripts/run_train_variant.slurm local configs/hydra/default.yaml
# ou la version array dediee
sbatch scripts/run_train_array.slurm configs/hydra/default.yaml
```
`<variant>` ∈ {local, global, nested} ; l'encodeur et les hyperparametres sont
definis dans le `<config>` YAML (`configs/hydra/`). Pour FlatHydra/HLPD, voir
les scripts `run_flat_*.slurm` / `run_hlpd_*.slurm` (cf. docs/PIPELINE.md).

### `cancel`

```bash
scancel $JOB_ID
```

### `logs`

Chercher et afficher les logs :
```bash
# Logs SLURM
cat slurm-$JOB_ID.out 2>/dev/null

# Logs par rank
ls -la logs/rank_*_*.log 2>/dev/null
tail -50 logs/rank_0_*.log 2>/dev/null
```

## Contexte cluster

- Partitions : `P100` (nodes avec 2 GPUs Tesla P100 chacun), `3090` (node40, 2x RTX 3090)
- QoS : max 2 GPUs/job, max 4 jobs simultanes
- Logs : `slurm-*.out` (racine)
