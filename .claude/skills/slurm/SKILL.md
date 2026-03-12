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

Utiliser le script existant :
```bash
cd /home/infres/tandriam-25/fil_rouge/rakuten-classification/03_production
./scripts/launch_distributed.sh $1 $2 $3 $4
```
Arguments : variant (local/global/nested), encoder (CamemBERT/XLM-RoBERTa/mBERT), nb_nodes, gpu_type (p100/rtx3090)

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
- DDP config : fichier `.ddp_config` genere par `launch_distributed.sh`
- Logs : `slurm-*.out` (racine) et `logs/rank_*_*.log` (par process)
