---
name: gpu-status
description: Affiche l'etat des GPUs sur le cluster (disponibilite, memoire, jobs en cours). Utiliser avant de lancer un training.
user-invocable: true
allowed-tools: Bash, Read
argument-hint: [partition]
---

# GPU Status

Etat des ressources GPU sur le cluster SLURM.

## Taches

### 1. Vue d'ensemble SLURM

```bash
sinfo -p P100,3090 -N --Format="NodeList:12,Partition:10,Gres:8,GresUsed:16,StateLong:12,CPUsState:14,FreeMem:10" --noheader 2>/dev/null
```

### 2. Jobs GPU en cours

```bash
squeue --format="%.10i %.8u %.15j %.8T %.10M %.4D %.4C %.20R %.10b" 2>/dev/null | head -30
```

### 3. Detail GPU par node (si accessible via SSH)

Pour chaque node idle/mix, tenter :
```bash
ssh -o ConnectTimeout=2 -o BatchMode=yes $NODE "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader" 2>/dev/null
```

### 4. Recommandation

Presenter un tableau :

| Node | Partition | GPUs Total | GPUs Libres | Etat | Memoire Libre |
|------|-----------|------------|-------------|------|---------------|

Conclure avec une recommandation :
- Combien de nodes/GPUs sont disponibles maintenant
- Quelle partition utiliser (P100 vs 3090)
- Batch size recommande selon la memoire disponible (P100: 16GB → BS=64, 3090: 24GB → BS=64)
