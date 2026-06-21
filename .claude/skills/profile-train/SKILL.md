---
name: profile-train
description: Profile un training run PyTorch (GPU, memoire, throughput, bottlenecks). Utiliser pour optimiser avant un run a grande echelle sur le cluster.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [script_or_log_path]
---

# Profile Train

Profile et optimise les performances d'un training run HYDRA.

## Arguments

- `$0` : chemin du script de training ou du fichier de log a analyser

## Taches

### 1. Analyse statique du code
- Lire le script de training (`$0` ou `rakuten/models/hydra/trainer.py` par defaut)
- Identifier : batch_size, max_length, nombre de parametres du modele, gradient accumulation
- Estimer la memoire GPU requise : `params * 4 bytes (fp32) * 3 (model + grads + optimizer)`

### 2. Profiling GPU (si nvidia-smi disponible)
```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

### 3. Analyse des logs de training (si fichier log fourni)
- Extraire : temps par batch, temps par epoque, loss progression
- Calculer : throughput (samples/sec), temps estime pour le dataset complet
- Detecter : stagnation de loss, spikes anormaux

### 4. Bottleneck detection
- Data loading vs forward pass vs backward pass (ratio ideal : <10% data loading)
- Verifier si `num_workers` du DataLoader est suffisant
- Verifier si `pin_memory=True` est active
- Verifier mixed precision (fp16/bf16) si GPU le supporte

### 5. Recommandations multi-GPU
- Calculer la taille optimale de batch par GPU
- Estimer le speedup lineaire attendu vs observe
- Verifier la config NCCL (backend, timeout) si training distribue actif

> **Note** : le pipeline DDP complet est archive via le tag git
> `archive/distributed-pipeline` et hors livrable.

## Output

Rapport markdown avec :
- Tableau : metrique | valeur | recommandation
- Estimation : temps total pour 8M samples sur la config actuelle
- Top 3 optimisations par impact attendu
