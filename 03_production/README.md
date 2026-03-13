# 03_production -- Training Distribué (DDP)

Training Distributed Data Parallel des modèles HYDRA sur cluster SLURM multi-nœuds (NCCL/RoCE).

## Usage

```bash
# Depuis la racine du projet :
./03_production/scripts/launch_distributed.sh local CamemBERT 4           # 4 nœuds P100
./03_production/scripts/launch_distributed.sh nested XLM-RoBERTa 1 rtx3090  # 1 nœud RTX 3090
```

## Apports

- **Gradient clipping** (`max_norm=1.0`) pour la stabilité
- **Gradient accumulation** avec `no_sync()` pour réduire l'overhead NCCL
- **Validation dual-speed** : Sniper (~15min, loss + ReduceLROnPlateau) et Audit (~1h, F1 complet + hard examples)
- **MLflow** : tracking temps réel (loss, LR, métriques), tableaux d'erreurs, artefacts modèle (rank 0)
- **Métriques hiérarchiques** (Constrained F1, HCR) via `shared.metrics`
- **Error tracking** inter-époques pour détecter le label noise (persistance cross-epochs)
- **Batch prediction** pour l'inférence multi-échantillons

## Structure

```
03_production/
  src/
    main.py               # Point d'entrée CLI pour le training DDP
    ddp_utils.py          # Setup/teardown DDP, print rank-aware
    data_sharding.py      # Sharding des données (rank 0)
    trainer_ddp.py        # Trainer DDP + MLflow
    error_analysis.py     # Analyse d'erreurs par sample et cross-epochs
    predict.py            # Inférence avec chargement de modèles DDP
  scripts/
    launch_distributed.sh # Lanceur multi-nœuds (détection GPUs, soumission SLURM)
    launch_node.sh        # Script SLURM par nœud (config NCCL)
```

## Prérequis

- Cluster SLURM avec partitions GPU (P100 / RTX 3090)
- RoCE pour NCCL multi-nœuds
- Environnement Python (voir `scripts/setup_cluster.sh`)
- Cache HuggingFace pré-chargé (nœuds offline)
