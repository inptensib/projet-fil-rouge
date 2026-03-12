# 03_production -- Training Distribue (DDP)

Training Distributed Data Parallel des modeles HYDRA sur cluster SLURM multi-noeuds (NCCL/RoCE).

## Usage

```bash
./scripts/launch_distributed.sh local CamemBERT 4           # 4 noeuds P100
./scripts/launch_distributed.sh nested XLM-RoBERTa 1 rtx3090  # 1 noeud RTX 3090
```

## Apports

- **Gradient clipping** (`max_norm=1.0`) pour la stabilite
- **Metriques hierarchiques** (Constrained F1, HCR) via `shared.metrics`
- **Error tracking** inter-epoques pour detecter le label noise
- **Batch prediction** pour l'inference multi-echantillons

## Structure

```
03_production/
  src/
    main.py               # Point d'entree CLI pour le training DDP
    ddp_utils.py          # Setup/teardown DDP, print rank-aware
    data_sharding.py      # Sharding des donnees (rank 0)
    trainer_ddp.py        # Trainer DDP + MLflow
    error_analysis.py     # Analyse d'erreurs par sample et cross-epochs
    predict.py            # Inference avec chargement de modeles DDP
  scripts/
    launch_distributed.sh # Lanceur multi-noeuds (detection GPUs, soumission SLURM)
    launch_node.sh        # Script SLURM par noeud (config NCCL)
```

## Prerequis

- Cluster SLURM avec partitions GPU (P100 / RTX 3090)
- RoCE pour NCCL multi-noeuds
- Environnement Python (voir `scripts/setup_cluster.sh`)
- Cache HuggingFace pre-charge (noeuds offline)
