# 03_production -- Distributed Training

Distributed Data Parallel (DDP) training for the HYDRA hierarchical classification models, designed for multi-node SLURM clusters with NCCL/RoCE interconnect.

## Structure

```
03_production/
  src/
    ddp_utils.py         -- DDP setup/teardown, rank-aware printing
    data_sharding.py     -- Data sharding for distributed training (rank 0)
    trainer_ddp.py       -- DDP trainer with gradient clipping, MLflow, error tracking
    error_analysis.py    -- Per-sample and cross-epoch error analysis
    predict.py           -- Production inference with DDP model loading support
  scripts/
    launch_distributed.sh -- Multi-node launcher (detects GPUs, submits SLURM jobs)
    launch_node.sh        -- Per-node SLURM script (configures NCCL, launches processes)
```

## Key improvements over the original

- **Gradient clipping** (`max_norm=1.0`) for training stability.
- **Hierarchy-aware metrics** (constrained F1, HCR) from `shared.metrics`.
- **Clean imports**: models from `02_hydra.src.models`, utilities from `shared/`.
- **Merged error analysis**: combines per-sample and global error reports.
- **Batch prediction**: `predict_batch()` for efficient multi-sample inference.

## Usage

```bash
# From the project root:
cd 03_production
./scripts/launch_distributed.sh local CamemBERT 4       # 4 P100 nodes
./scripts/launch_distributed.sh nested XLM-RoBERTa 1 rtx3090  # 1 RTX 3090 node
```

## Requirements

- SLURM cluster with GPU partitions (P100 and/or 3090)
- RoCE (RDMA over Converged Ethernet) for multi-node NCCL
- Python environment at `/home/infres/tandriam-25/fil_rouge/venv/`
- HuggingFace model cache pre-populated (nodes run offline)
