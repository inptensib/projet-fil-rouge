"""DDP entry point for SLURM-launched HYDRA training.

Called by launch_node.sh as: python -m src.main --variant local --encoder CamemBERT

Environment variables (set by launch_node.sh):
    GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE, CUDA_VISIBLE_DEVICES
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path for shared/ imports
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_HYDRA_ROOT = str(Path(__file__).resolve().parents[2] / "02_hydra")
if _HYDRA_ROOT not in sys.path:
    sys.path.insert(0, _HYDRA_ROOT)

from .ddp_utils import setup_ddp, cleanup_ddp
from .trainer_ddp import HYDRATrainerDDP


def main():
    parser = argparse.ArgumentParser(description="HYDRA DDP training entry point")
    parser.add_argument("--variant", type=str, default="local",
                        choices=["local", "global", "nested"])
    parser.add_argument("--encoder", type=str, default="CamemBERT",
                        choices=["CamemBERT", "XLM-RoBERTa", "mBERT"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--shard_dir", type=str, default="shards")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to single JSONL file (will sample + split)")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory with train_split.jsonl, val_split.jsonl, test_split.jsonl")
    parser.add_argument("--n_samples", type=int, default=10_000,
                        help="Number of samples to load")
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device("cuda", 0)

    # Rank 0 prepares shards before training
    if rank == 0 and (args.data or args.split_dir):
        if args.split_dir:
            from .data_sharding import prepare_shards_from_splits
            split_dir = Path(args.split_dir)
            prepare_shards_from_splits(
                train_path=str(split_dir / "train_split.jsonl"),
                val_path=str(split_dir / "val_split.jsonl"),
                test_path=str(split_dir / "test_split.jsonl"),
                world_size=world_size,
                shard_dir=args.shard_dir,
                n_samples=args.n_samples if args.data else None,
            )
        elif args.data:
            from .data_sharding import prepare_shards
            prepare_shards(
                filename=args.data,
                n_samples=args.n_samples,
                world_size=world_size,
                shard_dir=args.shard_dir,
            )

    # Wait for shards to be ready
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    try:
        trainer = HYDRATrainerDDP(
            shard_dir=args.shard_dir,
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
        )
        trainer.run_full_benchmark(
            encoders=[args.encoder],
            variants=[args.variant],
        )
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
