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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--shard_dir", type=str, default="shards")
    parser.add_argument("--save_dir", type=str, default="models")
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device("cuda", 0)

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
