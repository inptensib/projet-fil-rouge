"""Distributed Data Parallel (DDP) utilities.

Provides setup/teardown for PyTorch DDP using file-based rendezvous,
suitable for multi-node SLURM environments with NCCL backend.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def setup_ddp(rendezvous_dir: str = None) -> tuple[int, int]:
    """Initialize DDP process group via NFS file-based rendezvous.

    Reads from environment variables set by the launch script:
        GLOBAL_RANK  -- unique rank across all GPUs (0..WORLD_SIZE-1)
        LOCAL_RANK   -- GPU index on this node (0 or 1)
        WORLD_SIZE   -- total number of GPU processes

    Each process should see a single GPU via CUDA_VISIBLE_DEVICES,
    so ``torch.cuda.set_device(0)`` is always correct.

    Args:
        rendezvous_dir: Directory for the rendezvous file.
            Defaults to the parent of this file's directory.

    Returns:
        Tuple of (rank, world_size).
    """
    rank = int(os.environ["GLOBAL_RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    if rendezvous_dir is None:
        rendezvous_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rendezvous_file = f"file://{rendezvous_dir}/rendezvous_hydra"

    hostname = os.environ.get("HOSTNAME", "unknown")
    print(
        f"[Rank {rank} | local GPU {local_rank}] on {hostname} "
        f"-- initializing DDP (world_size={world_size})..."
    )

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=rendezvous_file,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(hours=3),
        )
        print(f"[Rank {rank} | local GPU {local_rank}] DDP initialized.")
    else:
        print(f"[Rank {rank} | local GPU {local_rank}] DDP bypassed (single process).")

    torch.cuda.set_device(0)
    return rank, world_size


def cleanup_ddp() -> None:
    """Destroy the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int) -> None:
    """Print a message only from rank 0.

    Args:
        msg: Message to print.
        rank: Current process rank.
    """
    if rank == 0:
        print(msg)
