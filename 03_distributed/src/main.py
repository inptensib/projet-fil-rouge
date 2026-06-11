"""DDP entry point for SLURM-launched HYDRA training.

Called by launch_node.sh as: python -m src.main --variant local --encoder CamemBERT
Hyperparamètres : --config <yaml> (02_hydra/configs/*.yaml), surchargés par
les flags CLI explicites. Priorité : CLI > YAML > défauts du trainer.

Environment variables (set by launch_node.sh):
    GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE, CUDA_VISIBLE_DEVICES
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is on sys.path for shared/ imports
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .ddp_utils import cleanup_ddp, setup_ddp
from .hydra_imports import load_hydra_module
from .trainer_ddp import HYDRATrainerDDP

# Clés HYDRATrainer → équivalents HYDRATrainerDDP
_DDP_KEY_ALIASES = {
    "max_grad_norm": "grad_clip_max_norm",
    "check_hierarchy": "check_hierarchy_consistency",
}

# Flags CLI qui priment sur le YAML quand fournis explicitement
_CLI_OVERRIDE_KEYS = ("batch_size", "epochs", "lr", "save_dir")


def build_trainer_kwargs(cfg: dict | None, args: argparse.Namespace) -> dict:
    """Fusionne config YAML et CLI en kwargs HYDRATrainerDDP.

    Priorité : CLI explicite > YAML > défauts du constructeur. Les clés YAML
    sans équivalent DDP (variant, encoder_name, fp16, seed...) sont écartées.
    """
    kwargs: dict = {}
    if cfg:
        mapped = load_hydra_module("config").map_config(cfg)
        for src_key, dst_key in _DDP_KEY_ALIASES.items():
            if src_key in mapped:
                mapped[dst_key] = mapped.pop(src_key)
        accepted = set(inspect.signature(HYDRATrainerDDP.__init__).parameters) - {
            "self", "shard_dir", "rank", "world_size", "device",
        }
        kwargs = {k: v for k, v in mapped.items() if k in accepted}

    for cli_key in _CLI_OVERRIDE_KEYS:
        value = getattr(args, cli_key)
        if value is not None:
            kwargs[cli_key] = value

    return kwargs


def resolve_variant_encoder(cfg: dict | None, args: argparse.Namespace) -> tuple[str, str]:
    """Variante et encodeur : CLI explicite > YAML > défauts historiques."""
    variant = args.variant or (cfg or {}).get("variant") or "local"
    encoder = args.encoder
    if encoder is None and cfg and "encoder" in cfg:
        encoder_map = load_hydra_module("config").ENCODER_MAP
        encoder = encoder_map.get(cfg["encoder"], cfg["encoder"])
    return variant, encoder or "CamemBERT"


def main():
    parser = argparse.ArgumentParser(description="HYDRA DDP training entry point")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (02_hydra/configs/*.yaml) — les flags "
                             "CLI explicites priment sur ses valeurs")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["local", "global", "nested"])
    parser.add_argument("--encoder", type=str, default=None,
                        choices=["CamemBERT", "XLM-RoBERTa", "mBERT"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--shard_dir", type=str, default="shards")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to single JSONL file (will sample + split)")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory with train_split.jsonl, val_split.jsonl, test_split.jsonl")
    parser.add_argument("--n_samples", type=int, default=10_000,
                        help="Number of samples to load")
    args = parser.parse_args()

    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    variant, encoder = resolve_variant_encoder(cfg, args)
    trainer_kwargs = build_trainer_kwargs(cfg, args)

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
            **trainer_kwargs,
        )
        trainer.run_full_benchmark(
            encoders=[encoder],
            variants=[variant],
        )
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
