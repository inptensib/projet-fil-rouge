"""CLI entry point for single-GPU training via Docker or direct invocation.

Maps the YAML config keys to HYDRATrainer constructor parameters.

Usage:
    python scripts/run_train.py --data shared/dataset/products.jsonl
    python scripts/run_train.py --data shared/dataset/products.jsonl --n-samples 50000
    python scripts/run_train.py --config configs/hydra/default.yaml --data shared/dataset/products.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from rakuten.models.hydra.config import map_config
from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.common.data_loader import prepare_hierarchical_data, prepare_hierarchical_data_from_splits


def main():
    parser = argparse.ArgumentParser(description="HYDRA single-GPU training")
    parser.add_argument(
        "--config",
        default="configs/hydra/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a single JSONL dataset file (will sample + split)",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Directory containing train_split.jsonl, val_split.jsonl, test_split.jsonl",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of samples to load (default: 10000)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["local", "global", "nested"],
        help="Override variant from YAML config",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint_latest.pt to resume training",
    )
    args = parser.parse_args()

    if not args.data and not args.split_dir:
        parser.error("Provide either --data or --split-dir")
    if args.data and args.split_dir:
        parser.error("Provide only one of --data or --split-dir")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.variant:
        cfg["variant"] = args.variant

    seed = cfg.get("seed", 42)

    if args.split_dir:
        split_dir = Path(args.split_dir)
        train_path = split_dir / "train_split.jsonl"
        val_path = split_dir / "val_split.jsonl"
        test_path = split_dir / "test_split.jsonl"
        for p in (train_path, val_path, test_path):
            if not p.exists():
                print(f"ERROR: Split file not found: {p}")
                sys.exit(1)
        print(f"Loading from pre-split files in {split_dir} ...")
        data = prepare_hierarchical_data_from_splits(
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            n_samples=args.n_samples if args.n_samples != 10_000 else None,
            seed=seed,
            use_description=cfg.get("use_description", True),
            synthetic_policy=cfg.get("synthetic_policy", "keep_all"),
            normalize=cfg.get("normalize_text", False),
        )
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"ERROR: Data file not found: {data_path}")
            sys.exit(1)
        print(f"Loading {args.n_samples} samples from {data_path} ...")
        data = prepare_hierarchical_data(
            file_path=str(data_path),
            n_samples=args.n_samples,
            seed=seed,
            use_description=cfg.get("use_description", True),
            normalize=cfg.get("normalize_text", False),
        )

    trainer_kwargs = map_config(cfg)
    print(f"Starting training: variant={trainer_kwargs.get('variant', 'local')}, "
          f"encoder={trainer_kwargs.get('encoder_name', 'CamemBERT')}")

    trainer = HYDRATrainer(data=data, resume_from=args.resume, **trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
