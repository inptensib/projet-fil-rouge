"""CLI entry point for single-GPU training via Docker or direct invocation.

Maps the YAML config keys to HYDRATrainer constructor parameters.

Usage:
    python scripts/run_train.py --data shared/dataset/products.jsonl
    python scripts/run_train.py --data shared/dataset/products.jsonl --n-samples 50000
    python scripts/run_train.py --config 02_hydra/configs/default.yaml --data shared/dataset/products.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / "02_hydra"))

from src.trainer import HYDRATrainer

from shared.data_loader import prepare_hierarchical_data, prepare_hierarchical_data_from_splits


def _map_config(cfg: dict) -> dict:
    """Map YAML config keys to HYDRATrainer constructor kwargs."""
    kwargs = {}

    # Direct mappings (same name)
    for key in ("variant", "batch_size", "epochs", "lr", "weight_decay",
                "max_length", "seed", "condition_l2_on_l1", "check_hierarchy",
                "save_dir", "use_focal", "focal_gamma",
                "gradient_accumulation_steps", "warmup_ratio", "fp16",
                "experiment_suffix"):
        if key in cfg:
            kwargs[key] = cfg[key]

    # Renamed mappings
    if "encoder" in cfg:
        # Config uses HF model name; trainer expects our key (CamemBERT, etc.)
        encoder_map = {
            "camembert-base": "CamemBERT",
            "xlm-roberta-base": "XLM-RoBERTa",
            "bert-base-multilingual-cased": "mBERT",
        }
        kwargs["encoder_name"] = encoder_map.get(cfg["encoder"], cfg["encoder"])

    for src, dst in [("gradient_clip", "max_grad_norm"), ("max_grad_norm", "max_grad_norm")]:
        if src in cfg:
            kwargs[dst] = cfg[src]

    if "level_weights" in cfg and len(cfg["level_weights"]) >= 1:
        kwargs["level_weight_l1"] = cfg["level_weights"][0]

    # Note: dropout and hidden_dim are model-level params handled internally
    # by HYDRATrainer via create_hydra_model, not constructor kwargs.

    return kwargs


def main():
    parser = argparse.ArgumentParser(description="HYDRA single-GPU training")
    parser.add_argument(
        "--config",
        default="02_hydra/configs/default.yaml",
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
        )

    trainer_kwargs = _map_config(cfg)
    print(f"Starting training: variant={trainer_kwargs.get('variant', 'local')}, "
          f"encoder={trainer_kwargs.get('encoder_name', 'CamemBERT')}")

    trainer = HYDRATrainer(data=data, resume_from=args.resume, **trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
