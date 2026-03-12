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

from shared.data_loader import prepare_hierarchical_data
from src.trainer import HYDRATrainer


def _map_config(cfg: dict) -> dict:
    """Map YAML config keys to HYDRATrainer constructor kwargs."""
    kwargs = {}

    # Direct mappings (same name)
    for key in ("variant", "batch_size", "epochs", "lr", "weight_decay",
                "max_length", "seed"):
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

    if "gradient_clip" in cfg:
        kwargs["max_grad_norm"] = cfg["gradient_clip"]

    if "level_weights" in cfg and len(cfg["level_weights"]) >= 1:
        kwargs["level_weight_l1"] = cfg["level_weights"][0]

    if "loss" in cfg:
        kwargs["use_focal"] = cfg["loss"] == "focal"

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
        required=True,
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of samples to load (default: 10000)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading {args.n_samples} samples from {data_path} ...")
    data = prepare_hierarchical_data(
        file_path=str(data_path),
        n_samples=args.n_samples,
        seed=cfg.get("seed", 42),
    )

    trainer_kwargs = _map_config(cfg)
    print(f"Starting training: variant={trainer_kwargs.get('variant', 'local')}, "
          f"encoder={trainer_kwargs.get('encoder_name', 'CamemBERT')}")

    trainer = HYDRATrainer(data=data, **trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
