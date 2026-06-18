"""CLI entry point for HLPD training — local CPU/GPU or SLURM cluster.

Usage:
    # Depuis les splits pré-générés (recommandé pour le dataset complet)
    python scripts/run_hlpd.py \\
        --split-dir /path/to/dataset/ \\
        --config 02b_hlpd/configs/hlpd_3090.yaml

    # Smoke test local (N=1000 samples, 1 epoch)
    python scripts/run_hlpd.py \\
        --split-dir /path/to/dataset/ \\
        --config 02b_hlpd/configs/hlpd_3090.yaml \\
        --n-samples 1000 \\
        --epochs 1

    # Reprendre un training interrompu
    python scripts/run_hlpd.py \\
        --split-dir /path/to/dataset/ \\
        --config 02b_hlpd/configs/hlpd_3090.yaml \\
        --resume models/hlpd/checkpoint_latest.pt
"""
# NOTE : les configs YAML restent dans 02b_hlpd/configs/ (non migré).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rakuten.models.hlpd import HLPDConfig, HLPDTrainer  # noqa: E402
from rakuten.common.data_loader import (                   # noqa: E402
    prepare_hierarchical_data,
    prepare_hierarchical_data_from_splits,
)

_DEFAULT_CONFIG = _ROOT / "02b_hlpd" / "configs" / "hlpd_3090.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HLPD — Hierarchical Label Prediction Decoder training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Source de données ---
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data",
        metavar="JSONL",
        help="Chemin vers products.jsonl (sample + split automatique)",
    )
    data_group.add_argument(
        "--split-dir",
        metavar="DIR",
        help="Dossier contenant train_split.jsonl, val_split.jsonl, test_split.jsonl",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Nombre de samples à charger (None = dataset complet)",
    )

    # --- Config ---
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Chemin vers le fichier YAML de config",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Chemin vers checkpoint_latest.pt pour reprendre un training",
    )

    # --- Overrides ---
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--save-dir",    default=None)

    args = parser.parse_args()

    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.save_dir is not None:
        overrides["save_dir"] = args.save_dir

    config = HLPDConfig.from_yaml(args.config, **overrides)

    # --- Chargement des données ---
    if args.split_dir:
        split_dir = Path(args.split_dir)
        for fname in ("train_split.jsonl", "val_split.jsonl", "test_split.jsonl"):
            if not (split_dir / fname).exists():
                print(f"[ERROR] Fichier manquant : {split_dir / fname}")
                sys.exit(1)
        sample_msg = f"{args.n_samples:,} samples" if args.n_samples else "dataset complet"
        print(f"[Data] Chargement depuis {split_dir} — {sample_msg}")
        data = prepare_hierarchical_data_from_splits(
            train_path=str(split_dir / "train_split.jsonl"),
            val_path=str(split_dir / "val_split.jsonl"),
            test_path=str(split_dir / "test_split.jsonl"),
            n_samples=args.n_samples,
            seed=config.seed,
        )
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"[ERROR] Fichier introuvable : {data_path}")
            sys.exit(1)
        n = args.n_samples or 10_000
        print(f"[Data] Chargement de {n:,} samples depuis {data_path}")
        data = prepare_hierarchical_data(
            file_path=str(data_path),
            n_samples=n,
            seed=config.seed,
        )

    print(
        f"[Config] encoder={config.encoder} | layers={config.num_decoder_layers} | "
        f"nhead={config.nhead} | batch={config.batch_size} | "
        f"lr={config.lr} | epochs={config.epochs} | "
        f"precision={config.precision} | grad_ckpt={config.gradient_checkpointing}"
    )

    trainer = HLPDTrainer(data=data, config=config, resume_from=args.resume)
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
