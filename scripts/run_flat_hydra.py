"""CLI entry point for FlatHydra training — CPU local ou GPU cluster.

Usage:
    # Entraînement local CPU sur N samples réels (via Docker)
    #   docker compose run --rm --profile baselines flat-hydra-cpu
    #   N_SAMPLES=2000 N_EPOCHS=3 docker compose run --rm --profile baselines flat-hydra-cpu

    # Depuis un seul fichier JSONL (sample + split automatique)
    python scripts/run_flat_hydra.py \
        --data shared/dataset/products.jsonl \
        --n-samples 1000 \
        --epochs 2 \
        --batch-size 16

    # Depuis des splits pré-générés (recommandé pour le dataset complet)
    python scripts/run_flat_hydra.py \
        --split-dir /path/to/dataset/ \
        --config 01_baselines/configs/flat_hydra.yaml

    # Reprendre un training interrompu
    python scripts/run_flat_hydra.py \
        --data shared/dataset/products.jsonl \
        --resume models/flat_hydra_cpu/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _ROOT / "01_baselines"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))

from src.flat_trainer import FlatHydraConfig, FlatHydraTrainer  # noqa: E402
from shared.data_loader import (                                  # noqa: E402
    prepare_hierarchical_data,
    prepare_hierarchical_data_from_splits,
)

_DEFAULT_CONFIG = _BASELINES_ROOT / "configs" / "flat_hydra.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlatHydra baseline — entraînement CPU local ou GPU cluster",
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
        default=10_000,
        help="Nombre de samples à charger depuis --data",
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

    # --- Overrides config (utiles pour les tests CPU locaux) ---
    parser.add_argument("--epochs",      type=int,   default=None, help="Nombre d'epochs")
    parser.add_argument("--batch-size",  type=int,   default=None, help="Batch size")
    parser.add_argument("--lr",          type=float, default=None, help="Learning rate AdamW")
    parser.add_argument("--dropout",     type=float, default=None, help="Dropout")
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean", "max", "last"],
        default=None,
        help="Stratégie de pooling",
    )
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Désactiver la projection d'embedding",
    )
    parser.add_argument(
        "--use-focal",
        action="store_true",
        help="Utiliser FocalLoss au lieu de CrossEntropy",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Répertoire de sauvegarde des artefacts",
    )

    args = parser.parse_args()

    # --- Construction des overrides ---
    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.pooling is not None:
        overrides["pooling"] = args.pooling
    if args.no_project:
        overrides["project_embedding"] = False
    if args.use_focal:
        overrides["use_focal"] = True
    if args.save_dir is not None:
        overrides["save_dir"] = args.save_dir

    config = FlatHydraConfig.from_yaml(args.config, **overrides)

    # --- Chargement des données ---
    if args.split_dir:
        split_dir = Path(args.split_dir)
        for fname in ("train_split.jsonl", "val_split.jsonl", "test_split.jsonl"):
            if not (split_dir / fname).exists():
                print(f"[ERROR] Fichier manquant : {split_dir / fname}")
                sys.exit(1)
        print(f"[Data] Chargement depuis les splits : {split_dir}")
        data = prepare_hierarchical_data_from_splits(
            train_path=str(split_dir / "train_split.jsonl"),
            val_path=str(split_dir / "val_split.jsonl"),
            test_path=str(split_dir / "test_split.jsonl"),
            seed=config.seed,
        )
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"[ERROR] Fichier introuvable : {data_path}")
            sys.exit(1)
        print(f"[Data] Chargement de {args.n_samples} samples depuis {data_path}")
        data = prepare_hierarchical_data(
            file_path=str(data_path),
            n_samples=args.n_samples,
            seed=config.seed,
        )

    print(
        f"[Config] encoder={config.encoder} | pooling={config.pooling} | "
        f"project={config.project_embedding} | dropout={config.dropout} | "
        f"lr={config.lr} | batch={config.batch_size} | "
        f"epochs={config.epochs} | fp16={config.fp16}"
    )

    trainer = FlatHydraTrainer(data=data, config=config, resume_from=args.resume)
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
