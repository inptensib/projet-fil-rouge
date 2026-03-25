"""Split du dataset Rakuten train.jsonl en train/val/test (70/15/15).

Stratégie deux passes pour gérer les fichiers volumineux :
  - Passe 1 : extraction des labels L2 en streaming (faible empreinte mémoire)
  - Split stratifié des indices via sklearn
  - Passe 2 : écriture des 3 fichiers JSONL en streaming
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

from sklearn.model_selection import train_test_split

from shared.data_loader import extract_level_2

logger = logging.getLogger(__name__)


def collect_labels(file_path: str) -> tuple[list[str], list[str], list[int]]:
    """Passe 1 : extraire les labels L1/L2 et les indices valides."""
    l1_labels: list[str] = []
    l2_labels: list[str] = []
    valid_indices: list[int] = []
    skipped = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0 and i % 1_000_000 == 0:
                logger.info("Passe 1 : %dM lignes lues...", i // 1_000_000)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            l2 = extract_level_2(record.get("producttype"))
            if l2 is None:
                skipped += 1
                continue

            l1 = str(record.get("typelabel", "Unknown")).strip() or "Unknown"
            l1_labels.append(l1)
            l2_labels.append(l2)
            valid_indices.append(i)

    logger.info(
        "Passe 1 terminée : %d échantillons valides, %d ignorés.",
        len(valid_indices),
        skipped,
    )
    return l2_labels, l1_labels, valid_indices


def compute_split_indices(
    labels: list[str],
    valid_indices: list[int],
    test_size: float,
    val_size: float,
    seed: int,
    min_class_count: int = 3,
) -> tuple[set[int], set[int], set[int]]:
    """Split stratifié en train/val/test. Regroupe les classes rares."""
    holdout_size = test_size + val_size

    # Regrouper les classes rares pour la stratification
    counts = Counter(labels)
    strat_labels = [
        lab if counts[lab] >= min_class_count else "__rare__" for lab in labels
    ]

    # Split 1 : train / (val+test)
    try:
        train_idx, holdout_idx, _, strat_holdout = train_test_split(
            valid_indices,
            strat_labels,
            test_size=holdout_size,
            stratify=strat_labels,
            random_state=seed,
        )
    except ValueError:
        logger.warning("Stratification échouée au split 1, fallback non-stratifié.")
        train_idx, holdout_idx, _, strat_holdout = train_test_split(
            valid_indices,
            strat_labels,
            test_size=holdout_size,
            random_state=seed,
        )

    # Split 2 : val / test (50/50 sur le holdout)
    val_fraction = val_size / holdout_size
    try:
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=1.0 - val_fraction,
            stratify=strat_holdout,
            random_state=seed,
        )
    except ValueError:
        logger.warning("Stratification échouée au split 2, fallback non-stratifié.")
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=1.0 - val_fraction,
            random_state=seed,
        )

    return set(train_idx), set(val_idx), set(test_idx)


def write_splits(
    file_path: str,
    output_dir: str,
    train_indices: set[int],
    val_indices: set[int],
    test_indices: set[int],
) -> dict[str, int]:
    """Passe 2 : écrire chaque ligne dans le bon fichier de sortie."""
    paths = {
        "train": os.path.join(output_dir, "train_split.jsonl"),
        "val": os.path.join(output_dir, "val_split.jsonl"),
        "test": os.path.join(output_dir, "test_split.jsonl"),
    }
    counts = {"train": 0, "val": 0, "test": 0}

    with (
        open(paths["train"], "w", encoding="utf-8") as f_train,
        open(paths["val"], "w", encoding="utf-8") as f_val,
        open(paths["test"], "w", encoding="utf-8") as f_test,
    ):
        with open(file_path, "r", encoding="utf-8") as f_in:
            for i, line in enumerate(f_in):
                if i > 0 and i % 1_000_000 == 0:
                    logger.info("Passe 2 : %dM lignes traitées...", i // 1_000_000)
                if i in train_indices:
                    f_train.write(line)
                    counts["train"] += 1
                elif i in val_indices:
                    f_val.write(line)
                    counts["val"] += 1
                elif i in test_indices:
                    f_test.write(line)
                    counts["test"] += 1

    for split, path in paths.items():
        logger.info("Écrit %s : %s (%d lignes)", split, path, counts[split])

    return counts


def log_statistics(
    counts: dict[str, int],
    l1_labels: list[str],
    l2_labels: list[str],
    train_indices: set[int],
    val_indices: set[int],
    test_indices: set[int],
    valid_indices: list[int],
) -> None:
    """Afficher les statistiques du split."""
    total = sum(counts.values())
    logger.info("=" * 60)
    logger.info("Statistiques du split")
    logger.info("=" * 60)
    for split, count in counts.items():
        pct = 100.0 * count / total if total > 0 else 0
        logger.info("  %-6s : %10d  (%.1f%%)", split, count, pct)
    logger.info("  %-6s : %10d", "total", total)

    # Classes uniques par split
    idx_to_pos = {idx: pos for pos, idx in enumerate(valid_indices)}
    for split_name, idx_set in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        positions = [idx_to_pos[idx] for idx in idx_set]
        unique_l1 = len({l1_labels[p] for p in positions})
        unique_l2 = len({l2_labels[p] for p in positions})
        logger.info("  %s — L1: %d classes, L2: %d classes", split_name, unique_l1, unique_l2)

    # Vérifier couverture L1 dans chaque split
    all_l1 = set(l1_labels)
    for split_name, idx_set in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        positions = [idx_to_pos[idx] for idx in idx_set]
        split_l1 = {l1_labels[p] for p in positions}
        missing = all_l1 - split_l1
        if missing:
            logger.warning(
                "%s : %d classes L1 manquantes : %s",
                split_name,
                len(missing),
                missing,
            )


def verify_splits(
    train_indices: set[int],
    val_indices: set[int],
    test_indices: set[int],
    valid_indices: list[int],
    test_size: float,
    val_size: float,
) -> None:
    """Vérifications d'intégrité post-split."""
    total = len(valid_indices)
    n_train = len(train_indices)
    n_val = len(val_indices)
    n_test = len(test_indices)

    assert train_indices.isdisjoint(val_indices), "Overlap train/val détecté"
    assert train_indices.isdisjoint(test_indices), "Overlap train/test détecté"
    assert val_indices.isdisjoint(test_indices), "Overlap val/test détecté"
    assert n_train + n_val + n_test == total, (
        f"Échantillons perdus : {n_train} + {n_val} + {n_test} != {total}"
    )

    train_pct = n_train / total
    val_pct = n_val / total
    test_pct = n_test / total
    expected_train = 1.0 - test_size - val_size

    assert abs(train_pct - expected_train) < 0.02, f"Train ratio {train_pct:.3f} hors tolérance"
    assert abs(val_pct - val_size) < 0.02, f"Val ratio {val_pct:.3f} hors tolérance"
    assert abs(test_pct - test_size) < 0.02, f"Test ratio {test_pct:.3f} hors tolérance"

    logger.info("Toutes les vérifications passées.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split du dataset Rakuten en train/val/test."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers le fichier JSONL source.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Répertoire de sortie (défaut : même répertoire que l'input).",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output_dir)
    logger.info("Ratios : train=%.0f%% / val=%.0f%% / test=%.0f%%",
                (1 - args.test_size - args.val_size) * 100,
                args.val_size * 100,
                args.test_size * 100)
    logger.info("Seed   : %d", args.seed)

    t0 = time.time()

    # Passe 1
    l2_labels, l1_labels, valid_indices = collect_labels(args.input)

    # Split
    train_indices, val_indices, test_indices = compute_split_indices(
        l2_labels,
        valid_indices,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        min_class_count=args.min_class_count,
    )

    # Vérifications
    verify_splits(
        train_indices, val_indices, test_indices,
        valid_indices, args.test_size, args.val_size,
    )

    # Passe 2
    counts = write_splits(
        args.input, args.output_dir,
        train_indices, val_indices, test_indices,
    )

    # Statistiques
    log_statistics(
        counts, l1_labels, l2_labels,
        train_indices, val_indices, test_indices,
        valid_indices,
    )

    elapsed = time.time() - t0
    logger.info("Terminé en %.1f secondes.", elapsed)


if __name__ == "__main__":
    main()
