#!/usr/bin/env python3
"""Précalcule results/class_metadata.json : support test par classe + parent L1 pour L2.

Usage:
    python3 scripts/build_class_metadata.py
    python3 scripts/build_class_metadata.py --dataset-dir /path/to/splits
"""
import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET_DIR = str(Path.home() / "fil_rouge" / "dataset")


def main():
    parser = argparse.ArgumentParser(description="Génère class_metadata.json")
    parser.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET_DIR,
        help="Répertoire des splits (train_split.jsonl, val_split.jsonl, test_split.jsonl)",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    # Mapping nom_L2 → nom_L1 depuis train + val (exhaustif pour couvrir toutes les classes)
    l2_to_l1: dict[str, str] = {}
    for fname in ("train_split.jsonl", "val_split.jsonl"):
        fpath = dataset_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath} introuvable")
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                pt = item["producttype"]
                l2_name = pt[0] if isinstance(pt, list) else str(pt)
                l1_name = item.get("typelabel", "")
                if l2_name and l2_name not in l2_to_l1:
                    l2_to_l1[l2_name] = l1_name

    # Support sur le jeu de test
    l1_support: Counter = Counter()
    l2_support: Counter = Counter()
    test_path = dataset_dir / "test_split.jsonl"
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                l1_name = item.get("typelabel", "")
                pt = item["producttype"]
                l2_name = pt[0] if isinstance(pt, list) else str(pt)
                if l1_name:
                    l1_support[l1_name] += 1
                if l2_name:
                    l2_support[l2_name] += 1
                    if l2_name not in l2_to_l1:
                        l2_to_l1[l2_name] = l1_name
    else:
        print(f"  [SKIP] {test_path} introuvable")

    all_l2 = set(l2_to_l1) | set(l2_support)
    meta = {
        "l1": {
            name: {"support_test": l1_support[name]}
            for name in sorted(l1_support)
        },
        "l2": {
            name: {
                "parent_l1": l2_to_l1.get(name, ""),
                "support_test": l2_support.get(name, 0),
            }
            for name in sorted(all_l2)
        },
    }

    out = RESULTS_DIR / "class_metadata.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    print(f"  L1 : {len(meta['l1'])} classes (test)")
    print(f"  L2 : {len(meta['l2'])} classes (train+val+test)")


if __name__ == "__main__":
    main()
