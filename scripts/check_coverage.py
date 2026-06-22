"""Vérifie que les splits train/val/test couvrent toutes les catégories de la taxonomie.

Deux modes :
  1. Vérification uniquement (défaut) — compare taxonomie vs splits existants
  2. Re-split (--resplit) — recrée train/val/test depuis le dataset brut

Usage :
    # Vérification
    python scripts/check_coverage.py \\
        --taxonomy product_classification_taxonomy.csv \\
        --split-dir /path/to/splits

    # Re-split 70/15/15
    python scripts/check_coverage.py \\
        --taxonomy product_classification_taxonomy.csv \\
        --split-dir /path/to/splits \\
        --resplit --input /path/to/products.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def load_taxonomy_l2(taxonomy_path: Path) -> set[str]:
    """Retourne l'ensemble des noms L2 définis dans la taxonomie."""
    l2s: set[str] = set()
    with open(taxonomy_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["type_de_produit"].strip()
            if name:
                l2s.add(name)
    return l2s


def load_split_l2(jsonl_path: Path) -> Counter:
    """Retourne un Counter {l2_name: count} pour un fichier JSONL."""
    counts: Counter = Counter()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pt = rec.get("producttype")
            if isinstance(pt, list) and pt:
                l2 = str(pt[0]).strip()
            elif isinstance(pt, str):
                l2 = pt.strip()
            else:
                continue
            if l2 and l2.lower() not in ("", "none", "nan", "unknown"):
                counts[l2] += 1
    return counts


def check_coverage(taxonomy_path: Path, split_dir: Path) -> int:
    """Compare taxonomie vs splits. Retourne le nombre de L2 manquantes."""
    print(f"Taxonomie : {taxonomy_path}")
    taxonomy_l2 = load_taxonomy_l2(taxonomy_path)
    print(f"  → {len(taxonomy_l2)} catégories L2 définies\n")

    split_files = {
        "train": split_dir / "train_split.jsonl",
        "val":   split_dir / "val_split.jsonl",
        "test":  split_dir / "test_split.jsonl",
    }

    split_counts: dict[str, Counter] = {}
    for name, path in split_files.items():
        if not path.exists():
            print(f"  [WARN] {path} introuvable — ignoré")
            continue
        print(f"Chargement {name} : {path}")
        split_counts[name] = load_split_l2(path)
        print(f"  → {sum(split_counts[name].values()):,} lignes, {len(split_counts[name])} L2 uniques")

    combined_l2: set[str] = set()
    for c in split_counts.values():
        combined_l2.update(c.keys())

    print(f"\nUnion train+val+test : {len(combined_l2)} L2 uniques")

    # Catégories de la taxonomie absentes du dataset
    missing_from_dataset = taxonomy_l2 - combined_l2
    # Catégories dans le dataset mais absentes de la taxonomie
    extra_in_dataset = combined_l2 - taxonomy_l2

    print(f"\n{'='*60}")
    print(f"L2 dans taxonomie mais ABSENTES des splits : {len(missing_from_dataset)}")
    if missing_from_dataset:
        for name in sorted(missing_from_dataset)[:30]:
            print(f"  - {name}")
        if len(missing_from_dataset) > 30:
            print(f"  ... et {len(missing_from_dataset) - 30} autres")

    print(f"\nL2 dans les splits mais ABSENTES de la taxonomie : {len(extra_in_dataset)}")
    if extra_in_dataset:
        for name in sorted(extra_in_dataset)[:10]:
            print(f"  - {name}")
        if len(extra_in_dataset) > 10:
            print(f"  ... et {len(extra_in_dataset) - 10} autres")

    # Couverture par split
    print(f"\n{'='*60}")
    print("Couverture par split (% des L2 taxonomie présentes) :")
    for name, counts in split_counts.items():
        present = taxonomy_l2 & set(counts.keys())
        pct = 100 * len(present) / len(taxonomy_l2)
        missing_here = taxonomy_l2 - set(counts.keys())
        print(f"  {name:5s} : {len(present):5d}/{len(taxonomy_l2)} ({pct:.1f}%) — {len(missing_here)} absentes")

    print(f"{'='*60}")
    if not missing_from_dataset:
        print("OK — tous les L2 de la taxonomie sont présents dans les splits combinés.")
    else:
        print(f"ATTENTION — {len(missing_from_dataset)} L2 de la taxonomie ne sont dans aucun split.")
        print("  → Ces catégories n'ont aucun exemplaire dans le dataset brut.")
        print("  → Un re-split ne changera rien pour elles.")

    return len(missing_from_dataset)


def resplit(input_path: Path, split_dir: Path, seed: int, min_class_count: int) -> None:
    """Lance scripts/split_dataset.py en sous-processus."""
    import subprocess
    cmd = [
        sys.executable, str(Path(__file__).parent / "split_dataset.py"),
        "--input", str(input_path),
        "--output-dir", str(split_dir),
        "--test-size", "0.15",
        "--val-size", "0.15",
        "--seed", str(seed),
        "--min-class-count", str(min_class_count),
    ]
    print(f"\nLancement du re-split :\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie la couverture taxonomie/splits et optionnellement re-split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--taxonomy", default="product_classification_taxonomy.csv",
                        help="Fichier CSV de taxonomie")
    parser.add_argument("--split-dir", required=True,
                        help="Répertoire contenant train/val/test_split.jsonl")
    parser.add_argument("--resplit", action="store_true",
                        help="Re-créer les splits depuis le dataset brut après vérification")
    parser.add_argument("--input", default=None,
                        help="Dataset brut JSONL (requis si --resplit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=3,
                        help="Seuil pour regrouper les classes rares lors de la stratification")
    args = parser.parse_args()

    taxonomy_path = Path(args.taxonomy)
    split_dir = Path(args.split_dir)

    if not taxonomy_path.exists():
        print(f"Taxonomie introuvable : {taxonomy_path}", file=sys.stderr)
        sys.exit(1)

    n_missing = check_coverage(taxonomy_path, split_dir)

    if args.resplit:
        if not args.input:
            print("\n--resplit requiert --input /path/to/products.jsonl", file=sys.stderr)
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Dataset brut introuvable : {input_path}", file=sys.stderr)
            sys.exit(1)
        if n_missing == 0:
            print("\nTous les L2 sont déjà couverts — re-split non nécessaire.")
        else:
            print(f"\n{n_missing} L2 manquantes dans les splits existants.")
            print("Note : si ces L2 sont absentes du dataset brut, le re-split ne les ajoutera pas.")
        resplit(input_path, split_dir, args.seed, args.min_class_count)

    sys.exit(0 if n_missing == 0 else 1)


if __name__ == "__main__":
    main()
