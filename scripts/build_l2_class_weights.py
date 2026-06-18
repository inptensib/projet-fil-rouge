"""Build L2 effective-number class weights from training data.

Outputs:
    {save_dir}/l2_class_weights.pt
    {save_dir}/l2_class_counts.json
    {save_dir}/l2_frequency_buckets.json

Usage:
    python scripts/build_l2_class_weights.py \\
        --split-dir /path/to/dataset \\
        --save-dir models/flat_hydra_dual_expert \\
        --beta 0.9999 \\
        --max-weight 10.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _ROOT / "01_baselines"

for p in (_ROOT, _BASELINES_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.data_loader import prepare_hierarchical_data_from_splits  # noqa: E402
from src.dual_expert.losses import (  # noqa: E402
    build_effective_number_weights,
    save_class_weight_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build effective-number L2 class weights for Dual-Expert training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--split-dir", metavar="DIR",
                            help="Dir with train/val/test_split.jsonl")
    data_group.add_argument("--data", metavar="JSONL",
                            help="Single JSONL file (sample + auto-split)")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--save-dir", default="models/flat_hydra_dual_expert",
                        help="Output directory for weight artifacts")
    parser.add_argument("--beta", type=float, default=0.9999,
                        help="Effective-number beta parameter")
    parser.add_argument("--max-weight", type=float, default=10.0,
                        help="Maximum allowed class weight")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[ClassWeights] Loading training data...")
    if args.split_dir:
        split_dir = Path(args.split_dir)
        data = prepare_hierarchical_data_from_splits(
            train_path=str(split_dir / "train_split.jsonl"),
            val_path=str(split_dir / "val_split.jsonl"),
            test_path=str(split_dir / "test_split.jsonl"),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    else:
        from shared.data_loader import prepare_hierarchical_data
        data = prepare_hierarchical_data(
            file_path=args.data,
            n_samples=args.n_samples or 10_000,
            seed=args.seed,
        )

    labels_l2 = list(data["y_l2_train"])
    num_classes = data["num_classes_l2"]

    print(f"[ClassWeights] {len(labels_l2):,} training samples, {num_classes} L2 classes")
    print(f"[ClassWeights] Building with beta={args.beta}, max_weight={args.max_weight}")

    weights = build_effective_number_weights(
        labels_l2, num_classes, beta=args.beta, max_weight=args.max_weight
    )

    print(
        f"[ClassWeights] Weight stats: "
        f"min={weights.min():.4f}, max={weights.max():.4f}, "
        f"mean={weights.mean():.4f}"
    )

    save_class_weight_artifacts(weights, labels_l2, num_classes, args.save_dir)
    print("[ClassWeights] Done.")


if __name__ == "__main__":
    main()
