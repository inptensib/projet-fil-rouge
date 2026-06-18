"""Evaluate top-K L2 recall on validation or test set.

Loads a trained FlatHydra checkpoint, runs inference with hierarchy-aware
scoring, and computes recall@K for K in {1, 3, 5, 10, 20} globally and
per frequency bucket.

Outputs:
    {report_dir}/topk_recall_report.json

Usage:
    python scripts/evaluate_topk.py \\
        --checkpoint models/flat_hydra_dual_expert \\
        --split-dir /path/to/dataset \\
        --split val \\
        --report-dir models/flat_hydra_dual_expert \\
        --top-k 20 \\
        --hierarchy-alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _ROOT / "01_baselines"

for p in (_ROOT, _BASELINES_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.data_loader import (  # noqa: E402
    HierarchicalProductDataset,
    prepare_hierarchical_data_from_splits,
)
from src.dual_expert.candidate_generator import get_topk_l2_candidates  # noqa: E402
from src.dual_expert.hierarchy_scorer import make_l2_to_l1_tensor  # noqa: E402
from src.dual_expert.topk_metrics import (  # noqa: E402
    build_topk_recall_report,
    save_topk_recall_report,
)
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-K L2 recall evaluation for FlatHydra Domain Expert.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="DIR",
                        help="FlatHydra checkpoint directory")
    parser.add_argument("--split-dir", required=True, metavar="DIR",
                        help="Dir with train/val/test_split.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--report-dir", default=None,
                        help="Output directory (defaults to checkpoint dir)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hierarchy-alpha", type=float, default=0.5)
    parser.add_argument("--no-hierarchy-scoring", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EvalTopK] Device: {device}")

    ckpt_dir = Path(args.checkpoint)
    report_dir = Path(args.report_dir or args.checkpoint)

    # Load config
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    # Load data
    split_dir = Path(args.split_dir)
    print(f"[EvalTopK] Loading data from {split_dir}...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if args.split == "val":
        texts_eval = data["X_val"]
        labels_l1_eval = data["y_l1_val"]
        labels_l2_eval = list(data["y_l2_val"])
    else:
        texts_eval = data["X_test"]
        labels_l1_eval = data["y_l1_test"]
        labels_l2_eval = list(data["y_l2_test"])

    train_labels_l2 = list(data["y_l2_train"])
    num_classes = data["num_classes_l2"]
    l2_to_l1_mapping = data["l2_to_l1_mapping"]

    # Load model
    print(f"[EvalTopK] Loading model from {ckpt_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir / "tokenizer"))

    baselines_root = _ROOT / "01_baselines"
    if str(baselines_root) not in sys.path:
        sys.path.insert(0, str(baselines_root))
    from src.flat_model import FlatHydra  # noqa: E402

    model = FlatHydra(
        encoder_name=cfg.get("encoder", "camembert-base"),
        num_classes_l1=data["num_classes_l1"],
        num_classes_l2=num_classes,
        pooling=cfg.get("pooling", "cls"),
        project_embedding=cfg.get("project_embedding", True),
        dropout=cfg.get("dropout", 0.2),
    )
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    # Build l2_to_l1 tensor
    l2_to_l1_tensor = make_l2_to_l1_tensor(l2_to_l1_mapping, num_classes)

    # Build DataLoader
    ds = HierarchicalProductDataset(
        texts_eval, labels_l1_eval, labels_l2_eval, tokenizer,
        max_length=cfg.get("max_length", 256),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    use_fp16 = device.type == "cuda"
    all_topk_preds: list[list[int]] = []

    print(f"[EvalTopK] Running inference (top-K={args.top_k}, alpha={args.hierarchy_alpha})...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=use_fp16):
                _, unified_logits = model(input_ids, mask)

            num_l1 = data["num_classes_l1"]
            logits_l1 = unified_logits[:, :num_l1]
            logits_l2 = unified_logits[:, num_l1:]

            candidates = get_topk_l2_candidates(
                logits_l1, logits_l2,
                l2_to_l1=l2_to_l1_tensor,
                k=args.top_k,
                hierarchy_scoring_enabled=not args.no_hierarchy_scoring,
                alpha=args.hierarchy_alpha,
            )
            for cands in candidates:
                all_topk_preds.append([c["l2_id"] for c in cands])

    k_values = [k for k in [1, 3, 5, 10, 20] if k <= args.top_k]
    report = build_topk_recall_report(
        labels_l2=labels_l2_eval,
        topk_preds=all_topk_preds,
        train_labels_l2=train_labels_l2,
        num_classes=num_classes,
        k_values=k_values,
    )

    # Print summary
    print("\n[EvalTopK] === Top-K Recall Report ===")
    for k, v in report["global"].items():
        print(f"  {k}: {v:.4f}")
    print("\n  By bucket:")
    for bucket, metrics in report["by_bucket"].items():
        r10 = metrics.get("l2_top_10_recall", 0.0)
        count = metrics.get("count_examples", 0)
        print(f"    {bucket}: top-10={r10:.4f} ({count} samples)")

    save_topk_recall_report(report, report_dir / "topk_recall_report.json")


if __name__ == "__main__":
    main()
