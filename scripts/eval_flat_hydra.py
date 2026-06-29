#!/usr/bin/env python3
"""Standalone TEST evaluation of a FlatHydra checkpoint — no SLURM, any GPU/CPU.

Loads the splits, loads ``best_model.pt``, evaluates on the test set, prints the
metrics and writes ``<results-dir>/test_metrics.json``. The device is auto-detected
(CUDA if available, else CPU). This is the direct equivalent of the inline eval in
``scripts/slurm/run_eval_flat_hydra_generic.slurm`` (which now calls this script).

    python3 scripts/eval_flat_hydra.py --split-dir <dataset>/ \\
        --model-pt models/v2/flat_hydra/best_model.pt \\
        --config configs/baselines/flat_hydra_camembert_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.models.baselines.flat_trainer import FlatHydraConfig, FlatHydraTrainer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a FlatHydra checkpoint on the test split (no SLURM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--split-dir", required=True,
                    help="Directory with train/val/test _split.jsonl.")
    ap.add_argument("--model-pt", default="models/v2/flat_hydra/best_model.pt",
                    help="Path to the trained checkpoint (best_model.pt).")
    ap.add_argument("--config", default="configs/baselines/flat_hydra_camembert_v2.yaml",
                    help="Model architecture YAML (must match the checkpoint).")
    ap.add_argument("--results-dir", default="results/flat_hydra_v2",
                    help="Where to write test_metrics.json.")
    ap.add_argument("--eval-batch-size", type=int, default=1536,
                    help="No-grad eval batch (lower it on a small GPU).")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = FlatHydraConfig.from_yaml(args.config)

    print("[Data] Loading splits...")
    t0 = time.time()
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        use_description=config.use_description,
        seed=42,
    )
    print(f"[Data] Loaded in {time.time() - t0:.0f}s — test: {len(data['X_test']):,}, "
          f"L1: {data['num_classes_l1']}, L2: {data['num_classes_l2']}")

    config.batch_size = args.eval_batch_size
    trainer = FlatHydraTrainer(data=data, config=config)

    print(f"[Model] Loading {args.model_pt}...")
    ckpt = torch.load(args.model_pt, map_location=trainer.device, weights_only=False)
    trainer._base_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print("[Model] Weights loaded.")

    test_loader = trainer._make_loader(
        data["X_test"], data["y_l1_test"], data["y_l2_test"], shuffle=False
    )
    print("[Eval] Evaluating on the TEST set...")
    t1 = time.time()
    metrics = trainer.evaluate(test_loader)
    elapsed = time.time() - t1

    print(f"\n{'=' * 60}")
    print(f"  FlatHydra — TEST set ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  F1 L1 macro       : {metrics['f1_l1_macro']:.4f}")
    print(f"  F1 L2 macro       : {metrics['f1_l2_macro']:.4f}")
    print(f"  Constrained F1 L2 : {metrics.get('constrained_f1_macro', 0.0):.4f}")
    print(f"  HCR               : {metrics.get('hcr', 0.0):.4f}")
    print(f"{'=' * 60}")

    summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
    summary["eval_time_s"] = round(elapsed, 1)
    summary["num_classes_l2"] = data["num_classes_l2"]
    out_path = results_dir / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
