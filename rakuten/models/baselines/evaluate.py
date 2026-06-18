"""Common evaluation for all baseline approaches.

Loads prediction CSV files produced by each baseline pipeline, computes
metrics (F1-macro, F1-weighted), and prints a comparison table.

Usage:
    python evaluate.py --results_dirs models_tfidf models_word2vec models_bert_flat_camembert_fine_tuning
    python evaluate.py --results_dirs models_tfidf --detailed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Cross-directory import from shared/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_predictions(results_dir: str) -> Optional[pd.DataFrame]:
    """Load a predictions CSV from a results directory.

    Expects a file named ``predictions.csv`` with columns ``y_true`` and ``y_pred``.

    Args:
        results_dir: Path to the directory containing predictions.csv.

    Returns:
        DataFrame with y_true and y_pred columns, or None if not found.
    """
    csv_path = os.path.join(results_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        print(f"[Eval] Warning: {csv_path} not found, skipping.")
        return None
    return pd.read_csv(csv_path)


def compute_metrics(preds_df: pd.DataFrame) -> dict[str, float]:
    """Compute F1-macro and F1-weighted from a predictions DataFrame.

    Args:
        preds_df: DataFrame with 'y_true' and 'y_pred' columns.

    Returns:
        Dictionary with 'f1_macro' and 'f1_weighted'.
    """
    return {
        "f1_macro": f1_score(
            preds_df["y_true"], preds_df["y_pred"], average="macro", zero_division=0,
        ),
        "f1_weighted": f1_score(
            preds_df["y_true"], preds_df["y_pred"], average="weighted", zero_division=0,
        ),
    }


def load_config(results_dir: str) -> dict:
    """Load config.json or training_config.json from a results directory.

    Args:
        results_dir: Path to the directory.

    Returns:
        Configuration dictionary, or empty dict if not found.
    """
    for name in ("config.json", "training_config.json"):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _detect_method_name(results_dir: str, config: dict) -> str:
    """Derive a human-readable method name from config or directory name.

    Args:
        results_dir: Path used as fallback.
        config: Loaded configuration dict.

    Returns:
        Short method name string.
    """
    if "model_name" in config:
        return config["model_name"]
    if "backbone" in config and "mode" in config:
        return f"BERT-{config['backbone']} ({config['mode']})"
    if "classifier" in config:
        embedding = config.get("embedding", config.get("vectorizer", ""))
        if "vector_size" in config:
            return f"Word2Vec + {config['classifier']}"
        return f"TF-IDF + {config['classifier']}"
    return os.path.basename(results_dir.rstrip("/"))


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_baselines(
    results_dirs: list[str],
    detailed: bool = False,
) -> pd.DataFrame:
    """Build a comparison table across all baseline methods.

    Args:
        results_dirs: List of directories, each containing predictions.csv
            and a config JSON.
        detailed: If True, also print per-class classification reports.

    Returns:
        DataFrame with one row per method showing key metrics.
    """
    rows: list[dict] = []

    for rdir in results_dirs:
        preds_df = load_predictions(rdir)
        if preds_df is None:
            continue

        config = load_config(rdir)
        metrics = compute_metrics(preds_df)
        method = _detect_method_name(rdir, config)

        row = {
            "Method": method,
            "F1-Macro": metrics["f1_macro"],
            "F1-Weighted": metrics["f1_weighted"],
            "Train Time (s)": config.get("train_time_s", config.get("train_time", "")),
            "Samples": config.get("n_samples", ""),
            "Classes": config.get("num_classes", ""),
        }
        rows.append(row)

        if detailed:
            print(f"\n{'=' * 60}")
            print(f"Classification Report: {method}")
            print("=" * 60)
            print(classification_report(preds_df["y_true"], preds_df["y_pred"]))

    if not rows:
        print("[Eval] No valid results found.")
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("F1-Macro", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON TABLE")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)

    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare baseline models for Rakuten classification",
    )
    parser.add_argument(
        "--results_dirs", nargs="+", required=True,
        help="Directories containing predictions.csv and config JSON files",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Print per-class classification reports for each method",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Optional path to save comparison table as CSV",
    )

    args = parser.parse_args()

    comparison = compare_baselines(args.results_dirs, detailed=args.detailed)

    if args.output_csv and not comparison.empty:
        comparison.to_csv(args.output_csv, index=False)
        print(f"\n[Eval] Comparison table saved to {args.output_csv}")


if __name__ == "__main__":
    main()
