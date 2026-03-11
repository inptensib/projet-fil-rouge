"""Error analysis for HYDRA hierarchical classification models.

Provides two main capabilities:
1. Per-sample error analysis: Classifies predictions by difficulty level.
2. Global error report: Tracks error persistence and top confusion pairs
   across epochs.

Usage:
    python -m src.error_analysis --model-path models/.../hydra_model.pt \\
                                 --shard-dir shards/ --n-samples 5000
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Cross-directory imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.data_loader import HierarchicalProductDataset

_HYDRA_ROOT = os.path.join(_PROJECT_ROOT, "02_hydra")
if _HYDRA_ROOT not in sys.path:
    sys.path.insert(0, _HYDRA_ROOT)

from src.models import create_hydra_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODERS = {
    "CamemBERT": "camembert-base",
    "XLM-RoBERTa": "xlm-roberta-base",
    "mBERT": "bert-base-multilingual-cased",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_metadata(
    model_path: str,
    shard_dir: str,
    device: torch.device,
) -> tuple:
    """Load a trained HYDRA model and its associated metadata.

    Handles DDP-wrapped state dicts by stripping the 'module.' prefix.

    Args:
        model_path: Path to the .pt checkpoint file.
        shard_dir: Directory containing metadata.pkl.
        device: Device to load the model on.

    Returns:
        Tuple of (model, metadata_dict, config_dict).
    """
    print(f"[Analysis] Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    with open(os.path.join(shard_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    encoder_name = config.get("encoder_name", "CamemBERT")
    encoder_path = ENCODERS.get(encoder_name, encoder_name)
    variant = config.get("variant", "local")

    model = create_hydra_model(
        variant=variant,
        encoder_name=encoder_path,
        num_classes_l1=config["num_classes_l1"],
        num_classes_l2=config["num_classes_l2"],
        pooling=config.get("pooling", "cls"),
        project_embedding=config.get("project_embedding", True),
    )

    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metadata, config


# ---------------------------------------------------------------------------
# Per-sample error analysis
# ---------------------------------------------------------------------------

def run_sample_analysis(
    model_path: str,
    shard_dir: str,
    n_samples: int = 5000,
    output_file: str = "hard_examples_report.csv",
) -> pd.DataFrame:
    """Analyze model predictions and classify samples by difficulty.

    Categories:
    - HARD (L1 Error): Level 1 prediction is wrong.
    - HARD (Overconfident L2 Error): L2 wrong but confidence > 0.8.
    - MEDIUM (L2 Error): L2 wrong with moderate confidence.
    - AMBIGUOUS (Low Confidence): Correct but confidence < 0.4.
    - EASY: Correct with sufficient confidence.

    Args:
        model_path: Path to the .pt checkpoint file.
        shard_dir: Directory containing data shards.
        n_samples: Number of validation samples to analyze.
        output_file: Path to save the error report CSV.

    Returns:
        DataFrame with per-sample analysis results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, config = load_model_and_metadata(model_path, shard_dir, device)

    encoder_name = config.get("encoder_name", "CamemBERT")
    encoder_path = ENCODERS.get(encoder_name, encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    # Load validation data
    print("[Analysis] Loading validation data...")
    with open(os.path.join(shard_dir, "val.pkl"), "rb") as f:
        val_data = pickle.load(f)

    texts = val_data["texts"][:n_samples]
    labels_l1 = val_data["labels_l1"][:n_samples]
    labels_l2 = val_data["labels_l2"][:n_samples]

    dataset = HierarchicalProductDataset(texts, labels_l1, labels_l2, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    l1_enc = metadata["l1_encoder"]
    l2_enc = metadata["l2_encoder"]

    results = []
    idx = 0

    print(f"[Analysis] Analyzing {len(texts)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            level_outputs, _ = model(input_ids, mask)
            logits_l1, logits_l2 = level_outputs

            probs_l1 = F.softmax(logits_l1, dim=-1)
            probs_l2 = F.softmax(logits_l2, dim=-1)
            conf_l1, pred_l1 = torch.max(probs_l1, dim=-1)
            conf_l2, pred_l2 = torch.max(probs_l2, dim=-1)

            for i in range(len(input_ids)):
                real_l1 = labels_l1[idx]
                real_l2 = labels_l2[idx]
                p_l1 = pred_l1[i].item()
                p_l2 = pred_l2[i].item()

                correct_l1 = real_l1 == p_l1
                correct_l2 = real_l2 == p_l2
                c_l2 = conf_l2[i].item()

                if not correct_l1:
                    difficulty = "HARD (L1 Error)"
                elif not correct_l2 and c_l2 > 0.80:
                    difficulty = "HARD (Overconfident L2 Error)"
                elif not correct_l2:
                    difficulty = "MEDIUM (L2 Error)"
                elif c_l2 < 0.40:
                    difficulty = "AMBIGUOUS (Low Confidence)"
                else:
                    difficulty = "EASY"

                results.append({
                    "text": texts[idx][:150] + "...",
                    "true_l1": l1_enc.inverse_transform([real_l1])[0],
                    "pred_l1": l1_enc.inverse_transform([p_l1])[0],
                    "conf_l1": round(conf_l1[i].item(), 4),
                    "correct_l1": correct_l1,
                    "true_l2": l2_enc.inverse_transform([real_l2])[0],
                    "pred_l2": l2_enc.inverse_transform([p_l2])[0],
                    "conf_l2": round(c_l2, 4),
                    "correct_l2": correct_l2,
                    "difficulty": difficulty,
                })
                idx += 1

    df = pd.DataFrame(results)

    # Summary
    print("\n[Analysis] Difficulty distribution:")
    print(df["difficulty"].value_counts().to_string())

    # Save errors for human review
    errors_df = df[df["difficulty"] != "EASY"].sort_values(
        by="conf_l2", ascending=False
    )
    errors_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[Analysis] Error report saved to: {output_file}")

    return df


# ---------------------------------------------------------------------------
# Global error report across epochs
# ---------------------------------------------------------------------------

def generate_global_report(model_dir: str) -> None:
    """Generate a global error analysis report across training epochs.

    Analyzes error persistence (samples that are consistently misclassified)
    and top confusion pairs over time.

    Args:
        model_dir: Directory containing per-epoch error CSV files.
    """
    print(f"[Global] Analyzing errors in: {model_dir}")

    train_files = sorted(
        glob.glob(os.path.join(model_dir, "hard_train_examples_epoch_*.csv"))
    )
    val_files = sorted(
        glob.glob(os.path.join(model_dir, "hard_examples_epoch_*.csv"))
    )

    if not train_files and not val_files:
        print("[Global] No error CSV files found.")
        return

    # Training error persistence
    if train_files:
        print("[Global] Analyzing training error persistence...")
        train_dfs = []
        for f in train_files:
            epoch = f.split("_")[-1].replace(".csv", "")
            df = pd.read_csv(f)
            df["epoch"] = int(epoch)
            train_dfs.append(df)

        all_train_errors = pd.concat(train_dfs)
        persistence = (
            all_train_errors.groupby("Sample_Idx")
            .size()
            .reset_index(name="frequency")
        )
        persistent_errors = persistence[persistence["frequency"] > 1].sort_values(
            by="frequency", ascending=False
        )

        print(
            f"   {len(persistent_errors)} samples are persistently "
            f"misclassified during training."
        )
        output_path = os.path.join(model_dir, "persistent_train_errors.csv")
        persistent_errors.to_csv(output_path, index=False)

    # Validation error trends
    if val_files:
        print("[Global] Analyzing validation error trends...")
        val_dfs = []
        for f in val_files:
            epoch = f.split("_")[-1].replace(".csv", "")
            df = pd.read_csv(f)
            df["epoch"] = int(epoch)
            val_dfs.append(df)

        all_val_errors = pd.concat(val_dfs)

        # Error type evolution plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=all_val_errors, x="epoch", hue="Type", ax=ax)
        ax.set_title("Validation Error Types by Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Count")
        plot_path = os.path.join(model_dir, "error_trends.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Top confusion pairs
        all_val_errors["confusion"] = (
            all_val_errors["Vrai_L2"] + " -> " + all_val_errors["Pred_L2"]
        )
        top_confusions = all_val_errors["confusion"].value_counts().head(15)

        print("\n[Global] Top 15 confusion pairs:")
        print(top_confusions.to_string())

        confusion_path = os.path.join(model_dir, "top_confusions.csv")
        top_confusions.to_csv(confusion_path)

    print(f"\n[Global] Report complete. Files saved to {model_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for HYDRA models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Per-sample analysis
    sample_parser = subparsers.add_parser("samples", help="Per-sample error analysis")
    sample_parser.add_argument("--model-path", required=True, help="Path to .pt checkpoint")
    sample_parser.add_argument("--shard-dir", required=True, help="Shard directory")
    sample_parser.add_argument("--n-samples", type=int, default=5000)
    sample_parser.add_argument("--output", default="hard_examples_report.csv")

    # Global report
    global_parser = subparsers.add_parser("global", help="Cross-epoch error report")
    global_parser.add_argument("--model-dir", required=True, help="Model directory")

    args = parser.parse_args()

    if args.command == "samples":
        run_sample_analysis(
            args.model_path, args.shard_dir, args.n_samples, args.output,
        )
    elif args.command == "global":
        generate_global_report(args.model_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
