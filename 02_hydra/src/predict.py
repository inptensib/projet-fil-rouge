"""Inference module for HYDRA models.

Supports loading trained checkpoints and running interactive or batch predictions
with optional hierarchical enforcement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils import ENCODERS
from .models import create_hydra_model


class _TextDataset(Dataset):
    """Simple dataset for batch inference."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )


def load_model(model_path: str, device: torch.device = None):
    """Load a trained HYDRA model from checkpoint.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, config_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    encoder_path = ENCODERS.get(config["encoder_name"], config["encoder_name"])
    model = create_hydra_model(
        variant=config["variant"],
        encoder_name=encoder_path,
        num_classes_l1=config["num_classes_l1"],
        num_classes_l2=config["num_classes_l2"],
    )

    # Handle DDP-wrapped state dicts
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def _enforce_hierarchy(
    logits_l1: torch.Tensor,
    logits_l2: torch.Tensor,
    parent_child_matrix: torch.Tensor,
) -> torch.Tensor:
    """Mask invalid L2 predictions based on predicted L1 parent."""
    preds_l1 = torch.argmax(logits_l1, dim=-1)
    valid_l2_mask = parent_child_matrix[preds_l1]
    masked = logits_l2.clone()
    masked[valid_l2_mask == 0] = float("-inf")
    return masked


def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    l1_encoder,
    l2_encoder,
    device: torch.device,
    max_length: int = 128,
    parent_child_matrix: torch.Tensor | None = None,
) -> dict:
    """Run inference on a single text input.

    Args:
        text: Product description text.
        model: Trained HYDRA model.
        tokenizer: HuggingFace tokenizer.
        l1_encoder: Fitted LabelEncoder for L1.
        l2_encoder: Fitted LabelEncoder for L2.
        device: Computation device.
        max_length: Maximum token length.
        parent_child_matrix: Optional parent-child matrix for hierarchy enforcement.

    Returns:
        Dictionary with L1 and L2 predictions and confidence scores.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_length, padding="max_length",
    ).to(device)

    with torch.no_grad():
        level_outputs, _ = model(inputs["input_ids"], inputs["attention_mask"])
        logits_l1, logits_l2 = level_outputs

        if parent_child_matrix is not None:
            logits_l2 = _enforce_hierarchy(logits_l1, logits_l2, parent_child_matrix)

        probs_l1 = F.softmax(logits_l1, dim=-1)
        probs_l2 = F.softmax(logits_l2, dim=-1)

        conf_l1, pred_l1 = torch.max(probs_l1, dim=-1)
        conf_l2, pred_l2 = torch.max(probs_l2, dim=-1)

    return {
        "level_1": {
            "label": l1_encoder.inverse_transform([pred_l1.item()])[0],
            "confidence": conf_l1.item(),
        },
        "level_2": {
            "label": l2_encoder.inverse_transform([pred_l2.item()])[0],
            "confidence": conf_l2.item(),
        },
    }


def predict_batch(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    l1_encoder,
    l2_encoder,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
    parent_child_matrix: torch.Tensor | None = None,
) -> list[dict]:
    """Run inference on a batch of texts using DataLoader.

    Args:
        texts: List of product description texts.
        model: Trained HYDRA model.
        tokenizer: HuggingFace tokenizer.
        l1_encoder: Fitted LabelEncoder for L1.
        l2_encoder: Fitted LabelEncoder for L2.
        device: Computation device.
        max_length: Maximum token length.
        batch_size: Batch size for inference.
        parent_child_matrix: Optional parent-child matrix for hierarchy enforcement.

    Returns:
        List of prediction dictionaries.
    """
    dataset = _TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_results = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)

            level_outputs, _ = model(input_ids, attention_mask)
            logits_l1, logits_l2 = level_outputs

            if parent_child_matrix is not None:
                logits_l2 = _enforce_hierarchy(logits_l1, logits_l2, parent_child_matrix)

            probs_l1 = F.softmax(logits_l1, dim=-1)
            probs_l2 = F.softmax(logits_l2, dim=-1)

            confs_l1, preds_l1 = torch.max(probs_l1, dim=-1)
            confs_l2, preds_l2 = torch.max(probs_l2, dim=-1)

            for i in range(input_ids.size(0)):
                all_results.append({
                    "level_1": {
                        "label": l1_encoder.inverse_transform([preds_l1[i].item()])[0],
                        "confidence": confs_l1[i].item(),
                    },
                    "level_2": {
                        "label": l2_encoder.inverse_transform([preds_l2[i].item()])[0],
                        "confidence": confs_l2[i].item(),
                    },
                })

    return all_results
