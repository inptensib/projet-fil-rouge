"""Production inference for HYDRA models.

Supports loading from both standard and DDP-wrapped checkpoints.
Models are imported from 02_hydra.src.models.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Cross-directory imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_HYDRA_ROOT = os.path.join(_PROJECT_ROOT, "02_hydra")
if _HYDRA_ROOT not in sys.path:
    sys.path.insert(0, _HYDRA_ROOT)

from src.models import create_hydra_model

from shared.utils import ENCODERS


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a trained HYDRA model from checkpoint.

    Handles DDP-wrapped state dicts by stripping the 'module.' prefix
    automatically.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: Device to load the model on. Defaults to best available.

    Returns:
        Tuple of (model, config_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    encoder_name = config.get("encoder_name", "CamemBERT")
    encoder_path = ENCODERS.get(encoder_name, encoder_name)
    model = create_hydra_model(
        variant=config["variant"],
        encoder_name=encoder_path,
        num_classes_l1=config["num_classes_l1"],
        num_classes_l2=config["num_classes_l2"],
        pooling=config.get("pooling", "cls"),
        project_embedding=config.get("project_embedding", True),
    )

    # Handle DDP-wrapped state dicts
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def load_label_encoders(shard_dir: str) -> tuple:
    """Load label encoders from shard metadata.

    Args:
        shard_dir: Directory containing metadata.pkl.

    Returns:
        Tuple of (l1_encoder, l2_encoder).
    """
    with open(os.path.join(shard_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return metadata["l1_encoder"], metadata["l2_encoder"]


def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    l1_encoder,
    l2_encoder,
    device: torch.device,
    max_length: int = 128,
) -> dict:
    """Run inference on a single text input.

    Args:
        text: Product description text.
        model: Trained HYDRA model (already in eval mode).
        tokenizer: HuggingFace tokenizer.
        l1_encoder: Fitted LabelEncoder for L1.
        l2_encoder: Fitted LabelEncoder for L2.
        device: Computation device.
        max_length: Maximum token length.

    Returns:
        Dictionary with L1 and L2 predictions and confidence scores.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        level_outputs, _ = model(inputs["input_ids"], inputs["attention_mask"])
        logits_l1, logits_l2 = level_outputs

        probs_l1 = F.softmax(logits_l1, dim=-1)
        probs_l2 = F.softmax(logits_l2, dim=-1)

        conf_l1, pred_l1 = torch.max(probs_l1, dim=-1)
        conf_l2, pred_l2 = torch.max(probs_l2, dim=-1)

    return {
        "level_1": {
            "label": l1_encoder.inverse_transform([pred_l1.item()])[0],
            "confidence": round(conf_l1.item(), 4),
        },
        "level_2": {
            "label": l2_encoder.inverse_transform([pred_l2.item()])[0],
            "confidence": round(conf_l2.item(), 4),
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
    batch_size: int = 32,
) -> list[dict]:
    """Run inference on a batch of texts.

    Args:
        texts: List of product description texts.
        model: Trained HYDRA model (already in eval mode).
        tokenizer: HuggingFace tokenizer.
        l1_encoder: Fitted LabelEncoder for L1.
        l2_encoder: Fitted LabelEncoder for L2.
        device: Computation device.
        max_length: Maximum token length.
        batch_size: Batch size for inference.

    Returns:
        List of prediction dictionaries.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            level_outputs, _ = model(
                inputs["input_ids"], inputs["attention_mask"]
            )
            logits_l1, logits_l2 = level_outputs

            probs_l1 = F.softmax(logits_l1, dim=-1)
            probs_l2 = F.softmax(logits_l2, dim=-1)

            conf_l1, pred_l1 = torch.max(probs_l1, dim=-1)
            conf_l2, pred_l2 = torch.max(probs_l2, dim=-1)

        for j in range(len(batch_texts)):
            results.append({
                "level_1": {
                    "label": l1_encoder.inverse_transform([pred_l1[j].item()])[0],
                    "confidence": round(conf_l1[j].item(), 4),
                },
                "level_2": {
                    "label": l2_encoder.inverse_transform([pred_l2[j].item()])[0],
                    "confidence": round(conf_l2[j].item(), 4),
                },
            })

    return results
