"""Inference module for HYDRA models.

Supports loading trained checkpoints and running interactive or batch predictions
with optional hierarchical enforcement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rakuten.common.utils import ENCODERS, enforce_hierarchy

from .models import create_hydra_model


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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    encoder_path = ENCODERS.get(config["encoder_name"], config["encoder_name"])
    model = create_hydra_model(
        variant=config["variant"],
        encoder_name=encoder_path,
        num_classes_l1=config["num_classes_l1"],
        num_classes_l2=config["num_classes_l2"],
        pooling=config.get("pooling", "cls"),
        project_embedding=config.get("project_embedding", True),
        dropout=config.get("dropout", 0.3),
        hidden_dim=config.get("hidden_dim", 256),
    )

    # Handle DDP-wrapped state dicts
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    l1_encoder,
    l2_encoder,
    device: torch.device,
    max_length: int = 256,
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
            logits_l2 = enforce_hierarchy(logits_l1, logits_l2, parent_child_matrix)

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
