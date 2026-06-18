"""FlatHydra model — fidèle à HYDRA baselines.py officiel (EMNLP 2025).

Une seule tête unifiée prédit L1 + L2 simultanément.
Différence clé avec HYDRABase : la projection est appliquée sur
last_hidden_state AVANT le pooling (pas après).

Interface de retour : ([], unified_logits) pour compatibilité HYDRALoss.

Reference: https://github.com/FKarl/HYDRA/blob/main/modeling/baselines.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

POOLING_METHODS = ("cls", "mean", "max", "last")


class FlatHydra(nn.Module):
    """Flat classifier with a single unified head for all hierarchy levels.

    Args:
        encoder_name: HuggingFace model identifier (e.g. 'camembert-base').
        num_classes_l1: Number of L1 classes (157).
        num_classes_l2: Number of L2 classes (~5 777 en v2 ; dérivé des données).
        pooling: Pooling strategy applied after projection ('cls', 'mean', 'max', 'last').
        project_embedding: Project last_hidden_state from d to d*num_levels before pooling.
        dropout: Dropout rate in the unified classifier head.
        freeze_encoder: Freeze encoder weights (feature-extraction mode).
    """

    NUM_LEVELS = 2

    def __init__(
        self,
        encoder_name: str,
        num_classes_l1: int,
        num_classes_l2: int,
        pooling: str = "cls",
        project_embedding: bool = True,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        if pooling not in POOLING_METHODS:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from {POOLING_METHODS}")

        self.pooling = pooling
        self.project_embedding = project_embedding
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2

        self.encoder = AutoModel.from_pretrained(encoder_name)
        d: int = self.encoder.config.hidden_size  # 768 for CamemBERT

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection applied on last_hidden_state (B, seq, d) → (B, seq, d*2)
        # before pooling — faithful to the official implementation.
        if project_embedding:
            self.embedding_projector = nn.Linear(d, d * self.NUM_LEVELS)
            total_input = d * self.NUM_LEVELS  # 1536
        else:
            self.embedding_projector = None
            total_input = d  # 768

        total_labels = num_classes_l1 + num_classes_l2  # 157 + num_classes_l2 (= 5 934 en v2)

        self.unified_classifier = nn.Sequential(
            nn.Linear(total_input, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, total_labels),
        )

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool (B, seq, dim) → (B, dim)."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1).values
        elif self.pooling == "last":
            last_idx = attention_mask.sum(dim=1) - 1
            return hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_idx,
            ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list, torch.Tensor]:
        """Run the flat forward pass.

        Args:
            input_ids: Token ids of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).

        Returns:
            Tuple ([], unified_logits) where unified_logits has shape
            (B, num_classes_l1 + num_classes_l2). The empty list is kept
            for interface compatibility with HYDRALoss.
        """
        hidden_states = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, seq, d)

        # Project BEFORE pooling — key difference from HYDRABase
        if self.project_embedding:
            hidden_states = self.embedding_projector(hidden_states)  # (B, seq, d*2)

        embedding = self._pool(hidden_states, attention_mask)  # (B, total_input)
        unified_logits = self.unified_classifier(embedding)    # (B, total_labels)

        return [], unified_logits
