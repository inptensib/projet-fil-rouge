"""HYDRA model architectures for hierarchical product classification.

Three variants as described in the HYDRA paper (EMNLP 2025):
- HYDRALocal: Independent parallel heads for L1 and L2.
- HYDRAGlobal: Local heads + a global head over concatenated logits.
- HYDRANested: Local heads + a nested head for hierarchy constraints.

Verified against the official implementation: https://github.com/FKarl/HYDRA
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HYDRABase(nn.Module):
    """Base class with shared encoder and pooling logic."""

    POOLING_METHODS = ("cls", "mean", "max", "last")

    def __init__(self, pooling: str = "cls"):
        super().__init__()
        if pooling not in self.POOLING_METHODS:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Choose from {self.POOLING_METHODS}"
            )
        self.pooling = pooling

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool encoder hidden states into a single vector per sample."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            return hidden_states.max(dim=1).values
        elif self.pooling == "last":
            return hidden_states[:, -1, :]

    @staticmethod
    def _make_classification_head(
        input_size: int, num_classes: int, dropout: float = 0.3
    ) -> nn.Sequential:
        """Create a classification head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear."""
        return nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LayerNorm(input_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size * 2, num_classes),
        )


class HYDRALocal(HYDRABase):
    """HYDRA with independent local heads for L1 and L2.

    Args:
        encoder_name: HuggingFace model identifier (e.g., 'camembert-base').
        num_classes_l1: Number of Level 1 classes.
        num_classes_l2: Number of Level 2 classes.
        pooling: Pooling strategy ('cls', 'mean', 'max', 'last').
        project_embedding: Whether to project encoder output to a larger space.
        dropout: Dropout rate for classification heads.
        freeze_encoder: Whether to freeze encoder parameters.
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes_l1: int,
        num_classes_l2: int,
        pooling: str = "cls",
        project_embedding: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__(pooling=pooling)

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_levels = 2
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.project_embedding = project_embedding

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if project_embedding:
            self.embedding_projector = nn.Linear(
                self.hidden_size, self.hidden_size * self.num_levels
            )
            self.embedding_size = self.hidden_size * self.num_levels
        else:
            self.embedding_size = self.hidden_size

        self.level_1_head = self._make_classification_head(
            self.embedding_size, num_classes_l1, dropout
        )
        self.level_2_head = self._make_classification_head(
            self.embedding_size, num_classes_l2, dropout
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], None]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_output.last_hidden_state
        if self.project_embedding:
            hidden_states = self.embedding_projector(hidden_states)
        embedding = self._pool(hidden_states, attention_mask)

        logits_l1 = self.level_1_head(embedding)
        logits_l2 = self.level_2_head(embedding)
        return [logits_l1, logits_l2], None


class HYDRAGlobal(HYDRABase):
    """HYDRA with local heads + a global head that sees all labels jointly.

    The global head takes the same embedding and produces logits for all
    (L1 + L2) classes, providing a unified hierarchical view.
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes_l1: int,
        num_classes_l2: int,
        pooling: str = "cls",
        project_embedding: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__(pooling=pooling)

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_levels = 2
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.project_embedding = project_embedding

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if project_embedding:
            self.embedding_projector = nn.Linear(
                self.hidden_size, self.hidden_size * self.num_levels
            )
            self.embedding_size = self.hidden_size * self.num_levels
        else:
            self.embedding_size = self.hidden_size

        self.level_1_head = self._make_classification_head(
            self.embedding_size, num_classes_l1, dropout
        )
        self.level_2_head = self._make_classification_head(
            self.embedding_size, num_classes_l2, dropout
        )

        total_labels = num_classes_l1 + num_classes_l2
        self.global_head = self._make_classification_head(
            self.embedding_size, total_labels, dropout
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_output.last_hidden_state
        if self.project_embedding:
            hidden_states = self.embedding_projector(hidden_states)
        embedding = self._pool(hidden_states, attention_mask)

        logits_l1 = self.level_1_head(embedding)
        logits_l2 = self.level_2_head(embedding)
        unified_logits = self.global_head(embedding)
        return [logits_l1, logits_l2], unified_logits


class HYDRANested(HYDRABase):
    """HYDRA with local heads + a nested head for hierarchy-aware refinement.

    The nested head takes the concatenated L1 and L2 logits and refines
    them through an additional MLP, enforcing implicit hierarchy constraints.
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes_l1: int,
        num_classes_l2: int,
        pooling: str = "cls",
        project_embedding: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__(pooling=pooling)

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_levels = 2
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.project_embedding = project_embedding

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if project_embedding:
            self.embedding_projector = nn.Linear(
                self.hidden_size, self.hidden_size * self.num_levels
            )
            self.embedding_size = self.hidden_size * self.num_levels
        else:
            self.embedding_size = self.hidden_size

        self.level_1_head = self._make_classification_head(
            self.embedding_size, num_classes_l1, dropout
        )
        self.level_2_head = self._make_classification_head(
            self.embedding_size, num_classes_l2, dropout
        )

        total_labels = num_classes_l1 + num_classes_l2
        self.nested_head = nn.Sequential(
            nn.Linear(total_labels, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, total_labels),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_output.last_hidden_state
        if self.project_embedding:
            hidden_states = self.embedding_projector(hidden_states)
        embedding = self._pool(hidden_states, attention_mask)

        logits_l1 = self.level_1_head(embedding)
        logits_l2 = self.level_2_head(embedding)
        concat_logits = torch.cat([logits_l1, logits_l2], dim=1)
        unified_logits = self.nested_head(concat_logits)
        return [logits_l1, logits_l2], unified_logits


# Factory for convenient instantiation
HYDRA_VARIANTS = {
    "local": HYDRALocal,
    "global": HYDRAGlobal,
    "nested": HYDRANested,
}


def create_hydra_model(variant: str, **kwargs) -> HYDRABase:
    """Create a HYDRA model by variant name.

    Args:
        variant: One of 'local', 'global', 'nested'.
        **kwargs: Arguments passed to the model constructor.

    Returns:
        HYDRA model instance.
    """
    if variant not in HYDRA_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(HYDRA_VARIANTS.keys())}")
    return HYDRA_VARIANTS[variant](**kwargs)
