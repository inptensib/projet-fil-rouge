"""HLPD -- Hierarchical Label Pre-populated Decoder.

Reference: Torba, Laclau et al., ECIR 2025.

Architecture overview:
    1. Text Encoder   -- Pretrained transformer (CamemBERT or T5 encoder).
    2. Label Encoder  -- Single self-attention layer + linear projection.
                         Encodes textual label descriptions at init time;
                         discarded at inference.
    3. HLP Decoder    -- Non-autoregressive decoder whose queries are the
                         label embeddings.  Self-attention uses a hierarchical
                         positional bias derived from the taxonomy tree.
                         Cross-attention attends to the text encoder output.
    4. Classification -- Binary logit per label node.

All labels are decoded in a single forward pass (non-autoregressive).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from .hierarchical_attention import (
    HierarchicalSelfAttention,
    build_taxonomy_distance_matrix,
)


# ---------------------------------------------------------------------------
# Label Encoder
# ---------------------------------------------------------------------------

class LabelEncoder(nn.Module):
    """Encode textual label descriptions into dense vectors.

    Uses a single Transformer self-attention layer followed by a linear
    projection.  Label descriptions follow the template:
        "<Child> is a sub-topic of <Parent>"

    This module is used *once* (before training or at init) to produce
    static label embeddings and can be discarded at inference time.

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        max_len: Maximum token length for label descriptions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode label descriptions.

        Args:
            input_ids: Token ids of shape (num_labels, seq_len).
            attention_mask: Mask of shape (num_labels, seq_len).

        Returns:
            Label embeddings of shape (num_labels, d_model).
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Transformer expects src_key_padding_mask: True = ignore
        pad_mask = attention_mask == 0
        x = self.self_attn(x, src_key_padding_mask=pad_mask)

        # Mean-pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return self.proj(pooled)  # (num_labels, d_model)


def generate_label_descriptions(
    l1_names: list[str],
    l2_names: list[str],
    l2_to_l1: dict[int, int],
) -> list[str]:
    """Generate textual descriptions for each label node.

    Template: "<Child> is a sub-topic of <Parent>"
    Root: "product classification root"

    Args:
        l1_names: List of L1 class names (length num_l1).
        l2_names: List of L2 class names (length num_l2).
        l2_to_l1: Mapping from L2 index to L1 index.

    Returns:
        List of descriptions with ordering [root, L1_0, ..., L2_0, ...].
    """
    descriptions = ["product classification root"]

    # L1 descriptions
    for name in l1_names:
        descriptions.append(f"{name} is a sub-topic of product classification")

    # L2 descriptions
    for l2_idx, name in enumerate(l2_names):
        parent_idx = l2_to_l1.get(l2_idx, 0)
        parent_name = l1_names[parent_idx] if parent_idx < len(l1_names) else "unknown"
        descriptions.append(f"{name} is a sub-topic of {parent_name}")

    return descriptions


# ---------------------------------------------------------------------------
# HLP Decoder Layer
# ---------------------------------------------------------------------------

class HLPDecoderLayer(nn.Module):
    """Single HLP decoder layer.

    Consists of:
        1. Hierarchical self-attention over label embeddings.
        2. Cross-attention to text encoder output.
        3. Feed-forward network.

    All with residual connections and layer normalization.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate.
        num_labels: Total number of label positions (for hierarchical bias).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        num_labels: int = 1,
    ) -> None:
        super().__init__()

        # Hierarchical self-attention
        self.self_attn = HierarchicalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            num_labels=num_labels,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to encoder output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        label_embeds: torch.Tensor,
        encoder_output: torch.Tensor,
        m_hiera: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            label_embeds: (batch, num_labels, d_model).
            encoder_output: (batch, seq_len, d_model) from text encoder.
            m_hiera: Hierarchical distance matrix (num_labels, num_labels).
            encoder_mask: Key padding mask for encoder output (batch, seq_len).
                          True = ignore.

        Returns:
            Updated label embeddings (batch, num_labels, d_model).
        """
        # 1. Hierarchical self-attention
        residual = label_embeds
        x = self.self_attn(label_embeds, m_hiera)
        x = self.dropout1(x)
        x = self.norm1(residual + x)

        # 2. Cross-attention to text encoder
        residual = x
        cross_out, _ = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=encoder_mask,
        )
        x = self.dropout2(cross_out)
        x = self.norm2(residual + x)

        # 3. Feed-forward
        residual = x
        x = self.norm3(residual + self.ffn(x))

        return x


# ---------------------------------------------------------------------------
# Full HLPD Model
# ---------------------------------------------------------------------------

class HLPDModel(nn.Module):
    """Hierarchical Label Pre-populated Decoder model.

    End-to-end model for hierarchical multi-label text classification.

    Args:
        encoder_name: HuggingFace model name for the text encoder
                      (e.g. "camembert-base", "google/mt5-small").
        num_l1: Number of L1 classes.
        num_l2: Number of L2 classes.
        l2_to_l1: Mapping from L2 index to L1 index.
        d_model: Decoder hidden dimension (label embeddings are projected
                 to this size if different from encoder).
        num_decoder_layers: Number of HLP decoder layers.
        num_heads: Number of attention heads in the decoder.
        d_ff: Feed-forward inner dimension in decoder.
        dropout: Dropout rate.
        freeze_encoder: Whether to freeze the text encoder weights.
        encoder_type: 'camembert' or 't5' to select the right model class.
    """

    def __init__(
        self,
        encoder_name: str = "camembert-base",
        num_l1: int = 157,
        num_l2: int = 6100,
        l2_to_l1: Optional[dict[int, int]] = None,
        d_model: int = 768,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        encoder_type: str = "camembert",
    ) -> None:
        super().__init__()

        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.num_labels = 1 + num_l1 + num_l2  # root + L1 + L2
        self.d_model = d_model

        # --- Text Encoder ---
        self.encoder_type = encoder_type
        if encoder_type == "t5":
            self.text_encoder = T5EncoderModel.from_pretrained(encoder_name)
        else:
            self.text_encoder = AutoModel.from_pretrained(encoder_name)

        # Get encoder hidden size for projection
        encoder_hidden = self.text_encoder.config.hidden_size
        self.encoder_proj = (
            nn.Linear(encoder_hidden, d_model)
            if encoder_hidden != d_model
            else nn.Identity()
        )

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # --- Hierarchical Distance Matrix (static, not a parameter) ---
        if l2_to_l1 is None:
            l2_to_l1 = {}
        self.l2_to_l1 = l2_to_l1

        m_hiera = build_taxonomy_distance_matrix(l2_to_l1, num_l1, num_l2)
        self.register_buffer("m_hiera", m_hiera)

        # --- Label Embeddings (learnable, initialized by label encoder) ---
        self.label_embeddings = nn.Parameter(
            torch.randn(self.num_labels, d_model) * 0.02
        )

        # --- HLP Decoder ---
        self.decoder_layers = nn.ModuleList([
            HLPDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                num_labels=self.num_labels,
            )
            for _ in range(num_decoder_layers)
        ])

        # --- Classification Head ---
        self.classifier = nn.Linear(d_model, 1)  # binary logit per label

    def initialize_label_embeddings(
        self,
        label_encoder: LabelEncoder,
        tokenizer,
        l1_names: list[str],
        l2_names: list[str],
        max_len: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize label embeddings from textual descriptions.

        Runs the label encoder on all label descriptions and writes
        the resulting vectors into self.label_embeddings. Call this
        once before training.

        Args:
            label_encoder: Trained or randomly initialized LabelEncoder.
            tokenizer: HuggingFace tokenizer for encoding descriptions.
            l1_names: L1 class name strings.
            l2_names: L2 class name strings.
            max_len: Max token length for descriptions.
            device: Device to run on.
        """
        if device is None:
            device = self.label_embeddings.device

        descriptions = generate_label_descriptions(
            l1_names, l2_names, self.l2_to_l1
        )

        label_encoder = label_encoder.to(device)
        label_encoder.eval()

        # Encode in batches to avoid OOM
        batch_size = 128
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(descriptions), batch_size):
                batch_desc = descriptions[i : i + batch_size]
                tokens = tokenizer(
                    batch_desc,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                embeds = label_encoder(tokens["input_ids"], tokens["attention_mask"])
                all_embeds.append(embeds.cpu())

        all_embeds = torch.cat(all_embeds, dim=0)  # (num_labels, d_model)
        with torch.no_grad():
            self.label_embeddings.copy_(all_embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tokenized text input of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, num_labels) -- one binary logit per
            label node (root + L1 + L2).
        """
        B = input_ids.size(0)

        # 1. Text Encoder
        if self.encoder_type == "t5":
            encoder_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        else:
            encoder_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        encoder_out = self.encoder_proj(encoder_out)  # (B, S, d_model)

        # 2. Prepare label queries (broadcast over batch)
        label_queries = self.label_embeddings.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, num_labels, d_model)

        # Encoder mask: True = padding positions to ignore
        encoder_mask = attention_mask == 0  # (B, S)

        # 3. HLP Decoder
        x = label_queries
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, self.m_hiera, encoder_mask)

        # 4. Classification
        logits = self.classifier(x).squeeze(-1)  # (B, num_labels)

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Run inference and extract L1/L2 predictions.

        Args:
            input_ids: Tokenized text of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len).
            threshold: Logit threshold for positive prediction.

        Returns:
            Dictionary with keys 'pred_l1' (batch,) and 'pred_l2' (batch,).
        """
        logits = self.forward(input_ids, attention_mask)  # (B, num_labels)

        # Split logits: positions [0] = root, [1..num_l1] = L1, [num_l1+1..] = L2
        l1_logits = logits[:, 1 : 1 + self.num_l1]          # (B, num_l1)
        l2_logits = logits[:, 1 + self.num_l1 :]             # (B, num_l2)

        pred_l1 = l1_logits.argmax(dim=-1)  # (B,)
        pred_l2 = l2_logits.argmax(dim=-1)  # (B,)

        return {"pred_l1": pred_l1, "pred_l2": pred_l2, "logits": logits}
