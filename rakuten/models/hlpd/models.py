"""HLPD — Hierarchical Label Prediction Decoder.

Reference: Torba, Laclau et al. ECIR 2025.
           https://github.com/FatosTorba/HLPD (Apache 2.0)

Architecture:
    Text → CamemBERT encoder → H_enc (B, seq_len, d_model)
    Labels (L = num_l1 + num_l2) → LabelEmbeddings → E (B, L, d_model)
        ↓  ×num_decoder_layers
    Cross-attention  (Q=E, K=V=H_enc)
    Hierarchical self-attention  (scores += W ⊙ M_hiera + b)
    FFN + LayerNorm (pre-norm)
        ↓
    logits[b, l] = h[b, l] · LabelEmbeddings[l]  (tied weights)  → (B, L)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoModel

from .attention_bias import build_M_hiera


class HLPDDecoderLayer(nn.Module):
    """One HLPD decoder layer: cross-attention + hierarchical self-attention + FFN.

    Pre-norm (LayerNorm before each sub-layer) for training stability.

    Args:
        d_model: Feature dimension (must match encoder hidden size).
        nhead:   Number of attention heads.
        ffn_dim: Inner dimension of the feed-forward network.
        dropout: Dropout probability.
        num_labels: Total number of labels L = num_l1 + num_l2.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        num_labels: int,
    ) -> None:
        super().__init__()

        # Cross-attention: labels (Q) attend to encoder output (K, V)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Hierarchical self-attention
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # W ∈ ℝ^{L×L} init Normal(1.0, 0.02) — scales hierarchy-aware pairs
        self.attn_weight = nn.Parameter(torch.empty(num_labels, num_labels))
        nn.init.normal_(self.attn_weight, mean=1.0, std=0.02)
        # b ∈ ℝ^{L×L} init 0 — unconstrained additive offset
        self.attn_bias = nn.Parameter(torch.zeros(num_labels, num_labels))

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        label_repr: torch.Tensor,    # (B, L, d_model)
        encoder_out: torch.Tensor,   # (B, seq_len, d_model)
        encoder_mask: torch.Tensor,  # (B, seq_len) — 1=valid, 0=padding
        M_hiera: torch.Tensor,       # (L, L) float — hierarchical binary mask
    ) -> torch.Tensor:
        x = label_repr

        # 1. Cross-attention (pre-norm): labels attend to encoder
        key_padding_mask = (encoder_mask == 0)  # True = ignore (padding)
        x_normed = self.norm_cross(x)
        x_ca, _ = self.cross_attn(
            query=x_normed,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=key_padding_mask,
        )
        x = x + x_ca

        # 2. Hierarchical self-attention (pre-norm): W ⊙ M + b additive bias
        x_normed = self.norm_self(x)
        hiera_bias = self.attn_weight * M_hiera + self.attn_bias  # (L, L)
        x_sa, _ = self.self_attn(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            attn_mask=hiera_bias,
        )
        x = x + x_sa

        # 3. FFN (pre-norm)
        x_normed = self.norm_ffn(x)
        x = x + self.ffn(x_normed)

        return x


class HLPD(nn.Module):
    """Hierarchical Label Prediction Decoder (HLPD).

    Args:
        encoder_name:           HuggingFace model name for the text encoder.
        num_l1:                 Number of L1 classes.
        num_l2:                 Number of L2 classes.
        l2_to_l1_mapping:       Mapping l2_index → l1_index (0-based).
        num_decoder_layers:     Number of stacked HLPD decoder layers.
        nhead:                  Attention heads in each decoder layer.
        ffn_dim:                FFN inner dimension.
        decoder_dropout:        Dropout in decoder layers.
        gradient_checkpointing: Recompute decoder activations during backward to save VRAM.
        freeze_encoder:         Freeze CamemBERT weights.
    """

    def __init__(
        self,
        encoder_name: str,
        num_l1: int,
        num_l2: int,
        l2_to_l1_mapping: dict[int, int],
        num_decoder_layers: int = 2,
        nhead: int = 4,
        ffn_dim: int = 2048,
        decoder_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.num_labels = num_l1 + num_l2
        self.gradient_checkpointing_flag = gradient_checkpointing

        # Text encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d_model: int = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Label embeddings: shared input + output (tied weights)
        self.label_embeddings = nn.Embedding(self.num_labels, d_model)
        nn.init.normal_(self.label_embeddings.weight, mean=0.0, std=0.02)

        # Tied output projection: output_proj.weight IS label_embeddings.weight
        self.output_proj = nn.Linear(d_model, self.num_labels, bias=False)
        self.output_proj.weight = self.label_embeddings.weight  # share the tensor

        # Hierarchical mask (fixed binary buffer, not a parameter)
        M = build_M_hiera(l2_to_l1_mapping, num_l1, num_l2)  # (L, L)
        self.register_buffer("M_hiera", M)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            HLPDDecoderLayer(d_model, nhead, ffn_dim, decoder_dropout, self.num_labels)
            for _ in range(num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # Encode text
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H_enc = enc_out.last_hidden_state  # (B, seq_len, d_model)

        # Initial label representations: one embedding per label, expanded to batch
        label_ids = torch.arange(self.num_labels, device=input_ids.device)
        label_ids = label_ids.unsqueeze(0).expand(B, -1)  # (B, L)
        h = self.label_embeddings(label_ids)              # (B, L, d_model)

        # Pass through decoder layers (with optional gradient checkpointing)
        for layer in self.decoder_layers:
            if self.gradient_checkpointing_flag and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, H_enc, attention_mask, self.M_hiera,
                    use_reentrant=False,
                )
            else:
                h = layer(h, H_enc, attention_mask, self.M_hiera)

        h = self.final_norm(h)  # (B, L, d_model)

        # Tied projection: logits[b, l] = h[b, l] · label_emb[l]
        logits = torch.einsum("bld,ld->bl", h, self.label_embeddings.weight)  # (B, L)
        return logits
