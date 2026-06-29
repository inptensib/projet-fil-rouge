"""HLPD — Hierarchical Label Pre-populated Decoder (CamemBERT backbone).

Reference: Torba, Laclau et al., ECIR 2025.

Architecture:
    Text  -> CamemBERT encoder -> H_text (B, seq_len, d_model)
    Label descriptions -> CamemBERT [CLS] -> label embeddings (L, d_model), cached once
        ->  LabelPooler (per-label non-linear projection)
        ->  Hierarchical self-attention  (scores += alpha * M_hiera + beta)   [no batch dim]
        ->  expand to (B, L, d_model)
        ->  Cross-attention (Q=labels, K=V=H_text) + FFN   x num_decoder_layers
        ->  LabelWisePooler -> logits (B, L)

Design notes:
    - Label embeddings are computed once from their French descriptions and cached
      (frozen). This removes the per-step cost of re-encoding L descriptions.
    - The hierarchical self-attention runs without a batch dimension: the labels are
      shared across the batch, so the cost is O(L^2) rather than O(B * L^2).
    - The hierarchical bias is parameterised by two scalars (alpha, beta) instead of a
      full L*L learnable matrix, which keeps the parameter count independent of L.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .attention_bias import build_M_hiera


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class WeightMask(nn.Module):
    """Learnable scalar scaling of the hierarchical bias: ``alpha * M_hiera + beta``.

    Two parameters only, so the cost is independent of the number of labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, M_hiera: torch.Tensor) -> torch.Tensor:
        return self.alpha * M_hiera + self.beta


class LabelPooler(nn.Module):
    """Per-label non-linear projection (no weights shared across labels).

    For each label i:  out_i = tanh(H_i * dense_i) * output_i + bias_i
    """

    def __init__(self, num_labels: int, d_model: int) -> None:
        super().__init__()
        self.dense = nn.Parameter(torch.empty(num_labels, d_model))
        self.output = nn.Parameter(torch.empty(num_labels, d_model))
        self.bias = nn.Parameter(torch.zeros(num_labels, d_model))
        nn.init.normal_(self.dense, std=0.02)
        nn.init.normal_(self.output, std=0.02)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        dtype = H.dtype
        return torch.tanh(H * self.dense.to(dtype)) * self.output.to(dtype) + self.bias.to(dtype)


class LabelWisePooler(nn.Module):
    """Per-label classification head: one logit per label.

    For each label i:  logit_i = sum(tanh(H_i * dense_i) * output_i) + bias_i
    """

    def __init__(self, num_labels: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Parameter(torch.empty(num_labels, d_model))
        self.output = nn.Parameter(torch.empty(num_labels, d_model))
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.dense, std=0.02)
        nn.init.normal_(self.output, std=0.02)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        dtype = H.dtype
        out = torch.tanh(H * self.dense.to(dtype))          # (B, L, D)
        out = self.dropout(out)
        return (out * self.output.to(dtype)).sum(dim=-1) + self.bias.to(dtype)  # (B, L)


class HierarchicalSelfAttention(nn.Module):
    """Self-attention over the labels with an additive hierarchical distance bias.

    Input:  H (L, D)  — no batch dimension (labels are shared across the batch).
    Output: (L, D). Memory is O(num_heads * L^2), independent of batch size.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.weight_mask = WeightMask()

    def forward(self, H: torch.Tensor, M_hiera: torch.Tensor) -> torch.Tensor:
        N, D = H.shape
        h, hd = self.num_heads, self.head_dim

        Q = self.q_proj(H).reshape(N, h, hd).permute(1, 0, 2)  # (h, N, hd)
        K = self.k_proj(H).reshape(N, h, hd).permute(1, 0, 2)
        V = self.v_proj(H).reshape(N, h, hd).permute(1, 0, 2)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale          # (h, N, N)
        hiera_bias = self.weight_mask(M_hiera).unsqueeze(0).to(scores.dtype)
        scores = scores + hiera_bias

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.bmm(attn, V).permute(1, 0, 2).reshape(N, D)        # (N, D)
        out = self.out_proj(out)
        return self.norm(H + out)


class LabelTextCrossAttention(nn.Module):
    """Cross-attention: labels are queries, text tokens are keys/values.

    Input:  H_label (B, L, D), H_text (B, T, D), text_mask (B, T).
    Output: (B, L, D).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        H_label: torch.Tensor,    # (B, L, D)
        H_text: torch.Tensor,     # (B, T, D)
        text_mask: torch.Tensor,  # (B, T) — 1=token, 0=padding
    ) -> torch.Tensor:
        B, N, D = H_label.shape
        _, T, _ = H_text.shape
        h, hd = self.num_heads, self.head_dim

        Q = self.q_proj(H_label).reshape(B, N, h, hd).transpose(1, 2)  # (B, h, N, hd)
        K = self.k_proj(H_text).reshape(B, T, h, hd).transpose(1, 2)   # (B, h, T, hd)
        V = self.v_proj(H_text).reshape(B, T, h, hd).transpose(1, 2)

        key_mask = text_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T): True=keep
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return self.norm(H_label + out)


class FeedForwardLayer(nn.Module):
    """Standard transformer FFN: Linear -> GELU -> Dropout -> Linear, with residual."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class HLPCrossLayer(nn.Module):
    """One decoder layer: cross-attention (labels -> text) followed by an FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = LabelTextCrossAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        H_label: torch.Tensor,
        H_text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.ffn(self.cross_attn(H_label, H_text, text_mask))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HLPD(nn.Module):
    """Hierarchical Label Pre-populated Decoder with a CamemBERT text encoder.

    The label embeddings are derived once from their text descriptions (CamemBERT
    [CLS]) and cached. Call :meth:`encode_labels` after moving the model to its
    device and before the first forward pass.

    Args:
        encoder_name:          HuggingFace model name for the text encoder.
        num_l1:                Number of L1 classes.
        num_l2:                Number of L2 classes.
        l2_to_l1_mapping:      Mapping l2_index -> l1_index (0-based).
        label_input_ids:       Tokenised label descriptions, shape (L, label_len).
        label_attention_mask:  Attention mask for the label descriptions, shape (L, label_len).
        num_decoder_layers:    Number of stacked cross-attention layers.
        nhead:                 Number of attention heads.
        ffn_dim:               FFN inner dimension.
        decoder_dropout:       Dropout in the decoder layers.
        gradient_checkpointing:Enable gradient checkpointing on the encoder.
        freeze_encoder:        Freeze the text encoder entirely.
        use_hiera_mask:        Use the hierarchical bias (False = ablation, zero bias).
    """

    def __init__(
        self,
        encoder_name: str,
        num_l1: int,
        num_l2: int,
        l2_to_l1_mapping: dict[int, int],
        label_input_ids: torch.Tensor,
        label_attention_mask: torch.Tensor,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        ffn_dim: int = 3072,
        decoder_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        freeze_encoder: bool = False,
        use_hiera_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.num_labels = num_l1 + num_l2
        self.use_hiera_mask = use_hiera_mask

        # Text encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d_model: int = self.encoder.config.hidden_size
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Tokenised label descriptions: only needed to (re)compute the cache, so they
        # are not part of the saved state_dict.
        self.register_buffer("label_input_ids", label_input_ids, persistent=False)
        self.register_buffer("label_attention_mask", label_attention_mask, persistent=False)
        # The cached label embeddings ARE saved, so a reloaded model needs no re-encode.
        self.register_buffer("label_embeds_cache", torch.zeros(self.num_labels, d_model))
        self.register_buffer("labels_encoded", torch.zeros(1, dtype=torch.uint8))

        # Per-label projection + hierarchical self-attention + decoder + head
        self.label_pooler = LabelPooler(self.num_labels, d_model)
        self.hier_self_attn = HierarchicalSelfAttention(d_model, nhead, decoder_dropout)
        self.decoder_layers = nn.ModuleList([
            HLPCrossLayer(d_model, nhead, ffn_dim, decoder_dropout)
            for _ in range(num_decoder_layers)
        ])
        self.label_wise_pooler = LabelWisePooler(self.num_labels, d_model, decoder_dropout)

        # Fixed hierarchical bias matrix (buffer)
        M = build_M_hiera(l2_to_l1_mapping, num_l1, num_l2)
        self.register_buffer("M_hiera", M)

    @torch.no_grad()
    def encode_labels(self, batch_size: int = 64) -> None:
        """Encode the label descriptions once (CamemBERT [CLS]) and cache them."""
        was_training = self.training
        self.eval()
        device = self.M_hiera.device
        embeds = []
        for start in range(0, self.num_labels, batch_size):
            end = min(start + batch_size, self.num_labels)
            ids = self.label_input_ids[start:end].to(device)
            mask = self.label_attention_mask[start:end].to(device)
            out = self.encoder(input_ids=ids, attention_mask=mask)
            embeds.append(out.last_hidden_state[:, 0, :].float())  # [CLS]
        self.label_embeds_cache = torch.cat(embeds, dim=0).to(device)
        self.labels_encoded.fill_(1)
        if was_training:
            self.train()

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:                 # (B, L)
        if int(self.labels_encoded.item()) == 0:
            # Lazy fallback (e.g. a model used directly without an explicit call).
            self.encode_labels()

        H_text = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, T, D)

        # Cached label embeddings -> per-label projection
        H_label = self.label_pooler(self.label_embeds_cache.to(H_text.dtype))  # (L, D)

        # Hierarchical self-attention (no batch dim)
        if self.use_hiera_mask:
            M = self.M_hiera.to(H_text.dtype)
        else:
            M = torch.zeros_like(self.M_hiera).to(H_text.dtype)
        H_label = self.hier_self_attn(H_label, M)  # (L, D)

        # Expand to the batch and run the cross-attention decoder
        H = H_label.unsqueeze(0).expand(input_ids.shape[0], -1, -1)  # (B, L, D)
        for layer in self.decoder_layers:
            H = layer(H, H_text, attention_mask)

        return self.label_wise_pooler(H)  # (B, L)
