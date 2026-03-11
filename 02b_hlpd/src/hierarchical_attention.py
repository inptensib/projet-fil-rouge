"""Hierarchical Self-Attention module for HLPD.

Implements the hierarchical positional bias from the ECIR 2025 paper
(Torba, Laclau et al.). Standard self-attention is augmented with a
learnable bias derived from taxonomy tree distances:

    Attn_hiera(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + W * M_hiera) V

where M_hiera[i,j] = -dist(v_i, v_j) in the taxonomy tree and W is a
learnable weight matrix (Hadamard product).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_taxonomy_distance_matrix(
    l2_to_l1: dict[int, int],
    num_l1: int,
    num_l2: int,
) -> torch.Tensor:
    """Build the static hierarchical distance matrix M_hiera.

    The taxonomy is a 2-level tree with a virtual root:
        root -> L1_i -> L2_j

    Distance rules (shortest path in the undirected tree):
        - dist(root, root) = 0
        - dist(root, L1_i) = 1
        - dist(root, L2_j) = 2
        - dist(L1_i, L1_j) = 2          (via root)
        - dist(L1_i, L2_j) = 1          if L2_j is child of L1_i
        - dist(L1_i, L2_j) = 3          otherwise (L1_i -> root -> L1_k -> L2_j)
        - dist(L2_i, L2_j) = 2          if same parent
        - dist(L2_i, L2_j) = 4          if different parents

    The label ordering in the decoder is: [root, L1_0, ..., L1_{n1-1}, L2_0, ..., L2_{n2-1}].
    We include the virtual root node as position 0.

    Args:
        l2_to_l1: Mapping from L2 index to L1 index.
        num_l1: Number of L1 classes.
        num_l2: Number of L2 classes.

    Returns:
        Float tensor of shape (1 + num_l1 + num_l2, 1 + num_l1 + num_l2)
        with M_hiera[i,j] = -dist(node_i, node_j).
    """
    total = 1 + num_l1 + num_l2  # root + L1 nodes + L2 nodes

    # Build adjacency list for BFS
    adj: dict[int, list[int]] = {i: [] for i in range(total)}

    # root (0) connects to all L1 nodes (1..num_l1)
    for l1 in range(num_l1):
        node_l1 = 1 + l1
        adj[0].append(node_l1)
        adj[node_l1].append(0)

    # L1 nodes connect to their L2 children
    for l2_idx, l1_idx in l2_to_l1.items():
        node_l1 = 1 + l1_idx
        node_l2 = 1 + num_l1 + l2_idx
        if node_l2 < total:
            adj[node_l1].append(node_l2)
            adj[node_l2].append(node_l1)

    # BFS from each node to compute shortest paths
    dist_matrix = torch.zeros(total, total, dtype=torch.float32)
    for src in range(total):
        visited = torch.full((total,), -1, dtype=torch.long)
        visited[src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if visited[neighbor] == -1:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)
        # Nodes not reachable (disconnected L2 without mapping) get max distance
        visited[visited == -1] = total
        dist_matrix[src] = visited.float()

    # M_hiera = -distance
    return -dist_matrix


class HierarchicalSelfAttention(nn.Module):
    """Multi-head self-attention with hierarchical positional bias.

    Replaces standard relative position embeddings with a learnable bias
    derived from taxonomy tree distances.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        num_labels: Total number of label positions (1 + num_l1 + num_l2).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Learnable weight matrix W for hierarchical bias (shared across heads)
        # Shape: (num_labels, num_labels) -- broadcast over heads
        self.W_hiera = nn.Parameter(torch.zeros(num_labels, num_labels))
        nn.init.xavier_uniform_(self.W_hiera)

    def forward(
        self,
        x: torch.Tensor,
        m_hiera: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Label embeddings of shape (batch, num_labels, d_model).
            m_hiera: Static distance matrix of shape (num_labels, num_labels).
                     Values are -dist(v_i, v_j).
            key_padding_mask: Optional boolean mask of shape (batch, num_labels).
                              True means the position should be ignored.

        Returns:
            Output tensor of shape (batch, num_labels, d_model).
        """
        B, N, _ = x.shape

        # Project to Q, K, V and split heads
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention scores: (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Hierarchical bias: W * M_hiera (Hadamard product)
        # Both are (N, N), broadcast over (B, H)
        hiera_bias = self.W_hiera * m_hiera.to(x.device)  # (N, N)
        scores = scores + hiera_bias.unsqueeze(0).unsqueeze(0)  # (B, H, N, N)

        if key_padding_mask is not None:
            # Expand mask: (B, 1, 1, N) -> mask columns
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)

        return out
