"""Hierarchical attention bias matrix M_hiera for HLPD self-attention.

M ∈ {0,1}^{L×L} (L = num_l1 + num_l2) encodes three types of relationships:

1. Self (diagonal): every label attends to itself.
2. Parent-child: each L2 class ↔ its L1 parent (bidirectional).
3. Siblings: L2 classes sharing the same L1 parent ↔ each other.

Label index layout:
    [0, num_l1)              → L1 classes (indices 0 … num_l1-1)
    [num_l1, num_l1+num_l2)  → L2 classes (indices num_l1 … L-1)
"""

from __future__ import annotations

import torch


def build_M_hiera(
    l2_to_l1_mapping: dict[int, int],
    num_l1: int,
    num_l2: int,
) -> torch.Tensor:
    """Build the hierarchical binary mask M of shape (L, L).

    Args:
        l2_to_l1_mapping: Mapping l2_index → l1_index (both 0-based).
        num_l1: Number of L1 classes.
        num_l2: Number of L2 classes.

    Returns:
        Float32 tensor of shape (num_l1 + num_l2, num_l1 + num_l2) with 0/1 entries.
        Symmetric by construction.
    """
    L = num_l1 + num_l2
    M = torch.zeros(L, L)

    # 1. Diagonal (self-connection for all labels)
    M.fill_diagonal_(1.0)

    # 2. Parent-child edges (bidirectional)
    for l2_idx, l1_idx in l2_to_l1_mapping.items():
        l2_full = num_l1 + l2_idx
        M[l1_idx, l2_full] = 1.0
        M[l2_full, l1_idx] = 1.0

    # 3. Siblings: L2 classes sharing the same L1 parent
    l1_to_l2s: dict[int, list[int]] = {}
    for l2_idx, l1_idx in l2_to_l1_mapping.items():
        l1_to_l2s.setdefault(l1_idx, []).append(l2_idx)

    for siblings in l1_to_l2s.values():
        if len(siblings) < 2:
            continue
        full_idx = torch.tensor([num_l1 + j for j in siblings], dtype=torch.long)
        # Vectorised all-pairs assignment
        M[full_idx.unsqueeze(1), full_idx.unsqueeze(0)] = 1.0

    return M
