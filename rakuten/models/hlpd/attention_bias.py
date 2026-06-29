"""Hierarchical distance bias M_hiera for HLPD self-attention.

M ∈ ℝ^{L×L} (L = num_l1 + num_l2) holds the *negative* shortest-path distance
between labels in the taxonomy tree ROOT → L1 → L2. Closer labels get a larger
(less negative) bias, so they are penalised less in the attention softmax.

Distances in a two-level tree (ROOT virtual node):
    self                       0   ->   0
    L1 <-> its child L2        1   ->  -1
    L1 <-> L1 (via ROOT)       2   ->  -2
    L2 <-> sibling L2          2   ->  -2
    L1 <-> non-child L2        3   ->  -3
    L2 <-> L2 (other parent)   4   ->  -4

Label index layout:
    [0, num_l1)              -> L1 classes
    [num_l1, num_l1+num_l2)  -> L2 classes
"""

from __future__ import annotations

import torch


def build_M_hiera(
    l2_to_l1_mapping: dict[int, int],
    num_l1: int,
    num_l2: int,
) -> torch.Tensor:
    """Build the hierarchical distance-bias matrix M of shape (L, L).

    Args:
        l2_to_l1_mapping: Mapping l2_index -> l1_index (both 0-based).
        num_l1: Number of L1 classes.
        num_l2: Number of L2 classes.

    Returns:
        Float32 tensor of shape (num_l1 + num_l2, num_l1 + num_l2) holding
        ``-shortest_path_distance``. Symmetric, zero diagonal.
    """
    L = num_l1 + num_l2
    M = torch.zeros(L, L, dtype=torch.float32)

    # Parent L1 index for each L2 (local index); -1 if unknown.
    parent = torch.full((num_l2,), -1, dtype=torch.long)
    for l2_idx, l1_idx in l2_to_l1_mapping.items():
        parent[int(l2_idx)] = int(l1_idx)

    # L1 <-> L1 block: 0 on the diagonal, -2 elsewhere (path through ROOT).
    l1_block = torch.full((num_l1, num_l1), -2.0)
    l1_block.fill_diagonal_(0.0)
    M[:num_l1, :num_l1] = l1_block

    # L1 <-> L2 block: -1 if the L1 is the L2's parent, else -3.
    a_idx = torch.arange(num_l1).unsqueeze(1)          # (num_l1, 1)
    p_row = parent.unsqueeze(0)                         # (1, num_l2)
    l1_l2 = torch.where(p_row == a_idx, -1.0, -3.0)     # (num_l1, num_l2)
    M[:num_l1, num_l1:] = l1_l2
    M[num_l1:, :num_l1] = l1_l2.t()

    # L2 <-> L2 block: 0 on the diagonal, -2 for siblings, -4 otherwise.
    p_col = parent.unsqueeze(1)                         # (num_l2, 1)
    p_rowv = parent.unsqueeze(0)                        # (1, num_l2)
    l2_block = torch.where(p_col == p_rowv, -2.0, -4.0)
    l2_block.fill_diagonal_(0.0)
    M[num_l1:, num_l1:] = l2_block

    return M
