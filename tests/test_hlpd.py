"""Tests for 02b_hlpd/src/ modules."""

from __future__ import annotations

import pytest
import torch

from helpers import import_from_hlpd

# Import HLPD modules using the helper
_ha_mod = import_from_hlpd("hierarchical_attention")
HierarchicalSelfAttention = _ha_mod.HierarchicalSelfAttention
build_taxonomy_distance_matrix = _ha_mod.build_taxonomy_distance_matrix

_losses_mod = import_from_hlpd("losses")
ZLPRLoss = _losses_mod.ZLPRLoss
HierarchicalBCELoss = _losses_mod.HierarchicalBCELoss
build_multi_hot_targets = _losses_mod.build_multi_hot_targets


class TestBuildTaxonomyDistanceMatrix:
    def test_shape(self):
        l2_to_l1 = {0: 0, 1: 0, 2: 1}
        matrix = build_taxonomy_distance_matrix(l2_to_l1, num_l1=2, num_l2=3)
        total = 1 + 2 + 3
        assert matrix.shape == (total, total)

    def test_root_self_distance(self):
        matrix = build_taxonomy_distance_matrix({0: 0}, num_l1=1, num_l2=1)
        assert matrix[0, 0].item() == 0.0

    def test_root_to_l1_distance(self):
        matrix = build_taxonomy_distance_matrix({0: 0}, num_l1=1, num_l2=1)
        assert matrix[0, 1].item() == -1.0

    def test_symmetry(self):
        l2_to_l1 = {0: 0, 1: 1}
        matrix = build_taxonomy_distance_matrix(l2_to_l1, num_l1=2, num_l2=2)
        assert torch.allclose(matrix, matrix.T)

    def test_sibling_l2_distance(self):
        l2_to_l1 = {0: 0, 1: 0}
        matrix = build_taxonomy_distance_matrix(l2_to_l1, num_l1=1, num_l2=2)
        assert matrix[2, 3].item() == -2.0

    def test_distant_l2_distance(self):
        l2_to_l1 = {0: 0, 1: 1}
        matrix = build_taxonomy_distance_matrix(l2_to_l1, num_l1=2, num_l2=2)
        assert matrix[3, 4].item() == -4.0


class TestHierarchicalSelfAttention:
    def test_output_shape(self):
        d_model, num_heads, num_labels, batch = 32, 4, 6, 2
        attn = HierarchicalSelfAttention(
            d_model=d_model, num_heads=num_heads, num_labels=num_labels,
        )
        x = torch.randn(batch, num_labels, d_model)
        m_hiera = torch.randn(num_labels, num_labels)
        out = attn(x, m_hiera)
        assert out.shape == (batch, num_labels, d_model)

    def test_gradient_flows(self):
        d_model, num_labels = 16, 4
        attn = HierarchicalSelfAttention(
            d_model=d_model, num_heads=2, num_labels=num_labels,
        )
        x = torch.randn(1, num_labels, d_model, requires_grad=True)
        m_hiera = torch.randn(num_labels, num_labels)
        out = attn(x, m_hiera)
        out.sum().backward()
        assert x.grad is not None

    def test_d_model_divisibility_check(self):
        with pytest.raises(AssertionError):
            HierarchicalSelfAttention(d_model=15, num_heads=4, num_labels=3)


class TestZLPRLoss:
    def test_output_shape(self):
        loss_fn = ZLPRLoss(reduction="mean")
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_reduction_none(self):
        loss_fn = ZLPRLoss(reduction="none")
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)

    def test_perfect_prediction_low_loss(self):
        loss_fn = ZLPRLoss(reduction="mean")
        logits = torch.zeros(2, 5)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        targets = torch.zeros(2, 5)
        targets[0, 0] = 1.0
        targets[1, 1] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.item() < 5.0

    def test_gradient_flows(self):
        logits = torch.randn(4, 8, requires_grad=True)
        targets = torch.zeros(4, 8)
        targets[:, 0] = 1.0
        loss = ZLPRLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestHierarchicalBCELoss:
    def test_output_shape(self):
        loss_fn = HierarchicalBCELoss(reduction="mean")
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_with_pos_weight(self):
        loss_fn = HierarchicalBCELoss(pos_weight=2.0)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0


class TestBuildMultiHotTargets:
    def test_shape(self):
        labels_l1 = torch.tensor([0, 1, 0])
        labels_l2 = torch.tensor([0, 2, 1])
        targets = build_multi_hot_targets(labels_l1, labels_l2, num_l1=3, num_l2=5)
        assert targets.shape == (3, 1 + 3 + 5)

    def test_root_always_active(self):
        labels_l1 = torch.tensor([0, 1])
        labels_l2 = torch.tensor([0, 1])
        targets = build_multi_hot_targets(labels_l1, labels_l2, num_l1=2, num_l2=3)
        assert targets[:, 0].sum() == 2.0

    def test_no_root(self):
        labels_l1 = torch.tensor([0])
        labels_l2 = torch.tensor([0])
        targets = build_multi_hot_targets(
            labels_l1, labels_l2, num_l1=2, num_l2=3, include_root=False,
        )
        assert targets[0, 0] == 0.0

    def test_correct_positions(self):
        labels_l1 = torch.tensor([1])
        labels_l2 = torch.tensor([2])
        targets = build_multi_hot_targets(labels_l1, labels_l2, num_l1=3, num_l2=5)
        assert targets[0, 0] == 1.0   # root
        assert targets[0, 2] == 1.0   # L1 class 1 -> position 1+1=2
        assert targets[0, 6] == 1.0   # L2 class 2 -> position 1+3+2=6
        assert targets[0].sum() == 3.0

    def test_num_active_labels(self):
        B = 8
        labels_l1 = torch.randint(0, 5, (B,))
        labels_l2 = torch.randint(0, 10, (B,))
        targets = build_multi_hot_targets(labels_l1, labels_l2, num_l1=5, num_l2=10)
        assert torch.all(targets.sum(dim=1) == 3.0)
