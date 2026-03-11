"""Tests for 02_hydra/src/losses.py."""

from __future__ import annotations

import pytest
import torch

from helpers import import_from_hydra

# Import at module level using helper
_losses_mod = import_from_hydra("losses")
FocalLoss = _losses_mod.FocalLoss
HYDRALoss = _losses_mod.HYDRALoss


class TestFocalLoss:
    def test_output_shape_mean(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_output_shape_none(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.shape == (4,)

    def test_output_shape_sum(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.shape == ()

    def test_gamma_zero_matches_ce(self):
        """With gamma=0, FocalLoss should approximate CrossEntropyLoss."""
        torch.manual_seed(42)
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))

        focal = FocalLoss(gamma=0.0, reduction="mean")
        ce = torch.nn.CrossEntropyLoss(reduction="mean")

        focal_loss = focal(logits, labels)
        ce_loss = ce(logits, labels)
        assert torch.allclose(focal_loss, ce_loss, atol=1e-5)

    def test_gradient_flows(self):
        logits = torch.randn(4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (4,))
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestHYDRALoss:
    def _make_inputs(self, num_l1=5, num_l2=10, batch=4):
        logits_l1 = torch.randn(batch, num_l1)
        logits_l2 = torch.randn(batch, num_l2)
        labels_l1 = torch.randint(0, num_l1, (batch,))
        labels_l2 = torch.randint(0, num_l2, (batch,))
        return logits_l1, logits_l2, labels_l1, labels_l2

    def test_local_variant(self):
        loss_fn = HYDRALoss(architecture="local")
        l1, l2, y1, y2 = self._make_inputs()
        total, loss_l1, loss_l2 = loss_fn([l1, l2], None, y1, y2)
        assert total.item() >= 0
        assert loss_l1.item() >= 0
        assert loss_l2.item() >= 0

    def test_global_variant(self):
        num_l1, num_l2 = 5, 10
        loss_fn = HYDRALoss(
            architecture="global", alpha=1.0,
            num_classes_l1=num_l1, num_classes_l2=num_l2,
        )
        l1, l2, y1, y2 = self._make_inputs(num_l1, num_l2)
        unified = torch.randn(4, num_l1 + num_l2)
        total, loss_l1, loss_l2 = loss_fn([l1, l2], unified, y1, y2)
        assert total.item() >= 0

    def test_nested_variant(self):
        num_l1, num_l2 = 5, 10
        loss_fn = HYDRALoss(
            architecture="nested", alpha=0.5,
            num_classes_l1=num_l1, num_classes_l2=num_l2,
        )
        l1, l2, y1, y2 = self._make_inputs(num_l1, num_l2)
        unified = torch.randn(4, num_l1 + num_l2)
        total, _, _ = loss_fn([l1, l2], unified, y1, y2)
        assert total.item() >= 0

    def test_use_ce_instead_of_focal(self):
        loss_fn = HYDRALoss(architecture="local", use_focal=False)
        l1, l2, y1, y2 = self._make_inputs()
        total, _, _ = loss_fn([l1, l2], None, y1, y2)
        assert total.item() >= 0

    def test_gradient_flows(self):
        l1 = torch.randn(4, 5, requires_grad=True)
        l2 = torch.randn(4, 10, requires_grad=True)
        y1 = torch.randint(0, 5, (4,))
        y2 = torch.randint(0, 10, (4,))
        loss_fn = HYDRALoss(architecture="local")
        total, _, _ = loss_fn([l1, l2], None, y1, y2)
        total.backward()
        assert l1.grad is not None
        assert l2.grad is not None
