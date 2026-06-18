"""Tests for rakuten/models/hydra/losses.py."""

from __future__ import annotations

import pytest
import torch
from rakuten.models.hydra.losses import FocalLoss, HYDRALoss


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

    def test_focal_loss_gamma_zero_equals_cross_entropy(self):
        """FocalLoss(gamma=0) must be numerically equal to CrossEntropyLoss."""
        torch.manual_seed(123)
        logits = torch.randn(32, 8)
        labels = torch.randint(0, 8, (32,))

        focal = FocalLoss(gamma=0.0, reduction="mean")
        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        assert torch.allclose(focal(logits, labels), ce(logits, labels), atol=1e-5)

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

    def test_use_focal_true(self):
        """HYDRALoss(use_focal=True) doit produire une loss finie non-NaN."""
        loss_fn = HYDRALoss(architecture="local", use_focal=True, focal_gamma=2.0)
        l1, l2, y1, y2 = self._make_inputs()
        total, _, _ = loss_fn([l1, l2], None, y1, y2)
        assert total.item() >= 0
        assert not torch.isnan(total)

    def test_l1_always_uses_crossentropy(self):
        """L1 doit toujours utiliser CrossEntropy, même si use_focal=True."""
        torch.manual_seed(42)
        logits = torch.zeros(4, 10)
        logits[:, 0] = 10.0          # exemples faciles — p_t ≈ 1
        labels = torch.zeros(4, dtype=torch.long)

        fn_focal = HYDRALoss(architecture="local", use_focal=True, focal_gamma=2.0)
        fn_ce    = HYDRALoss(architecture="local", use_focal=False)

        _, loss_l1_focal, _ = fn_focal([logits, logits], None, labels, labels)
        _, loss_l1_ce,    _ = fn_ce(   [logits, logits], None, labels, labels)

        # L1 toujours CE → valeurs identiques
        assert torch.allclose(loss_l1_focal, loss_l1_ce, atol=1e-5)

    def test_focal_downweights_easy_examples(self):
        """FocalLoss doit être < CrossEntropy sur exemples faciles (p_t ≈ 1)."""
        torch.manual_seed(42)
        logits = torch.zeros(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 10.0  # confiance max sur classe 0 → exemple facile
        focal = HYDRALoss(architecture="local", use_focal=True, focal_gamma=2.0)
        ce    = HYDRALoss(architecture="local", use_focal=False)
        focal_loss, _, _ = focal([logits, logits], None, labels, labels)
        ce_loss,    _, _ = ce(   [logits, logits], None, labels, labels)
        assert focal_loss.item() < ce_loss.item()

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


class TestRebalancedLosses:
    """class_weights / logit_adjustment: None = bit-identical, tensors = effective."""

    def _make_inputs(self, num_l1=5, num_l2=10, batch=4, seed=7):
        torch.manual_seed(seed)
        logits_l1 = torch.randn(batch, num_l1)
        logits_l2 = torch.randn(batch, num_l2)
        labels_l1 = torch.randint(0, num_l1, (batch,))
        labels_l2 = torch.randint(0, num_l2, (batch,))
        return logits_l1, logits_l2, labels_l1, labels_l2

    def test_none_defaults_are_neutral(self):
        """Explicit None tensors give the exact same loss as before."""
        l1, l2, y1, y2 = self._make_inputs()
        plain = HYDRALoss(architecture="local")
        explicit = HYDRALoss(
            architecture="local",
            class_weights_l1=None, class_weights_l2=None,
            logit_adjustment_l1=None, logit_adjustment_l2=None,
        )
        total_a, l1_a, l2_a = plain([l1, l2], None, y1, y2)
        total_b, l1_b, l2_b = explicit([l1, l2], None, y1, y2)
        assert torch.equal(total_a, total_b)
        assert torch.equal(l1_a, l1_b)
        assert torch.equal(l2_a, l2_b)

    def test_focal_class_weights_match_weighted_ce_at_gamma_zero(self):
        """FocalLoss(gamma=0, class_weights=w) == CrossEntropyLoss(weight=w)."""
        l1, _, y1, _ = self._make_inputs()
        weights = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.8])
        focal = FocalLoss(gamma=0.0, class_weights=weights)
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        assert torch.allclose(focal(l1, y1), ce(l1, y1), atol=1e-5)

    def test_focal_logit_adjustment_added_before_loss(self):
        """FocalLoss(logit_adjustment=adj) == FocalLoss plain on logits + adj."""
        _, l2, _, y2 = self._make_inputs()
        adj = torch.linspace(-2.0, 0.0, 10)
        adjusted_fn = FocalLoss(gamma=2.0, logit_adjustment=adj)
        plain_fn = FocalLoss(gamma=2.0)
        assert torch.allclose(adjusted_fn(l2, y2), plain_fn(l2 + adj, y2))

    def test_hydra_class_weights_l1(self):
        """L1 stays CrossEntropy, now weighted."""
        l1, l2, y1, y2 = self._make_inputs()
        weights = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.8])
        loss_fn = HYDRALoss(architecture="local", class_weights_l1=weights)
        _, loss_l1, _ = loss_fn([l1, l2], None, y1, y2)
        expected = torch.nn.CrossEntropyLoss(weight=weights)(l1, y1)
        assert torch.allclose(loss_l1, expected, atol=1e-5)

    def test_hydra_logit_adjustment_l2(self):
        """The L2 adjustment is added to the L2 logits before the loss."""
        l1, l2, y1, y2 = self._make_inputs()
        adj = torch.linspace(-3.0, 0.0, 10)
        loss_fn = HYDRALoss(architecture="local", logit_adjustment_l2=adj)
        plain = HYDRALoss(architecture="local")
        _, loss_l1_adj, loss_l2_adj = loss_fn([l1, l2], None, y1, y2)
        _, loss_l1_ref, loss_l2_ref = plain([l1, l2 + adj], None, y1, y2)
        assert torch.allclose(loss_l2_adj, loss_l2_ref, atol=1e-5)
        assert torch.allclose(loss_l1_adj, loss_l1_ref, atol=1e-5)

    def test_flat_path_rebalanced(self):
        """FlatHydra path (empty level_outputs): unified slices are adjusted."""
        num_l1, num_l2 = 5, 10
        _, _, y1, y2 = self._make_inputs(num_l1, num_l2)
        torch.manual_seed(11)
        unified = torch.randn(4, num_l1 + num_l2)
        adj_l2 = torch.linspace(-3.0, 0.0, num_l2)
        weights_l1 = torch.rand(num_l1) + 0.5
        loss_fn = HYDRALoss(
            architecture="global",
            num_classes_l1=num_l1, num_classes_l2=num_l2,
            class_weights_l1=weights_l1, logit_adjustment_l2=adj_l2,
        )
        total, loss_l1, loss_l2 = loss_fn([], unified, y1, y2)
        expected_l1 = torch.nn.CrossEntropyLoss(weight=weights_l1)(
            unified[:, :num_l1], y1
        )
        expected_l2 = torch.nn.CrossEntropyLoss()(unified[:, num_l1:] + adj_l2, y2)
        assert torch.allclose(loss_l1, expected_l1, atol=1e-5)
        assert torch.allclose(loss_l2, expected_l2, atol=1e-5)
        assert torch.allclose(total, 0.5 * expected_l1 + expected_l2, atol=1e-5)

    def test_rebalanced_gradient_flows(self):
        l1 = torch.randn(4, 5, requires_grad=True)
        l2 = torch.randn(4, 10, requires_grad=True)
        y1 = torch.randint(0, 5, (4,))
        y2 = torch.randint(0, 10, (4,))
        loss_fn = HYDRALoss(
            architecture="local",
            class_weights_l1=torch.rand(5) + 0.5,
            class_weights_l2=torch.rand(10) + 0.5,
            logit_adjustment_l1=torch.randn(5),
            logit_adjustment_l2=torch.randn(10),
        )
        total, _, _ = loss_fn([l1, l2], None, y1, y2)
        total.backward()
        assert l1.grad is not None
        assert l2.grad is not None
