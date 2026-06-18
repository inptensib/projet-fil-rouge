"""Tests for shared/training_utils.py.

Uses mocked tokenizer and mocked mlflow module to stay offline.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pytest
import torch

from rakuten.common.training_utils import (
    EarlyStopping,
    MLflowSession,
    compute_class_weights,
    compute_logit_adjustment,
    get_linear_schedule_with_warmup,
    make_hierarchical_loader,
    resolve_rebalancing,
)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _make_optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


class TestLinearScheduleWithWarmup:
    def test_lambda_values(self):
        """Lambda: 0 at step 0, linear up to 1 at warmup end, linear down to 0."""
        scheduler = get_linear_schedule_with_warmup(
            _make_optimizer(), num_warmup_steps=10, num_training_steps=110
        )
        lam = scheduler.lr_lambdas[0]
        assert lam(0) == 0.0
        assert lam(5) == pytest.approx(0.5)
        assert lam(10) == pytest.approx(1.0)
        assert lam(60) == pytest.approx(0.5)
        assert lam(110) == pytest.approx(0.0)
        assert lam(200) == 0.0  # clamped, never negative

    def test_zero_warmup(self):
        """num_warmup_steps=0 must not divide by zero."""
        scheduler = get_linear_schedule_with_warmup(
            _make_optimizer(), num_warmup_steps=0, num_training_steps=10
        )
        lam = scheduler.lr_lambdas[0]
        assert lam(0) == pytest.approx(1.0)
        assert lam(5) == pytest.approx(0.5)
        assert lam(10) == pytest.approx(0.0)

    def test_lr_applied_to_optimizer(self):
        """Stepping the scheduler scales the optimizer lr."""
        optimizer = _make_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=4, num_training_steps=8
        )
        assert optimizer.param_groups[0]["lr"] == 0.0
        for _ in range(4):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_parity_with_trainer_implementations(self):
        """The shared copy must match the per-trainer implementations."""
        from helpers import (
            DISTRIBUTED_ROOT,
            HLPD_ROOT,
            HYDRA_ROOT,
            _import_from,
            import_from_hlpd,
            import_from_hydra,
        )

        flat_trainer = _import_from(
            str(_PROJECT_ROOT / "01_baselines"),
            "flat_trainer",
            exclude_roots=[HYDRA_ROOT, HLPD_ROOT, DISTRIBUTED_ROOT],
        )
        hydra_trainer = import_from_hydra("trainer")
        hlpd_trainer = import_from_hlpd("trainer")

        variants = [
            hydra_trainer.get_linear_schedule_with_warmup,
            flat_trainer._get_linear_schedule_with_warmup,
            hlpd_trainer._get_linear_schedule_with_warmup,
        ]
        ref = get_linear_schedule_with_warmup(
            _make_optimizer(), num_warmup_steps=7, num_training_steps=53
        ).lr_lambdas[0]
        for factory in variants:
            lam = factory(_make_optimizer(), 7, 53).lr_lambdas[0]
            for step in range(0, 60):
                assert lam(step) == pytest.approx(ref(step))


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_scheme_none_returns_none(self):
        assert compute_class_weights(Counter({0: 4}), [0], "none") is None

    def test_inv_sqrt_values(self):
        weights = compute_class_weights(
            Counter({0: 4, 1: 1}), [0, 1], "inv_sqrt", normalize=False
        )
        assert torch.allclose(weights, torch.tensor([0.5, 1.0]))

    def test_inv_values(self):
        weights = compute_class_weights(
            Counter({0: 4, 1: 1}), [0, 1], "inv", normalize=False
        )
        assert torch.allclose(weights, torch.tensor([0.25, 1.0]))

    def test_normalized_mean_is_one(self):
        weights = compute_class_weights(
            Counter({0: 4, 1: 1}), [0, 1], "inv_sqrt"
        )
        assert torch.allclose(weights, torch.tensor([2 / 3, 4 / 3]))
        assert weights.mean().item() == pytest.approx(1.0)

    def test_zero_count_class_gets_min_nonzero_weight(self):
        """100% synthetic classes (ORPH): weight of the smallest real count."""
        weights = compute_class_weights(
            Counter({0: 4, 2: 16}), [0, 1, 2], "inv_sqrt", normalize=False
        )
        assert torch.allclose(weights, torch.tensor([0.5, 0.5, 0.25]))

    def test_classes_order_defines_tensor_order(self):
        weights = compute_class_weights(
            Counter({0: 1, 1: 4}), [1, 0], "inv_sqrt", normalize=False
        )
        assert torch.allclose(weights, torch.tensor([0.5, 1.0]))

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme"):
            compute_class_weights(Counter({0: 4}), [0], "sqrt")

    def test_all_zero_counts_raises(self):
        with pytest.raises(ValueError, match="non nul"):
            compute_class_weights(Counter(), [0, 1], "inv_sqrt")


# ---------------------------------------------------------------------------
# compute_logit_adjustment
# ---------------------------------------------------------------------------

class TestComputeLogitAdjustment:
    def test_tau_zero_returns_none(self):
        assert compute_logit_adjustment(Counter({0: 9}), [0], tau=0.0) is None

    def test_smoothed_log_priors(self):
        adj = compute_logit_adjustment(Counter({0: 9, 1: 4}), [0, 1], tau=1.0)
        expected = torch.tensor([math.log(10 / 15), math.log(5 / 15)])
        assert torch.allclose(adj, expected)

    def test_tau_scales_linearly(self):
        counts = Counter({0: 9, 1: 4})
        adj1 = compute_logit_adjustment(counts, [0, 1], tau=1.0)
        adj2 = compute_logit_adjustment(counts, [0, 1], tau=2.0)
        assert torch.allclose(adj2, 2.0 * adj1)

    def test_zero_count_class_is_finite(self):
        """Add-one smoothing: a class with no real sample keeps a finite prior."""
        adj = compute_logit_adjustment(Counter({0: 3}), [0, 1], tau=1.0)
        assert torch.isfinite(adj).all()
        assert torch.allclose(adj, torch.tensor([math.log(0.8), math.log(0.2)]))

    def test_rare_class_gets_lower_offset(self):
        """The rare-class offset is more negative: its training logit must
        compensate more (logit-adjusted loss)."""
        adj = compute_logit_adjustment(Counter({0: 100, 1: 1}), [0, 1], tau=1.0)
        assert adj[1] < adj[0]


# ---------------------------------------------------------------------------
# resolve_rebalancing
# ---------------------------------------------------------------------------

class TestResolveRebalancing:
    _KEYS = (
        "class_weights_l1", "class_weights_l2",
        "logit_adjustment_l1", "logit_adjustment_l2",
    )

    def _data(self) -> dict:
        return {
            "synthetic_policy": "drop_oversampled",
            "real_class_counts_l1": Counter({0: 6, 1: 3}),
            "real_class_counts_l2": Counter({0: 4, 1: 2, 2: 3}),
        }

    def test_disabled_returns_all_none_without_guard(self):
        """Options off: no tensor and no guard, even on a bare data dict."""
        tensors = resolve_rebalancing({}, 2, 3)
        assert set(tensors) == set(self._KEYS)
        assert all(tensors[k] is None for k in self._KEYS)

    def test_guard_requires_drop_oversampled(self):
        data = self._data()
        data["synthetic_policy"] = "keep_all"
        with pytest.raises(ValueError, match="drop_oversampled"):
            resolve_rebalancing(data, 2, 3, class_weighting="inv_sqrt")

    def test_guard_missing_policy_flag(self):
        with pytest.raises(ValueError, match="drop_oversampled"):
            resolve_rebalancing({}, 2, 3, logit_adjustment_tau=1.0)

    def test_guard_missing_counts(self):
        with pytest.raises(ValueError, match="real_class_counts"):
            resolve_rebalancing(
                {"synthetic_policy": "drop_oversampled"}, 2, 3,
                class_weighting="inv_sqrt",
            )

    def test_full_resolution_shapes(self):
        tensors = resolve_rebalancing(
            self._data(), 2, 3,
            class_weighting="inv_sqrt", logit_adjustment_tau=1.0,
        )
        assert tensors["class_weights_l1"].shape == (2,)
        assert tensors["class_weights_l2"].shape == (3,)
        assert tensors["logit_adjustment_l1"].shape == (2,)
        assert tensors["logit_adjustment_l2"].shape == (3,)

    def test_partial_activation(self):
        tensors = resolve_rebalancing(
            self._data(), 2, 3, class_weighting="inv_sqrt",
        )
        assert tensors["class_weights_l1"] is not None
        assert tensors["logit_adjustment_l1"] is None
        tensors = resolve_rebalancing(
            self._data(), 2, 3, logit_adjustment_tau=1.0,
        )
        assert tensors["class_weights_l1"] is None
        assert tensors["logit_adjustment_l1"] is not None


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2, min_delta=0.001)
        assert es.update(0.5) is True
        assert es.best == 0.5
        assert es.counter == 0
        assert es.update(0.4) is False
        assert es.counter == 1
        assert es.update(0.6) is True
        assert es.best == 0.6
        assert es.counter == 0

    def test_min_delta_threshold(self):
        """Improvement <= min_delta does not count as progress."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.update(0.5)
        assert es.update(0.505) is False
        assert es.best == 0.5
        assert es.counter == 1
        assert es.update(0.52) is True
        assert es.best == 0.52

    def test_should_stop_after_patience(self):
        es = EarlyStopping(patience=2, min_delta=0.0)
        es.update(0.5)
        assert not es.should_stop
        es.update(0.5)
        assert not es.should_stop
        es.update(0.5)
        assert es.should_stop

    def test_state_injection(self):
        """Resume injects best/counter restored from a checkpoint."""
        es = EarlyStopping(patience=2, min_delta=0.001)
        es.best = 0.7
        es.counter = 1
        assert es.update(0.65) is False
        assert es.counter == 2
        assert es.should_stop
        assert es.best == 0.7


# ---------------------------------------------------------------------------
# MLflowSession
# ---------------------------------------------------------------------------

class TestMLflowSession:
    def _session(self, enabled: bool = True) -> MLflowSession:
        return MLflowSession("sqlite:///test.db", "test_experiment", enabled=enabled)

    def test_disabled_never_touches_mlflow(self):
        fake = mock.MagicMock()
        session = self._session(enabled=False)
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start(run_name="run", params={"lr": 1e-4})
            session.log_metrics({"f1": 0.5}, step=1)
            session.log_figure(object(), "fig.png")
            session.log_text("html", "report.html")
            session.log_artifact("model.pt")
            session.end()
        assert session.active is False
        fake.start_run.assert_not_called()
        fake.log_metrics.assert_not_called()
        fake.end_run.assert_not_called()

    def test_log_before_start_is_noop(self):
        fake = mock.MagicMock()
        session = self._session()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.log_metrics({"f1": 0.5})
        fake.log_metrics.assert_not_called()

    def test_start_and_logging_success(self):
        fake = mock.MagicMock()
        session = self._session()
        fig, table = object(), object()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start(run_name="run", params={"lr": 1e-4})
            assert session.active is True
            session.log_metrics({"f1": 0.5}, step=3)
            session.log_metric("f1", 0.5, step=3)
            session.log_figure(fig, "fig.png")
            session.log_dict({"a": 1}, "a.json")
            session.log_text("html", "report.html")
            session.log_table(table, "errors.json")
            session.log_artifact("model.pt")
            session.end()

        fake.set_tracking_uri.assert_called_once_with("sqlite:///test.db")
        fake.set_experiment.assert_called_once_with("test_experiment")
        fake.start_run.assert_called_once_with(run_name="run")
        fake.log_params.assert_called_once_with({"lr": 1e-4})
        fake.log_metrics.assert_called_once_with({"f1": 0.5}, step=3)
        fake.log_metric.assert_called_once_with("f1", 0.5, step=3)
        fake.log_figure.assert_called_once_with(fig, "fig.png")
        fake.log_dict.assert_called_once_with({"a": 1}, "a.json")
        fake.log_text.assert_called_once_with("html", "report.html")
        fake.log_table.assert_called_once_with(table, "errors.json")
        fake.log_artifact.assert_called_once_with("model.pt")
        fake.end_run.assert_called_once_with()
        assert session.active is False

    def test_start_failure_deactivates_session(self):
        fake = mock.MagicMock()
        fake.start_run.side_effect = RuntimeError("server down")
        session = self._session()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start(run_name="run", params={"lr": 1e-4})
            assert session.active is False
            session.log_metrics({"f1": 0.5})
        fake.log_metrics.assert_not_called()

    def test_logging_exception_is_swallowed(self):
        fake = mock.MagicMock()
        fake.log_metrics.side_effect = RuntimeError("boom")
        fake.log_figure.side_effect = RuntimeError("boom")
        session = self._session()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start()
            session.log_metrics({"f1": 0.5}, step=1)
            session.log_figure(object(), "fig.png")
        assert session.active is True

    def test_end_exception_is_swallowed(self):
        fake = mock.MagicMock()
        fake.end_run.side_effect = RuntimeError("boom")
        session = self._session()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start()
            session.end()
        assert session.active is False

    def test_end_with_status(self):
        fake = mock.MagicMock()
        session = self._session()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            session.start()
            session.end(status="FAILED")
        fake.end_run.assert_called_once_with(status="FAILED")


# ---------------------------------------------------------------------------
# make_hierarchical_loader
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(seq: int = 8):
    tok = mock.MagicMock()

    def _tokenize(text, **kwargs):
        return {
            "input_ids": torch.randint(0, 100, (1, seq)),
            "attention_mask": torch.ones(1, seq, dtype=torch.long),
        }

    tok.side_effect = _tokenize
    return tok


class TestMakeHierarchicalLoader:
    def test_batches_have_expected_keys_and_shapes(self):
        texts = ["produit a", "produit b", "produit c", "produit d"]
        labels_l1 = np.array([0, 1, 0, 1])
        labels_l2 = np.array([0, 1, 2, 3])
        loader = make_hierarchical_loader(
            texts, labels_l1, labels_l2,
            tokenizer=_make_mock_tokenizer(),
            max_length=8,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        assert len(loader) == 2
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 8)
        assert torch.equal(batch["label_l1"], torch.tensor([0, 1]))
        assert torch.equal(batch["label_l2"], torch.tensor([0, 1]))

    def test_drop_last(self):
        texts = ["a", "b", "c", "d", "e"]
        labels = np.array([0, 1, 0, 1, 0])
        loader = make_hierarchical_loader(
            texts, labels, labels,
            tokenizer=_make_mock_tokenizer(),
            max_length=8,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        assert len(loader) == 2

    def test_dataset_size_matches_inputs(self):
        texts = ["x"] * 7
        labels = np.zeros(7, dtype=np.int64)
        loader = make_hierarchical_loader(
            texts, labels, labels,
            tokenizer=_make_mock_tokenizer(),
            max_length=8,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        assert len(loader.dataset) == 7
        assert len(loader) == 2
