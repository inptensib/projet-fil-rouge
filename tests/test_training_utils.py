"""Tests for shared/training_utils.py.

Uses mocked tokenizer and mocked mlflow module to stay offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.training_utils import (  # noqa: E402
    EarlyStopping,
    MLflowSession,
    get_linear_schedule_with_warmup,
    make_hierarchical_loader,
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
