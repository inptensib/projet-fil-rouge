"""Tests for rakuten/models/hydra/trainer.py (HYDRATrainer).

Mini-training (2 epochs) with a mocked encoder/tokenizer and a disabled
mlflow module. Freezes the on-disk checkpoint formats (checkpoint_latest.pt,
best_model.pt), the resume path, and the evaluate() metric keys.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
import torch

from rakuten.models.hydra.trainer import HYDRATrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN = 32
_SEQ = 8
_BATCH = 4
_L1 = 3
_L2 = 6
_N_TRAIN = 20  # 5 batches per epoch (drop_last=True)
_N_EVAL = 12   # every L1/L2 class present in val/test
_STEPS_PER_EPOCH = _N_TRAIN // _BATCH
_MODEL_KEY = "HYDRA_local_CamemBERT"

# Frozen checkpoint formats (order included)
_LATEST_KEYS = [
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "epoch",
    "step_in_epoch",
    "global_step",
    "best_val_f1",
    "epochs_without_improvement",
    "history",
    "config",
]
_BEST_KEYS = ["model_state_dict", "config", "best_val_f1_l2", "epoch"]
_CONFIG_KEYS = [
    "variant",
    "encoder_name",
    "num_classes_l1",
    "num_classes_l2",
    "pooling",
    "project_embedding",
    "dropout",
    "hidden_dim",
]


# ---------------------------------------------------------------------------
# Mocks (picklable tokenizer: the dataloaders use forked workers)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Deterministic tokenizer stub."""

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.randint(0, 100, (1, _SEQ)),
            "attention_mask": torch.ones(1, _SEQ, dtype=torch.long),
        }


def _make_mock_encoder():
    """Encoder returning zeros: constant logits per eval, so the predicted
    class always has true positives (every class is present in val/test) and
    f1_l2_macro > min_delta — best_model.pt is guaranteed to be written."""
    encoder = mock.MagicMock()
    encoder.config.hidden_size = _HIDDEN

    def fake_forward(*args, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None and args:
            input_ids = args[0]
        B, S = input_ids.shape
        result = mock.MagicMock()
        result.last_hidden_state = torch.zeros(B, S, _HIDDEN)
        return result

    encoder.side_effect = fake_forward
    return encoder


@contextmanager
def _patched_environment():
    """Mock HuggingFace factories and deactivate mlflow (no repo db writes)."""
    fake_mlflow = mock.MagicMock()
    fake_mlflow.set_tracking_uri.side_effect = RuntimeError("mlflow disabled in tests")
    with mock.patch("transformers.AutoModel.from_pretrained") as auto_model, \
            mock.patch("transformers.AutoTokenizer.from_pretrained") as auto_tok, \
            mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
        auto_model.return_value = _make_mock_encoder()
        auto_tok.return_value = _FakeTokenizer()
        yield


# ---------------------------------------------------------------------------
# Data / trainer factories
# ---------------------------------------------------------------------------

def _make_data() -> dict:
    from sklearn.preprocessing import LabelEncoder

    l2_to_l1 = {i: i % _L1 for i in range(_L2)}
    l2_train = [i % _L2 for i in range(_N_TRAIN)]
    l2_eval = [i % _L2 for i in range(_N_EVAL)]

    le1, le2 = LabelEncoder(), LabelEncoder()
    le1.fit(list(range(_L1)))
    le2.fit(list(range(_L2)))

    return {
        "X_train": [f"produit {i}" for i in range(_N_TRAIN)],
        "X_val": [f"produit eval {i}" for i in range(_N_EVAL)],
        "X_test": [f"produit eval {i}" for i in range(_N_EVAL)],
        "y_l1_train": [l2_to_l1[c] for c in l2_train],
        "y_l1_val": [l2_to_l1[c] for c in l2_eval],
        "y_l1_test": [l2_to_l1[c] for c in l2_eval],
        "y_l2_train": l2_train,
        "y_l2_val": l2_eval,
        "y_l2_test": l2_eval,
        "l1_encoder": le1,
        "l2_encoder": le2,
        "l2_to_l1_mapping": l2_to_l1,
        "num_classes_l1": _L1,
        "num_classes_l2": _L2,
    }


def _make_trainer(save_root: Path, **overrides) -> HYDRATrainer:
    kwargs = dict(
        data=_make_data(),
        encoder_name="CamemBERT",
        variant="local",
        batch_size=_BATCH,
        epochs=2,
        max_length=_SEQ,
        hidden_dim=_HIDDEN,
        early_stopping_patience=10,
        fp16=False,
        save_dir=str(save_root / "models"),
    )
    kwargs.update(overrides)
    return HYDRATrainer(**kwargs)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One shared 2-epoch mini-training."""
    save_root = tmp_path_factory.mktemp("hydra_trainer")
    with _patched_environment():
        trainer = _make_trainer(save_root)
        result = trainer.train()
    return trainer, result, save_root / "models" / _MODEL_KEY


# ---------------------------------------------------------------------------
# Checkpoint formats (frozen contract)
# ---------------------------------------------------------------------------

class TestCheckpointFormats:
    def test_checkpoint_latest_exact_keys(self, trained):
        _, _, model_dir = trained
        ckpt = torch.load(model_dir / "checkpoint_latest.pt", weights_only=False)
        assert list(ckpt.keys()) == _LATEST_KEYS
        assert list(ckpt["config"].keys()) == _CONFIG_KEYS

    def test_checkpoint_latest_values(self, trained):
        trainer, _, model_dir = trained
        ckpt = torch.load(model_dir / "checkpoint_latest.pt", weights_only=False)
        assert ckpt["epoch"] == 2
        assert ckpt["step_in_epoch"] == 0
        assert ckpt["global_step"] == 2 * _STEPS_PER_EPOCH
        assert ckpt["history"]["epoch"] == [1, 2]
        assert ckpt["config"] == {
            "variant": "local",
            "encoder_name": "CamemBERT",
            "num_classes_l1": _L1,
            "num_classes_l2": _L2,
            "pooling": "cls",
            "project_embedding": True,
            "dropout": 0.3,
            "hidden_dim": _HIDDEN,
        }
        trainer._base_model.load_state_dict(ckpt["model_state_dict"])

    def test_best_model_exact_keys(self, trained):
        _, result, model_dir = trained
        ckpt = torch.load(model_dir / "best_model.pt", weights_only=False)
        assert list(ckpt.keys()) == _BEST_KEYS
        assert list(ckpt["config"].keys()) == _CONFIG_KEYS
        assert ckpt["best_val_f1_l2"] == pytest.approx(result["best_val_f1_l2"])
        assert 1 <= ckpt["epoch"] <= 2

    def test_train_result_and_completion_marker(self, trained):
        _, result, model_dir = trained
        assert set(result.keys()) == {
            "history", "test_metrics", "best_val_f1_l2", "total_time",
        }
        assert result["history"]["epoch"] == [1, 2]
        assert len(result["history"]["val_f1_l2_macro"]) == 2
        assert result["best_val_f1_l2"] > 0.0
        assert (model_dir / "training_complete.txt").exists()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_restores_and_continues(self, trained, tmp_path, capsys):
        _, _, model_dir = trained
        ckpt1 = torch.load(model_dir / "checkpoint_latest.pt", weights_only=False)

        with _patched_environment():
            trainer2 = _make_trainer(
                tmp_path, epochs=3,
                resume_from=str(model_dir / "checkpoint_latest.pt"),
            )
            result = trainer2.train()

        out = capsys.readouterr().out
        assert "[Resume] Reprise à epoch 2" in out
        assert f"global_step {2 * _STEPS_PER_EPOCH}" in out
        assert f"best F1={ckpt1['best_val_f1']:.4f}" in out

        # History restored from the checkpoint, extended by the third epoch
        assert result["history"]["epoch"] == [1, 2, 3]

        ckpt2 = torch.load(
            tmp_path / "models" / _MODEL_KEY / "checkpoint_latest.pt",
            weights_only=False,
        )
        assert ckpt2["epoch"] == 3
        assert ckpt2["global_step"] == 3 * _STEPS_PER_EPOCH
        assert ckpt2["best_val_f1"] >= ckpt1["best_val_f1"]

    def test_resume_fast_forwards_mid_epoch(self, trained, tmp_path, capsys):
        _, _, model_dir = trained
        ckpt = torch.load(model_dir / "checkpoint_latest.pt", weights_only=False)
        ckpt["epoch"] = 1
        ckpt["step_in_epoch"] = 3
        ckpt["global_step"] = _STEPS_PER_EPOCH + 3
        mid_path = tmp_path / "checkpoint_mid.pt"
        torch.save(ckpt, mid_path)

        with _patched_environment():
            trainer = _make_trainer(tmp_path, epochs=2, resume_from=str(mid_path))
            trainer.train()

        out = capsys.readouterr().out
        assert "step_in_epoch 3" in out

        ckpt2 = torch.load(
            tmp_path / "models" / _MODEL_KEY / "checkpoint_latest.pt",
            weights_only=False,
        )
        # 3 of the 5 batches fast-forwarded: only 2 optimizer steps run
        assert ckpt2["epoch"] == 2
        assert ckpt2["global_step"] == _STEPS_PER_EPOCH + 3 + (_STEPS_PER_EPOCH - 3)
        assert ckpt2["step_in_epoch"] == 0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_metric_keys(self, trained):
        trainer, _, _ = trained
        data = trainer.data
        loader = trainer._create_dataloader(
            data["X_val"], data["y_l1_val"], data["y_l2_val"], shuffle=False,
        )
        metrics = trainer.evaluate(loader)

        for key in ("f1_l1_macro", "f1_l2_macro", "hcr", "constrained_f1_macro"):
            assert key in metrics, f"clé manquante: {key}"
        for key in (
            "_preds_l1", "_preds_l2", "_labels_l1",
            "_labels_l2", "_confs_l1", "_confs_l2",
        ):
            assert len(metrics[key]) == _N_EVAL

    def test_no_tranche_without_real_counts(self, trained):
        """v1 data (no real_class_counts_l2): no f1_l2_by_tranche key."""
        trainer, _, _ = trained
        data = trainer.data
        loader = trainer._create_dataloader(
            data["X_val"], data["y_l1_val"], data["y_l2_val"], shuffle=False,
        )
        assert "f1_l2_by_tranche" not in trainer.evaluate(loader)

    def test_tranche_with_real_counts(self, tmp_path):
        """v2 data: f1_l2_by_tranche is added, consistent with f1_l2_macro."""
        from collections import Counter

        data = _make_data()
        data["real_class_counts_l2"] = Counter(data["y_l2_train"])
        with _patched_environment():
            trainer = _make_trainer(tmp_path, data=data)
            loader = trainer._create_dataloader(
                data["X_val"], data["y_l1_val"], data["y_l2_val"], shuffle=False,
            )
            metrics = trainer.evaluate(loader)

        tranche = metrics["f1_l2_by_tranche"]
        assert set(tranche) == {"many", "medium", "few"}
        # All counts < 20: every class is in the 'few' tranche
        assert tranche["few"]["n_classes"] == _L2
        assert tranche["few"]["support"] == _N_EVAL
        assert tranche["few"]["f1_macro"] == pytest.approx(metrics["f1_l2_macro"])
        assert tranche["many"]["f1_macro"] is None


# ---------------------------------------------------------------------------
# Rebalancing (class_weighting / logit_adjustment_tau)
# ---------------------------------------------------------------------------

def _make_v2_data() -> dict:
    from collections import Counter

    data = _make_data()
    data["synthetic_policy"] = "drop_oversampled"
    data["real_class_counts_l1"] = Counter(data["y_l1_train"])
    data["real_class_counts_l2"] = Counter(data["y_l2_train"])
    return data


class TestRebalancing:
    def test_guard_without_drop_oversampled(self, tmp_path):
        with _patched_environment():
            with pytest.raises(ValueError, match="drop_oversampled"):
                _make_trainer(tmp_path, class_weighting="inv_sqrt")

    def test_guard_tau_without_drop_oversampled(self, tmp_path):
        with _patched_environment():
            with pytest.raises(ValueError, match="drop_oversampled"):
                _make_trainer(tmp_path, logit_adjustment_tau=1.0)

    def test_guard_missing_real_counts(self, tmp_path):
        data = _make_v2_data()
        del data["real_class_counts_l1"]
        with _patched_environment():
            with pytest.raises(ValueError, match="real_class_counts"):
                _make_trainer(tmp_path, data=data, class_weighting="inv_sqrt")

    def test_rebalancing_tensors_reach_criterion(self, tmp_path):
        with _patched_environment():
            trainer = _make_trainer(
                tmp_path, data=_make_v2_data(),
                class_weighting="inv_sqrt", logit_adjustment_tau=1.0,
            )
        assert trainer.criterion.loss_fn_l1.weight.shape == (_L1,)
        assert trainer.criterion.logit_adjustment_l2.shape == (_L2,)
        assert trainer.criterion.loss_fn_l1.weight.device == trainer.device
        assert trainer.criterion.logit_adjustment_l2.device == trainer.device

    def test_default_trainer_loss_unchanged(self, tmp_path):
        """Neutrality: without the options, same loss as before on a fixed batch."""
        import torch.nn.functional as F

        torch.manual_seed(0)
        logits_l1 = torch.randn(_BATCH, _L1)
        logits_l2 = torch.randn(_BATCH, _L2)
        y1 = torch.randint(0, _L1, (_BATCH,))
        y2 = torch.randint(0, _L2, (_BATCH,))

        with _patched_environment():
            trainer = _make_trainer(tmp_path, data=_make_v2_data())
        total, _, _ = trainer.criterion([logits_l1, logits_l2], None, y1, y2)
        expected = 0.5 * F.cross_entropy(logits_l1, y1) + F.cross_entropy(logits_l2, y2)
        assert torch.allclose(total, expected, atol=1e-6)
        assert trainer.criterion.loss_fn_l1.weight is None
        assert trainer.criterion.logit_adjustment_l2 is None
