"""Tests for 02_hydra/src/models.py.

Uses mocked HuggingFace models to avoid downloading weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest import mock

from helpers import import_from_hydra

_losses_mod = import_from_hydra("losses")
HYDRALoss = _losses_mod.HYDRALoss

_HIDDEN_SIZE = 64
_BATCH_SIZE = 2
_SEQ_LEN = 16
_NUM_L1 = 5
_NUM_L2 = 10


def _make_mock_encoder():
    """Create a mock encoder that returns a proper last_hidden_state."""
    encoder = mock.MagicMock()
    encoder.config.hidden_size = _HIDDEN_SIZE

    def fake_forward(*args, input_ids=None, attention_mask=None, **kwargs):
        # Handle both positional and keyword args
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        B, S = input_ids.shape
        result = mock.MagicMock()
        result.last_hidden_state = torch.randn(B, S, _HIDDEN_SIZE)
        return result

    encoder.side_effect = fake_forward
    return encoder


# Import models module from HYDRA using the helper
_models_mod = import_from_hydra("models")
HYDRALocal = _models_mod.HYDRALocal
HYDRAGlobal = _models_mod.HYDRAGlobal
HYDRANested = _models_mod.HYDRANested
create_hydra_model = _models_mod.create_hydra_model


@pytest.fixture(autouse=True)
def mock_auto_model():
    """Mock AutoModel.from_pretrained for all tests."""
    with mock.patch("transformers.AutoModel.from_pretrained") as m:
        m.return_value = _make_mock_encoder()
        yield m


class TestHYDRALocal:
    def test_output_shapes(self, mock_auto_model):
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, unified = model(input_ids, mask)
        assert len(level_outputs) == 2
        assert level_outputs[0].shape == (_BATCH_SIZE, _NUM_L1)
        assert level_outputs[1].shape == (_BATCH_SIZE, _NUM_L2)
        assert unified is None

    def test_no_projection(self, mock_auto_model):
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            project_embedding=False,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, _ = model(input_ids, mask)
        assert level_outputs[0].shape == (_BATCH_SIZE, _NUM_L1)


class TestHYDRAGlobal:
    def test_output_shapes(self, mock_auto_model):
        model = HYDRAGlobal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, unified = model(input_ids, mask)
        assert level_outputs[0].shape == (_BATCH_SIZE, _NUM_L1)
        assert unified is not None
        assert unified.shape == (_BATCH_SIZE, _NUM_L1 + _NUM_L2)


class TestHYDRANested:
    def test_output_shapes(self, mock_auto_model):
        model = HYDRANested(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, unified = model(input_ids, mask)
        assert level_outputs[0].shape == (_BATCH_SIZE, _NUM_L1)
        assert unified is not None
        assert unified.shape == (_BATCH_SIZE, _NUM_L1 + _NUM_L2)


class TestCreateHydraModel:
    def test_factory_local(self, mock_auto_model):
        model = create_hydra_model(
            variant="local", encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        assert isinstance(model, HYDRALocal)

    def test_factory_global(self, mock_auto_model):
        model = create_hydra_model(
            variant="global", encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        assert isinstance(model, HYDRAGlobal)

    def test_factory_nested(self, mock_auto_model):
        model = create_hydra_model(
            variant="nested", encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        assert isinstance(model, HYDRANested)

    def test_factory_invalid(self, mock_auto_model):
        with pytest.raises(ValueError, match="Unknown variant"):
            create_hydra_model(
                variant="invalid", encoder_name="mock-model",
                num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            )


class TestPoolingMethods:
    @pytest.mark.parametrize("pooling", ["cls", "mean", "max", "last"])
    def test_all_pooling(self, mock_auto_model, pooling):
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            pooling=pooling,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, _ = model(input_ids, mask)
        assert level_outputs[0].shape == (_BATCH_SIZE, _NUM_L1)

    def test_invalid_pooling(self, mock_auto_model):
        with pytest.raises(ValueError, match="Unknown pooling"):
            HYDRALocal(
                encoder_name="mock-model",
                num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
                pooling="invalid",
            )

    def test_max_pooling_ignores_padding(self, mock_auto_model):
        """Max pooling must ignore padding tokens (mask=0) via -1e9 fill."""
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            pooling="max", project_embedding=False,
        )
        # Manually test _pool with a controlled hidden_states
        B, S, H = 1, 4, _HIDDEN_SIZE
        hidden = torch.full((B, S, H), -5.0)
        hidden[0, 0, :] = 1.0  # only first token is real
        mask = torch.zeros(B, S, dtype=torch.long)
        mask[0, 0] = 1  # only first token unmasked
        pooled = model._pool(hidden, mask)
        # Should pick the unmasked token (value 1.0), not padded ones (-5.0)
        assert pooled.shape == (B, H)
        assert torch.allclose(pooled, torch.ones(B, H))


class TestNestedGradientBackprop:
    def test_nested_gradient_backprop(self, mock_auto_model):
        """Gradients must flow through the nested head back to the encoder."""
        model = HYDRANested(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        level_outputs, unified = model(input_ids, mask)
        loss = unified.sum()
        loss.backward()
        # Check that nested_head parameters have gradients
        for name, param in model.nested_head.named_parameters():
            assert param.grad is not None, f"No gradient for nested_head.{name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for nested_head.{name}"


class TestTrainerOneBatch:
    def test_trainer_one_batch(self, mock_auto_model):
        """One training step: forward + backward + optimizer step must not crash."""
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = HYDRALoss(architecture="local", use_focal=False)

        input_ids = torch.randint(0, 100, (_BATCH_SIZE, _SEQ_LEN))
        mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
        labels_l1 = torch.randint(0, _NUM_L1, (_BATCH_SIZE,))
        labels_l2 = torch.randint(0, _NUM_L2, (_BATCH_SIZE,))

        # Forward
        level_outputs, unified = model(input_ids, mask)
        loss, loss_l1, loss_l2 = criterion(level_outputs, unified, labels_l1, labels_l2)
        assert loss.item() > 0

        # Backward + step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second forward should produce different logits (weights updated)
        level_outputs_2, _ = model(input_ids, mask)
        assert not torch.allclose(level_outputs[0], level_outputs_2[0])


class TestLastPoolingIgnoresPadding:
    def test_last_pooling_ignores_padding(self, mock_auto_model):
        """Last pooling must use the real last token, not padding."""
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            pooling="last", project_embedding=False,
        )
        B, S, H = 2, 8, _HIDDEN_SIZE
        hidden = torch.randn(B, S, H)
        # Sample 0: 4 real tokens, sample 1: 6 real tokens
        mask = torch.zeros(B, S, dtype=torch.long)
        mask[0, :4] = 1
        mask[1, :6] = 1
        pooled = model._pool(hidden, mask)
        assert pooled.shape == (B, H)
        # Should pick hidden[0, 3] and hidden[1, 5] (last real tokens)
        assert torch.allclose(pooled[0], hidden[0, 3])
        assert torch.allclose(pooled[1], hidden[1, 5])


class TestCheckpointRoundTrip:
    def test_checkpoint_round_trip(self, mock_auto_model, tmp_path):
        """Save checkpoint and reload: state_dict must match exactly."""
        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
            pooling="mean", project_embedding=False, dropout=0.1, hidden_dim=32,
        )

        # Save checkpoint
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "variant": "local",
                "encoder_name": "mock-model",
                "num_classes_l1": _NUM_L1,
                "num_classes_l2": _NUM_L2,
                "pooling": "mean",
                "project_embedding": False,
                "dropout": 0.1,
                "hidden_dim": 32,
            },
            "best_val_f1_l2": 0.5,
            "epoch": 1,
        }, ckpt_path)

        # Reload using load_model from predict.py
        _predict_mod = import_from_hydra("predict")
        loaded_model, loaded_config = _predict_mod.load_model(
            str(ckpt_path), device=torch.device("cpu")
        )

        # Verify all parameters match
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key]), \
                f"Mismatch for {key}"
        assert loaded_config["pooling"] == "mean"
        assert loaded_config["project_embedding"] is False
        assert loaded_config["hidden_dim"] == 32


class TestPredictFunctions:
    def test_predict_single(self, mock_auto_model):
        """predict() returns correct structure on synthetic input."""
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        _predict_mod = import_from_hydra("predict")

        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        model.eval()

        l1_encoder = LabelEncoder()
        l1_encoder.classes_ = np.array([f"L1_{i}" for i in range(_NUM_L1)])

        l2_encoder = LabelEncoder()
        l2_encoder.classes_ = np.array([f"L2_{i}" for i in range(_NUM_L2)])

        class _FakeTokenizerOutput(dict):
            """Dict subclass with .to() for device transfer."""
            def to(self, device):
                return _FakeTokenizerOutput(
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in self.items()}
                )

        tokenizer = mock.MagicMock()
        tokenizer.side_effect = lambda text, **kwargs: _FakeTokenizerOutput({
            "input_ids": torch.randint(0, 100, (1, _SEQ_LEN)),
            "attention_mask": torch.ones(1, _SEQ_LEN, dtype=torch.long),
        })

        result = _predict_mod.predict(
            "produit test", model, tokenizer,
            l1_encoder, l2_encoder, torch.device("cpu"),
        )
        assert "level_1" in result and "level_2" in result
        assert "label" in result["level_1"] and "confidence" in result["level_1"]
        assert result["level_1"]["label"].startswith("L1_")
        assert 0.0 <= result["level_1"]["confidence"] <= 1.0

    def test_predict_batch(self, mock_auto_model):
        """predict_batch() returns one result per input text."""
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        _predict_mod = import_from_hydra("predict")

        model = HYDRALocal(
            encoder_name="mock-model",
            num_classes_l1=_NUM_L1, num_classes_l2=_NUM_L2,
        )
        model.eval()

        l1_encoder = LabelEncoder()
        l1_encoder.classes_ = np.array([f"L1_{i}" for i in range(_NUM_L1)])
        l2_encoder = LabelEncoder()
        l2_encoder.classes_ = np.array([f"L2_{i}" for i in range(_NUM_L2)])

        tokenizer = mock.MagicMock()
        tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": torch.randint(0, 100, (1, _SEQ_LEN)),
            "attention_mask": torch.ones(1, _SEQ_LEN, dtype=torch.long),
        }

        texts = ["produit A", "produit B", "produit C"]
        results = _predict_mod.predict_batch(
            texts, model, tokenizer,
            l1_encoder, l2_encoder, torch.device("cpu"),
            batch_size=2,
        )
        assert len(results) == 3
        for r in results:
            assert "level_1" in r and "level_2" in r


class TestEnforceHierarchy:
    def test_enforce_hierarchy(self):
        """_enforce_hierarchy must return a valid L2 child of predicted L1."""
        _predict_mod = import_from_hydra("predict")

        num_l1, num_l2 = 3, 6
        # L1=0 -> L2={0,1}, L1=1 -> L2={2,3}, L1=2 -> L2={4,5}
        parent_child_matrix = torch.zeros(num_l1, num_l2)
        parent_child_matrix[0, 0] = 1; parent_child_matrix[0, 1] = 1
        parent_child_matrix[1, 2] = 1; parent_child_matrix[1, 3] = 1
        parent_child_matrix[2, 4] = 1; parent_child_matrix[2, 5] = 1

        # Force L1 prediction to class 1 (valid L2: 2, 3)
        logits_l1 = torch.tensor([[0.0, 10.0, 0.0]])
        # L2 logits: highest is class 0, but it's invalid for L1=1
        logits_l2 = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])

        masked = _predict_mod.enforce_hierarchy(logits_l1, logits_l2, parent_child_matrix)
        pred_l2 = torch.argmax(masked, dim=-1).item()
        # Must be 2 or 3 (valid children of L1=1)
        assert pred_l2 in (2, 3)


class TestHYDRALossGlobalRaisesWithoutNumClasses:
    def test_raises_without_num_classes(self):
        """HYDRALoss(architecture='global') without num_classes must raise ValueError."""
        with pytest.raises(ValueError, match="num_classes_l1 and num_classes_l2 are required"):
            HYDRALoss(architecture="global")

    def test_raises_nested_without_num_classes(self):
        """HYDRALoss(architecture='nested') without num_classes must raise ValueError."""
        with pytest.raises(ValueError, match="num_classes_l1 and num_classes_l2 are required"):
            HYDRALoss(architecture="nested", num_classes_l1=5)
