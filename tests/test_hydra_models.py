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
