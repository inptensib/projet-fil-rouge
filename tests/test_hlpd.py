"""Tests for rakuten/models/hlpd/ — HLPD model, losses, attention bias, trainer.

Uses mocked HuggingFace models to avoid downloading weights.
All tests run on CPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from rakuten.models.hlpd.attention_bias import build_M_hiera
from rakuten.models.hlpd.losses import ZLPRLoss
from rakuten.models.hlpd.models import HLPD, HLPDDecoderLayer
from rakuten.models.hlpd.trainer import HLPDConfig, HLPDTrainer

# ---------------------------------------------------------------------------
# Constants — small enough for CPU tests
# ---------------------------------------------------------------------------

_HIDDEN = 64
_BATCH  = 2
_SEQ    = 16
_L1     = 5
_L2     = 12
_L      = _L1 + _L2  # 17


# ---------------------------------------------------------------------------
# Mock encoder (shared fixture)
# ---------------------------------------------------------------------------

def _make_mock_encoder(hidden: int = _HIDDEN):
    enc = mock.MagicMock()
    enc.config.hidden_size = hidden

    def _fwd(*args, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None and args:
            input_ids = args[0]
        B, S = input_ids.shape
        out = mock.MagicMock()
        out.last_hidden_state = torch.randn(B, S, hidden)
        return out

    enc.side_effect = _fwd
    enc.gradient_checkpointing_enable = mock.MagicMock()
    return enc


@pytest.fixture(autouse=True)
def mock_auto_model():
    with mock.patch("transformers.AutoModel.from_pretrained") as m:
        m.return_value = _make_mock_encoder()
        yield m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_to_l1(l1: int = _L1, l2: int = _L2) -> dict[int, int]:
    """Simple mapping: l2_idx -> l2_idx % l1."""
    return {i: i % l1 for i in range(l2)}


def _make_data(n_train: int = 20, n_val: int = 8):
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    rng = np.random.default_rng(0)
    texts_train = [f"produit {i}" for i in range(n_train)]
    texts_val   = [f"val {i}" for i in range(n_val)]
    l1_train = rng.integers(0, _L1, n_train).tolist()
    l2_train = rng.integers(0, _L2, n_train).tolist()
    l1_val   = rng.integers(0, _L1, n_val).tolist()
    l2_val   = rng.integers(0, _L2, n_val).tolist()

    le1, le2 = LabelEncoder(), LabelEncoder()
    le1.fit(list(range(_L1)))
    le2.fit(list(range(_L2)))

    return {
        "X_train": texts_train, "X_val": texts_val, "X_test": texts_val,
        "y_l1_train": l1_train, "y_l1_val": l1_val, "y_l1_test": l1_val,
        "y_l2_train": l2_train, "y_l2_val": l2_val, "y_l2_test": l2_val,
        "l1_encoder": le1, "l2_encoder": le2,
        "l2_to_l1_mapping": _l2_to_l1(),
        "num_classes_l1": _L1,
        "num_classes_l2": _L2,
    }


# ---------------------------------------------------------------------------
# Tests — build_M_hiera
# ---------------------------------------------------------------------------

class TestBuildMHiera:
    def _mapping(self) -> dict:
        return _l2_to_l1()

    def test_shape(self):
        M = build_M_hiera(self._mapping(), _L1, _L2)
        assert M.shape == (_L, _L)

    def test_diagonal_ones(self):
        M = build_M_hiera(self._mapping(), _L1, _L2)
        assert torch.all(M.diagonal() == 1.0)

    def test_symmetric(self):
        M = build_M_hiera(self._mapping(), _L1, _L2)
        assert torch.all(M == M.T)

    def test_parent_child_edges(self):
        mapping = _l2_to_l1()
        M = build_M_hiera(mapping, _L1, _L2)
        for l2_idx, l1_idx in mapping.items():
            l2_full = _L1 + l2_idx
            assert M[l1_idx, l2_full] == 1.0, f"Missing edge L1={l1_idx}↔L2={l2_full}"
            assert M[l2_full, l1_idx] == 1.0, f"Missing edge L2={l2_full}↔L1={l1_idx}"

    def test_siblings_connected(self):
        # With mapping i -> i%5: L2 classes 0 and 5 share parent L1=0
        mapping = _l2_to_l1()
        M = build_M_hiera(mapping, _L1, _L2)
        # Full indices: L2 class 0 → 5, L2 class 5 → 10
        idx_0 = _L1 + 0   # 5
        idx_5 = _L1 + 5   # 10
        assert M[idx_0, idx_5] == 1.0, "Sibling pair not connected"
        assert M[idx_5, idx_0] == 1.0, "Sibling pair not connected (symmetric)"

    def test_non_related_is_zero(self):
        # L1 class 0 and L1 class 1 share no direct relation → should be 0
        M = build_M_hiera(_l2_to_l1(), _L1, _L2)
        assert M[0, 1] == 0.0

    def test_values_binary(self):
        M = build_M_hiera(_l2_to_l1(), _L1, _L2)
        unique = M.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Tests — ZLPRLoss
# ---------------------------------------------------------------------------

class TestZLPRLoss:
    def test_non_negative(self):
        criterion = ZLPRLoss()
        y_pred = torch.randn(4, _L)
        y_true = (torch.rand(4, _L) > 0.5).float()
        loss = criterion(y_pred, y_true)
        assert loss.item() >= 0.0

    def test_high_confidence_gives_small_loss(self):
        """Perfect predictions (high positive logit, low negative logit) → small loss."""
        criterion = ZLPRLoss()
        y_pred = torch.full((4, _L), -10.0)
        y_true = torch.zeros(4, _L)
        # One positive per sample at index 0
        y_true[:, 0] = 1.0
        y_pred[:, 0] = 10.0
        loss = criterion(y_pred, y_true)
        assert loss.item() < 0.1, f"Expected small loss, got {loss.item():.4f}"

    def test_gradient_flows(self):
        criterion = ZLPRLoss()
        y_pred = torch.randn(4, _L, requires_grad=True)
        y_true = (torch.rand(4, _L) > 0.5).float()
        loss = criterion(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert y_pred.grad.abs().sum() > 0.0

    def test_all_negative_labels_loss(self):
        """If y_true=0 everywhere and y_pred=0, loss = log(L+1) per sample.

        Trace:
          y_pred_neg = 0s (size L), cat zero → L+1 zeros → logsumexp = log(L+1)
          y_pred_pos = -1e12 (size L), cat zero → dominated by 0 → logsumexp ≈ 0
          loss ≈ log(L+1)
        """
        criterion = ZLPRLoss()
        y_pred = torch.zeros(2, _L)
        y_true = torch.zeros(2, _L)
        loss = criterion(y_pred, y_true)
        expected = torch.tensor(float(_L + 1)).log()
        assert abs(loss.item() - expected.item()) < 1e-4


# ---------------------------------------------------------------------------
# Tests — HLPD model
# ---------------------------------------------------------------------------

class TestHLPD:
    def _model(self, **kwargs) -> HLPD:
        return HLPD(
            encoder_name="camembert-base",
            num_l1=_L1,
            num_l2=_L2,
            l2_to_l1_mapping=_l2_to_l1(),
            num_decoder_layers=1,  # 1 layer for speed
            nhead=2,
            ffn_dim=32,
            decoder_dropout=0.0,
            gradient_checkpointing=False,  # disable for tests (non-leaf issues)
            **kwargs,
        )

    def _batch(self):
        ids  = torch.randint(0, 100, (_BATCH, _SEQ))
        mask = torch.ones(_BATCH, _SEQ, dtype=torch.long)
        return ids, mask

    def test_forward_shape(self):
        model = self._model()
        logits = model(*self._batch())
        assert logits.shape == (_BATCH, _L), f"Expected ({_BATCH}, {_L}), got {logits.shape}"

    def test_forward_no_nan(self):
        model = self._model()
        logits = model(*self._batch())
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    def test_tied_weights_shared(self):
        model = self._model()
        assert model.output_proj.weight is model.label_embeddings.weight

    def test_M_hiera_registered_as_buffer(self):
        model = self._model()
        assert "M_hiera" in dict(model.named_buffers())

    def test_M_hiera_shape(self):
        model = self._model()
        assert model.M_hiera.shape == (_L, _L)

    def test_freeze_encoder(self):
        model = self._model(freeze_encoder=True)
        for param in model.encoder.parameters():
            assert not param.requires_grad

    def test_gradient_flows_through_model(self):
        model = self._model()
        ids, mask = self._batch()
        logits = model(ids, mask)
        loss = logits.sum()
        loss.backward()
        # label_embeddings should have gradient
        assert model.label_embeddings.weight.grad is not None


# ---------------------------------------------------------------------------
# Tests — HLPDConfig
# ---------------------------------------------------------------------------

class TestHLPDConfig:
    def test_defaults(self):
        cfg = HLPDConfig()
        assert cfg.encoder == "camembert-base"
        assert cfg.num_decoder_layers == 2
        assert cfg.nhead == 4
        assert cfg.precision == "bfloat16"

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="lr"):
            HLPDConfig(lr=-1.0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="decoder_dropout"):
            HLPDConfig(decoder_dropout=1.5)

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            HLPDConfig(precision="fp16")

    def test_from_yaml(self, tmp_path):
        import yaml
        cfg_dict = {"epochs": 3, "lr": 2.0e-4, "batch_size": 16}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg_dict))
        cfg = HLPDConfig.from_yaml(str(p))
        assert cfg.epochs == 3
        assert cfg.batch_size == 16

    def test_from_yaml_with_overrides(self, tmp_path):
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"epochs": 5}))
        cfg = HLPDConfig.from_yaml(str(p), epochs=2)
        assert cfg.epochs == 2

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            HLPDConfig.from_yaml("/does/not/exist.yaml")

    def test_unknown_yaml_keys_ignored(self, tmp_path):
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"unknown_key": 99, "epochs": 2}))
        cfg = HLPDConfig.from_yaml(str(p))
        assert cfg.epochs == 2


# ---------------------------------------------------------------------------
# Tests — HLPDTrainer
# ---------------------------------------------------------------------------

def _make_mock_tokenizer():
    tok = mock.MagicMock()
    tok.side_effect = None

    def _tokenize(text, **kwargs):
        return {
            "input_ids": torch.randint(0, 100, (1, _SEQ)),
            "attention_mask": torch.ones(1, _SEQ, dtype=torch.long),
        }

    tok.side_effect = _tokenize
    tok.save_pretrained = mock.MagicMock()
    return tok


def _trainer_config(tmp_path=None, **kw) -> HLPDConfig:
    defaults: dict = dict(
        epochs=1,
        batch_size=4,
        num_workers=0,
        precision="float32",  # CPU tests
        gradient_checkpointing=False,
        num_decoder_layers=1,
        nhead=2,
        ffn_dim=32,
        checkpoint_every_n_steps=9999,
        early_stopping_patience=1,
        save_dir=str(tmp_path / "hlpd_model") if tmp_path else "models/hlpd_test",
    )
    defaults.update(kw)
    return HLPDConfig(**defaults)


class TestHLPDTrainer:

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_instantiation(self, mock_tok, mock_auto_model):
        mock_tok.return_value = _make_mock_tokenizer()
        data = _make_data()
        trainer = HLPDTrainer(data=data, config=_trainer_config(Path(tempfile.mkdtemp())))
        assert trainer.num_l1 == _L1
        assert trainer.num_l2 == _L2
        assert trainer.num_labels == _L

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_evaluate_returns_metrics(self, mock_tok, mock_auto_model):
        tok = _make_mock_tokenizer()
        mock_tok.return_value = tok

        data = _make_data()
        trainer = HLPDTrainer(
            data=data,
            config=_trainer_config(Path(tempfile.mkdtemp())),
        )

        from rakuten.common.data_loader import HierarchicalProductDataset
        from torch.utils.data import DataLoader

        ds = HierarchicalProductDataset(
            data["X_val"], data["y_l1_val"], data["y_l2_val"],
            tok, max_length=_SEQ,
        )
        loader = DataLoader(ds, batch_size=4)
        metrics = trainer.evaluate(loader)

        assert "f1_l1_macro" in metrics
        assert "f1_l2_macro" in metrics
        assert "hcr" in metrics
        assert "_preds_l1" in metrics
        assert "_preds_l2" in metrics

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_train_writes_metrics_json(self, mock_tok, mock_auto_model, tmp_path):
        mock_tok.return_value = _make_mock_tokenizer()
        data = _make_data()
        cfg = _trainer_config(tmp_path)
        trainer = HLPDTrainer(data=data, config=cfg)
        trainer.train()

        metrics_path = tmp_path / "hlpd_model" / "metrics.json"
        assert metrics_path.exists()
        with open(metrics_path) as f:
            hist = json.load(f)
        assert "val_f1_l2_macro" in hist
        assert len(hist["epoch"]) >= 1

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_checkpoint_intra_epoch_save_and_resume(self, mock_tok, mock_auto_model, tmp_path):
        """Intra-epoch checkpoint must contain global_step and step_in_epoch."""
        mock_tok.return_value = _make_mock_tokenizer()
        data = _make_data()

        # Use checkpoint_every_n_steps=1 to force a checkpoint after 1 optim step
        cfg = _trainer_config(
            tmp_path,
            checkpoint_every_n_steps=1,
            gradient_accumulation_steps=1,
        )
        trainer = HLPDTrainer(data=data, config=cfg)
        trainer.train()

        ckpt_path = tmp_path / "hlpd_model" / "checkpoint_latest.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "global_step" in ckpt
        assert "step_in_epoch" in ckpt
        assert "epoch" in ckpt
        assert "best_val_f1" in ckpt

        # Verify resume: create new trainer from checkpoint
        trainer2 = HLPDTrainer(
            data=data,
            config=cfg,
            resume_from=str(ckpt_path),
        )
        # Should not raise
        assert trainer2._base_model is not None

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_save_artefacts(self, mock_tok, mock_auto_model, tmp_path):
        tok = _make_mock_tokenizer()
        mock_tok.return_value = tok

        data = _make_data()
        cfg = _trainer_config(tmp_path)
        trainer = HLPDTrainer(data=data, config=cfg)
        trainer.save()

        dest = tmp_path / "hlpd_model"
        assert (dest / "l1_label_encoder.pkl").exists()
        assert (dest / "l2_label_encoder.pkl").exists()
        assert (dest / "l2_to_l1_mapping.json").exists()
        assert (dest / "config.json").exists()

        with open(dest / "config.json") as f:
            saved = json.load(f)
        assert saved["num_l1"] == _L1
        assert saved["num_l2"] == _L2
