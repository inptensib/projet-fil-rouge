"""Tests for rakuten/models/baselines/flat_model.py and flat_trainer.py.

Uses mocked HuggingFace models to avoid downloading weights.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from rakuten.models.baselines.flat_model import FlatHydra
from rakuten.models.baselines.flat_trainer import FlatHydraConfig, FlatHydraTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN = 64
_BATCH = 2
_SEQ = 16
_L1 = 5
_L2 = 12
_TOTAL = _L1 + _L2


# ---------------------------------------------------------------------------
# Mock encoder factory (shared with HYDRA tests pattern)
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
    return enc


@pytest.fixture(autouse=True)
def mock_auto_model():
    with mock.patch("transformers.AutoModel.from_pretrained") as m:
        m.return_value = _make_mock_encoder()
        yield m


# ---------------------------------------------------------------------------
# Helper: minimal data dict (mirrors prepare_hierarchical_data output)
# ---------------------------------------------------------------------------

def _make_data(n_train: int = 20, n_val: int = 8):
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    rng = np.random.default_rng(0)
    texts_train = [f"produit {i}" for i in range(n_train)]
    texts_val = [f"produit val {i}" for i in range(n_val)]
    l1_train = rng.integers(0, _L1, n_train).tolist()
    l2_train = rng.integers(0, _L2, n_train).tolist()
    l1_val = rng.integers(0, _L1, n_val).tolist()
    l2_val = rng.integers(0, _L2, n_val).tolist()

    l2_to_l1 = {i: i % _L1 for i in range(_L2)}

    le1, le2 = LabelEncoder(), LabelEncoder()
    le1.fit(list(range(_L1)))
    le2.fit(list(range(_L2)))

    return {
        "X_train": texts_train, "X_val": texts_val, "X_test": texts_val,
        "y_l1_train": l1_train, "y_l1_val": l1_val, "y_l1_test": l1_val,
        "y_l2_train": l2_train, "y_l2_val": l2_val, "y_l2_test": l2_val,
        "l1_encoder": le1, "l2_encoder": le2,
        "l2_to_l1_mapping": l2_to_l1,
        "num_classes_l1": _L1,
        "num_classes_l2": _L2,
    }


# ---------------------------------------------------------------------------
# FlatHydra model tests
# ---------------------------------------------------------------------------

class TestFlatHydra:
    def _model(self, **kwargs) -> FlatHydra:
        return FlatHydra(
            encoder_name="camembert-base",
            num_classes_l1=_L1,
            num_classes_l2=_L2,
            **kwargs,
        )

    def _batch(self):
        ids = torch.randint(0, 100, (_BATCH, _SEQ))
        mask = torch.ones(_BATCH, _SEQ, dtype=torch.long)
        return ids, mask

    def test_output_interface(self):
        """forward must return ([], unified_logits)."""
        model = self._model()
        level_outputs, unified = model(*self._batch())
        assert isinstance(level_outputs, list)
        assert len(level_outputs) == 0
        assert unified.shape == (_BATCH, _TOTAL)

    def test_output_with_project_embedding(self):
        model = self._model(project_embedding=True)
        _, unified = model(*self._batch())
        assert unified.shape == (_BATCH, _TOTAL)

    def test_output_without_project_embedding(self):
        model = self._model(project_embedding=False)
        _, unified = model(*self._batch())
        assert unified.shape == (_BATCH, _TOTAL)

    @pytest.mark.parametrize("pooling", ["cls", "mean", "max", "last"])
    def test_pooling_strategies(self, pooling):
        model = self._model(pooling=pooling)
        _, unified = model(*self._batch())
        assert unified.shape == (_BATCH, _TOTAL)

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError, match="Unknown pooling"):
            self._model(pooling="attention")

    def test_freeze_encoder(self):
        model = self._model(freeze_encoder=True)
        for param in model.encoder.parameters():
            assert not param.requires_grad

    def test_classifier_head_not_frozen(self):
        model = self._model(freeze_encoder=True)
        classifier_params = list(model.unified_classifier.parameters())
        assert any(p.requires_grad for p in classifier_params)

    def test_l1_l2_slicing(self):
        """L1 logits are first num_classes_l1 columns of unified output."""
        model = self._model()
        _, unified = model(*self._batch())
        logits_l1 = unified[:, :_L1]
        logits_l2 = unified[:, _L1:]
        assert logits_l1.shape == (_BATCH, _L1)
        assert logits_l2.shape == (_BATCH, _L2)

    def test_no_nan_in_output(self):
        model = self._model()
        _, unified = model(*self._batch())
        assert not torch.isnan(unified).any()

    def test_projection_before_pooling(self):
        """Embedding projector must exist on last_hidden_state (seq dim present)."""
        model = self._model(project_embedding=True)
        assert model.embedding_projector is not None
        # Input is (B, seq, hidden), output is (B, seq, hidden*2)
        h = _HIDDEN
        assert model.embedding_projector.in_features == h
        assert model.embedding_projector.out_features == h * FlatHydra.NUM_LEVELS


# ---------------------------------------------------------------------------
# FlatHydraConfig tests
# ---------------------------------------------------------------------------

class TestFlatHydraConfig:
    def test_defaults(self):
        cfg = FlatHydraConfig()
        assert cfg.encoder == "camembert-base"
        assert cfg.dropout == 0.2
        assert cfg.project_embedding is True
        assert cfg.use_focal is False
        assert cfg.level_weight_l1 == 0.5
        assert cfg.normalize_text is False
        assert cfg.synthetic_policy == "keep_all"
        assert cfg.class_weighting == "none"
        assert cfg.logit_adjustment_tau == 0.0

    def test_invalid_class_weighting_raises(self):
        with pytest.raises(ValueError, match="class_weighting"):
            FlatHydraConfig(class_weighting="sqrt")

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="lr"):
            FlatHydraConfig(lr=-1.0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            FlatHydraConfig(dropout=1.5)

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError, match="Unknown pooling"):
            FlatHydraConfig(pooling="attention")

    def test_from_yaml(self, tmp_path):
        cfg_dict = {
            "encoder": "camembert-base",
            "dropout": 0.3,
            "epochs": 5,
            "lr": 1e-4,
        }
        p = tmp_path / "cfg.yaml"
        import yaml
        p.write_text(yaml.dump(cfg_dict))
        cfg = FlatHydraConfig.from_yaml(str(p))
        assert cfg.dropout == 0.3
        assert cfg.epochs == 5

    def test_from_yaml_with_overrides(self, tmp_path):
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"epochs": 5}))
        cfg = FlatHydraConfig.from_yaml(str(p), epochs=3)
        assert cfg.epochs == 3

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            FlatHydraConfig.from_yaml("/does/not/exist.yaml")

    def test_unknown_yaml_keys_ignored(self, tmp_path):
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"device": "auto", "epochs": 2}))
        cfg = FlatHydraConfig.from_yaml(str(p))
        assert cfg.epochs == 2


# ---------------------------------------------------------------------------
# FlatHydraTrainer tests
# ---------------------------------------------------------------------------

class TestFlatHydraTrainer:
    def _config(self, **kw) -> FlatHydraConfig:
        return FlatHydraConfig(
            epochs=1,
            batch_size=4,
            num_workers=0,
            fp16=False,
            early_stopping_patience=1,
            **kw,
        )

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_instantiation(self, mock_tok, mock_auto_model):
        mock_tok.return_value = mock.MagicMock()
        mock_tok.return_value.side_effect = None
        data = _make_data()
        trainer = FlatHydraTrainer(data=data, config=self._config())
        assert trainer.num_classes_l1 == _L1
        assert trainer.num_classes_l2 == _L2
        assert isinstance(trainer.model, (FlatHydra, nn.DataParallel))

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_evaluate_returns_metrics(self, mock_tok, mock_auto_model):
        """evaluate() must return f1_l2_macro, hcr, and raw preds."""
        tok = mock.MagicMock()
        tok.side_effect = None

        def _tokenize(text, **kwargs):
            seq = 8
            return {
                "input_ids": torch.randint(0, 100, (1, seq)),
                "attention_mask": torch.ones(1, seq, dtype=torch.long),
            }

        tok.side_effect = _tokenize
        mock_tok.return_value = tok

        data = _make_data()
        trainer = FlatHydraTrainer(data=data, config=self._config())

        from rakuten.common.data_loader import HierarchicalProductDataset
        from torch.utils.data import DataLoader

        ds = HierarchicalProductDataset(
            data["X_val"],
            data["y_l1_val"],
            data["y_l2_val"],
            tok,
            max_length=8,
        )
        loader = DataLoader(ds, batch_size=4)
        metrics = trainer.evaluate(loader)

        assert "f1_l2_macro" in metrics
        assert "f1_l1_macro" in metrics
        assert "hcr" in metrics
        assert "_preds_l1" in metrics
        assert "_preds_l2" in metrics

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_metrics_json_written_after_epoch(self, mock_tok, mock_auto_model, tmp_path):
        """metrics.json must be written at the end of each epoch."""
        tok = mock.MagicMock()
        tok.side_effect = None

        def _tokenize(text, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

        tok.side_effect = _tokenize
        mock_tok.return_value = tok

        data = _make_data()
        cfg = self._config(save_dir=str(tmp_path / "model"))
        trainer = FlatHydraTrainer(data=data, config=cfg)
        trainer.train()

        metrics_path = tmp_path / "model" / "metrics.json"
        assert metrics_path.exists(), "metrics.json must be written after training"
        with open(metrics_path) as f:
            hist = json.load(f)
        assert "val_f1_l2_macro" in hist
        assert len(hist["epoch"]) >= 1

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_rebalancing_guard_raises(self, mock_tok, mock_auto_model):
        """class_weighting without drop_oversampled data: explicit ValueError."""
        mock_tok.return_value = mock.MagicMock()
        with pytest.raises(ValueError, match="drop_oversampled"):
            FlatHydraTrainer(
                data=_make_data(),
                config=self._config(class_weighting="inv_sqrt"),
            )

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_rebalancing_tensors_reach_criterion(self, mock_tok, mock_auto_model):
        from collections import Counter

        mock_tok.return_value = mock.MagicMock()
        data = _make_data()
        data["synthetic_policy"] = "drop_oversampled"
        data["real_class_counts_l1"] = Counter(data["y_l1_train"])
        data["real_class_counts_l2"] = Counter(data["y_l2_train"])
        trainer = FlatHydraTrainer(
            data=data,
            config=self._config(class_weighting="inv_sqrt", logit_adjustment_tau=1.0),
        )
        assert trainer.criterion.loss_fn_l1.weight.shape == (_L1,)
        assert trainer.criterion.logit_adjustment_l2.shape == (_L2,)

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_evaluate_tranche_with_real_counts(self, mock_tok, mock_auto_model):
        """v2 data: evaluate() adds f1_l2_by_tranche."""
        from collections import Counter

        tok = mock.MagicMock()

        def _tokenize(text, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

        tok.side_effect = _tokenize
        mock_tok.return_value = tok

        data = _make_data()
        data["real_class_counts_l2"] = Counter(data["y_l2_train"])
        trainer = FlatHydraTrainer(data=data, config=self._config())

        from rakuten.common.data_loader import HierarchicalProductDataset
        from torch.utils.data import DataLoader

        ds = HierarchicalProductDataset(
            data["X_val"], data["y_l1_val"], data["y_l2_val"], tok, max_length=8,
        )
        metrics = trainer.evaluate(DataLoader(ds, batch_size=4))

        tranche = metrics["f1_l2_by_tranche"]
        assert set(tranche) == {"many", "medium", "few"}
        assert tranche["many"]["f1_macro"] is None  # all counts < 20

    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    def test_save_artefacts(self, mock_tok, mock_auto_model, tmp_path):
        """save() must produce tokenizer, encoders, mapping, and config.json."""
        tok = mock.MagicMock()
        tok.side_effect = None
        tok.save_pretrained = mock.MagicMock()
        mock_tok.return_value = tok

        data = _make_data()
        cfg = self._config(save_dir=str(tmp_path / "model"))
        trainer = FlatHydraTrainer(data=data, config=cfg)
        trainer.save()

        dest = tmp_path / "model"
        assert (dest / "l1_label_encoder.pkl").exists()
        assert (dest / "l2_label_encoder.pkl").exists()
        assert (dest / "l2_to_l1_mapping.json").exists()
        assert (dest / "config.json").exists()

        with open(dest / "config.json") as f:
            cfg_saved = json.load(f)
        assert cfg_saved["num_classes_l1"] == _L1
        assert cfg_saved["num_classes_l2"] == _L2
