"""Tests for 03_distributed/src/ modules."""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import pytest
import torch
from helpers import import_from_distributed


def _cli_args(**overrides) -> argparse.Namespace:
    """Namespace identique aux défauts argparse de src/main.py."""
    defaults = dict(
        config=None, variant=None, encoder=None, batch_size=None,
        epochs=None, lr=None, shard_dir="shards", save_dir=None,
        data=None, split_dir=None, n_samples=10_000,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildTrainerKwargs:
    """Fusion config YAML + CLI du point d'entrée DDP (priorité CLI > YAML)."""

    def test_sans_config_ni_cli(self):
        main = import_from_distributed("main")
        assert main.build_trainer_kwargs(None, _cli_args()) == {}

    def test_cli_seul(self):
        main = import_from_distributed("main")
        kwargs = main.build_trainer_kwargs(None, _cli_args(batch_size=128, lr=1e-4))
        assert kwargs == {"batch_size": 128, "lr": 1e-4}

    def test_yaml_mapping_et_alias_ddp(self):
        main = import_from_distributed("main")
        cfg = {
            "batch_size": 32, "max_grad_norm": 0.5, "check_hierarchy": True,
            "variant": "nested", "encoder": "camembert-base", "fp16": True,
            "dropout": 0.3, "hidden_dim": 256, "seed": 42,
        }
        kwargs = main.build_trainer_kwargs(cfg, _cli_args())
        assert kwargs["batch_size"] == 32
        assert kwargs["grad_clip_max_norm"] == 0.5
        assert kwargs["check_hierarchy_consistency"] is True
        assert kwargs["dropout"] == 0.3
        assert kwargs["hidden_dim"] == 256
        # Clés sans équivalent DDP écartées
        for absent in ("variant", "encoder_name", "fp16", "seed", "max_grad_norm"):
            assert absent not in kwargs

    def test_cli_prime_sur_yaml(self):
        main = import_from_distributed("main")
        kwargs = main.build_trainer_kwargs({"batch_size": 32}, _cli_args(batch_size=128))
        assert kwargs["batch_size"] == 128


class TestResolveVariantEncoder:
    def test_defauts_historiques(self):
        main = import_from_distributed("main")
        assert main.resolve_variant_encoder(None, _cli_args()) == ("local", "CamemBERT")

    def test_yaml_traduit_l_encodeur_hf(self):
        main = import_from_distributed("main")
        cfg = {"variant": "nested", "encoder": "bert-base-multilingual-cased"}
        assert main.resolve_variant_encoder(cfg, _cli_args()) == ("nested", "mBERT")

    def test_cli_prime_sur_yaml(self):
        main = import_from_distributed("main")
        cfg = {"variant": "nested", "encoder": "bert-base-multilingual-cased"}
        args = _cli_args(variant="global", encoder="CamemBERT")
        assert main.resolve_variant_encoder(cfg, args) == ("global", "CamemBERT")


class TestDDPUtils:
    def test_print_rank0_rank0(self, capsys):
        ddp_utils = import_from_distributed("ddp_utils")
        ddp_utils.print_rank0("hello", rank=0)
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_print_rank0_other_rank(self, capsys):
        ddp_utils = import_from_distributed("ddp_utils")
        ddp_utils.print_rank0("hello", rank=1)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_cleanup_ddp_when_not_initialized(self):
        ddp_utils = import_from_distributed("ddp_utils")
        with mock.patch("torch.distributed.is_initialized", return_value=False), \
             mock.patch("torch.distributed.destroy_process_group") as mock_destroy:
            ddp_utils.cleanup_ddp()
            mock_destroy.assert_not_called()


class TestCheckEarlyStopping:
    """_check_early_stopping isolé : pas de GPU, mlflow mocké, dist mocké."""

    CONFIG_ATTRS = dict(
        num_classes_l1=157, num_classes_l2=6126, pooling="cls",
        project_embedding=True, dropout=0.3, hidden_dim=256,
    )

    def _make_rank0_trainer(self, trainer_ddp, patience=2, min_delta=0.001):
        trainer = trainer_ddp.HYDRATrainerDDP.__new__(trainer_ddp.HYDRATrainerDDP)
        trainer.rank = 0
        trainer.device = torch.device("cpu")
        trainer._global_step = 42
        trainer._early_stopper = trainer_ddp.EarlyStopping(patience, min_delta)
        for attr, value in self.CONFIG_ATTRS.items():
            setattr(trainer, attr, value)
        return trainer

    def test_improvement_saves_checkpoint_with_frozen_keys(self, tmp_path):
        trainer_ddp = import_from_distributed("trainer_ddp")
        trainer = self._make_rank0_trainer(trainer_ddp)
        model = torch.nn.Linear(2, 2)
        fake_mlflow = mock.MagicMock()

        with mock.patch.object(trainer_ddp, "mlflow", fake_mlflow):
            stop = trainer._check_early_stopping(
                model, 0.8, "local", "CamemBERT", str(tmp_path)
            )

        assert stop is False
        fake_mlflow.pytorch.log_model.assert_called_once()
        checkpoint = torch.load(tmp_path / "hydra_model.pt", weights_only=False)
        assert set(checkpoint) == {
            "model_state_dict", "global_step", "best_val_f1", "config",
        }
        assert checkpoint["global_step"] == 42
        assert checkpoint["best_val_f1"] == 0.8
        assert checkpoint["config"] == {
            "num_classes_l1": 157, "num_classes_l2": 6126,
            "variant": "local", "encoder_name": "CamemBERT",
            "pooling": "cls", "project_embedding": True,
            "dropout": 0.3, "hidden_dim": 256,
        }

    def test_no_improvement_stops_after_patience(self, tmp_path):
        trainer_ddp = import_from_distributed("trainer_ddp")
        trainer = self._make_rank0_trainer(trainer_ddp, patience=2)
        trainer._early_stopper.best = 0.9
        model = torch.nn.Linear(2, 2)

        with mock.patch.object(trainer_ddp, "mlflow", mock.MagicMock()):
            assert trainer._check_early_stopping(
                model, 0.5, "local", "CamemBERT", str(tmp_path)
            ) is False
            assert trainer._check_early_stopping(
                model, 0.5, "local", "CamemBERT", str(tmp_path)
            ) is True

        assert not (tmp_path / "hydra_model.pt").exists()

    def test_non_rank0_only_participates_in_broadcast(self, tmp_path):
        trainer_ddp = import_from_distributed("trainer_ddp")
        trainer = trainer_ddp.HYDRATrainerDDP.__new__(trainer_ddp.HYDRATrainerDDP)
        trainer.rank = 1
        trainer.device = torch.device("cpu")
        fake_dist = mock.MagicMock()
        fake_dist.is_initialized.return_value = True

        with mock.patch.object(trainer_ddp, "dist", fake_dist):
            stop = trainer._check_early_stopping(
                torch.nn.Linear(2, 2), None, "local", "CamemBERT", str(tmp_path)
            )

        assert stop is False
        fake_dist.broadcast.assert_called_once()
        assert fake_dist.broadcast.call_args.kwargs["src"] == 0
        fake_dist.barrier.assert_called_once()
        assert not (tmp_path / "hydra_model.pt").exists()
