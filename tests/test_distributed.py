"""Tests for 03_distributed/src/ modules."""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import pytest
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
