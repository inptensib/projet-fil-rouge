"""Verify that all modules can be imported without errors."""

from __future__ import annotations

from helpers import import_from_hydra


class TestSharedImports:
    def test_import_shared_utils(self):
        import rakuten.common.utils as utils
        assert hasattr(utils, "set_seed")
        assert hasattr(utils, "ENCODERS")

    def test_import_shared_metrics(self):
        import rakuten.common.metrics as metrics
        assert hasattr(metrics, "compute_all_metrics")

    def test_import_shared_data_loader(self):
        import rakuten.common.data_loader as data_loader
        assert hasattr(data_loader, "HierarchicalProductDataset")


class TestHydraImports:
    def test_import_hydra_models(self):
        models = import_from_hydra("models")
        assert hasattr(models, "HYDRALocal")
        assert hasattr(models, "HYDRAGlobal")
        assert hasattr(models, "HYDRANested")
        assert hasattr(models, "create_hydra_model")

    def test_import_hydra_losses(self):
        losses = import_from_hydra("losses")
        assert hasattr(losses, "FocalLoss")
        assert hasattr(losses, "HYDRALoss")
