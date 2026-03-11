"""Verify that all modules can be imported without errors."""

from __future__ import annotations

from helpers import import_from_hydra


class TestSharedImports:
    def test_import_shared_utils(self):
        import shared.utils
        assert hasattr(shared.utils, "set_seed")

    def test_import_shared_metrics(self):
        import shared.metrics
        assert hasattr(shared.metrics, "compute_all_metrics")

    def test_import_shared_data_loader(self):
        import shared.data_loader
        assert hasattr(shared.data_loader, "HierarchicalProductDataset")


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
