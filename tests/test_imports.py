"""Verify that all modules can be imported without errors."""

from __future__ import annotations


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
        import rakuten.models.hydra.models as models
        assert hasattr(models, "HYDRALocal")
        assert hasattr(models, "HYDRAGlobal")
        assert hasattr(models, "HYDRANested")
        assert hasattr(models, "create_hydra_model")

    def test_import_hydra_losses(self):
        import rakuten.models.hydra.losses as losses
        assert hasattr(losses, "FocalLoss")
        assert hasattr(losses, "HYDRALoss")

    def test_import_hydra_trainer(self):
        import rakuten.models.hydra.trainer as trainer
        assert hasattr(trainer, "HYDRATrainer")


class TestBaselinesImports:
    def test_import_baselines_flat_trainer(self):
        import rakuten.models.baselines.flat_trainer as flat_trainer
        assert hasattr(flat_trainer, "FlatHydraTrainer")

    def test_import_baselines_flat_model(self):
        import rakuten.models.baselines.flat_model as flat_model
        assert hasattr(flat_model, "FlatHydra")


class TestHlpdImports:
    def test_import_hlpd_models(self):
        import rakuten.models.hlpd.models as models
        assert hasattr(models, "HLPD")

    def test_import_hlpd_trainer(self):
        import rakuten.models.hlpd.trainer as trainer
        assert hasattr(trainer, "HLPDTrainer")
