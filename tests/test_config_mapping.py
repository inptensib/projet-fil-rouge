"""Tests du mapping YAML → HYDRATrainer (02_hydra/src/config.py).

Invariant de sécurité : pour chaque config YAML du repo, les hyperparamètres
d'architecture (dropout, hidden_dim, pooling, project_embedding, freeze_encoder)
mappés doivent être égaux aux défauts Python du modèle — garantit que la
correction de transmission (config historiquement inerte) ne change aucun
comportement, y compris pour les reprises de checkpoints en cours.

Note : imports via import_from_hydra() à l'intérieur des tests (pas au niveau
module) — convention du repo pour éviter la collision des packages `src` entre
01_baselines, 02_hydra, 02b_hlpd et 03_distributed à la collecte pytest.
"""
from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest
import yaml
from helpers import import_from_hydra

_ROOT = Path(__file__).resolve().parents[1]
HYDRA_CONFIGS = sorted((_ROOT / "02_hydra" / "configs").glob("*.yaml"))

ARCH_KEYS = ("dropout", "hidden_dim", "pooling", "project_embedding", "freeze_encoder")


@functools.lru_cache(maxsize=None)
def _hydra_modules():
    """Charge config/models/trainer une seule fois (imports lourds)."""
    config_mod = import_from_hydra("config")
    models_mod = import_from_hydra("models")
    trainer_mod = import_from_hydra("trainer")
    return config_mod, models_mod, trainer_mod


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _defaults(callable_obj) -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(callable_obj).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize("config_path", HYDRA_CONFIGS, ids=lambda p: p.name)
def test_arch_params_match_python_defaults(config_path):
    """Neutralité : les valeurs YAML == défauts du modèle et du trainer."""
    config_mod, models_mod, trainer_mod = _hydra_modules()
    mapped = config_mod.map_config(_load(config_path))
    model_defaults = _defaults(models_mod.HYDRABase.__init__)
    trainer_defaults = _defaults(trainer_mod.HYDRATrainer.__init__)

    for key in ARCH_KEYS:
        assert key in mapped, f"{config_path.name} : clé '{key}' absente du mapping"
        assert mapped[key] == model_defaults[key] == trainer_defaults[key], (
            f"{config_path.name} : {key}={mapped[key]} diverge des défauts Python "
            f"(modèle={model_defaults[key]}, trainer={trainer_defaults[key]}) — "
            f"le fix de transmission ne serait plus neutre"
        )


@pytest.mark.parametrize("config_path", HYDRA_CONFIGS, ids=lambda p: p.name)
def test_mapped_keys_accepted_by_trainer(config_path):
    """Chaque clé mappée doit exister dans la signature de HYDRATrainer."""
    config_mod, _, trainer_mod = _hydra_modules()
    mapped = config_mod.map_config(_load(config_path))
    accepted = set(inspect.signature(trainer_mod.HYDRATrainer.__init__).parameters)
    unknown = set(mapped) - accepted
    assert not unknown, f"{config_path.name} : clés inconnues du trainer : {unknown}"


def test_encoder_mapping():
    config_mod, _, _ = _hydra_modules()
    map_config = config_mod.map_config
    assert map_config({"encoder": "camembert-base"})["encoder_name"] == "CamemBERT"
    assert map_config({"encoder": "bert-base-multilingual-cased"})["encoder_name"] == "mBERT"
    assert map_config({"encoder": "xlm-roberta-base"})["encoder_name"] == "XLM-RoBERTa"
    # Encodeur inconnu : transmis tel quel
    assert map_config({"encoder": "flaubert-base"})["encoder_name"] == "flaubert-base"
    assert set(config_mod.ENCODER_MAP) == {
        "camembert-base", "xlm-roberta-base", "bert-base-multilingual-cased",
    }


def test_legacy_aliases():
    config_mod, _, _ = _hydra_modules()
    map_config = config_mod.map_config
    # gradient_clip (ancien nom) → max_grad_norm
    assert map_config({"gradient_clip": 0.5})["max_grad_norm"] == 0.5
    # loss: focal (ancien format) → use_focal, sans écraser use_focal explicite
    assert map_config({"loss": "focal"})["use_focal"] is True
    assert map_config({"loss": "ce"})["use_focal"] is False
    assert map_config({"loss": "focal", "use_focal": False})["use_focal"] is False


def test_level_weights():
    config_mod, _, _ = _hydra_modules()
    assert config_mod.map_config({"level_weights": [0.5]})["level_weight_l1"] == 0.5
    assert "level_weight_l1" not in config_mod.map_config({"level_weights": []})


def test_unknown_yaml_keys_ignored():
    config_mod, _, _ = _hydra_modules()
    mapped = config_mod.map_config({"num_levels": 2, "scheduler": "linear", "device": "auto"})
    assert mapped == {}
