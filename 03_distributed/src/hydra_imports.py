"""Chargement des modules de 02_hydra/src sans collision de package.

03_distributed/src et 02_hydra/src portent le même nom de package `src` :
quand ce projet est lancé via `python -m src.main`, `from src.losses import …`
résout vers 03_distributed/src et échoue. On charge donc les modules de
02_hydra par chemin de fichier explicite, sous un alias dédié.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HYDRA_SRC = Path(__file__).resolve().parents[2] / "02_hydra" / "src"


def load_hydra_module(name: str) -> ModuleType:
    """Charge 02_hydra/src/<name>.py sous l'alias _hydra_<name> (avec cache)."""
    alias = f"_hydra_{name}"
    if alias in sys.modules:
        return sys.modules[alias]
    path = _HYDRA_SRC / f"{name}.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Module 02_hydra/src/{name}.py introuvable ({path})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module
