"""Helper functions for tests - importing from conflicting src/ packages."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HYDRA_ROOT = os.path.join(PROJECT_ROOT, "02_hydra")
HLPD_ROOT = os.path.join(PROJECT_ROOT, "02b_hlpd")
DISTRIBUTED_ROOT = os.path.join(PROJECT_ROOT, "03_distributed")


def _import_from(root: str, module_name: str, exclude_roots: list[str] | None = None) -> ModuleType:
    """Import src.<module_name> from a specific project directory."""
    if exclude_roots is None:
        exclude_roots = []

    # Clear cached src.* modules
    for k in list(sys.modules):
        if k.startswith("src.") or k == "src":
            del sys.modules[k]

    saved_path = sys.path.copy()
    try:
        # Build a clean path: root first, project root, then everything else
        clean_path = [root, PROJECT_ROOT]
        for p in saved_path:
            if p not in clean_path and p not in exclude_roots:
                clean_path.append(p)
        sys.path = clean_path
        return importlib.import_module(f"src.{module_name}")
    finally:
        sys.path = saved_path


def import_from_hydra(module_name: str) -> ModuleType:
    """Import a module from 02_hydra/src/."""
    return _import_from(HYDRA_ROOT, module_name, exclude_roots=[HLPD_ROOT, DISTRIBUTED_ROOT])


def import_from_hlpd(module_name: str) -> ModuleType:
    """Import a module from 02b_hlpd/src/."""
    return _import_from(HLPD_ROOT, module_name, exclude_roots=[HYDRA_ROOT, DISTRIBUTED_ROOT])


def import_from_distributed(module_name: str) -> ModuleType:
    """Import a module from 03_distributed/src/."""
    return _import_from(DISTRIBUTED_ROOT, module_name, exclude_roots=[HYDRA_ROOT, HLPD_ROOT])
