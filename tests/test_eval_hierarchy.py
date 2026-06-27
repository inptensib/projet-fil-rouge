"""Tests de scripts/eval_hierarchy.py — résolution du mode de sortie JSON.

Le mode "auto" doit publier les métriques SANS masque pour un modèle standard
et AVEC masque pour un modèle hiérarchique (régression de l'incident mBERT :
des métriques masquées avaient été publiées pour un modèle standard).
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=None)
def _eval_hierarchy():
    """Charge scripts/eval_hierarchy.py (import lourd : trainer HYDRA)."""
    spec = importlib.util.spec_from_file_location(
        "eval_hierarchy", _ROOT / "scripts" / "eval_hierarchy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestResolveOutputMode:
    def test_explicit_base_wins(self):
        mod = _eval_hierarchy()
        assert mod._resolve_output_mode("base", {"check_hierarchy": True}) == "base"

    def test_explicit_hier_wins(self):
        mod = _eval_hierarchy()
        assert mod._resolve_output_mode("hier", {}) == "hier"

    def test_auto_standard_model_gives_base(self):
        mod = _eval_hierarchy()
        assert mod._resolve_output_mode("auto", {}) == "base"
        assert mod._resolve_output_mode(
            "auto", {"check_hierarchy": False, "condition_l2_on_l1": False}
        ) == "base"

    def test_auto_hierarchy_model_gives_hier(self):
        mod = _eval_hierarchy()
        assert mod._resolve_output_mode("auto", {"check_hierarchy": True}) == "hier"
        assert mod._resolve_output_mode("auto", {"condition_l2_on_l1": True}) == "hier"


class TestSerializeResults:
    _RESULTS = {
        "local": {
            "base": {
                "f1_l1_macro": 0.93, "f1_l2_macro": 0.66,
                "hcr": 0.86, "hcr_multiparent": 0.90, "constrained_f1_macro": 0.65,
            },
            "hier": {
                "f1_l1_macro": 0.91, "f1_l2_macro": 0.61,
                "hcr": 0.997, "hcr_multiparent": 1.0, "constrained_f1_macro": 0.61,
            },
            "time_base": 100.04,
            "time_hier": 200.06,
        },
    }

    def test_mode_base_picks_base_metrics(self):
        mod = _eval_hierarchy()
        out = mod._serialize_results(self._RESULTS, "base", "cfg.yaml")
        assert out["local"]["f1_l2_macro"] == 0.66
        assert out["local"]["hcr"] == 0.86
        assert out["local"]["hcr_multiparent"] == 0.90
        assert out["local"]["eval_time_s"] == 100.0

    def test_mode_hier_picks_hier_metrics(self):
        mod = _eval_hierarchy()
        out = mod._serialize_results(self._RESULTS, "hier", "cfg.yaml")
        assert out["local"]["f1_l2_macro"] == 0.61
        assert out["local"]["hcr"] == 0.997
        assert out["local"]["hcr_multiparent"] == 1.0
        assert out["local"]["eval_time_s"] == 200.1

    def test_metadata_keys_present_and_prefixed(self):
        mod = _eval_hierarchy()
        out = mod._serialize_results(self._RESULTS, "base", "configs/hydra/default.yaml")
        assert out["_mode"] == "base"
        assert out["_config"] == "configs/hydra/default.yaml"
        assert "_date" in out
        # Contrat generate_comparison.py : les variantes restent des clés plates
        non_meta = [k for k in out if not k.startswith("_")]
        assert non_meta == ["local"]
        assert set(out["local"]) == {
            "f1_l1_macro", "f1_l2_macro", "hcr", "hcr_multiparent",
            "constrained_f1_macro", "eval_time_s",
        }
