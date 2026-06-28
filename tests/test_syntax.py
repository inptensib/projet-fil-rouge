"""Smoke tests — syntax only.

Every tracked Python file must compile, and every notebook code cell must parse
as valid Python. This is intentionally minimal: the delivered repo keeps only
syntax checks (this file) and import checks (``test_imports.py``).
"""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Caches / venv / tooling — skipped at ANY depth.
_SKIP_ANYWHERE = {
    ".git", "venv", ".venv", "env", "ENV", "__pycache__", ".pytest_cache",
    ".ruff_cache", "node_modules", ".claude", ".ipynb_checkpoints", "build", "dist",
}
# Weights / data / runs / slides — skipped only at the repo ROOT, so that the
# source tree `rakuten/models/` is NOT excluded by the top-level `models/`.
_SKIP_TOPLEVEL = {
    "models", "dataset", "rakuten_classification.egg-info",
}


def _collect(pattern: str) -> list[Path]:
    found = []
    for path in ROOT.rglob(pattern):
        parts = path.relative_to(ROOT).parts
        if parts[0] in _SKIP_TOPLEVEL or parts[0].startswith("dataset"):
            continue
        if any(p in _SKIP_ANYWHERE or p.endswith(".egg-info") for p in parts):
            continue
        found.append(path)
    return sorted(found)


_PY_FILES = _collect("*.py")
_NOTEBOOKS = _collect("*.ipynb")


@pytest.mark.parametrize("path", _PY_FILES, ids=[str(p.relative_to(ROOT)) for p in _PY_FILES])
def test_python_file_compiles(path: Path) -> None:
    """The file is syntactically valid (compiled, not executed)."""
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=[str(p.relative_to(ROOT)) for p in _NOTEBOOKS])
def test_notebook_code_cells_parse(path: Path) -> None:
    """Every code cell parses as valid Python (IPython magics stripped)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        first = next((ln for ln in lines if ln.strip()), "")
        if first.lstrip().startswith(("%%", "!", "%")):  # cell magic / shell → skip
            continue
        # Drop line magics / shell / help lines, keep the rest.
        code = "".join("" if ln.lstrip().startswith(("%", "!", "?")) else ln for ln in lines)
        if not code.strip():
            continue
        try:
            ast.parse(code)
        except SyntaxError as exc:
            pytest.fail(f"{path.relative_to(ROOT)} — code cell {i}: {exc}")
