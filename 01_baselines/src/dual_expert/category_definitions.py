"""Category Definition Store for the Dual-Expert LLM reranker.

Loads offline-generated L2 category definitions and serves them by l2_id.
Falls back gracefully when a definition is missing or the store is empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_FALLBACK_DEFINITION: dict = {
    "definition_fr": (
        "Catégorie définie par le chemin taxonomique fourni. "
        "Utiliser le nom L1 et L2 pour déterminer l'usage principal."
    ),
    "included_products": [],
    "excluded_products": [],
    "positive_signals": [],
    "negative_signals": [],
    "ambiguities": [],
    "confidence": "low",
}


class CategoryDefinitionStore:
    """In-memory store for L2 category definitions loaded from a JSONL file.

    Each row in the JSONL file must contain at least 'l2_id' and
    'definition_fr'. Other fields (positive_signals, negative_signals,
    path_text, etc.) are stored and forwarded to the LLM reranker.

    Args:
        path: Path to the JSONL definitions file. If None or missing, the
              store operates in fallback-only mode (no LLM definitions).
        version: Expected definition version string for validation.
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        version: Optional[str] = None,
    ) -> None:
        self._definitions: dict[int, dict] = {}
        self._version = version
        self._path = path

        if path is not None:
            self._load(Path(path))

    def _load(self, path: Path) -> None:
        """Load definitions from JSONL file."""
        if not path.exists():
            print(f"[DefinitionStore] File not found: {path} — fallback-only mode.")
            return

        loaded = 0
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    l2_id = int(record["l2_id"])
                    self._definitions[l2_id] = record
                    loaded += 1
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    skipped += 1
                    if skipped <= 5:  # only log first few errors
                        print(f"[DefinitionStore] Line {line_no} skipped: {e}")

        print(
            f"[DefinitionStore] Loaded {loaded} definitions from {path} "
            f"({skipped} skipped)."
        )

    def get(self, l2_id: int) -> dict:
        """Return the definition for a given L2 id.

        Falls back to a minimal definition if the id is unknown.

        Args:
            l2_id: Integer L2 class index.

        Returns:
            Dict with at least 'definition_fr', 'positive_signals',
            'negative_signals', and 'confidence'.
        """
        if l2_id in self._definitions:
            return self._definitions[l2_id]
        return self._make_fallback(l2_id)

    def get_many(self, l2_ids: list[int]) -> list[dict]:
        """Return definitions for a list of L2 ids.

        Args:
            l2_ids: List of integer L2 class indices.

        Returns:
            List of definition dicts in the same order as l2_ids.
        """
        return [self.get(l2_id) for l2_id in l2_ids]

    def _make_fallback(self, l2_id: int) -> dict:
        """Generate a minimal fallback definition for an unknown L2 id."""
        fb = dict(_FALLBACK_DEFINITION)
        fb["l2_id"] = l2_id
        fb["_is_fallback"] = True
        return fb

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, l2_id: int) -> bool:
        return int(l2_id) in self._definitions
