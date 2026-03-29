# Contributing

## Setup

```bash
git clone <repo-url> && cd rakuten-classification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Dataset
export HF_TOKEN="hf_..."   # Demander a Tahiana
python scripts/download_dataset.py
```

## Conventions de code

- **Linter** : Ruff (config dans `pyproject.toml`)
  ```bash
  pip install ruff
  ruff check .           # Verifier
  ruff check . --fix     # Corriger automatiquement
  ```
- **Docstrings** : Google-style sur toutes les fonctions publiques
- **Type hints** : obligatoires sur les fonctions de `shared/`, `02_hydra/src/`, `03_production/src/`
- **Imports** : stdlib, puis tiers, puis locaux (enforce par Ruff isort)
- **`from __future__ import annotations`** : en haut de chaque fichier Python

## Workflow Git

1. Creer une branche : `git checkout -b feat/ma-feature`
2. Coder + tester
3. Verifier : `ruff check .` et `python -m pytest tests/ -v`
4. Commit : messages en anglais, prefixes `feat:`, `fix:`, `docs:`, `test:`
5. Push + Pull Request

## Tests

**77 tests doivent passer avant tout PR :**

```bash
python -m pytest tests/ -v --tb=short
```

Ajouter des tests pour tout nouveau module dans `tests/test_<module>.py`.

## Structure du projet

| Dossier | Quoi y mettre |
|---------|---------------|
| `shared/` | Code reutilisable (data loading, metrics, utils) |
| `02_hydra/src/` | Architecture HYDRA (modeles, losses, trainer) |
| `02_hydra/configs/` | Hyperparametres (source de verite) |
| `03_production/src/` | Training distribue DDP |
| `scripts/` | CLI, SLURM jobs, utilitaires |
| `tests/` | Tests pytest |

## Config

Source de verite : `02_hydra/configs/default.yaml`. Tout default Python doit matcher ce fichier.
