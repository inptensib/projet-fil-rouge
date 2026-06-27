# Contributing guide

## Branches and commits

- Create a branch from `main` before making any change; **never commit directly to `main`**.
- Commit format: **Conventional Commits in French**
  ```
  feat(scope): ajouter la variante nested hiérarchique
  fix(hydra): corriger le masque d'inférence pour les classes rares
  docs(pipeline): ajouter les commandes FlatHydra v2
  chore(deps): mettre à jour torch vers 2.7
  ```
  Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.

## Quality pipeline before committing

Install the development tools:

```bash
pip install -e ".[dev]"
```

Then, before any commit:

```bash
# 1. Tests (mandatory — 312 tests must pass)
python -m pytest tests/ -q

# 2. Lint (optional but recommended)
ruff check .
ruff format --check .
```

## Opening a Pull Request

Push the branch and open a PR against `main` through the GitHub interface (or `gh pr create`).
The PR must specify: context, changes made, and impacted metrics if applicable.

## Note on Claude Code tooling

The `/pre-commit`, `/sync-check`, and similar skills mentioned in `CLAUDE.md`, along with the
"auto-test hook", are specific to the **Claude Code** agent — a human contributor runs the
manual commands above instead.
