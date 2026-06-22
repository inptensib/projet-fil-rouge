# Guide de contribution

## Branches et commits

- Créer une branche depuis `main` avant toute modification ; **ne jamais committer directement sur `main`**.
- Format de commit : **Conventional Commits en français**
  ```
  feat(scope): ajouter la variante nested hiérarchique
  fix(hydra): corriger le masque d'inférence pour les classes rares
  docs(pipeline): ajouter les commandes FlatHydra v2
  chore(deps): mettre à jour torch vers 2.7
  ```
  Types courants : `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.

## Pipeline qualité avant commit

Installer les outils de développement :

```bash
pip install -e ".[dev]"
```

Puis, avant tout commit :

```bash
# 1. Tests (obligatoire — 305 tests doivent passer)
python -m pytest tests/ -q

# 2. Lint (optionnel mais recommandé)
ruff check .
ruff format --check .
```

## Ouvrir une Pull Request

Pousser la branche et ouvrir une PR vers `main` via l'interface GitHub (ou `gh pr create`).
La PR doit préciser : contexte, changements apportés, métriques impactées si applicable.

## Note sur les outils Claude Code

Les skills `/pre-commit`, `/sync-check`, etc. mentionnés dans `CLAUDE.md` et le « hook
auto-test » sont propres à l'agent **Claude Code** — un contributeur humain lance les
commandes manuelles ci-dessus à la place.
