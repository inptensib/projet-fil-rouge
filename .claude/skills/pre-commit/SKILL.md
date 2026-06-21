---
name: pre-commit
description: Pipeline de qualite avant commit (lint, tests, sync docs). Lancer avant chaque commit pour garantir la qualite.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

# Pre-Commit Pipeline

Verification complete avant commit. Execute dans l'ordre, s'arrete au premier echec.

## Etapes

### 1. Lint (syntaxe et style)

```bash
python -m py_compile <fichiers modifies>
```

Si `ruff` est installe :
```bash
python -m ruff check --select=E,F,W .
```

Sinon, verifier au minimum :
- Pas de `import *`
- Pas de variables inutilisees evidentes
- Pas de print() de debug oublies

### 2. Tests

```bash
python -m pytest tests/ -x -q --tb=short
```

Tous les tests doivent passer. Arret immediat si echec.

### 3. Sync Check

Invoquer le skill `/sync-check` pour verifier coherence code ↔ docs.
Points critiques :
- Variantes : jamais `hybrid`
- Defaults code = defaults config
- Trees dans READMEs a jour

### 4. Securite basique

Verifier qu'aucun fichier sensible n'est stage :
```bash
git diff --cached --name-only | grep -iE "\.(env|key|pem|credentials|secret)" || true
```

Verifier pas de tokens/passwords hardcodes :
```bash
grep -rn "password\|api_key\|secret\|token" --include="*.py" $(git diff --cached --name-only) 2>/dev/null | grep -v "argparse\|parser\|help=" || true
```

### 5. Rapport final

| Etape | Statut | Details |
|-------|--------|---------|
| Lint | PASS/FAIL | ... |
| Tests | PASS/FAIL | ... |
| Sync Check | PASS/FAIL | ... |
| Securite | PASS/FAIL | ... |

Verdict : READY TO COMMIT ou BLOCKERS FOUND
