---
name: dep-audit
description: Audit des dependances Python (imports reels vs requirements.txt). Detecte packages manquants ou inutiles.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

# Dependency Audit

Compare les imports Python reels avec `requirements.txt`.

## Taches

### 1. Collecter tous les imports

```bash
grep -rh "^import \|^from " --include="*.py" . | sort -u
```

Exclure les imports de la stdlib Python et les imports internes au projet (shared/, src/, etc.)

### 2. Lire requirements.txt

Lister tous les packages declares et leurs versions.

### 3. Cross-reference

Pour chaque package dans requirements.txt :
- Verifier qu'il est importe quelque part dans le code (.py ou .ipynb)
- Attention aux noms differents entre pip et import (ex: `scikit-learn` → `sklearn`, `Pillow` → `PIL`)

Pour chaque import tiers dans le code :
- Verifier qu'il est declare dans requirements.txt
- Attention aux dependances transitives (ex: `joblib` vient avec `scikit-learn`)

### 4. Mapping connu pour ce projet

| Package pip | Import Python | Utilise dans |
|-------------|---------------|--------------|
| `torch` | `torch` | 02_hydra/, 03_production/ |
| `transformers` | `transformers` | 02_hydra/src/models.py |
| `scikit-learn` | `sklearn` | 01_baselines/ |
| `mlflow` | `mlflow` | 03_production/src/trainer_ddp.py |
| `sentence-transformers` | `sentence_transformers` | notebooks/ |
| `pyyaml` | `yaml` | 02_hydra/src/trainer.py |

### 5. Verifier aussi les notebooks

```bash
grep -rh "import \|from " --include="*.ipynb" . 2>/dev/null
```

## Output

| Package | Dans requirements.txt | Importe dans le code | Statut |
|---------|-----------------------|----------------------|--------|

- MANQUANT : importe mais pas dans requirements.txt
- INUTILE : dans requirements.txt mais jamais importe
- TRANSITIF : pas importe directement mais dependance d'un autre package
- OK : present des deux cotes
