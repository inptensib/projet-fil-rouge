---
name: sync-check
description: Verifie la coherence entre le code Python et la documentation (READMEs, docs/, configs YAML). Detecte les drifts.
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

# Sync Check

Audit de coherence code ↔ documentation.

## Points de verification

### 1. Variantes HYDRA
- Grep `hybrid` dans tous les fichiers .md et .yaml — doit etre 0 occurrences (le bon terme est `nested`)
- Verifier que les variantes listees dans les READMEs matchent les classes dans `02_hydra/src/models.py`

### 2. Constantes numeriques
Verifier la coherence de ces valeurs entre code, configs, et docs :
- `157` (classes L1)
- `6100` (classes L2, peut etre 6100+)
- `128` (max_length tokenizer)
- `0.3` (dropout)
- `3.5e-5` (learning rate)
- `2.0` (focal loss gamma)
- `256` (hidden_dim)
- `128` (batch_size)

### 3. Formules
- FocalLoss dans README vs code `02_hydra/src/losses.py` — pas d'alpha dans le code
- HYDRALoss ponderation dans README vs code

### 4. Signatures de fonctions
Cross-verifier les appels entre modules :
- `create_hydra_model()` dans README vs `02_hydra/src/models.py`
- `HYDRALoss()` dans README vs `02_hydra/src/losses.py`
- `HYDRATrainerDDP` dans `03_distributed/src/main.py` vs `03_distributed/src/trainer_ddp.py`

### 5. Structure tree
Verifier que les trees dans chaque README listent tous les fichiers existants :
- `README.md` (racine)
- `02_hydra/README.md`
- `03_distributed/README.md`

### 6. Defaults code vs config
Comparer les valeurs par defaut dans `02_hydra/src/models.py` avec `02_hydra/configs/default.yaml`

### 7. Docker
- Services dans `docker-compose.yml` vs documentation dans README
- Dockerfile commands vs README Quick Start

## Output

Tableau markdown :

| # | Fichier | Probleme | Severite |
|---|---------|----------|----------|

Verdict : PASS (0 erreurs) ou FAIL (N erreurs a corriger)
