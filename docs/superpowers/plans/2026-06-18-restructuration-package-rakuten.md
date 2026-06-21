# Restructuration en package `rakuten` — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** transformer le code (dossiers numérotés + packages `src/` homonymes) en un package Python unique `rakuten`, installable et importable proprement, sans hack `sys.path`.

**Architecture:** migration incrémentale par couche, sur une branche dédiée. Chaque couche (`common`, puis chaque modèle) est déplacée par `git mv`, ses imports `from shared.`/`from src.` réécrits, ses tests passés en imports propres, puis commit. La suite de tests existante sert de filet à chaque étape.

**Tech Stack:** Python 3.13, pytest, setuptools (pyproject), ruff. Pas de nouvelle dépendance.

## Constat clé (conditionne tout le plan)

Les imports **internes** à chaque module sont **déjà relatifs** (`from .losses import`,
`from .models import`, `from .attention_bias import`, `from .flat_model import`).
Un `git mv` d'un dossier les **préserve** : le relatif ne dépend pas du nom absolu
du package. Donc on ne réécrit QUE :
1. `from shared.X` → `from rakuten.common.X` (vers le module commun)
2. les `from src.X` **externes** (scripts/, tests/) → `from rakuten.models.{...}.X`
3. on **supprime** les `sys.path.insert` et la gymnastique `tests/helpers.py`

## Global Constraints

- **Prérequis bloquants** : (1) les 3 jobs SLURM terminés, (2) `refactor/trainers`
  mergée dans `main`. Le plan s'exécute sur ce `main` consolidé, branche
  `restructure/package-rakuten`.
- **Invariant absolu** : `pytest tests/ -q` reste **vert à chaque commit**.
  Référence ~299 tests (309 du kit v2 − ~10 tests DDP retirés à la Task 0).
- **Aucun changement de logique** : déplacements + imports + packaging uniquement.
- **Venv** : `/home/infres/tandriam-25/fil_rouge/venv/bin/python` (et son `pytest`,
  `pip`). Toutes les commandes l'utilisent.
- **Pas de `console_scripts`**, pas de `from src.` ni `sys.path.insert` résiduels
  en fin de plan.
- **`hybrid_system/` et le DDP** : hors package (cf. spec).
- Mémoire commits : **pas de `Co-Authored-By`**.

---

### Task 0 : Prérequis, branche, archivage du DDP

**Files:**
- Delete: `03_distributed/` (tout le dossier), `tests/test_distributed.py`
- Tag git: `archive/distributed-pipeline`

**Interfaces:**
- Produces: branche `restructure/package-rakuten` partant du `main` consolidé,
  sans le pipeline DDP. Référence de tests post-Task : ~299.

- [ ] **Step 1 : Vérifier les prérequis**

Run :
```bash
squeue -u $USER --states=RUNNING --format="%.12i %.16j" | grep -v bash   # doit être vide (jobs finis)
git -C ~/fil_rouge/rakuten-classification branch --merged main | grep refactor/trainers  # doit lister la branche
```
Expected : aucun job actif ; `refactor/trainers` apparaît comme mergée. Si l'un
des deux échoue, **STOP** — les prérequis ne sont pas remplis.

- [ ] **Step 2 : Créer la branche de travail**

```bash
cd ~/fil_rouge/rakuten-classification
git checkout main && git pull
git checkout -b restructure/package-rakuten
```

- [ ] **Step 3 : Archiver le DDP par tag (récupérable à vie)**

```bash
git tag -a archive/distributed-pipeline -m "Pipeline DDP (03_distributed) avant exclusion du livrable — récupérable ici"
git push origin archive/distributed-pipeline
```

- [ ] **Step 4 : Supprimer le DDP du livrable**

```bash
git rm -r 03_distributed/
git rm tests/test_distributed.py
```

- [ ] **Step 5 : Vérifier que rien d'autre ne casse**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS, ~299 tests (les ~10 tests DDP ont disparu, aucun autre échec —
confirmé : aucun import fonctionnel externe vers `03_distributed`).

- [ ] **Step 6 : Commit**

```bash
git commit -m "chore(restructure): exclure le pipeline DDP du livrable (archivé via tag)"
```

---

### Task 1 : Scaffold du package `rakuten`

**Files:**
- Create: `rakuten/__init__.py`, `rakuten/models/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: package `rakuten` installable en editable ; `import rakuten` fonctionne
  depuis n'importe quel cwd.

- [ ] **Step 1 : Créer les __init__ de tête de package**

```bash
mkdir -p rakuten/models
printf '"""Classification hiérarchique de produits Rakuten."""\n' > rakuten/__init__.py
printf '"""Familles de modèles : baselines, hydra, hlpd."""\n' > rakuten/models/__init__.py
```

- [ ] **Step 2 : Ajouter [build-system] et [project] à pyproject.toml**

Ajouter en tête de `pyproject.toml` (avant le `[tool.ruff]` existant), en reprenant
les dépendances de `requirements.txt` :

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rakuten-classification"
version = "0.1.0"
description = "Classification hiérarchique de produits Rakuten (HYDRA, HLPD, baselines)"
requires-python = ">=3.10"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.setuptools.packages.find]
include = ["rakuten*"]
```

- [ ] **Step 3 : Installer en editable**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pip install -e .`
Expected : `Successfully installed rakuten-classification-0.1.0`.

- [ ] **Step 4 : Vérifier l'import du package vide depuis ailleurs**

Run : `cd /tmp && /home/infres/tandriam-25/fil_rouge/venv/bin/python -c "import rakuten; print('ok')" && cd -`
Expected : `ok`.

- [ ] **Step 5 : Vérifier que les tests existants passent encore**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299, inchangé — on n'a rien déplacé encore).

- [ ] **Step 6 : Commit**

```bash
git add rakuten/ pyproject.toml
git commit -m "feat(restructure): scaffold du package rakuten installable (pip install -e .)"
```

---

### Task 2 : Couche `common` (ex `shared/`)

**Files:**
- Move: `shared/` → `rakuten/common/`
- Modify (réécriture imports) : tous les `.py` du repo contenant `from shared.`
- Test: `tests/test_shared_data_loader.py`, `tests/test_shared_metrics.py`,
  `tests/test_shared_utils.py`, `tests/test_training_utils.py`,
  `tests/test_text_normalization.py`

**Interfaces:**
- Produces: `rakuten.common.{data_loader,metrics,utils,training_utils,text_normalization}`
  importables proprement.

- [ ] **Step 1 : Déplacer le dossier**

```bash
git mv shared rakuten/common
test -f rakuten/common/__init__.py || printf '"""Briques partagées : données, métriques, utilitaires."""\n' > rakuten/common/__init__.py
git add rakuten/common/__init__.py
```

- [ ] **Step 2 : Réécrire tous les `from shared.` → `from rakuten.common.`**

```bash
grep -rl "from shared\.\|import shared\b" --include="*.py" rakuten/ scripts/ tests/ \
  | xargs sed -i -E 's/from shared\./from rakuten.common./g; s/^import shared$/import rakuten.common as shared/g'
```

- [ ] **Step 3 : Réécrire les imports dans les tests de la couche common**

Dans `tests/test_shared_data_loader.py`, `tests/test_shared_metrics.py`,
`tests/test_shared_utils.py`, `tests/test_training_utils.py`,
`tests/test_text_normalization.py` : remplacer tout import via `conftest`/sys.path
ou `from shared.X` par des imports directs, ex :

```python
from rakuten.common.metrics import compute_all_metrics, compute_f1_by_tranche
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.common.training_utils import EarlyStopping, compute_class_weights
from rakuten.common.text_normalization import normalize_text
```

- [ ] **Step 4 : Lancer les tests de la couche**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/test_shared_data_loader.py tests/test_shared_metrics.py tests/test_shared_utils.py tests/test_training_utils.py tests/test_text_normalization.py -q`
Expected : PASS.

- [ ] **Step 5 : Lancer la suite COMPLÈTE (les modules encore en place importent maintenant rakuten.common)**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299). Les modules `02_hydra/src` etc. importent désormais
`rakuten.common.*` (résolu car le package est installé), tout en restant dans
leurs dossiers d'origine via `tests/helpers.py`.

- [ ] **Step 6 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): shared/ -> rakuten/common/ + imports rakuten.common"
```

---

### Task 3 : Couche `models/hydra` (ex `02_hydra/src/`)

**Files:**
- Move: `02_hydra/src/` → `rakuten/models/hydra/`
- Modify: `scripts/run_train.py`, `scripts/eval_hierarchy.py`,
  `scripts/generate_f1_per_class.py` (les `from src.X` → `from rakuten.models.hydra.X`,
  retrait de leurs `sys.path.insert`)
- Test: `tests/test_hydra_models.py`, `tests/test_hydra_losses.py`,
  `tests/test_hydra_trainer.py`, `tests/test_config_mapping.py`

**Interfaces:**
- Consumes: `rakuten.common.*` (Task 2)
- Produces: `rakuten.models.hydra.{models,losses,trainer,predict,config}`. Les
  imports relatifs internes (`from .losses`, `from .models`) sont préservés par
  le `git mv`.

- [ ] **Step 1 : Déplacer le dossier (les imports relatifs internes survivent)**

```bash
git mv 02_hydra/src rakuten/models/hydra
test -f rakuten/models/hydra/__init__.py || printf '"""Modèle HYDRA (multi-tête encoder-only, EMNLP 2025)."""\n' > rakuten/models/hydra/__init__.py
git add rakuten/models/hydra/__init__.py
rmdir 02_hydra 2>/dev/null || true   # si vide après déplacement des configs (Task 8)
```

- [ ] **Step 2 : Réécrire les consommateurs externes (scripts)**

Dans `scripts/run_train.py`, `scripts/eval_hierarchy.py`,
`scripts/generate_f1_per_class.py` : remplacer
```python
from src.config import map_config
from src.trainer import HYDRATrainer
```
par
```python
from rakuten.models.hydra.config import map_config
from rakuten.models.hydra.trainer import HYDRATrainer
```
et **supprimer** les lignes `sys.path.insert(0, ... "02_hydra")` et l'insertion
de `_ROOT` devenues inutiles.

- [ ] **Step 3 : Réécrire les tests hydra en imports propres**

Dans `tests/test_hydra_models.py`, `tests/test_hydra_losses.py`,
`tests/test_hydra_trainer.py`, `tests/test_config_mapping.py` : remplacer les
`import_from_hydra("X")` par des imports directs, ex :
```python
from rakuten.models.hydra.models import HYDRALocal, create_hydra_model
from rakuten.models.hydra.losses import HYDRALoss
from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.models.hydra.config import map_config, ENCODER_MAP
```

- [ ] **Step 4 : Lancer les tests hydra**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/test_hydra_models.py tests/test_hydra_losses.py tests/test_hydra_trainer.py tests/test_config_mapping.py -q`
Expected : PASS.

- [ ] **Step 5 : Suite complète**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 6 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): 02_hydra/src/ -> rakuten/models/hydra/ + imports propres"
```

---

### Task 4 : Couche `models/baselines` (ex `01_baselines/src/`)

**Files:**
- Move: `01_baselines/src/` → `rakuten/models/baselines/`
- Modify: `scripts/run_flat_hydra.py` + heredocs des SLURM baselines/éval
  (`from src.X` → `from rakuten.models.baselines.X`, retrait `sys.path.insert`)
- Test: `tests/test_flat_hydra.py`

**Interfaces:**
- Consumes: `rakuten.common.*`
- Produces: `rakuten.models.baselines.{flat_model,flat_trainer,tfidf_pipeline,word2vec_pipeline,bert_flat_classifier,common,evaluate}`.
  Import relatif `from .flat_model import FlatHydra` préservé.

- [ ] **Step 1 : Déplacer**

```bash
git mv 01_baselines/src rakuten/models/baselines
test -f rakuten/models/baselines/__init__.py || printf '"""Modèles de référence : TF-IDF, Word2Vec, fastText, FlatHydra."""\n' > rakuten/models/baselines/__init__.py
git add rakuten/models/baselines/__init__.py
```

- [ ] **Step 2 : Réécrire les consommateurs externes**

Dans `scripts/run_flat_hydra.py` : `from src.flat_trainer import ...` →
`from rakuten.models.baselines.flat_trainer import ...`, retirer `sys.path.insert`.
Dans les heredocs Python des SLURM `run_eval_flat_hydra*.slurm` : remplacer
`from src.flat_trainer import` par `from rakuten.models.baselines.flat_trainer import`
et retirer les `sys.path.insert(0, ... "01_baselines")`.

- [ ] **Step 3 : Réécrire les imports de test_flat_hydra.py**

```python
from rakuten.models.baselines.flat_model import FlatHydra
from rakuten.models.baselines.flat_trainer import FlatHydraConfig, FlatHydraTrainer
```
(retirer le bloc `import_from_distributed`/sys.path du fichier).

- [ ] **Step 4 : Tests baselines**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/test_flat_hydra.py -q`
Expected : PASS.

- [ ] **Step 5 : Suite complète**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 6 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): 01_baselines/src/ -> rakuten/models/baselines/"
```

---

### Task 5 : Couche `models/hlpd` (ex `02b_hlpd/src/`)

**Files:**
- Move: `02b_hlpd/src/` → `rakuten/models/hlpd/`
- Modify: `scripts/run_hlpd.py` (`from src import ...` → `from rakuten.models.hlpd import ...`)
- Test: `tests/test_hlpd.py`

**Interfaces:**
- Consumes: `rakuten.common.*`
- Produces: `rakuten.models.hlpd` (le `__init__.py` non-vide ré-exporte
  `HLPD, HLPDConfig, HLPDDecoderLayer, HLPDTrainer, ZLPRLoss, build_M_hiera` —
  ses imports relatifs `from .losses`/`from .models` survivent au `git mv`).

- [ ] **Step 1 : Déplacer (le __init__.py non-vide et ses imports relatifs suivent)**

```bash
git mv 02b_hlpd/src rakuten/models/hlpd
```

- [ ] **Step 2 : Réécrire le consommateur externe**

Dans `scripts/run_hlpd.py` : `from src import HLPDConfig, HLPDTrainer` →
`from rakuten.models.hlpd import HLPDConfig, HLPDTrainer`, retirer `sys.path.insert`.

- [ ] **Step 3 : Réécrire test_hlpd.py**

Remplacer `import_from_hlpd("X")` par les imports directs, ex :
```python
from rakuten.models.hlpd import HLPDConfig, HLPDTrainer
from rakuten.models.hlpd.losses import ZLPRLoss
from rakuten.models.hlpd.attention_bias import build_M_hiera
```

- [ ] **Step 4 : Tests HLPD**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/test_hlpd.py -q`
Expected : PASS (31 tests).

- [ ] **Step 5 : Suite complète**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 6 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): 02b_hlpd/src/ -> rakuten/models/hlpd/"
```

---

### Task 6 : Nettoyer scripts + heredocs SLURM restants

**Files:**
- Modify: tous les `scripts/*.slurm` (heredocs Python) et `scripts/*.py` ayant
  encore un `sys.path.insert` ou un `from src.`

**Interfaces:**
- Produces: scripts sans manipulation `sys.path`, imports `rakuten.*` partout.

- [ ] **Step 1 : Recenser les résidus**

Run :
```bash
grep -rn "sys.path.insert\|from src\.\|import src\b" scripts/
```
Expected : liste des occurrences restantes (heredocs SLURM non couverts par les
tasks précédentes).

- [ ] **Step 2 : Réécrire chaque résidu**

Pour chaque occurrence : remplacer `from src.X` par le `from rakuten.models.{…}.X`
correspondant à l'origine du module, et **supprimer** la/les ligne(s)
`sys.path.insert(...)` du heredoc.

- [ ] **Step 3 : Vérifier la syntaxe de tous les scripts**

Run :
```bash
for f in scripts/*.slurm scripts/*.sh; do bash -n "$f" || echo "KO: $f"; done
grep -rn "sys.path.insert\|from src\." scripts/ || echo "0 résidu"
```
Expected : aucun KO, `0 résidu`.

- [ ] **Step 4 : Smoke training (chemin réel d'un script)**

Run :
```bash
/home/infres/tandriam-25/fil_rouge/venv/bin/python scripts/run_train.py \
  --split-dir ~/fil_rouge/dataset_v2/datasets_finaux --n-samples 2000 \
  --config configs/hydra/default.yaml 2>&1 | head -20
```
(NB : si `configs/` pas encore déplacé — Task 8 — utiliser `configs/hydra/default.yaml`.)
Expected : le training démarre (chargement données + « Training HYDRA… »), pas
d'`ImportError`. Interrompre après confirmation (Ctrl-C).

- [ ] **Step 5 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): scripts + heredocs SLURM en imports rakuten (zéro sys.path)"
```

---

### Task 7 : Nettoyer `tests/helpers.py` et `conftest.py`

**Files:**
- Modify/Delete: `tests/helpers.py`, `tests/conftest.py`, `tests/test_imports.py`

**Interfaces:**
- Produces: une suite de tests sans gymnastique d'import (plus de
  `import_from_hydra/hlpd/distributed`, plus de `sys.path` dans conftest).

- [ ] **Step 1 : Confirmer que plus aucun test n'utilise les helpers**

Run : `grep -rn "import_from_\|from helpers" tests/`
Expected : 0 (toutes les tasks 2-5 ont migré leurs tests). S'il en reste,
les migrer en imports directs `from rakuten...`.

- [ ] **Step 2 : Supprimer helpers.py et nettoyer conftest.py**

```bash
git rm tests/helpers.py
```
Dans `tests/conftest.py` : retirer les `sys.path.insert` (rakuten est installé).
Garder le fichier minimal (ou le supprimer s'il devient vide).

- [ ] **Step 3 : Mettre à jour test_imports.py**

Remplacer ses vérifications d'import par des imports `rakuten.*`, ex :
```python
def test_package_imports():
    import rakuten.common.data_loader
    import rakuten.models.hydra.trainer
    import rakuten.models.baselines.flat_trainer
    import rakuten.models.hlpd
```

- [ ] **Step 4 : Suite complète**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 5 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): suppression de la gymnastique d'import des tests"
```

---

### Task 8 : Regrouper les configs YAML

**Files:**
- Move: `configs/hydra/` → `configs/hydra/`, `configs/baselines/` →
  `configs/baselines/`, `configs/hlpd/` → `configs/hlpd/`
- Modify: chemins de config dans `scripts/*.slurm`, `scripts/*.py` et docs

**Interfaces:**
- Produces: tous les YAML sous `configs/{hydra,baselines,hlpd}/`.

- [ ] **Step 1 : Déplacer les configs**

```bash
mkdir -p configs
git mv configs/hydra configs/hydra
git mv configs/baselines configs/baselines
git mv configs/hlpd configs/hlpd
rmdir 02_hydra 01_baselines 02b_hlpd 2>/dev/null || true
```

- [ ] **Step 2 : Réécrire les chemins de config**

```bash
grep -rl "configs/hydra\|configs/baselines\|configs/hlpd" scripts/ docs/ *.md \
  | xargs sed -i -E 's#configs/hydra#configs/hydra#g; s#configs/baselines#configs/baselines#g; s#configs/hlpd#configs/hlpd#g'
```

- [ ] **Step 3 : Vérifier qu'un script charge bien sa config**

Run :
```bash
/home/infres/tandriam-25/fil_rouge/venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/hydra/default.yaml'))['variant'])"
grep -rn "configs/hydra\|configs/baselines\|configs/hlpd" scripts/ docs/ || echo "0 chemin obsolète"
```
Expected : `local` ; `0 chemin obsolète`.

- [ ] **Step 4 : Suite complète**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 5 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): regrouper les YAML sous configs/{hydra,baselines,hlpd}/"
```

---

### Task 9 : Déplacer les notebooks

**Files:**
- Move: `00_eda/` → `notebooks/eda/`, notebooks baselines → `notebooks/baselines/`

**Interfaces:**
- Produces: `notebooks/` (hors package).

- [ ] **Step 1 : Déplacer**

```bash
mkdir -p notebooks
git mv 00_eda notebooks/eda
test -d 01_baselines/notebooks && git mv 01_baselines/notebooks notebooks/baselines || true
```

- [ ] **Step 2 : Vérifier qu'aucun code ne référence ces chemins**

Run : `grep -rn "00_eda\|01_baselines/notebooks" rakuten/ scripts/ tests/ || echo "0 référence"`
Expected : `0 référence`.

- [ ] **Step 3 : Suite complète (sanity)**

Run : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pytest tests/ -q`
Expected : PASS (~299).

- [ ] **Step 4 : Commit**

```bash
git add -A
git commit -m "refactor(restructure): notebooks regroupés sous notebooks/"
```

---

### Task 10 : Documentation (ARCHITECTURE + MAJ chemins + réconciliation v1→v2)

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Modify: `README.md`, `ONBOARDING.md`, `docs/PIPELINE.md`, `docs/METRICS.md`,
  `CLAUDE.md`

**Interfaces:**
- Produces: doc cohérente avec la nouvelle structure + chiffres v2.

- [ ] **Step 1 : Créer docs/ARCHITECTURE.md**

Contenu : la narration du projet (EDA → baselines → HYDRA → HLPD), la table de
correspondance ancien→nouveau (cf. spec), une note « entraînement distribué
exploré (DDP) mais écarté du livrable au profit du DataParallel — disponible au
tag git `archive/distributed-pipeline` », et un paragraphe sur `hybrid_system/`
(contribution séparée d'un coéquipier, CamemBERT+XGBoost, hors package).

- [ ] **Step 2 : MAJ des chemins/noms dans toute la doc**

Dans `README.md`, `ONBOARDING.md`, `docs/PIPELINE.md`, `docs/METRICS.md`,
`CLAUDE.md` : remplacer les arbres et chemins `02_hydra/src`, `shared/`,
`03_distributed/`, etc. par la structure `rakuten/…` ; commandes d'import et de
lancement mises à jour. Retirer les mentions du pipeline DDP comme composant actif.

- [ ] **Step 3 : Réconcilier les chiffres v1 → v2**

Remplacer les constantes dataset v1 par v2 dans `ONBOARDING.md`, `README.md`,
`docs/METRICS.md`, `CLAUDE.md` :
- train `5 573 659` → `5 259 035`
- classes L2 `6 126` → `5 777`
(garder l'avertissement fuite v1 déjà présent dans METRICS/comparison.html).

- [ ] **Step 4 : Vérifications doc**

Run :
```bash
grep -rn "02_hydra\|03_distributed\|from src\.\|5 573 659\|6 126" README.md ONBOARDING.md docs/PIPELINE.md docs/METRICS.md CLAUDE.md || echo "0 référence obsolète"
```
Expected : `0 référence obsolète`.

- [ ] **Step 5 : Commit**

```bash
git add -A
git commit -m "docs(restructure): ARCHITECTURE.md + chemins rakuten + chiffres v2"
```

---

## Vérification finale (avant merge dans main)

- [ ] **`pip install -e .` propre** : `/home/infres/tandriam-25/fil_rouge/venv/bin/python -m pip install -e .` → succès.
- [ ] **Import hors repo** : `cd /tmp && python -c "from rakuten.models.hydra.trainer import HYDRATrainer; print('ok')"` → `ok`.
- [ ] **Suite complète** : `pytest tests/ -q` → vert (~299).
- [ ] **Zéro résidu** : `grep -rn "sys.path.insert\|from src\.\|import_from_hydra" rakuten/ scripts/ tests/` → vide.
- [ ] **Smoke training** : `run_train.py --n-samples 2000 --config configs/hydra/default.yaml` démarre sans `ImportError`.
- [ ] **Lint** : `ruff check .` → propre.
- [ ] **`/pre-commit`** complet → READY.
- [ ] **Arbre** : plus de `00_eda 01_baselines 02_hydra 02b_hlpd 03_distributed shared` à la racine ; présence de `rakuten/ configs/ notebooks/`.

## Self-Review (effectué)

- **Couverture spec** : chaque section du spec a sa task (structure→T1-9,
  packaging→T1, imports→T2-7, suppressions DDP→T0, configs→T8, notebooks→T9,
  docs+v1/v2→T10, vérif→section finale). ✓
- **Placeholders** : aucun « TODO/TBD ». Commandes et imports explicites. ✓
- **Cohérence des types** : noms d'imports cibles cohérents entre tasks
  (`rakuten.common.*`, `rakuten.models.{hydra,baselines,hlpd}.*`). ✓
- **Risque résiduel** : les numéros de ligne ne sont pas figés (le `main`
  consolidé post-merge diffère) → le plan opère par motifs (`grep`/`sed`) +
  filet de tests, ce qui est robuste à l'état exact des fichiers.
