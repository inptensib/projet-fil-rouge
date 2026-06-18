# Restructuration en package `rakuten` — Design

## Contexte

Le projet doit être livré à une entreprise (fil rouge Télécom Paris × Rakuten).
La prise en main par un nouveau développeur est aujourd'hui freinée par un
défaut structurel : les dossiers numérotés (`02_hydra`, `02b_hlpd`…) ne peuvent
pas être des packages Python (`from 02_hydra import` → `SyntaxError`). Conséquence
en cascade : **4 packages `src/` homonymes** qui s'entrechoquent, **25 fichiers**
qui bricolent `sys.path`, un module `hydra_imports.py` de contournement, et une
gymnastique d'import dans `tests/helpers.py`. Un dev ne peut pas faire un simple
`import` propre, et le projet n'est pas installable (`pip install`).

**Objectif** : transformer le code en un **package `rakuten` unique et
installable**, avec des imports nets et zéro hack `sys.path`, organisé de façon
auto-explicative pour un nouveau venu.

## Décisions (validées en brainstorming)

1. **Cible** : package `rakuten/` installable (`pip install -e .`), pas de
   conservation de la numérotation (la narration passe dans un doc).
2. **Organisation** : par famille de modèle regroupée sous `models/`
   (« Variante A ») — top-level lisible **et** code de chaque modèle cohésif.
3. **Stratégie de migration** : incrémentale par couche, sur une branche dédiée,
   un commit vérifié par étape (309 tests comme filet).
4. **`hybrid_system/`** : hors périmètre, reste un dossier séparé, documenté
   comme la contribution d'un coéquipier.
5. **Configs YAML** : regroupées dans un `configs/` top-level (pas dans le package).
6. **`03_distributed/` (pipeline DDP) EXCLU du livrable** : ~2000 lignes qui
   n'ont produit aucun résultat publié (supplanté par le DataParallel single-node).
   Préservé via un tag git avant suppression + noté dans `ARCHITECTURE.md`.
   Conséquence : pas de `rakuten/distributed/`, plus de dépendance croisée, et
   `hydra_imports.py` disparaît avec le dossier (sa seule raison d'être).

## Structure cible

```
rakuten-classification/
├── pyproject.toml          # [build-system] + [project] + deps + [tool.ruff]
├── README.md  ONBOARDING.md  CLAUDE.md
├── rakuten/                # LE package installable
│   ├── __init__.py
│   ├── common/             # briques partagées par tout
│   │   ├── data_loader.py · text_normalization.py
│   │   ├── metrics.py · utils.py · training_utils.py
│   ├── models/             # les 3 familles comparées, chacune COMPLÈTE
│   │   ├── baselines/      # tfidf, word2vec, fasttext, flat_model, flat_trainer,
│   │   │                     common, evaluate
│   │   ├── hydra/          # models · losses · trainer · predict · config
│   │   └── hlpd/           # models · losses · attention_bias · trainer
│   └── (pas de distributed/ — pipeline DDP exclu du livrable, cf. décision 6)
├── configs/                # configs/{hydra,baselines,hlpd}/*.yaml
├── scripts/                # CLI + SLURM (imports propres)
├── notebooks/              # ex 00_eda + notebooks baselines (hors package)
├── tests/                  # helpers.py simplifié
├── docs/
│   ├── ARCHITECTURE.md     # NOUVEAU : narration 00→03 + table de correspondance
│   └── PIPELINE.md  METRICS.md  flat_hydra_conception.md
├── hybrid_system/          # INCHANGÉ, hors package (note d'avertissement)
└── results/
```

Imports cibles :

```python
from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits
from rakuten.models.hydra.models import HYDRALocal   # importé proprement par distributed/
```

### Correspondance ancien → nouveau

| Ancien | Nouveau |
|---|---|
| `shared/` | `rakuten/common/` |
| `02_hydra/src/` | `rakuten/models/hydra/` |
| `01_baselines/src/` | `rakuten/models/baselines/` |
| `02b_hlpd/src/` | `rakuten/models/hlpd/` |
| `03_distributed/src/` | **exclu** (tag git d'archive + note ARCHITECTURE) |
| `configs/hydra/` etc. | `configs/{hydra,baselines,hlpd}/` |
| `00_eda/` | `notebooks/` + narration dans `docs/ARCHITECTURE.md` |

## Packaging & imports

- **`pyproject.toml`** : ajouter `[build-system]` (setuptools), `[project]`
  (nom `rakuten-classification`, dépendances reprises de `requirements.txt`,
  `requires-python = ">=3.10"`), découverte du package `rakuten`. Conserver
  `[tool.ruff]`.
- **Règles de réécriture** (mécaniques) :
  - `from shared.X` → `from rakuten.common.X`
  - `from src.X` selon le dossier d'origine → `from rakuten.models.{hydra|baselines|hlpd}.X`
- **Suppressions** :
  - tout `03_distributed/` (pipeline DDP exclu — cf. décision 6), avec
    `hydra_imports.py`, `main.py` (`python -m src.main`), `trainer_ddp.py`, etc.,
    et le test `tests/test_distributed.py`
  - les `sys.path.insert` restants (ceux des `.py`/heredocs non-distributed)
  - la gymnastique de `tests/helpers.py` (`import_from_hydra/hlpd/distributed`,
    `_import_from`) → remplacée par de vrais imports
  - `conftest.py` : retirer les manipulations `sys.path` (gérées par
    `pip install -e .`)
- **Points d'entrée** : `scripts/*.py` gardent leur rôle de CLI mais avec imports
  propres. Plus de `python -m src.main` (DDP exclu). Pas de `console_scripts`
  pour l'instant (YAGNI).

## Séquence de migration (incrémentale, 1 commit vérifié par étape)

Sur une branche neuve `restructure/package-rakuten`, créée **après** les prérequis :

0. **Archiver puis exclure le DDP** : tag git `archive/distributed-pipeline` sur
   le `main` consolidé, puis `git rm -r 03_distributed/` + `tests/test_distributed.py`.
1. **Scaffold** : `rakuten/` + tous les `__init__.py`, `pyproject.toml`
   `[project]`/`[build-system]`, `pip install -e .`.
2. **`common/`** (ex `shared/`) : déplacer, corriger les `from shared.` → tests.
3. **`models/hydra/`** (ex `02_hydra/src/`) → tests.
4. **`models/baselines/`** (ex `01_baselines/src/`) → tests.
5. **`models/hlpd/`** (ex `02b_hlpd/src/`) → tests.
6. **`scripts/` + heredocs SLURM** : imports propres, retrait des scripts DDP
   (`launch_*.sh`) → `bash -n` + smoke training.
7. **`tests/`** : simplifier `helpers.py` + `conftest.py` → suite complète verte.
8. **`configs/`** : déplacer les YAML vers `configs/{hydra,baselines,hlpd}/`,
   mettre à jour les chemins dans scripts + docs.
9. **`notebooks/`** : déplacer `00_eda` + notebooks baselines.
10. **Docs** : créer `docs/ARCHITECTURE.md` (dont la note « distribué exploré,
    écarté du livrable »), mettre à jour les chemins et noms dans
    README/ONBOARDING/PIPELINE/METRICS/CLAUDE, **réconcilier v1→v2**
    (train 5 259 035, L2 5 777).

## Tests & vérification

- **Invariant** : la suite de tests reste verte à **chaque** commit. Référence :
  309 tests après le merge du kit v2, **moins les ~10 tests DDP retirés
  volontairement** (`test_distributed.py`) → ~299 attendus, et le compte ne
  baisse plus ensuite.
- **Vérification finale end-to-end** :
  - `pip install -e .` réussit
  - `python -c "from rakuten.models.hydra.trainer import HYDRATrainer"` depuis
    un dossier quelconque (hors repo)
  - `python -m pytest tests/ -q` → vert
  - smoke training : `python scripts/run_train.py --split-dir <v2> --n-samples 5000`
  - `ruff check .` + pipeline `/pre-commit`
  - `grep -rn "sys.path.insert\|from src\.\|import_from_hydra" rakuten/ scripts/ tests/`
    → 0 occurrence

## Prérequis bloquants (porte dure)

La restructuration **ne démarre pas** tant que :
1. Les 3 jobs SLURM en cours sont terminés (ils rechargent
   `02_hydra/src/trainer.py` etc. à la reprise — un renommage casserait leur
   resume).
2. La branche `refactor/trainers` (refactor des trainers + kit v2) est mergée
   dans `main` (sinon le renommage de dossiers entre en conflit frontal avec
   la branche).

La branche `restructure/package-rakuten` part de ce `main` consolidé.

## Hors périmètre

- `hybrid_system/` : non touché, non intégré au package. Ajout d'un paragraphe
  dans README/ARCHITECTURE indiquant que c'est la contribution séparée d'un
  coéquipier (CamemBERT + XGBoost).
- `03_distributed/` (pipeline DDP) : exclu du livrable, archivé via tag git
  (cf. décision 6). Non détruit, récupérable depuis l'historique.
- Aucun changement de logique de training/éval (refonte purement structurelle :
  déplacements + imports + packaging).
- Pas de `console_scripts` (peut s'ajouter plus tard).
- Pas de conversion DataParallel → DDP (sans objet : le DDP est retiré).

## Risques & mitigations

| Risque | Mitigation |
|---|---|
| Rename = tout-ou-rien sur les imports | Migration par couche dans l'ordre des dépendances, tests à chaque étape |
| Conflit avec `refactor/trainers` | Prérequis : merger la branche AVANT (porte dure) |
| Casser la reprise des jobs en cours | Prérequis : jobs terminés AVANT (porte dure) |
| Chemins cassés dans scripts/docs | Étapes 9-11 dédiées + grep de vérification final |
| Régression silencieuse | 309 tests verts exigés à chaque commit + smoke training |
