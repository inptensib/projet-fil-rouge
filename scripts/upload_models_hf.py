#!/usr/bin/env python3
"""
Upload des modèles entraînés Rakuten vers HuggingFace Hub.

Usage :
  # Vérification sans upload (recommandé en premier)
  python scripts/upload_models_hf.py --dry-run

  # Upload complet
  python scripts/upload_models_hf.py --token hf_xxx

  # Upload d'un seul sous-dossier (debug / reprise)
  python scripts/upload_models_hf.py --token hf_xxx --target hydra-local-camembert

  # Suppression locale après vérification (destructif)
  python scripts/upload_models_hf.py --token hf_xxx --verify-and-delete
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "TahianaAndriambahoaka/rakuten-models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mapping source locale → destination dans le repo HF
UPLOAD_MAPPING = [
    # HYDRA standard CamemBERT
    (PROJECT_ROOT / "models" / "HYDRA_local_CamemBERT",         "hydra-local-camembert"),
    (PROJECT_ROOT / "models" / "HYDRA_nested_CamemBERT",        "hydra-nested-camembert"),
    (PROJECT_ROOT / "models" / "HYDRA_global_CamemBERT",        "hydra-global-camembert"),
    # HYDRA mBERT
    (PROJECT_ROOT / "models" / "HYDRA_local_mBERT",             "hydra-local-mbert"),
    (PROJECT_ROOT / "models" / "HYDRA_global_mBERT",            "hydra-global-mbert"),
    (PROJECT_ROOT / "models" / "HYDRA_nested_mBERT",            "hydra-nested-mbert"),
    # HYDRA hierarchy CamemBERT
    (PROJECT_ROOT / "models/v2" / "HYDRA_local_CamemBERT",   "hydra-local-hierarchy-camembert"),
    (PROJECT_ROOT / "models/v2" / "HYDRA_nested_CamemBERT",  "hydra-nested-hierarchy-camembert"),
    (PROJECT_ROOT / "models/v2" / "HYDRA_global_CamemBERT",  "hydra-global-hierarchy-camembert"),
    # HYDRA title-only standard
    (PROJECT_ROOT / "models/v1/title" / "HYDRA_local_CamemBERT_title",  "hydra-local-title-camembert"),
    (PROJECT_ROOT / "models/v1/title" / "HYDRA_nested_CamemBERT_title", "hydra-nested-title-camembert"),
    (PROJECT_ROOT / "models/v1/title" / "HYDRA_global_CamemBERT_title", "hydra-global-title-camembert"),
    # HYDRA title-only hierarchy
    (PROJECT_ROOT / "models/v1/title_hierarchy" / "HYDRA_local_CamemBERT_title_hierarchy",  "hydra-local-title-hierarchy-camembert"),
    (PROJECT_ROOT / "models/v1/title_hierarchy" / "HYDRA_nested_CamemBERT_title_hierarchy", "hydra-nested-title-hierarchy-camembert"),
    (PROJECT_ROOT / "models/v1/title_hierarchy" / "HYDRA_global_CamemBERT_title_hierarchy", "hydra-global-title-hierarchy-camembert"),
    # FlatHydra
    (PROJECT_ROOT / "models" / "flat_hydra",       "flat-hydra"),
    (PROJECT_ROOT / "models" / "flat_hydra_title", "flat-hydra-title"),
    # FlatHydra mBERT + dataset v2 (entraînés post-restructure, méthode naïve)
    (PROJECT_ROOT / "models" / "flat_hydra_mbert",    "flat-hydra-mbert"),
    (PROJECT_ROOT / "models/v2" / "flat_hydra",       "flat-hydra-camembert-v2"),
    (PROJECT_ROOT / "models/v2" / "flat_hydra_mbert", "flat-hydra-mbert-v2"),
    # HLPD
    (PROJECT_ROOT / "models" / "hlpd", "hlpd"),
    # Baselines
    (PROJECT_ROOT / "models" / "baselines" / "tfidf",          "baselines/tfidf"),
    (PROJECT_ROOT / "models" / "baselines" / "word2vec",        "baselines/word2vec"),
    (PROJECT_ROOT / "models" / "baselines" / "fasttext",        "baselines/fasttext"),
    (PROJECT_ROOT / "models" / "baselines" / "fasttext_softmax","baselines/fasttext_softmax"),
]

IGNORE_PATTERNS = ["_data_cache.pkl", "*.tmp", ".gitkeep", "__pycache__", "*.pyc"]

MODEL_CARD = """\
---
language: fr
license: other
tags:
  - hierarchical-classification
  - e-commerce
  - rakuten
  - french
  - text-classification
  - pytorch
datasets:
  - TahianaAndriambahoaka/rakuten-dataset
---

# Rakuten Product Classification — Modèles entraînés (HYDRA / HLPD / Baselines)

Archive complète des modèles entraînés dans le cadre du projet **fil rouge** de
classification hiérarchique de produits Rakuten France (EMNLP 2025 — architecture HYDRA).

## Tâche

Classification hiérarchique à 2 niveaux :
- **L1** : 157 catégories parentes
- **L2** : 6 126 catégories feuilles (dataset v1 d'origine ; 5 777 sur le dataset v2 dé-fuité)

Données : 8M+ produits Rakuten France, texte en français (`designation + description`).
Dataset : [TahianaAndriambahoaka/rakuten-dataset](https://huggingface.co/datasets/TahianaAndriambahoaka/rakuten-dataset)

## Résultats test (F1 macro, set de test ~1,17M produits)

### HYDRA (multi-tête encoder-only, CamemBERT-base)

| Modèle | Entrée | F1 L1 | F1 L2 | HCR |
|---|---|---|---|---|
| `flat-hydra` | Title + Desc | 93.24 % | **71.70 %** | 85.49 % |
| `hydra-local-camembert` | Title + Desc | 93.50 % | 66.65 % | 85.82 % |
| `hydra-nested-camembert` | Title + Desc | 93.62 % | 64.26 % | 85.83 % |
| `hydra-global-camembert` | Title + Desc | 93.56 % | 66.20 % | 85.87 % |
| `hydra-local-hierarchy-camembert` | Title + Desc | 91.33 % | 61.18 % | **99.74 %** |
| `hydra-nested-hierarchy-camembert` | Title + Desc | 92.06 % | 61.14 % | **99.74 %** |
| `hydra-global-hierarchy-camembert` | Title + Desc | 92.59 % | 62.70 % | **99.74 %** |
| `hydra-local-mbert` | Title + Desc | 93.04 % | 66.80 % | 85.81 % |

### FlatHydra mBERT & dataset v2 (méthode naïve, ajoutés post-restructure)

> v2 = dataset dé-fuité (5 777 L2). **Non comparable terme à terme** à v1 (6 126 L2) — cf. dataset.

| Modèle | Dataset | F1 L1 | F1 L2 | HCR |
|---|---|---|---|---|
| `flat-hydra-camembert-v2` | **v2** | 92.47 % | **78.93 %** | 85.17 % |
| `flat-hydra-mbert-v2` | **v2** | 92.09 % | 77.40 % | 85.20 % |
| `flat-hydra-mbert` | v1 | 93.38 % | 71.78 % | 85.61 % |
| `hydra-global-mbert` | v1 | — | 66.28 % (val) | — |
| `hydra-nested-mbert` | v1 | — | 64.31 % (val) | — |

Note : « (val) » = F1 L2 **validation** (set test non évalué pour ces deux modèles).

### HLPD (Hierarchical Label-aware Pre-trained Decoder)

| Modèle | Entrée | F1 L1 | F1 L2 | HCR |
|---|---|---|---|---|
| `hlpd` | Title + Desc | 80.70 % | 61.50 % | 97.80 % |

### Baselines

| Modèle | Entrée | F1 L1 | F1 L2 | HCR |
|---|---|---|---|---|
| FastText Softmax | Title + Desc | 82.61 % | 42.87 % | 67.41 % |
| TF-IDF + LinearSVC | Title + Desc | 83.26 % | 42.75 % | 63.41 % |
| Word2Vec + LinearSVC | Title + Desc | 60.24 % | 22.27 % | 25.06 % |
| FastText Supervised | Title + Desc | 82.60 % | 10.01 % | 29.29 % |

## Structure du repo

```
rakuten-models/
├── hydra-local-camembert/
│   ├── best_model.pt             # Poids inférence (~436 Mo)
│   ├── checkpoint_latest.pt      # État complet + optimiseur (~1.3 Go)
│   └── training_complete.txt
├── hydra-nested-camembert/
├── hydra-global-camembert/
├── hydra-local-mbert/
├── hydra-global-mbert/
├── hydra-nested-mbert/
├── hydra-local-hierarchy-camembert/
├── hydra-nested-hierarchy-camembert/
├── hydra-global-hierarchy-camembert/
├── hydra-local-title-camembert/
├── hydra-nested-title-camembert/
├── hydra-global-title-camembert/
├── hydra-local-title-hierarchy-camembert/
├── hydra-nested-title-hierarchy-camembert/
├── hydra-global-title-hierarchy-camembert/
├── flat-hydra/                   # + tokenizer/, label_encoders, mappings
├── flat-hydra-title/
├── flat-hydra-mbert/
├── flat-hydra-camembert-v2/      # meilleur F1 L2 : 78.9 % (dataset v2)
├── flat-hydra-mbert-v2/
├── hlpd/                         # checkpoint_latest.pt uniquement
└── baselines/
    ├── tfidf/
    ├── word2vec/
    ├── fasttext/
    └── fasttext_softmax/
```

## Charger un modèle

```python
from huggingface_hub import snapshot_download
import torch

# Télécharger uniquement les fichiers d'inférence d'une variante
local_dir = snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-models",
    allow_patterns="hydra-local-camembert/*",
)
model = torch.load(f"{local_dir}/hydra-local-camembert/best_model.pt", map_location="cpu")
```

Pour recharger avec l'architecture complète, cloner le repo du projet et utiliser
`rakuten/models/hydra/models.py::create_hydra_model()`.
"""


def get_dir_size(path: Path) -> int:
    """Retourne la taille totale d'un répertoire en octets."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file() and not any(f.match(p) for p in IGNORE_PATTERNS):
            total += f.stat().st_size
    return total


def fmt_size(n_bytes: int) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} To"


def print_inventory(mapping: list) -> int:
    """Affiche l'inventaire et retourne la taille totale."""
    total = 0
    print(f"\n{'Source locale':<60} {'Destination HF':<45} {'Taille':>10}")
    print("-" * 118)
    for src, dst in mapping:
        if src.exists():
            size = get_dir_size(src)
            total += size
            print(f"  {str(src.relative_to(PROJECT_ROOT)):<58} {dst:<45} {fmt_size(size):>10}")
        else:
            print(f"  {str(src.relative_to(PROJECT_ROOT)):<58} {dst:<45} {'ABSENT':>10}")
    print("-" * 118)
    print(f"{'TOTAL':<105} {fmt_size(total):>10}")
    return total


def upload_model_card(api, dry_run: bool) -> None:
    """Crée/met à jour le README.md du repo avec la model card."""
    if dry_run:
        print("\n[dry-run] Créerait le fichier README.md (model card) à la racine du repo")
        return
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        tmp_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
        print("  ✓ README.md (model card) uploadé")
    finally:
        os.unlink(tmp_path)


def upload_one(api, src: Path, dst: str, dry_run: bool) -> None:
    """Upload un sous-dossier vers le repo."""
    if not src.exists():
        print(f"  [SKIP] {src} n'existe pas localement")
        return
    size = get_dir_size(src)
    if dry_run:
        print(f"  [dry-run] uploadrait {src.relative_to(PROJECT_ROOT)} → {dst}/ ({fmt_size(size)})")
        return
    print(f"  → Upload {src.relative_to(PROJECT_ROOT)} → {dst}/ ({fmt_size(size)}) ...", flush=True)
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(src),
        path_in_repo=dst,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Upload {dst}",
    )
    print(f"  ✓ {dst}/ uploadé")


def verify_upload(api, mapping: list) -> bool:
    """Compare la liste des fichiers distants vs locaux. Retourne True si OK."""
    print("\nVérification de l'intégrité de l'upload...")
    remote_files = set(api.list_repo_files(REPO_ID, repo_type="model"))
    all_ok = True
    for src, dst in mapping:
        if not src.exists():
            continue
        local_files = [
            f.relative_to(src)
            for f in src.rglob("*")
            if f.is_file() and not any(f.match(p) for p in IGNORE_PATTERNS)
        ]
        missing = []
        for lf in local_files:
            expected_remote = f"{dst}/{lf.as_posix()}"
            if expected_remote not in remote_files:
                missing.append(expected_remote)
        if missing:
            all_ok = False
            print(f"  ✗ {dst}/ — {len(missing)} fichier(s) manquant(s) sur HF :")
            for m in missing[:5]:
                print(f"      {m}")
            if len(missing) > 5:
                print(f"      ... et {len(missing) - 5} autres")
        else:
            print(f"  ✓ {dst}/ — {len(local_files)} fichier(s) présents sur HF")
    return all_ok


def delete_local(mapping: list) -> None:
    """Supprime les dossiers locaux après confirmation."""
    import shutil

    print("\n⚠  Suppression locale — cette opération est IRRÉVERSIBLE.")
    print("   Les modèles doivent être correctement uploadés sur HF avant de continuer.")
    answer = input("   Taper 'OUI JE CONFIRME' pour procéder : ").strip()
    if answer != "OUI JE CONFIRME":
        print("   Annulé.")
        return

    # Grouper les dossiers racines à supprimer
    roots_to_delete = set()
    for src, _ in mapping:
        # Garder seulement les dossiers de premier niveau sous models*/
        parts = src.relative_to(PROJECT_ROOT).parts
        roots_to_delete.add(PROJECT_ROOT / parts[0] / parts[1] if len(parts) > 2 else src)

    for path in sorted(roots_to_delete):
        if path.exists():
            shutil.rmtree(path)
            print(f"  ✓ Supprimé : {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [skip] Déjà absent : {path.relative_to(PROJECT_ROOT)}")

    # Supprimer aussi les dossiers racines vides (models/v2, models/v1/title, etc.)
    for root_dir in ["models/v2", "models/v1/title", "models/v1/title_hierarchy"]:
        root_path = PROJECT_ROOT / root_dir
        if root_path.exists():
            remaining = list(root_path.iterdir())
            if not remaining:
                root_path.rmdir()
                print(f"  ✓ Répertoire vide supprimé : {root_dir}/")

    print("\nSuppression terminée. Vérifiez avec : du -sh models* models/v2 models/v1/title*")


def main():
    parser = argparse.ArgumentParser(description="Upload des modèles Rakuten vers HuggingFace Hub")
    parser.add_argument("--token", help="Token HF (sinon lit HF_TOKEN env ou ~/.cache/huggingface/token)")
    parser.add_argument("--dry-run", action="store_true", help="Afficher ce qui serait uploadé sans faire l'upload")
    parser.add_argument("--target", help="Uploader uniquement ce sous-dossier (ex: hydra-local-camembert)")
    parser.add_argument("--verify-and-delete", action="store_true",
                        help="Vérifier l'intégrité puis proposer la suppression locale")
    args = parser.parse_args()

    # Résoudre le token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token and not args.dry_run:
        print("Erreur : aucun token HF trouvé.")
        print("Options :")
        print("  1. Exporter : export HF_TOKEN=hf_xxx")
        print("  2. Passer en argument : --token hf_xxx")
        print("  3. Se connecter : hf auth login")
        sys.exit(1)

    # Filtrer le mapping si --target
    mapping = UPLOAD_MAPPING
    if args.target:
        mapping = [(src, dst) for src, dst in UPLOAD_MAPPING if dst == args.target or dst.startswith(args.target + "/")]
        if not mapping:
            print(f"Erreur : cible '{args.target}' introuvable dans le mapping.")
            print("Destinations disponibles :")
            for _, dst in UPLOAD_MAPPING:
                print(f"  {dst}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Repo cible : {REPO_ID}")
    print(f"  Mode       : {'dry-run' if args.dry_run else 'upload réel'}")
    if args.target:
        print(f"  Filtre     : {args.target}")
    print(f"{'=' * 60}")

    total_size = print_inventory(mapping)
    print(f"\nEstimation durée upload : {total_size / 1024**3 / 5:.0f}–{total_size / 1024**3 / 2:.0f} min (selon débit)")

    if args.dry_run:
        print("\n[dry-run] Aucun upload effectué.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Vérifier l'authentification
    try:
        user = api.whoami()
        print(f"\nConnecté en tant que : {user['name']}")
    except Exception as e:
        print(f"Erreur d'authentification : {e}")
        sys.exit(1)

    if args.verify_and_delete:
        ok = verify_upload(api, mapping)
        if ok:
            delete_local(mapping)
        else:
            print("\n✗ Des fichiers manquent sur HF. Suppression locale annulée par sécurité.")
            print("  Relancez l'upload des dossiers manquants, puis réessayez --verify-and-delete.")
        return

    # Création du repo (idempotente)
    print(f"\nCréation du repo {REPO_ID} (si nécessaire)...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
    print(f"  ✓ Repo prêt")

    # Upload model card
    upload_model_card(api, dry_run=False)

    # Upload des dossiers
    print(f"\nUpload de {len(mapping)} dossier(s)...")
    errors = []
    for i, (src, dst) in enumerate(mapping, 1):
        print(f"\n[{i}/{len(mapping)}] {dst}")
        try:
            upload_one(api, src, dst, dry_run=False)
        except Exception as e:
            errors.append((dst, str(e)))
            print(f"  ✗ Erreur sur {dst} : {e}")
            print("  → Continuons avec le suivant...")

    print(f"\n{'=' * 60}")
    if errors:
        print(f"Upload terminé avec {len(errors)} erreur(s) :")
        for dst, err in errors:
            print(f"  ✗ {dst} : {err}")
        print("\nPour reprendre un dossier spécifique :")
        print("  python scripts/upload_models_hf.py --token hf_xxx --target <nom>")
    else:
        print("Upload complet sans erreur.")
        print(f"\nVérifiez sur : https://huggingface.co/{REPO_ID}")
        print("\nPour vérifier et supprimer localement :")
        print("  python scripts/upload_models_hf.py --token hf_xxx --verify-and-delete")


if __name__ == "__main__":
    main()
