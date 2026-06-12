# Téléchargement du dataset Rakuten (v2) depuis Hugging Face Hub
#
# Depuis le redressement du 12/06/2026, le repo HF est organisé en
# sous-dossiers ; les splits à utiliser sont dans datasets_finaux/
# (fuite train↔éval purgée, plancher 100/classe, manifeste synthétique).
# Ce script ne télécharge QUE les fichiers finaux (~7 Go sur les 14,7 Go
# du repo) et les place à plat dans <repo>/dataset/ comme attendu par
# les scripts d'entraînement ($DATA_DIR/train_split.jsonl).
#
# Utilisation :
#   pip install huggingface_hub
#   export HF_TOKEN="hf_..."   # token read-only fourni par Tahiana
#   python scripts/download_dataset.py

import os
import shutil
import sys

from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Erreur : variable d'environnement HF_TOKEN non définie.")
    print("Demande le token read-only à Tahiana, puis :")
    print('  export HF_TOKEN="hf_..."')
    sys.exit(1)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
os.makedirs(output_dir, exist_ok=True)

snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-dataset",
    repo_type="dataset",
    allow_patterns=["datasets_finaux/*"],
    local_dir=output_dir,
    token=HF_TOKEN,
)

# Remonter les fichiers de datasets_finaux/ à la racine de dataset/
finaux_dir = os.path.join(output_dir, "datasets_finaux")
if os.path.isdir(finaux_dir):
    for fname in os.listdir(finaux_dir):
        shutil.move(os.path.join(finaux_dir, fname), os.path.join(output_dir, fname))
    os.rmdir(finaux_dir)

print(f"Dataset v2 téléchargé dans : {output_dir}")
for fname in ("train_split.jsonl", "val_split.jsonl", "test_split.jsonl", "synthetic_ids.txt"):
    path = os.path.join(output_dir, fname)
    status = f"{os.path.getsize(path) / 1024**3:.2f} Go" if os.path.exists(path) else "MANQUANT"
    print(f"  {fname:<22} {status}")
