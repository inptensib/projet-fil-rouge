# Téléchargement du dataset Rakuten depuis Hugging Face Hub
#
# Repo privé : TahianaAndriambahoaka/rakuten-dataset (organisé en sous-dossiers)
#   - datasets_finaux/  -> dataset v2 (actuel, fuite purgée)  — PAR DÉFAUT
#   - datasets_v1/      -> dataset v1 (ancien)                — option --old
# Les fichiers sont placés à plat dans <repo>/dataset/ (attendu par les scripts
# d'entraînement, $DATA_DIR/train_split.jsonl).
#
# Utilisation :
#   pip install huggingface_hub
#   export HF_TOKEN="hf_..."                    # token read-only fourni par Tahiana
#   python scripts/download_dataset.py          # v2 -> <repo>/dataset/
#   python scripts/download_dataset.py --old    # v1 -> <repo>/dataset/old/

import argparse
import os
import shutil
import sys

from huggingface_hub import snapshot_download

REPO_ID = "TahianaAndriambahoaka/rakuten-dataset"

parser = argparse.ArgumentParser(description="Télécharge le dataset Rakuten depuis Hugging Face.")
parser.add_argument(
    "--old",
    action="store_true",
    help="Télécharge l'ancien dataset (v1) dans dataset/old/ au lieu du v2 dans dataset/",
)
args = parser.parse_args()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Erreur : variable d'environnement HF_TOKEN non définie.")
    print("Demande le token read-only à Tahiana, puis :")
    print('  export HF_TOKEN="hf_..."')
    sys.exit(1)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

if args.old:
    subfolder, output_dir, label = "datasets_v1", os.path.join(base_dir, "old"), "v1 (ancien)"
else:
    subfolder, output_dir, label = "datasets_finaux", base_dir, "v2 (actuel)"

os.makedirs(output_dir, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[f"{subfolder}/*"],
    local_dir=output_dir,
    token=HF_TOKEN,
)

# Remonter les fichiers de <subfolder>/ à la racine de output_dir
src_dir = os.path.join(output_dir, subfolder)
if os.path.isdir(src_dir):
    for fname in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, fname), os.path.join(output_dir, fname))
    os.rmdir(src_dir)

print(f"Dataset {label} téléchargé dans : {output_dir}")
for fname in sorted(os.listdir(output_dir)):
    path = os.path.join(output_dir, fname)
    if os.path.isfile(path):
        print(f"  {fname:<22} {os.path.getsize(path) / 1024**3:.2f} Go")
