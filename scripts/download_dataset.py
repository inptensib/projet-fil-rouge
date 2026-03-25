# Téléchargement du dataset Rakuten depuis Hugging Face Hub
#
# Utilisation :
#   pip install huggingface_hub
#   export HF_TOKEN="hf_..."   # token read-only fourni par Tahiana
#   python scripts/download_dataset.py

import os
import sys

from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Erreur : variable d'environnement HF_TOKEN non définie.")
    print("Demande le token read-only à Tahiana, puis :")
    print('  export HF_TOKEN="hf_..."')
    sys.exit(1)

output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(output_dir, exist_ok=True)

snapshot_download(
    repo_id="TahianaAndriambahoaka/rakuten-dataset",
    repo_type="dataset",
    local_dir=output_dir,
    token=HF_TOKEN,
)

print(f"Dataset téléchargé dans : {os.path.abspath(output_dir)}")
