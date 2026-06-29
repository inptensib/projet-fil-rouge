# Téléchargement des modèles entraînés depuis Hugging Face Hub
#
# Repo privé : TahianaAndriambahoaka/rakuten-models (~63 Go au total).
# Place chaque modèle dans <repo>/models/<nom-du-modèle>/.
#
# Utilisation :
#   pip install huggingface_hub
#   export HF_TOKEN="hf_..."                      # token read fourni par Tahiana
#   python scripts/download_models.py             # TOUS les modèles (~63 Go)
#   python scripts/download_models.py flat-hydra-camembert-v2   # un seul modèle

import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "TahianaAndriambahoaka/rakuten-models"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Erreur : variable d'environnement HF_TOKEN non définie.")
    print("Demande le token read-only à Tahiana, puis :")
    print('  export HF_TOKEN="hf_..."')
    sys.exit(1)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(output_dir, exist_ok=True)

# Argument optionnel : un seul modèle (sous-dossier HF) ; sinon tout le repo.
allow_patterns = None
if len(sys.argv) > 1:
    allow_patterns = [f"{sys.argv[1]}/*"]
    print(f"Téléchargement du modèle « {sys.argv[1]} » dans : {output_dir}")
else:
    print(f"Téléchargement de TOUS les modèles (~63 Go) dans : {output_dir}")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    allow_patterns=allow_patterns,  # None = repo complet
    local_dir=output_dir,
    token=HF_TOKEN,
)

print(f"Terminé. Modèles disponibles dans : {output_dir}")
