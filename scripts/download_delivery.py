# Download the Rakuten delivery bundle from the Hugging Face Hub.
#
# Private repo: TahianaAndriambahoaka/rakuten-delivery (~9 GB total).
# A single, self-contained handoff: everything needed to RUN the models, plus
# the v2 dataset, in one download.
#
#   onnx/flat_v2/    retained model (CamemBERT Flat v2) in ONNX + tokenizer,
#                    label encoders and a standalone predict_onnx.py
#   onnx/hydra_v2/   the 3 HYDRA hierarchy models in ONNX + the hierarchical
#                    mask matrix and a standalone predict_onnx.py
#   dataset_v2/      train / val / test splits (JSONL) + synthetic_ids.txt
#   README.md        full usage; requirements.txt
#
# Inference runs fully offline: onnxruntime only, no PyTorch and no Hugging Face
# at runtime (the tokenizer and label encoders are bundled). See the bundle's
# README.md and predict_onnx.py.
#
# Usage:
#   pip install huggingface_hub
#   export HF_TOKEN="hf_..."                          # read token provided by the owner
#   python3 scripts/download_delivery.py               # everything (~9 GB)
#   python3 scripts/download_delivery.py --onnx-only       # ONNX models only (~1.8 GB)
#   python3 scripts/download_delivery.py --dataset-only    # dataset v2 only (~7.2 GB)
#   python3 scripts/download_delivery.py --output /path/to/dir

import argparse
import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "TahianaAndriambahoaka/rakuten-delivery"

parser = argparse.ArgumentParser(
    description="Download the Rakuten delivery bundle (ONNX models + dataset v2) from Hugging Face.",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--onnx-only",
    action="store_true",
    help="Download only the ONNX models (onnx/ + README + requirements), ~1.8 GB",
)
group.add_argument(
    "--dataset-only",
    action="store_true",
    help="Download only the v2 dataset (dataset_v2/), ~7.2 GB",
)
parser.add_argument(
    "--output",
    default=None,
    help="Destination directory (default: <repo>/rakuten-delivery/)",
)
args = parser.parse_args()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: environment variable HF_TOKEN is not set.")
    print("Ask the owner for the read-only token, then:")
    print('  export HF_TOKEN="hf_..."')
    sys.exit(1)

if args.output:
    output_dir = os.path.abspath(args.output)
else:
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rakuten-delivery"))
os.makedirs(output_dir, exist_ok=True)

if args.onnx_only:
    allow_patterns = ["onnx/*", "README.md", "requirements.txt"]
    label = "ONNX models only (~1.8 GB)"
elif args.dataset_only:
    allow_patterns = ["dataset_v2/*", "README.md"]
    label = "dataset v2 only (~7.2 GB)"
else:
    allow_patterns = None  # full repo
    label = "full bundle (~9 GB)"

print(f"Downloading {label} into: {output_dir}")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    allow_patterns=allow_patterns,
    local_dir=output_dir,
    token=HF_TOKEN,
)

print(f"\nDone. Delivery bundle available in: {output_dir}")
print("Next: see README.md for offline ONNX inference (onnxruntime, no PyTorch).")
