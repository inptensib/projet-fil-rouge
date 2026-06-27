#!/usr/bin/env python3
"""Download the retained Flat v2 model from the Hugging Face Hub.

    export HF_TOKEN=hf_...                  # the model repo is private
    python download_weights.py [target_dir]   # default: current directory

Fetches the ``flat-hydra-camembert-v2/`` folder (~497 MB: best_model.pt,
tokenizer/, label encoders, mapping, config) from the model repo and prints the
local model directory to pass to ``predict.py --model-dir``.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "TahianaAndriambahoaka/rakuten-models"
SUBDIR = "flat-hydra-camembert-v2"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("warning: no HF_TOKEN/HUGGINGFACE_TOKEN set — this will fail if the "
              "repo is private.", file=sys.stderr)
    local_root = snapshot_download(
        repo_id=REPO,
        allow_patterns=f"{SUBDIR}/*",
        local_dir=target,
        token=token,
    )
    model_dir = os.path.join(local_root, SUBDIR)
    print(model_dir)


if __name__ == "__main__":
    main()
