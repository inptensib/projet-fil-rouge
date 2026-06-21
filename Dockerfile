# ==============================================================================
# Rakuten Hierarchical Classification -- Base Image
# La source de verite des dependances est requirements.txt / pyproject.toml.
# Cette image les reproduit pour le dev, la CI et l'entrainement GPU.
# ==============================================================================

# Image de base CUDA. NB : requirements.txt epingle torch==2.7.1, qui est donc
# reinstalle par-dessus le torch fourni par l'image de base (le wheel torch
# embarque ses propres libs CUDA, l'execution reste correcte).
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL project="rakuten-classification"
LABEL org="Télécom Paris × Rakuten France"

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Env vars (set BEFORE downloads so cache goes to the right place) ---
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# --- Python deps (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

# --- Pre-download CamemBERT tokenizer + model weights (cached in image) ---
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('camembert-base'); \
AutoModel.from_pretrained('camembert-base')"

# --- Copy project + installation editable du package rakuten ---
COPY . .
RUN pip install --no-cache-dir -e .
