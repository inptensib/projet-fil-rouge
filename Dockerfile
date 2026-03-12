# ==============================================================================
# Rakuten Hierarchical Classification -- Base Image
# Source de verite pour l'environnement Python (dev, CI, et production)
# ==============================================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="tandriam-25"
LABEL project="rakuten-classification"

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

# --- Copy project ---
COPY . .
