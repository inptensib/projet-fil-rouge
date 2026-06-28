#!/usr/bin/env python3
"""Standalone inference for the retained Rakuten classifier — CamemBERT Flat (v2).

Classifies a French product listing (designation + description) into:
  - L1 category (157 classes, ``typelabel``)
  - L2 category (5 775 classes, ``producttype``)

A single CamemBERT-base encoder -> projection (768->1536) applied BEFORE pooling
-> CLS pooling -> one unified MLP head -> logits of size 157 + 5 775 = 5 932,
sliced into L1 ``[:157]`` and L2 ``[157:]``.

This file is self-contained: it copies the minimal model definition so it runs
without the training package. Source of truth, kept in sync with this copy:
``rakuten/models/baselines/flat_model.py``.

Usage:
    # single product
    python predict.py --model-dir flat-hydra-camembert-v2 \\
        --designation "Cage a hamster 3 etages" --description "Metal, 60x40 cm"

    # batch: JSONL in ({"designation": ..., "description": ...} per line) -> JSONL out
    python predict.py --model-dir flat-hydra-camembert-v2 \\
        --input products.jsonl --output predictions.jsonl

The model weights are not shipped in git (~493 MB). Fetch them first with
``download_weights.py`` (see README.md).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

POOLING_METHODS = ("cls", "mean", "max", "last")


class FlatHydra(nn.Module):
    """Single unified head predicting L1 + L2 simultaneously.

    Faithful copy of ``rakuten/models/baselines/flat_model.py`` so this script is
    self-contained. The projection is applied on ``last_hidden_state`` BEFORE
    pooling, matching the official HYDRA baseline.
    """

    NUM_LEVELS = 2

    def __init__(self, encoder_name, num_classes_l1, num_classes_l2,
                 pooling="cls", project_embedding=True, dropout=0.2):
        super().__init__()
        if pooling not in POOLING_METHODS:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from {POOLING_METHODS}")
        self.pooling = pooling
        self.project_embedding = project_embedding
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2

        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size  # 768 for CamemBERT

        if project_embedding:
            self.embedding_projector = nn.Linear(d, d * self.NUM_LEVELS)
            total_input = d * self.NUM_LEVELS
        else:
            self.embedding_projector = None
            total_input = d

        total_labels = num_classes_l1 + num_classes_l2
        self.unified_classifier = nn.Sequential(
            nn.Linear(total_input, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, total_labels),
        )

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            return hidden_states.masked_fill(mask == 0, -1e9).max(dim=1).values
        last_idx = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        if self.project_embedding:
            hidden = self.embedding_projector(hidden)
        embedding = self._pool(hidden, attention_mask)
        return [], self.unified_classifier(embedding)


class FlatClassifier:
    """Loads the model bundle once and classifies product listings."""

    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        if not (model_dir / "best_model.pt").exists():
            raise FileNotFoundError(
                f"best_model.pt not found in '{model_dir}'. Download the model first "
                "(python download_weights.py) — see README.md.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(model_dir / "best_model.pt", map_location=self.device,
                          weights_only=False)
        cfg = ckpt["config"]
        self.max_length = int(cfg.get("max_length", 256))
        self.use_description = bool(cfg.get("use_description", True))
        self.num_l1 = int(ckpt["num_classes_l1"])
        self.num_l2 = int(ckpt["num_classes_l2"])

        self.model = FlatHydra(
            encoder_name=cfg["encoder"],
            num_classes_l1=self.num_l1,
            num_classes_l2=self.num_l2,
            pooling=cfg.get("pooling", "cls"),
            project_embedding=cfg.get("project_embedding", True),
            dropout=cfg.get("dropout", 0.2),
        )
        # strip a possible DDP "module." prefix
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        with open(model_dir / "l1_label_encoder.pkl", "rb") as f:
            self.l1_labels = [str(c) for c in pickle.load(f).classes_]
        with open(model_dir / "l2_label_encoder.pkl", "rb") as f:
            self.l2_labels = [str(c) for c in pickle.load(f).classes_]

        # l2 index -> primary L1 parent index (for optional hierarchy enforcement)
        mapping = json.loads((model_dir / "l2_to_l1_mapping.json").read_text(encoding="utf-8"))
        self.l2_parent = torch.full((self.num_l2,), -1, dtype=torch.long)
        for l2_idx, l1_idx in mapping.items():
            self.l2_parent[int(l2_idx)] = int(l1_idx)
        self.l2_parent = self.l2_parent.to(self.device)

    def _build_text(self, designation, description=""):
        designation = (designation or "").strip()
        description = (description or "").strip()
        if self.use_description and description:
            return f"{designation}. {description}"
        return designation

    @torch.no_grad()
    def classify(self, designation, description="", enforce_hierarchy=False, top_k=1):
        text = self._build_text(designation, description)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt").to(self.device)
        _, logits = self.model(enc["input_ids"], enc["attention_mask"])
        logits = logits[0]
        logits_l1, logits_l2 = logits[:self.num_l1], logits[self.num_l1:]

        probs_l1 = F.softmax(logits_l1, dim=-1)
        l1_idx = int(probs_l1.argmax())

        if enforce_hierarchy:
            allowed = self.l2_parent == l1_idx
            if bool(allowed.any()):
                logits_l2 = logits_l2.masked_fill(~allowed, float("-inf"))
        probs_l2 = F.softmax(logits_l2, dim=-1)
        k = max(1, min(top_k, self.num_l2))
        l2_conf, l2_idx = torch.topk(probs_l2, k)

        result = {
            "text": text,
            "level1": {"label": self.l1_labels[l1_idx],
                       "confidence": round(float(probs_l1[l1_idx]), 4)},
            "level2": {"label": self.l2_labels[int(l2_idx[0])],
                       "confidence": round(float(l2_conf[0]), 4)},
            "hierarchy_enforced": bool(enforce_hierarchy),
        }
        if top_k > 1:
            result["level2_topk"] = [
                {"label": self.l2_labels[int(i)], "confidence": round(float(c), 4)}
                for c, i in zip(l2_conf, l2_idx)]
        return result


def main():
    ap = argparse.ArgumentParser(
        description="Classify Rakuten products into L1/L2 with CamemBERT Flat (v2).")
    ap.add_argument("--model-dir", default="models/v2/flat_hydra",
                    help="Folder with best_model.pt, tokenizer/, encoders, mapping.")
    ap.add_argument("--designation", help="Product title (single-item mode).")
    ap.add_argument("--description", default="", help="Product description (optional).")
    ap.add_argument("--input", help="JSONL with {designation, description} per line (batch mode).")
    ap.add_argument("--output", help="JSONL output path for batch mode (default: stdout).")
    ap.add_argument("--enforce-hierarchy", action="store_true",
                    help="Constrain L2 to children of the predicted L1 (single primary parent).")
    ap.add_argument("--top-k", type=int, default=1, help="Return top-k L2 candidates.")
    ap.add_argument("--device", help="cpu | cuda (default: auto).")
    args = ap.parse_args()

    clf = FlatClassifier(args.model_dir, device=args.device)

    if args.input:
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        n = 0
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                r = clf.classify(o.get("designation", ""), o.get("description", ""),
                                 enforce_hierarchy=args.enforce_hierarchy, top_k=args.top_k)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        if args.output:
            out.close()
        print(f"[done] {n} products -> {args.output or 'stdout'}", file=sys.stderr)
    elif args.designation:
        print(json.dumps(
            clf.classify(args.designation, args.description,
                         enforce_hierarchy=args.enforce_hierarchy, top_k=args.top_k),
            ensure_ascii=False, indent=2))
    else:
        ap.error("provide --designation (single mode) or --input (batch mode).")


if __name__ == "__main__":
    main()
