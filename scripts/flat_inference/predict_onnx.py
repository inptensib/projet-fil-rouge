#!/usr/bin/env python3
"""Standalone ONNX inference for the CamemBERT Flat (v2) model — no torch needed.

Runs the exported ``flat_hydra_v2.onnx`` with onnxruntime. Same I/O as predict.py,
but inference needs only onnxruntime + the tokenizer + the label files (not the
torch/transformers model stack) — lighter and faster to start, portable.

    # single
    python predict_onnx.py --model-dir ../../models_v2/flat_hydra \\
        --designation "Cage à hamster 3 étages" --description "Métal, 60x40 cm"
    # batch (JSONL in -> JSONL out)
    python predict_onnx.py --model-dir ../../models_v2/flat_hydra \\
        --input products.jsonl --output predictions.jsonl

The ONNX file is produced by export_onnx.py (parity-checked against PyTorch).
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class OnnxFlatClassifier:
    """Loads the ONNX graph + tokenizer + label files and classifies listings."""

    def __init__(self, model_dir, onnx_path=None, max_length=256):
        model_dir = Path(model_dir)
        onnx_path = Path(onnx_path) if onnx_path else model_dir / "flat_hydra_v2.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} introuvable — lancez d'abord export_onnx.py.")
        so = ort.SessionOptions()
        # explicit thread count avoids onnxruntime CPU-affinity warnings on big nodes
        so.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", "4"))
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        with open(model_dir / "l1_label_encoder.pkl", "rb") as f:
            self.l1_labels = [str(c) for c in pickle.load(f).classes_]
        with open(model_dir / "l2_label_encoder.pkl", "rb") as f:
            self.l2_labels = [str(c) for c in pickle.load(f).classes_]
        self.num_l1 = len(self.l1_labels)
        mapping = json.loads((model_dir / "l2_to_l1_mapping.json").read_text(encoding="utf-8"))
        self.l2_parent = np.full(len(self.l2_labels), -1, dtype=np.int64)
        for k, v in mapping.items():
            self.l2_parent[int(k)] = int(v)
        self.max_length = max_length

    def _build_text(self, designation, description=""):
        designation = (designation or "").strip()
        description = (description or "").strip()
        return f"{designation}. {description}" if description else designation

    def classify(self, designation, description="", enforce_hierarchy=False, top_k=1):
        text = self._build_text(designation, description)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="np")
        logits = self.sess.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0][0]
        logits_l1, logits_l2 = logits[:self.num_l1], logits[self.num_l1:]

        probs_l1 = _softmax(logits_l1)
        l1_idx = int(probs_l1.argmax())
        if enforce_hierarchy:
            allowed = self.l2_parent == l1_idx
            if allowed.any():
                logits_l2 = np.where(allowed, logits_l2, -np.inf)
        probs_l2 = _softmax(logits_l2)
        order = np.argsort(probs_l2)[::-1][:max(1, top_k)]

        result = {
            "text": text,
            "level1": {"label": self.l1_labels[l1_idx], "confidence": round(float(probs_l1[l1_idx]), 4)},
            "level2": {"label": self.l2_labels[int(order[0])], "confidence": round(float(probs_l2[order[0]]), 4)},
            "hierarchy_enforced": bool(enforce_hierarchy),
        }
        if top_k > 1:
            result["level2_topk"] = [
                {"label": self.l2_labels[int(i)], "confidence": round(float(probs_l2[i]), 4)} for i in order]
        return result


def main():
    ap = argparse.ArgumentParser(description="ONNX inference (CamemBERT Flat v2, sans torch).")
    ap.add_argument("--model-dir", default="../../models_v2/flat_hydra")
    ap.add_argument("--onnx", default=None, help="défaut : <model-dir>/flat_hydra_v2.onnx")
    ap.add_argument("--designation")
    ap.add_argument("--description", default="")
    ap.add_argument("--input", help="JSONL {designation, description} par ligne (batch)")
    ap.add_argument("--output", help="JSONL de sortie (défaut: stdout)")
    ap.add_argument("--enforce-hierarchy", action="store_true")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    clf = OnnxFlatClassifier(args.model_dir, args.onnx)
    if args.input:
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        n = 0
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                out.write(json.dumps(clf.classify(o.get("designation", ""), o.get("description", ""),
                          enforce_hierarchy=args.enforce_hierarchy, top_k=args.top_k), ensure_ascii=False) + "\n")
                n += 1
        if args.output:
            out.close()
        print(f"[done] {n} produits -> {args.output or 'stdout'}", file=sys.stderr)
    elif args.designation:
        print(json.dumps(clf.classify(args.designation, args.description,
              enforce_hierarchy=args.enforce_hierarchy, top_k=args.top_k), ensure_ascii=False, indent=2))
    else:
        ap.error("fournir --designation (single) ou --input (batch)")


if __name__ == "__main__":
    main()
