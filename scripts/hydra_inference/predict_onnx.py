#!/usr/bin/env python3
"""Standalone ONNX inference for a HYDRA hierarchy model (v2) — no torch.

Runs ``hydra_<variant>_v2.onnx`` with onnxruntime (outputs logits_l1, logits_l2),
then applies the **hierarchical mask** in numpy — same behaviour as the PyTorch
eval (the HCR-via-masking that defines these models): keep only L2 that are valid
children of the predicted L1, using the multi-parent matrix.

Needs: the .onnx, a tokenizer (camembert-base — reused from the Flat v2 bundle's
tokenizer/), the v2 label encoders (l1/l2_label_encoder.pkl, reused from Flat v2),
and parent_child_matrix_v2.npy (built by build_hierarchy_matrix.py). No torch.

    python3 predict_onnx.py \\
        --onnx ../../models/v2/HYDRA_local_CamemBERT/hydra_local_v2.onnx \\
        --encoders ../../models/v2/flat_hydra \\
        --matrix parent_child_matrix_v2.npy \\
        --designation "40 Pcs Automatique Poulet Buveur D'eau Volaille Abreuvoir Potable Mamelon Élevage Accessoires"
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


class OnnxHydraClassifier:
    """ONNX HYDRA inference with hierarchical masking (multi-parent)."""

    def __init__(self, onnx_path, encoders_dir, matrix_path,
                 tokenizer="camembert-base", max_length=256):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", "4"))
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        tdir = Path(encoders_dir) / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tdir) if tdir.exists() else tokenizer)
        with open(Path(encoders_dir) / "l1_label_encoder.pkl", "rb") as f:
            self.l1 = [str(c) for c in pickle.load(f).classes_]
        with open(Path(encoders_dir) / "l2_label_encoder.pkl", "rb") as f:
            self.l2 = [str(c) for c in pickle.load(f).classes_]
        self.matrix = np.load(matrix_path)  # (num_l1, num_l2) uint8
        assert self.matrix.shape == (len(self.l1), len(self.l2)), \
            f"matrice {self.matrix.shape} incohérente avec encoders ({len(self.l1)},{len(self.l2)})"
        self.max_length = max_length

    def _build_text(self, designation, description=""):
        designation = (designation or "").strip()
        description = (description or "").strip()
        return f"{designation}. {description}" if description else designation

    def classify(self, designation, description="", enforce_hierarchy=True, top_k=1):
        text = self._build_text(designation, description)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="np")
        logits_l1, logits_l2 = self.sess.run(
            ["logits_l1", "logits_l2"],
            {"input_ids": enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)},
        )
        logits_l1, logits_l2 = logits_l1[0], logits_l2[0]

        probs_l1 = _softmax(logits_l1)
        l1_idx = int(probs_l1.argmax())
        if enforce_hierarchy:
            allowed = self.matrix[l1_idx]  # valid L2 children of predicted L1
            logits_l2 = np.where(allowed > 0, logits_l2, -np.inf)
        probs_l2 = _softmax(logits_l2)
        order = np.argsort(probs_l2)[::-1][:max(1, top_k)]

        result = {
            "text": text,
            "level1": {"label": self.l1[l1_idx], "confidence": round(float(probs_l1[l1_idx]), 4)},
            "level2": {"label": self.l2[int(order[0])], "confidence": round(float(probs_l2[order[0]]), 4)},
            "hierarchy_enforced": bool(enforce_hierarchy),
        }
        if top_k > 1:
            result["level2_topk"] = [
                {"label": self.l2[int(i)], "confidence": round(float(probs_l2[i]), 4)} for i in order]
        return result


def main():
    ap = argparse.ArgumentParser(description="ONNX inference HYDRA v2 (masquage hiérarchique, sans torch).")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--encoders", default="../../models/v2/flat_hydra",
                    help="dossier avec l1/l2_label_encoder.pkl + tokenizer/ (réutilisés du Flat v2)")
    ap.add_argument("--matrix", default="parent_child_matrix_v2.npy")
    ap.add_argument("--designation")
    ap.add_argument("--description", default="")
    ap.add_argument("--input", help="JSONL {designation, description} par ligne (batch)")
    ap.add_argument("--output", help="JSONL de sortie (défaut: stdout)")
    ap.add_argument("--no-hierarchy", action="store_true", help="désactiver le masquage hiérarchique")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    clf = OnnxHydraClassifier(args.onnx, args.encoders, args.matrix)
    enforce = not args.no_hierarchy
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
                          enforce_hierarchy=enforce, top_k=args.top_k), ensure_ascii=False) + "\n")
                n += 1
        if args.output:
            out.close()
        print(f"[done] {n} produits -> {args.output or 'stdout'}", file=sys.stderr)
    elif args.designation:
        print(json.dumps(clf.classify(args.designation, args.description,
              enforce_hierarchy=enforce, top_k=args.top_k), ensure_ascii=False, indent=2))
    else:
        ap.error("fournir --designation (single) ou --input (batch)")


if __name__ == "__main__":
    main()
