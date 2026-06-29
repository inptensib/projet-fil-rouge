#!/usr/bin/env python3
"""Export the CamemBERT Flat (v2) model to ONNX, and verify parity with PyTorch.

Produces a single ONNX graph that takes (input_ids, attention_mask) and returns
the unified logits (157 L1 + 5 775 L2). The graph runs with onnxruntime WITHOUT
torch/transformers — only the tokenizer and the label files are still needed
alongside (they are pre/post-processing, not part of the net). See predict_onnx.py.

    python3 export_onnx.py --model-dir ../../models/v2/flat_hydra \\
        --out ../../models/v2/flat_hydra/flat_hydra_v2.onnx

Dynamic axes: batch and sequence length are both flexible.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from predict import FlatClassifier  # reuse the loader + the FlatHydra definition


class _ExportWrapper(torch.nn.Module):
    """Return only the unified logits tensor (ONNX dislikes the ([], logits) tuple)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[1]


def export(model_dir, out_path, opset=17):
    clf = FlatClassifier(model_dir, device="cpu")
    wrapper = _ExportWrapper(clf.model).eval()
    dummy_ids = torch.ones(1, clf.max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, clf.max_length, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return clf, out_path


def validate(clf, onnx_path, tol=1e-2):
    """Compare ONNX vs PyTorch logits + predicted L1/L2 on a few sentences."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    samples = [
        ("Cage à hamster 3 étages tubes", "Cage en métal pour rongeurs, 60x40 cm"),
        ("Ballon de football taille 5", "Cuir synthétique cousu main"),
        ("Andouillette de Troyes lot de 4", "Spécialité charcutière à griller"),
    ]
    max_diff = 0.0
    mismatches = 0
    for desig, desc in samples:
        text = clf._build_text(desig, desc)
        enc = clf.tokenizer(text, truncation=True, max_length=clf.max_length,
                            padding="max_length", return_tensors="pt")
        with torch.no_grad():
            pt = clf.model(enc["input_ids"], enc["attention_mask"])[1][0].numpy()
        onnx_logits = sess.run(
            ["logits"],
            {"input_ids": enc["input_ids"].numpy(),
             "attention_mask": enc["attention_mask"].numpy()},
        )[0][0]
        max_diff = max(max_diff, float(np.abs(pt - onnx_logits).max()))
        # predicted L1 / L2 must be identical
        for lo in (pt, onnx_logits):
            pass
        pt_l1, on_l1 = pt[:clf.num_l1].argmax(), onnx_logits[:clf.num_l1].argmax()
        pt_l2, on_l2 = pt[clf.num_l1:].argmax(), onnx_logits[clf.num_l1:].argmax()
        same = (pt_l1 == on_l1) and (pt_l2 == on_l2)
        mismatches += 0 if same else 1
        print(f"  '{desig[:35]}' -> L1 {clf.l1_labels[on_l1]!r} | L2 {clf.l2_labels[on_l2]!r}"
              f"  {'✓' if same else '✗ MISMATCH'}")
    print(f"\n  écart max de logits PyTorch vs ONNX : {max_diff:.2e}")
    ok = max_diff < tol and mismatches == 0
    print("  PARITÉ OK ✓" if ok else f"  PARITÉ KO (diff>{tol} ou {mismatches} mismatch)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Export Flat v2 to ONNX + parity check.")
    ap.add_argument("--model-dir", default="../../models/v2/flat_hydra")
    ap.add_argument("--out", default="../../models/v2/flat_hydra/flat_hydra_v2.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"Export ONNX (opset {args.opset}) : {args.model_dir} -> {args.out}")
    clf, out = export(args.model_dir, args.out, args.opset)
    import os
    print(f"  écrit : {out} ({os.path.getsize(out)/1e6:.0f} Mo)\n")
    print("Validation de parité PyTorch vs ONNX :")
    ok = validate(clf, out)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
