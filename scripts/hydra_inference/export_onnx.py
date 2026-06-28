#!/usr/bin/env python3
"""Export a HYDRA hierarchy model (v2) to ONNX, with PyTorch-parity check.

Inference for the 3 HYDRA variants (local/nested/global) uses the **local heads**
``(logits_l1, logits_l2)`` — the global/nested extra head is only used at training.
So the ONNX graph exposes exactly these two outputs (identical wrapper for all
variants). The hierarchical mask is NOT in the graph; it is applied in
post-processing (see predict_onnx.py), preserving the "HCR via masking" behaviour.

This is a DEV / build tool: it imports the ``rakuten`` package to load the
checkpoint. The shipped inference (predict_onnx.py) needs only onnxruntime.

    python export_onnx.py \\
        --ckpt ../../models/v2/HYDRA_local_CamemBERT/best_model.pt \\
        --out  ../../models/v2/HYDRA_local_CamemBERT/hydra_local_v2.onnx
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rakuten.models.hydra.predict import load_model


class _Wrap(torch.nn.Module):
    """Expose only the local heads (logits_l1, logits_l2) for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        (logits_l1, logits_l2), _ = self.model(input_ids, attention_mask)
        return logits_l1, logits_l2


def export(ckpt, out, opset=17, max_length=256):
    model, config = load_model(ckpt, device=torch.device("cpu"))
    wrapper = _Wrap(model).eval()
    dummy = (torch.ones(1, max_length, dtype=torch.long),
             torch.ones(1, max_length, dtype=torch.long))
    torch.onnx.export(
        wrapper, dummy, out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits_l1", "logits_l2"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits_l1": {0: "batch"},
            "logits_l2": {0: "batch"},
        },
        opset_version=opset, do_constant_folding=True,
    )
    return model, config


def validate(model, onnx_path, max_length=256, tol=1e-2):
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("camembert-base")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    samples = [
        "Cage à hamster 3 étages, métal, 60x40 cm",
        "Ballon de football cuir synthétique taille 5",
        "Andouillette de Troyes à griller, lot de 4",
    ]
    maxd = 0.0
    for t in samples:
        enc = tok(t, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            (l1, l2), _ = model(enc["input_ids"], enc["attention_mask"])
            l1, l2 = l1[0].numpy(), l2[0].numpy()
        o1, o2 = sess.run(
            ["logits_l1", "logits_l2"],
            {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
        )
        maxd = max(maxd, float(np.abs(l1 - o1[0]).max()), float(np.abs(l2 - o2[0]).max()))
    print(f"  écart max logits PyTorch vs ONNX : {maxd:.2e}  ->  {'PARITÉ OK' if maxd < tol else 'KO'}")
    return maxd < tol


def main():
    ap = argparse.ArgumentParser(description="Export HYDRA v2 -> ONNX (logits_l1, logits_l2).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"Export {args.ckpt} -> {args.out}")
    model, config = export(args.ckpt, args.out, args.opset)
    print(f"  variant={config['variant']} | num_l1={config['num_classes_l1']} | num_l2={config['num_classes_l2']}")
    print(f"  écrit : {os.path.getsize(args.out) / 1e6:.0f} Mo")
    ok = validate(model, args.out)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
