#!/usr/bin/env python3
"""Build the multi-parent L1->L2 matrix (157 x 5775) used for HYDRA masking.

Replicates ``build_parent_child_matrix_from_labels`` over the v2 splits: marks
every (L1, L2) edge observed in the data (multi-parent — an L2 can be a valid
child of several L1). Reuses the Flat v2 label encoders (same v2 label space, so
the indices match the model heads). Saves ``parent_child_matrix_v2.npy``
(uint8, 157x5775, ~900 KB) — needed by predict_onnx.py for the hierarchical mask.

    python3 build_hierarchy_matrix.py \\
        --ds ~/fil_rouge/dataset_v2/datasets_finaux \\
        --encoders ../../models/v2/flat_hydra \\
        --out parent_child_matrix_v2.npy
"""
import argparse
import json
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Build multi-parent L1->L2 matrix for HYDRA masking.")
    ap.add_argument("--ds", default=os.path.expanduser("~/fil_rouge/dataset_v2/datasets_finaux"))
    ap.add_argument("--encoders", default="../../models/v2/flat_hydra")
    ap.add_argument("--out", default="parent_child_matrix_v2.npy")
    args = ap.parse_args()

    with open(os.path.join(args.encoders, "l1_label_encoder.pkl"), "rb") as f:
        l1 = pickle.load(f)
    with open(os.path.join(args.encoders, "l2_label_encoder.pkl"), "rb") as f:
        l2 = pickle.load(f)
    l1i = {str(c): i for i, c in enumerate(l1.classes_)}
    l2i = {str(c): i for i, c in enumerate(l2.classes_)}
    matrix = np.zeros((len(l1.classes_), len(l2.classes_)), dtype=np.uint8)

    n = edges = 0
    for split in ("train_split.jsonl", "val_split.jsonl", "test_split.jsonl"):
        path = os.path.join(args.ds, split)
        if not os.path.exists(path):
            print(f"  (absent, ignoré) {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                n += 1
                ia = l1i.get(o.get("typelabel"))
                pt = o.get("producttype")
                key = pt[0] if isinstance(pt, list) and pt else pt
                ib = l2i.get(key)
                if ia is not None and ib is not None and matrix[ia, ib] == 0:
                    matrix[ia, ib] = 1
                    edges += 1

    np.save(args.out, matrix)
    per_l2 = matrix.sum(axis=0)
    print(f"{n:,} lignes scannées | {edges} arêtes (L1,L2) | matrice {matrix.shape} -> {args.out}")
    print(f"L2 multi-parents (>1 parent L1) : {(per_l2 > 1).sum()} / {matrix.shape[1]}")
    print(f"L2 sans parent observé : {(per_l2 == 0).sum()} (attendu petit)")


if __name__ == "__main__":
    main()
