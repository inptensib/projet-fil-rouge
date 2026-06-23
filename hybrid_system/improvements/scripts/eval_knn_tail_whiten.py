#!/usr/bin/env python3
"""
kNN/NCM tail expert + EMBEDDING WHITENING sweep.

The base NCM tail voter (eval_knn_tail.py) gave +1pp F1-macro, capped by the
frozen embedding space: VAL recall@1 on tail = 0.704, and the detector can only
fire at high precision on ~30% of tail samples (BERT embeddings are anisotropic
— everything is moderately similar, which is exactly why a MARGIN was needed).

Whitening directly attacks that anisotropy:
  - center : subtract the global mean embedding
  - pcaN   : center, then remove the top-N principal components ("all-but-the-top",
             Mu & Viswanath 2017) — the top PCs carry the shared/anisotropic
             directions that wash out class separation.

Whitening is LINEAR, so it commutes with the class mean: whitened centroid =
W(raw centroid). We build raw centroids ONCE (one train pass) and just re-project
them per config — cheap. Then we evaluate every config on VAL (recall@1 +
calibrated operating point) and run the full TEST override curve for the baseline
plus the best whitening config.

CPU only. Reuses embeddings/{split}_emb_concat.npz. No GPU, no training.

Run:
    .venv/bin/python eval_knn_tail_whiten.py
    # optional: --configs none,center,pca1,pca5,pca10  --emb l2  --tail-max 100
"""
import os, sys, json, pickle, argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir",    default="embeddings")
ap.add_argument("--counts",     default="analysis/l2_subcategory_counts.json")
ap.add_argument("--base-cache", default="predictions_cache_combined_v3_balanced.pkl")
ap.add_argument("--emb",        default="l2", choices=["l2", "concat", "l1", "full"],
                help="which slice of the cached vector to use ('full' = whole vector, "
                     "e.g. 768d SetFit embeddings)")
ap.add_argument("--emb-name",   default="emb_concat",
                help="npz suffix: file = {split}_{emb_name}.npz (use emb_setfit for SetFit)")
ap.add_argument("--tail-max",   type=int, default=100)
ap.add_argument("--configs",    default="none,center,pca1,pca5,pca10")
ap.add_argument("--pca-sample", type=int, default=200000, help="train rows for PCA fit")
args = ap.parse_args()

import zipfile, struct
from numpy.lib import format as npf

SLICE = {"l1": (0, 768), "l2": (768, 1536), "concat": (0, 1536),
         "full": (0, 1 << 30)}[args.emb]
def emb_slice(X): return np.ascontiguousarray(X[:, SLICE[0]:SLICE[1]]).astype(np.float32)
def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; return M / n

def _npz_path(split):
    p = os.path.join(args.emb_dir, f"{split}_{args.emb_name}.npz")
    if not os.path.isfile(p): sys.exit(f"ERROR: missing {p}")
    return p

def memmap_X(split):
    """Memory-map the (uncompressed) X member of {split}_emb_concat.npz so the
    17GB train file is NOT loaded into RAM. Falls back to a normal load if the
    member happens to be compressed."""
    p = _npz_path(split)
    zf = zipfile.ZipFile(p, "r"); zi = zf.getinfo("X.npy")
    if zi.compress_type != zipfile.ZIP_STORED:
        return np.load(p)["X"]                       # fallback (compressed)
    with open(p, "rb") as f:
        f.seek(zi.header_offset + 26)
        namelen, extralen = struct.unpack("<HH", f.read(4))
        f.seek(zi.header_offset + 30 + namelen + extralen)
        ver = npf.read_magic(f)
        shape, fortran, dtype = npf._read_array_header(f, ver)
        data_start = f.tell()
    return np.memmap(p, mode="r", dtype=dtype, shape=shape,
                     order="F" if fortran else "C", offset=data_start)

def load_y(split):
    # load only the small label member (does not touch the 17GB X)
    return np.asarray(np.load(_npz_path(split), allow_pickle=True)["y_l2"])

counts = json.load(open(args.counts))
tail_set = {k for k, v in counts.items() if v < args.tail_max}
print(f"emb={args.emb} (dim {SLICE[1]-SLICE[0]})  tail<{args.tail_max}: {len(tail_set)} classes")

# ── raw per-class centroids + global mean + PCA basis (one train pass) ──────
print("Train pass: raw centroids + PCA basis (memmap, low RAM) ...")
y_tr = load_y("train")
all_labels = np.array(sorted(set(y_tr.tolist())))
lab2idx = {l: i for i, l in enumerate(all_labels)}
codes = np.fromiter((lab2idx[l] for l in y_tr), dtype=np.int64, count=len(y_tr))
del y_tr
n_cls = len(all_labels)
X_mm = memmap_X("train")                                 # NOT loaded into RAM
D = emb_slice(np.asarray(X_mm[:1])).shape[1]             # real dim (handles 'full')
sums = np.zeros((n_cls, D), dtype=np.float64)            # accumulator (small)
cnt  = np.bincount(codes, minlength=n_cls).astype(np.float64)
N = len(codes)
stride = max(1, N // args.pca_sample)
pca_rows = []
CH = 100000                                              # smaller chunk -> lower peak RAM
for s in range(0, N, CH):
    Xc = emb_slice(np.asarray(X_mm[s:s+CH]))             # float32 chunk only
    np.add.at(sums, codes[s:s+CH], Xc.astype(np.float64))
    pca_rows.append(Xc[::stride].copy())                 # float32 strided sample
    del Xc
raw_cent = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)   # n_cls x D
mu = (sums.sum(0) / N).astype(np.float32)                            # global mean
Xs = np.vstack(pca_rows).astype(np.float32); del pca_rows, X_mm
print(f"  {n_cls} centroids; PCA sample {Xs.shape[0]:,}")
# top-K principal components of centered sample
Xc = Xs - mu
# economy SVD via covariance eigen (D=768 small)
cov = (Xc.T @ Xc) / len(Xc)
evals, evecs = np.linalg.eigh(cov)            # ascending
PCs = evecs[:, ::-1].T.astype(np.float32)     # rows = components, descending
del Xs, Xc, cov, sums
is_tail_centroid = np.array([l in tail_set for l in all_labels])

# ── whitening transform (linear) applied to any matrix of rows ─────────────
def make_W(cfg):
    cfg = cfg.strip()
    if cfg == "none":  return ("none", 0)
    if cfg == "center":return ("center", 0)
    if cfg.startswith("pca"): return ("pca", int(cfg[3:]))
    sys.exit(f"unknown config {cfg}")

def whiten(M, kind, k):
    if kind == "none": return l2norm(M.astype(np.float32))
    Mc = M.astype(np.float32) - mu
    if kind == "pca" and k > 0:
        B = PCs[:k]                       # k x D
        Mc = Mc - (Mc @ B.T) @ B          # remove top-k PCs
    return l2norm(Mc)

# precompute whitened centroids per config (cheap: n_cls x D)
configs = [c for c in args.configs.split(",") if c.strip()]
WC = {}
for c in configs:
    kind, k = make_W(c)
    Pw = whiten(raw_cent, kind, k)
    WC[c] = (Pw[is_tail_centroid].copy(), Pw[~is_tail_centroid].copy(), kind, k)
tail_labels = all_labels[is_tail_centroid]

def ncm(Xmm, c):
    """Xmm = full-concat memmap; slices + whitens per chunk (low RAM)."""
    Pt, Pnt, kind, k = WC[c]
    n = len(Xmm)
    pred = np.empty(n, dtype=object); marg = np.empty(n, np.float32)
    B = PCs[:k] if kind == "pca" else None
    CHq = 40000
    for s in range(0, n, CHq):
        M = emb_slice(np.asarray(Xmm[s:s+CHq]))     # (chunk, D) float32, from disk
        if kind != "none":
            M = M - mu
            if B is not None: M = M - (M @ B.T) @ B
        M = l2norm(M)
        st = M @ Pt.T; snt = M @ Pnt.T
        at = st.argmax(1)
        pred[s:s+CHq] = tail_labels[at]
        marg[s:s+CHq] = st[np.arange(len(at)), at] - snt.max(1)
    return pred, marg

# ── VAL: per-config recall@1 + operating point at strict-precision>=0.5 ────
print("\nLoading VAL (memmap) ...")
Xv = memmap_X("val"); yv = load_y("val")
v_is_tail = np.isin(yv, list(tail_set)); n_tail_v = int(v_is_tail.sum())
grid = [round(x, 3) for x in np.arange(-0.05, 0.41, 0.025)]

print(f"\n{'config':<10}{'recall@1':>10}{'tau@P.5':>10}{'tail-rec@P.5':>14}")
print("-" * 46)
val_pick = {}
for c in configs:
    vp, vm = ncm(Xv, c)
    r1 = float((vp[v_is_tail] == yv[v_is_tail]).mean())
    tau5, rec5 = None, 0.0
    for tau in grid:
        fire = vm > tau; nf = int(fire.sum())
        if nf == 0: continue
        strict = float((vp[fire] == yv[fire]).mean())
        if strict >= 0.5:
            tau5 = tau
            rec5 = float((fire & v_is_tail & (vp == yv)).sum() / max(n_tail_v, 1))
            break
    val_pick[c] = (r1, tau5, rec5)
    print(f"{c:<10}{r1:>10.3f}{str(tau5):>10}{rec5:>14.3f}")
del Xv

# rank configs: higher recall@1 (tie-break tail-recall at P>=0.5)
ranked = sorted(configs, key=lambda c: (val_pick[c][0], val_pick[c][2]), reverse=True)
best = ranked[0]
test_configs = list(dict.fromkeys(["none", best]))   # baseline + best
print(f"\nBest whitening config by VAL: {best}  -> running TEST for: {test_configs}")

# ── TEST: override curve for baseline + best config ────────────────────────
base = pickle.load(open(args.base_cache, "rb"))
cam_p = np.asarray(base["cam_preds"]); cam_c = np.asarray(base["cam_confs"], np.float32)
xgb_p = np.asarray(base["xgb_preds"]); xgb_c = np.asarray(base["xgb_confs"], np.float32)
true  = np.asarray(base["true_labels"]); has = xgb_c > 0.0
final = cam_p.copy(); final[has & (xgb_c > cam_c)] = xgb_p[has & (xgb_c > cam_c)]
base_acc = accuracy_score(true, final)
labels_all = np.array(sorted(set(true) | set(final)))
_, _, f1c, _ = precision_recall_fscore_support(true, final, labels=labels_all, average=None, zero_division=0)
base_mac = f1c.mean()
print(f"\ncurrent standard: acc={base_acc:.4f}  f1_macro={base_mac:.4f}")

Xt = memmap_X("test")                              # memmap, not loaded into RAM
true_is_tail = np.isin(true, list(tail_set))
scan = [round(x, 3) for x in np.arange(0.0, 0.41, 0.025)]
for c in test_configs:
    tp, tm = ncm(Xt, c)
    print("\n" + "=" * 86)
    print(f"TEST override curve — whitening='{c}'  (recall@1={val_pick[c][0]:.3f})")
    print("=" * 86)
    print(f"{'tau':>7}{'#override':>11}{'ovr-prec':>10}{'acc':>9}{'Δacc':>9}{'f1_macro':>10}{'Δmac':>9}")
    print("-" * 86)
    bestrow = None
    for tau in scan:
        fire = tm > tau
        new = final.copy(); new[fire] = tp[fire]
        nov = int(fire.sum())
        op = float((tp[fire] == true[fire]).mean()) if nov else 0.0
        a = accuracy_score(true, new)
        _, _, f2, _ = precision_recall_fscore_support(true, new, labels=labels_all, average=None, zero_division=0)
        m = f2.mean()
        print(f"{tau:>7.3f}{nov:>11,}{op:>10.3f}{a:>9.4f}{a-base_acc:>+9.4f}{m:>10.4f}{m-base_mac:>+9.4f}")
        if (a - base_acc) >= -0.001 and (bestrow is None or (m - base_mac) > bestrow[1]):
            bestrow = (tau, m - base_mac, a - base_acc)
    if bestrow:
        print(f"  best @ Δacc>=-0.001: tau={bestrow[0]}  Δf1m={bestrow[1]:+.4f}  Δacc={bestrow[2]:+.4f}")

print("\nCompare the best-row Δf1m for 'none' vs the whitened config — that is the whitening gain.")
