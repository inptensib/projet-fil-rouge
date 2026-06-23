#!/usr/bin/env python3
"""
kNN / class-prototype TAIL EXPERT — third voter for the long tail.

The diagnostic showed huge accuracy-safe F1-macro headroom in the tail
(ultra-rare + rare, <100 train samples): +15.8pp oracle macro at ~flat accuracy,
and 77-99% of tail errors are "both base experts wrong" (need a NEW model).

This expert is FLAT (global) — it indexes tail-class TRAIN embeddings and predicts
by nearest class prototype, so it BYPASSES L1 routing entirely (unlike the XGBoost
branch). It works at 1-5 samples/class where gradient models can't.

Pipeline (fit on VAL, evaluate on TEST — no test leakage):
  1. Build one L2-normalized centroid per tail class from cached TRAIN embeddings.
  2. VAL: recall@1 on tail samples (embedding-quality go/no-go) + precision/coverage
     of the "is this a tail class X" detector across cosine thresholds tau.
  3. Calibrate tau at precision targets on VAL (protects accuracy — only override
     when the detector is right >= P of the time).
  4. TEST: reconstruct the v3_balanced max_conf prediction, override with the kNN
     tail prediction where cosine-sim > tau, report net accuracy / F1-macro vs the
     current standard.

CPU-only. Reuses embeddings/{split}_emb_concat.npz — NO GPU, NO retraining.

Run:
    .venv/bin/python eval_knn_tail.py            # default: L2-half embeddings, tail<100
    .venv/bin/python eval_knn_tail.py --emb concat --tail-max 100
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir",   default="embeddings")
ap.add_argument("--counts",    default="analysis/l2_subcategory_counts.json")
ap.add_argument("--base-cache",default="predictions_cache_combined_v3_balanced.pkl")
ap.add_argument("--emb",       default="l2", choices=["l2", "concat", "l1"],
                help="which slice of the 1536d concat to use for the kNN")
ap.add_argument("--tail-max",  type=int, default=100, help="tail = train count < this")
ap.add_argument("--taus", default="", help="comma list of cosine thresholds to also eval on TEST")
args = ap.parse_args()

SLICE = {"l1": (0, 768), "l2": (768, 1536), "concat": (0, 1536)}[args.emb]

def emb_slice(X):
    return np.ascontiguousarray(X[:, SLICE[0]:SLICE[1]]).astype(np.float32)

def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n

def load_npz(split):
    p = os.path.join(args.emb_dir, f"{split}_emb_concat.npz")
    if not os.path.isfile(p):
        sys.exit(f"ERROR: missing {p} (run train_xgboost_v2.py once to cache concat embeddings)")
    return np.load(p, allow_pickle=True)

# ── tail label set ─────────────────────────────────────────────────────────
counts = json.load(open(args.counts))
tail_set = {k for k, v in counts.items() if v < args.tail_max}
print(f"emb={args.emb} (dim {SLICE[1]-SLICE[0]})  |  tail = train count < {args.tail_max}: "
      f"{len(tail_set)} classes")

# ── 1) build per-class centroids over ALL classes from TRAIN (NCM) ──────────
# An absolute sim-to-nearest-tail-centroid threshold has no precision (BERT
# embeddings are anisotropic — everything is moderately similar to everything).
# Fix: build centroids for ALL classes and fire the tail override only when the
# nearest centroid OVERALL is a tail class, by a MARGIN over the nearest
# non-tail centroid. That removes the global-similarity baseline.
print("Building per-class centroids over ALL classes from TRAIN ...")
tr = load_npz("train")
y_tr = np.asarray(tr["y_l2"])
all_labels = np.array(sorted(set(y_tr.tolist())))
lab2idx = {l: i for i, l in enumerate(all_labels)}
codes = np.fromiter((lab2idx[l] for l in y_tr), dtype=np.int64, count=len(y_tr))
n_cls = len(all_labels)
D = SLICE[1] - SLICE[0]
sums = np.zeros((n_cls, D), dtype=np.float64)
cnt = np.bincount(codes, minlength=n_cls).astype(np.float64)
X_tr_full = tr["X"]
CH = 200000
for s in range(0, len(codes), CH):
    Xc = emb_slice(X_tr_full[s:s+CH]).astype(np.float64)
    np.add.at(sums, codes[s:s+CH], Xc)
P = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
P = l2norm(P)
is_tail_centroid = np.array([l in tail_set for l in all_labels])
P_tail  = np.ascontiguousarray(P[is_tail_centroid])
P_nt    = np.ascontiguousarray(P[~is_tail_centroid])
tail_labels = all_labels[is_tail_centroid]
print(f"  {n_cls} class centroids ({is_tail_centroid.sum()} tail / "
      f"{(~is_tail_centroid).sum()} non-tail) from {len(codes):,} train samples")
del X_tr_full, tr, sums

# ── NCM predict: best tail label, and margin = best_tail_sim - best_nontail_sim
def ncm_predict(X):
    Xn = l2norm(emb_slice(X))
    pred = np.empty(len(Xn), dtype=object)
    marg = np.empty(len(Xn), dtype=np.float32)
    CH = 40000
    for s in range(0, len(Xn), CH):
        chunk = Xn[s:s+CH]
        st = chunk @ P_tail.T            # (c, n_tail)
        snt = chunk @ P_nt.T             # (c, n_nontail)
        at = st.argmax(1)
        best_tail = st[np.arange(len(at)), at]
        best_nt = snt.max(1)
        pred[s:s+CH] = tail_labels[at]
        marg[s:s+CH] = best_tail - best_nt   # >0 => nearest overall is a tail class
    return pred, marg

# ── 2) VAL: recall@1 on tail + precision/coverage by MARGIN ────────────────
print("\nLoading VAL ...")
va = load_npz("val")
yv = np.asarray(va["y_l2"])
v_pred, v_marg = ncm_predict(va["X"])
del va
v_is_tail = np.isin(yv, list(tail_set))
n_tail_v = int(v_is_tail.sum())
recall1 = float((v_pred[v_is_tail] == yv[v_is_tail]).mean()) if n_tail_v else 0.0
print(f"  VAL tail samples: {n_tail_v:,}")
print(f"  VAL recall@1 on tail (nearest tail centroid == true class): {recall1:.3f}")

print("\nVAL tail detector by MARGIN tau (best_tail_sim - best_nontail_sim):")
print(f"{'tau':>6}{'fired':>10}{'strict-prec':>13}{'tail-prec':>11}{'tail-recall':>13}")
grid = [round(x, 3) for x in np.arange(-0.05, 0.31, 0.025)]
cal = []
for tau in grid:
    fire = v_marg > tau
    nf = int(fire.sum())
    if nf == 0:
        print(f"{tau:>6.3f}{0:>10}"); continue
    strict = float((v_pred[fire] == yv[fire]).mean())        # exact tail class right
    tprec  = float(v_is_tail[fire].mean())                   # fired sample is really tail
    trec   = float((fire & v_is_tail & (v_pred == yv)).sum() / max(n_tail_v, 1))
    cal.append((tau, strict, nf))
    print(f"{tau:>6.3f}{nf:>10,}{strict:>12.3f}{tprec:>11.3f}{trec:>13.3f}")

def tau_for(pmin):
    ok = [t for t, s, nf in cal if s >= pmin]
    return min(ok) if ok else None
targets = [0.4, 0.5, 0.6, 0.7]
chosen = {p: tau_for(p) for p in targets}
extra = [float(x) for x in args.taus.split(",") if x.strip()] if args.taus else []
print("\nCalibrated margin tau (lowest tau meeting VAL strict-precision target):")
for p in targets:
    print(f"  precision>={p}: tau={chosen[p]}")

# ── 4) TEST: net effect with the kNN override ──────────────────────────────
print("\nLoading TEST base cache + embeddings ...")
if not os.path.isfile(args.base_cache):
    sys.exit(f"ERROR: base cache {args.base_cache} not found")
base = pickle.load(open(args.base_cache, "rb"))
cam_p = np.asarray(base["cam_preds"]); cam_c = np.asarray(base["cam_confs"], np.float32)
xgb_p = np.asarray(base["xgb_preds"]); xgb_c = np.asarray(base["xgb_confs"], np.float32)
true  = np.asarray(base["true_labels"])
has   = xgb_c > 0.0
final = cam_p.copy()
ov = has & (xgb_c > cam_c)
final[ov] = xgb_p[ov]
base_acc = accuracy_score(true, final)
labels_all = np.array(sorted(set(true) | set(final)))
_, _, f1c, _ = precision_recall_fscore_support(true, final, labels=labels_all, average=None, zero_division=0)
base_mac = f1c.mean()
print(f"  current standard: acc={base_acc:.4f}  f1_macro={base_mac:.4f}")

te = load_npz("test")
t_pred, t_marg = ncm_predict(te["X"]); del te

# always also scan a margin grid so we see the full operating curve on TEST
scan = [round(x, 3) for x in np.arange(0.0, 0.31, 0.025)]
run_taus = sorted(set(scan) | {chosen[p] for p in targets if chosen[p] is not None} | set(extra))
print("\n" + "=" * 90)
print("TEST: override base prediction with NCM tail-pred where margin > tau")
print("  (margin = sim to nearest tail centroid - sim to nearest non-tail centroid)")
print("=" * 90)
print(f"{'tau':>7}{'#override':>11}{'ovr-prec':>10}{'tail-prec':>11}{'acc':>9}{'Δacc':>9}{'f1_macro':>10}{'Δmac':>9}")
print("-" * 90)
true_is_tail = np.isin(true, list(tail_set))
for tau in run_taus:
    fire = t_marg > tau
    new = final.copy()
    new[fire] = t_pred[fire]
    nov = int(fire.sum())
    ovr_prec  = float((t_pred[fire] == true[fire]).mean()) if nov else 0.0
    tailprec  = float(true_is_tail[fire].mean()) if nov else 0.0
    a = accuracy_score(true, new)
    _, _, f1c2, _ = precision_recall_fscore_support(true, new, labels=labels_all, average=None, zero_division=0)
    m = f1c2.mean()
    star = "  <-- VAL-cal" if tau in {chosen[p] for p in targets if chosen[p] is not None} else ""
    print(f"{tau:>7.3f}{nov:>11,}{ovr_prec:>10.3f}{tailprec:>11.3f}{a:>9.4f}{a-base_acc:>+9.4f}"
          f"{m:>10.4f}{m-base_mac:>+9.4f}{star}")

print("\nGuide: pick the tau with the best Δf1_macro while Δacc stays >= 0 (or barely negative).")
print("That operating point = the kNN tail expert folded into max_conf as a tail-only third voter.")
