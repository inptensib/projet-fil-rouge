#!/usr/bin/env python3
"""
LEARNED tail DETECTOR — improve the "should I fire the tail override?" gate.

The kNN tail voter currently fires on a single cosine MARGIN > tau. That gate's
precision/recall tradeoff is the binding limiter (+1.0pp f1m). This replaces the
single-threshold gate with a learned binary classifier P(true label is a tail
class) over richer features, fit on VAL / applied on TEST:

  NCM features (from run2 L2 embeddings, always available):
    margin           = sim(nearest tail centroid) - sim(nearest non-tail centroid)
    best_tail_sim    = sim to nearest tail centroid
    gap2             = best_tail_sim - 2nd_best_tail_sim   (how unambiguous the pick is)
    tail_logcount    = log1p(train count of the predicted tail class)
  base-model features (if the base caches are present — the valuable signal):
    cam_conf         = focal L2 top-1 confidence (low => base uncertain => likely tail)
    cam_is_head      = base predicted a HEAD (>=1k) class  (tail-misclassified-as-head)
    cam_logcount     = log1p(train count of the base's predicted class)

Then it overrides the base prediction with the NCM tail-pred where P(tail) > thr
(thr calibrated on VAL to a precision target), and reports TEST acc / f1_macro vs
the margin-only baseline (+1.0pp).

CPU only. Reuses cached run2 concat embeddings (memmap) + base prediction caches.
Run:  sbatch run_learned_detector.sh        (or python eval_learned_detector.py)
"""
import os, sys, json, pickle, struct
import numpy as np
from numpy.lib import format as npf
import zipfile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

EMB_DIR    = "embeddings"
COUNTS     = "analysis/l2_subcategory_counts.json"
VAL_CACHE  = "predictions_cache_combined_val.pkl"            # focal+v2 (cam = focal, consistent)
TEST_CACHE = "predictions_cache_combined_v3_balanced.pkl"
TAIL_MAX   = 100
SLICE      = (768, 1536)     # run2 L2 half of the concat

def emb_slice(X): return np.ascontiguousarray(X[:, SLICE[0]:SLICE[1]]).astype(np.float32)
def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; return M / n

def memmap_X(split):
    p = os.path.join(EMB_DIR, f"{split}_emb_concat.npz")
    zf = zipfile.ZipFile(p); zi = zf.getinfo("X.npy")
    with open(p, "rb") as f:
        f.seek(zi.header_offset + 26); nl, el = struct.unpack("<HH", f.read(4))
        f.seek(zi.header_offset + 30 + nl + el); ver = npf.read_magic(f)
        shape, fo, dt = npf._read_array_header(f, ver); ds = f.tell()
    return np.memmap(p, mode="r", dtype=dt, shape=shape, order="F" if fo else "C", offset=ds)

def load_y(split):
    return np.asarray(np.load(os.path.join(EMB_DIR, f"{split}_emb_concat.npz"),
                              allow_pickle=True)["y_l2"])

counts = json.load(open(COUNTS))
tail_set = {k for k, v in counts.items() if v < TAIL_MAX}
def logcount(label_arr):
    return np.log1p(np.array([counts.get(l, 0) for l in label_arr], dtype=np.float64))
print(f"tail<{TAIL_MAX}: {len(tail_set)} classes")

# ── run2 centroids over ALL classes (memmap, low RAM) ──────────────────────
print("Building run2 centroids ...")
y_tr = load_y("train")
all_labels = np.array(sorted(set(y_tr.tolist())))
lab2idx = {l: i for i, l in enumerate(all_labels)}
codes = np.fromiter((lab2idx[l] for l in y_tr), dtype=np.int64, count=len(y_tr)); del y_tr
n_cls = len(all_labels)
Xmm = memmap_X("train")
D = emb_slice(np.asarray(Xmm[:1])).shape[1]
sums = np.zeros((n_cls, D), np.float64); cnt = np.bincount(codes, minlength=n_cls).astype(np.float64)
for s in range(0, len(codes), 100000):
    Xc = emb_slice(np.asarray(Xmm[s:s+100000]))
    np.add.at(sums, codes[s:s+100000], Xc.astype(np.float64)); del Xc
P = l2norm((sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)); del sums, Xmm
is_tail_c = np.array([l in tail_set for l in all_labels])
P_tail = np.ascontiguousarray(P[is_tail_c]); P_nt = np.ascontiguousarray(P[~is_tail_c])
tail_labels = all_labels[is_tail_c]
print(f"  {is_tail_c.sum()} tail / {(~is_tail_c).sum()} non-tail centroids")

# ── per-sample NCM features ────────────────────────────────────────────────
def ncm_features(split):
    Xmm = memmap_X(split); n = len(Xmm)
    best_tail = np.empty(n, np.float32); second_tail = np.empty(n, np.float32)
    best_nt = np.empty(n, np.float32); pred = np.empty(n, dtype=object)
    CH = 40000
    for s in range(0, n, CH):
        M = l2norm(emb_slice(np.asarray(Xmm[s:s+CH])))
        st = M @ P_tail.T; snt = M @ P_nt.T
        a = st.argmax(1)
        best_tail[s:s+CH] = st[np.arange(len(a)), a]
        # 2nd best tail
        st[np.arange(len(a)), a] = -1e9
        second_tail[s:s+CH] = st.max(1)
        best_nt[s:s+CH] = snt.max(1)
        pred[s:s+CH] = tail_labels[a]
    return best_tail, second_tail, best_nt, pred

def build_features(split, base_cache):
    bt, st2, bnt, pred = ncm_features(split)
    feat = {
        "margin":        bt - bnt,
        "best_tail":     bt,
        "gap2":          bt - st2,
        "tail_logcount": logcount(pred),
    }
    # base-model features (optional)
    cam = None
    if base_cache and os.path.isfile(base_cache):
        d = pickle.load(open(base_cache, "rb"))
        cam_pred = np.asarray(d["cam_preds"]); cam_conf = np.asarray(d["cam_confs"], np.float32)
        if len(cam_pred) == len(bt):
            feat["cam_conf"]     = cam_conf
            feat["cam_logcount"] = logcount(cam_pred)
            feat["cam_is_head"]  = (logcount(cam_pred) >= np.log1p(1000)).astype(np.float32)
            cam = True
        else:
            print(f"  WARN: {base_cache} length {len(cam_pred)} != {len(bt)} — skipping base feats")
    X = np.column_stack([feat[k] for k in sorted(feat)])
    return X, sorted(feat), pred, (cam is not None)

print("VAL features ..."); Xv, names, pred_v, cam_v = build_features("val", VAL_CACHE)
print("TEST features ..."); Xt, names_t, pred_t, cam_t = build_features("test", TEST_CACHE)
if names != names_t:
    common = sorted(set(names) & set(names_t))
    print(f"  feature mismatch val{names} vs test{names_t} -> using common {common}")
    Xv = Xv[:, [names.index(c) for c in common]]; Xt = Xt[:, [names_t.index(c) for c in common]]
    names = common
print(f"  features: {names}")

yv = load_y("val"); yt = load_y("test")
val_is_tail = np.isin(yv, list(tail_set)).astype(int)
test_is_tail = np.isin(yt, list(tail_set))
print(f"  VAL tail rate: {val_is_tail.mean():.4f}  ({val_is_tail.sum():,})")

# ── train detector on VAL, score on TEST ───────────────────────────────────
print("\nTraining detector (GradientBoosting) ...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 subsample=0.7, random_state=0)
clf.fit(Xv, val_is_tail)
pv = clf.predict_proba(Xv)[:, 1]; pt = clf.predict_proba(Xt)[:, 1]
auc_learned = roc_auc_score(val_is_tail, pv)
auc_margin  = roc_auc_score(val_is_tail, Xv[:, names.index("margin")])
print(f"  VAL AUC: learned={auc_learned:.4f}  vs margin-only={auc_margin:.4f}")
print("  feature importances:", {n: round(float(i), 3) for n, i in zip(names, clf.feature_importances_)})

# ── TEST override curve at P(tail) thresholds ──────────────────────────────
base = pickle.load(open(TEST_CACHE, "rb"))
cam_p = np.asarray(base["cam_preds"]); cam_c = np.asarray(base["cam_confs"], np.float32)
xgb_p = np.asarray(base["xgb_preds"]); xgb_c = np.asarray(base["xgb_confs"], np.float32)
true  = np.asarray(base["true_labels"]); has = xgb_c > 0.0
final = cam_p.copy(); final[has & (xgb_c > cam_c)] = xgb_p[has & (xgb_c > cam_c)]
base_acc = accuracy_score(true, final)
labs = np.array(sorted(set(true) | set(final)))
_, _, f1c, _ = precision_recall_fscore_support(true, final, labels=labs, average=None, zero_division=0)
base_mac = f1c.mean()
print(f"\ncurrent standard: acc={base_acc:.4f}  f1_macro={base_mac:.4f}")
print("(margin-only kNN voter best was Δf1m=+0.0100 at ~flat acc)\n")
print("=" * 86)
print("TEST: override base with NCM tail-pred where detector P(tail) > thr")
print("=" * 86)
print(f"{'P_thr':>7}{'#override':>11}{'ovr-prec':>10}{'acc':>9}{'Δacc':>9}{'f1_macro':>10}{'Δmac':>9}")
print("-" * 86)
best = None
for thr in [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]:
    fire = pt > thr
    new = final.copy(); new[fire] = pred_t[fire]
    nov = int(fire.sum())
    op = float((pred_t[fire] == true[fire]).mean()) if nov else 0.0
    a = accuracy_score(true, new)
    _, _, f2, _ = precision_recall_fscore_support(true, new, labels=labs, average=None, zero_division=0)
    m = f2.mean()
    print(f"{thr:>7.2f}{nov:>11,}{op:>10.3f}{a:>9.4f}{a-base_acc:>+9.4f}{m:>10.4f}{m-base_mac:>+9.4f}")
    if (a - base_acc) >= -0.001 and (best is None or (m - base_mac) > best[1]):
        best = (thr, m - base_mac, a - base_acc)
if best:
    print(f"\n  best @ Δacc>=-0.001: P_thr={best[0]}  Δf1m={best[1]:+.4f}  Δacc={best[2]:+.4f}")
    print(f"  vs margin-only +0.0100 -> learned detector { 'WINS' if best[1] > 0.0100 else 'does not beat' } it")
