#!/usr/bin/env python3
"""
Contrastive PROJECTION HEAD on frozen run2 embeddings (improvement #5, safe form).

The kNN tail voter is capped by run2's recall@1 (0.704). Whitening (linear) didn't
help; SetFit-from-MPNet hurt (weak base + forgetting). This learns a NON-linear
metric on the *frozen* run2 space — a small MLP trained with supervised-contrastive
(SupCon) loss to pull same-tail-class embeddings together and push classes apart.
Because the encoder is frozen (we only have its cached vectors), it CANNOT degrade
the base — it can only add a learned transform. Directly tests "can a learned metric
on run2 beat raw run2?".

Trains on cached run2 L2 embeddings (no transformer forward passes -> fast), then
projects train(centroid pool)/val/test and saves to embeddings_proj/{split}_emb_proj.npz
in the layout eval_knn_tail_whiten.py reads.

Run:  sbatch run_proj_head.sh
Eval: python eval_knn_tail_whiten.py --emb-dir embeddings_proj --emb-name emb_proj --emb full --configs none
"""
import os, sys, json, struct, zipfile, time
import numpy as np
from numpy.lib import format as npf
import torch, torch.nn as nn, torch.nn.functional as F

EMB_DIR   = "embeddings"
OUT_DIR   = "embeddings_proj"
COUNTS    = "analysis/l2_subcategory_counts.json"
TAIL_MAX  = 100
SLICE     = (768, 1536)                 # run2 L2 half of concat
NONTAIL_PER_CLASS = 20                  # non-tail samples/class in the training pool (negatives)
PROJ_DIM  = 768
HID       = 1024
EPOCHS    = 30
P_CLASSES = 64                          # classes per batch
K_SAMPLES = 4                           # samples per class per batch  (batch = P*K)
STEPS_PER_EPOCH = 400
TEMP      = 0.1
LR        = 1e-3
DEBUG = "--debug" in sys.argv
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

def emb_slice(X): return np.ascontiguousarray(X[:, SLICE[0]:SLICE[1]]).astype(np.float32)
def memmap_X(split):
    p = os.path.join(EMB_DIR, f"{split}_emb_concat.npz")
    zf = zipfile.ZipFile(p); zi = zf.getinfo("X.npy")
    with open(p, "rb") as f:
        f.seek(zi.header_offset+26); nl,el = struct.unpack("<HH", f.read(4))
        f.seek(zi.header_offset+30+nl+el); ver = npf.read_magic(f)
        shape, fo, dt = npf._read_array_header(f, ver); ds = f.tell()
    return np.memmap(p, mode="r", dtype=dt, shape=shape, order="F" if fo else "C", offset=ds)
def load_y(split):
    return np.asarray(np.load(os.path.join(EMB_DIR, f"{split}_emb_concat.npz"), allow_pickle=True)["y_l2"])

counts = json.load(open(COUNTS))
tail_set = {k for k, v in counts.items() if v < TAIL_MAX}
print(f"tail<{TAIL_MAX}: {len(tail_set)} classes | device={dev}")

# ── gather training embeddings: ALL tail + capped non-tail (from cached run2) ─
print("Gathering run2 training embeddings (memmap) ...")
y_tr = load_y("train"); Xmm = memmap_X("train")
keep_idx = []; per_nt = {}
for i, lab in enumerate(y_tr):
    if lab in tail_set:
        keep_idx.append(i)
    elif per_nt.get(lab, 0) < NONTAIL_PER_CLASS:
        keep_idx.append(i); per_nt[lab] = per_nt.get(lab, 0) + 1
keep_idx = np.array(keep_idx)
if DEBUG: keep_idx = keep_idx[:20000]
X = emb_slice(np.asarray(Xmm[keep_idx]))          # (M, 768) float32
ylab = y_tr[keep_idx]
del Xmm, y_tr
# integer codes; only classes with >=2 samples can anchor (need positives)
uniq, y = np.unique(ylab, return_inverse=True)
print(f"  {len(X):,} train embeddings, {len(uniq)} classes")
Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
Xt = torch.from_numpy(Xn).float()

# per-class index lists (for balanced sampling); usable = classes with >=2 samples
from collections import defaultdict
cls_idx = defaultdict(list)
for i, c in enumerate(y): cls_idx[c].append(i)
usable = [c for c, ix in cls_idx.items() if len(ix) >= 2]
print(f"  usable anchor classes (>=2 samples): {len(usable)}")

# ── model: MLP projection head ──────────────────────────────────────────────
class Proj(nn.Module):
    def __init__(self, d=768, h=HID, o=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
model = Proj().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def supcon(z, labels, temp=TEMP):
    # z: (B, d) normalized; labels: (B,)
    sim = z @ z.T / temp
    B = z.size(0)
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, -1e9)
    logits = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(logits)
    denom = exp.sum(1)
    pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos = pos & ~self_mask
    log_prob = logits - torch.log(denom.unsqueeze(1) + 1e-12)
    ppos = (log_prob * pos).sum(1)
    npos = pos.sum(1).clamp(min=1)
    has = pos.sum(1) > 0
    return -(ppos / npos)[has].mean() if has.any() else torch.tensor(0.0, device=z.device)

def sample_batch():
    cls = np.random.choice(usable, size=min(P_CLASSES, len(usable)), replace=False)
    idx = []
    for c in cls:
        pool = cls_idx[c]
        idx += list(np.random.choice(pool, size=K_SAMPLES, replace=len(pool) < K_SAMPLES))
    idx = np.array(idx)
    return Xt[idx].to(dev), torch.from_numpy(y[idx]).to(dev)

print("Training SupCon projection head ...")
t0 = time.time()
spe = 50 if DEBUG else STEPS_PER_EPOCH
for ep in range(2 if DEBUG else EPOCHS):
    model.train(); tot = 0.0
    for _ in range(spe):
        xb, yb = sample_batch()
        z = model(xb); loss = supcon(z, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"  epoch {ep+1}: loss={tot/spe:.4f}  ({(time.time()-t0)/60:.1f} min)")

# ── project + save train(centroid pool)/val/test ────────────────────────────
model.eval()
@torch.no_grad()
def project_np(Xnp):
    out = np.zeros((len(Xnp), PROJ_DIM), dtype=np.float16)
    for s in range(0, len(Xnp), 50000):
        xb = torch.from_numpy(Xnp[s:s+50000]).float().to(dev)
        xb = F.normalize(xb, dim=1)
        out[s:s+50000] = model(xb).cpu().numpy().astype(np.float16)
    return out

# train: project the gathered tail+nontail pool (for centroids), keep ALL tail
def save_split(split):
    if split == "train":
        Xs = X; Y2 = ylab; Y1 = np.array([""]*len(ylab))
    else:
        mm = memmap_X(split); Xs = emb_slice(np.asarray(mm)); Y2 = load_y(split); Y1 = np.array([""]*len(Y2)); del mm
    Z = project_np(Xs)
    out = os.path.join(OUT_DIR, f"{split}_emb_proj.npz"); tmp = out + ".tmp.npz"
    np.savez(tmp, X=Z, y_l1=Y1, y_l2=Y2); os.replace(tmp, out)
    print(f"  saved {out}  shape={Z.shape}")

for split in ["train", "val", "test"]:
    print(f"Projecting {split} ..."); save_split(split)

print("Done. Eval:")
print("  python eval_knn_tail_whiten.py --emb-dir embeddings_proj --emb-name emb_proj --emb full --configs none")
