#!/usr/bin/env python3
"""
FLAT SetFit tail expert — one contrastive model over the tail label space.

Unlike the old per-L1 SetFit (27 gated models, rejected), this trains ONE model:
  - flat: no L1 routing — embeds text, the tail voter finds the nearest tail
    centroid globally (drop-in replacement for the kNN voter's embedding source)
  - target: tail subcats only (train count < TAIL_MAX), each a class
  - trained WITH a non-tail negative pool so the tail-vs-non-tail margin (our
    detector's limiter) is sharp

After training it EXTRACTS + caches embeddings (the SetFit contrastive body) for
tail-train / a non-tail subsample / val / test, in the same .npz layout the eval
already reads (X, y_l1, y_l2) — so eval_knn_tail_whiten.py runs unchanged with
  --emb-dir embeddings_setfit --emb-name emb_setfit --emb full

One GPU job does train + extract. Output:
  checkpoints/setfit_flat/body/         (SentenceTransformer body, ~1GB)
  embeddings_setfit/{train,val,test}_emb_setfit.npz   (float16, disk-light)

Run:  sbatch run_setfit_flat.sh        (GPU)
      python train_setfit_flat.py --debug   (tiny, CPU/GPU smoke test)
"""
import os, sys, json, time, gc, unicodedata
import numpy as np
import torch

# ── setfit compatibility monkey-patches (same as train_setfit.py) ──────────
import huggingface_hub
if not hasattr(huggingface_hub, "DatasetFilter"):
    huggingface_hub.DatasetFilter = None
import setfit.model_card as _mc
_orig = getattr(_mc.SetFitModelCardData, "set_widget_examples", None)
if _orig:
    def _patched(self, dataset):
        try: _orig(self, dataset)
        except (AttributeError, TypeError): pass
    _mc.SetFitModelCardData.set_widget_examples = _patched

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset, load_dataset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEBUG = "--debug" in sys.argv

# ── config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
BODY_DIR = os.path.join(BASE_DIR, "checkpoints/setfit_flat/body")
EMB_DIR  = os.path.join(BASE_DIR, "embeddings_setfit")
os.makedirs(os.path.dirname(BODY_DIR), exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

ST_MODEL          = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TAIL_MAX          = 100      # tail = train count < this
MAX_PER_CLASS     = 12       # few-shot cap per tail class (training) — SetFit needs only 8-16
NONTAIL_POOL      = 4000     # non-tail train samples mixed in as negatives
NONTAIL_PER_CLASS = 4        # cap per non-tail class in the pool
NUM_ITERATIONS    = 10       # contrastive pair sampling (pairs = 2*iter*n_samples; keep RAM sane)
NUM_EPOCHS_BODY   = 1
BATCH_SIZE        = 32
ENCODE_BATCH      = 256
NONTAIL_CENTROID_CAP = 80    # non-tail train samples/class kept for margin centroids
MAXLEN = 512

def get_text(ex):
    t = ex.get("designation", "") or ""; d = ex.get("description", "") or ""
    return (f"{t} {d}"[:MAXLEN]) if d else t

def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

# ── tail label set + canonical name map (merge encoding dupes) ─────────────
counts = json.load(open(os.path.join(ANALYSIS_DIR, "l2_subcategory_counts.json")))
tail_set = {k for k, v in counts.items() if v < TAIL_MAX}
canon = {}
for name in counts:
    canon.setdefault(normalize(name), name)
def canonical(sub): return canon.get(normalize(sub), sub)

print(f"Flat SetFit tail expert | tail<{TAIL_MAX}: {len(tail_set)} classes")

print("Loading dataset ...")
ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train_split.jsonl"),
    "val":   os.path.join(DATA_DIR, "val_split.jsonl"),
    "test":  os.path.join(DATA_DIR, "test_split.jsonl"),
})
if DEBUG:
    ds = {k: v.select(range(min(20000, len(v)))) for k, v in ds.items()}

# ── collect flat training set: tail (capped) + non-tail negative pool ──────
print("Collecting training samples ...")
tail_texts, tail_labels = [], []
per_tail = {}
nontail_texts, nontail_labels = [], []
per_nontail = {}
rng = np.random.default_rng(0)
for ex in ds["train"]:
    pt = ex.get("producttype");
    if not pt: continue
    sub = canonical(pt[0]); txt = get_text(ex)
    if not txt.strip(): continue
    if sub in tail_set:
        if per_tail.get(sub, 0) < MAX_PER_CLASS:
            tail_texts.append(txt); tail_labels.append(sub)
            per_tail[sub] = per_tail.get(sub, 0) + 1
    else:
        if len(nontail_texts) < NONTAIL_POOL and per_nontail.get(sub, 0) < NONTAIL_PER_CLASS \
           and rng.random() < 0.5:
            nontail_texts.append(txt); nontail_labels.append(sub)
            per_nontail[sub] = per_nontail.get(sub, 0) + 1

texts  = tail_texts + nontail_texts
labels = tail_labels + nontail_labels
uniq = sorted(set(labels)); lab2id = {l: i for i, l in enumerate(uniq)}
y = [lab2id[l] for l in labels]
print(f"  train: {len(tail_texts):,} tail ({len(per_tail)} classes) + "
      f"{len(nontail_texts):,} non-tail ({len(per_nontail)} classes) = {len(texts):,} samples, "
      f"{len(uniq)} labels")
train_ds = Dataset.from_dict({"text": texts, "label": y})

# ── train (contrastive body; head is fit but unused downstream) ────────────
print("Training flat SetFit body ...")
t0 = time.time()
model = SetFitModel.from_pretrained(ST_MODEL)
args = TrainingArguments(
    batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS,
    num_epochs=NUM_EPOCHS_BODY, body_learning_rate=2e-5, seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
print(f"  trained in {(time.time()-t0)/60:.1f} min")
model.model_body.save(BODY_DIR)
print(f"  saved body -> {BODY_DIR}")

# ── extract + cache embeddings (the contrastive body) ──────────────────────
from sentence_transformers import SentenceTransformer
body = model.model_body
dev = "cuda" if torch.cuda.is_available() else "cpu"

def encode(texts):
    return body.encode(texts, batch_size=ENCODE_BATCH, convert_to_numpy=True,
                       show_progress_bar=False, normalize_embeddings=False).astype(np.float16)

def collect_train_for_centroids():
    T, Y1, Y2 = [], [], []; pernt = {}
    for ex in ds["train"]:
        pt = ex.get("producttype")
        if not pt: continue
        sub = canonical(pt[0]); txt = get_text(ex)
        if not txt.strip(): continue
        if sub in tail_set:
            T.append(txt); Y1.append(ex.get("typelabel","")); Y2.append(sub)       # ALL tail
        else:
            if pernt.get(sub, 0) < NONTAIL_CENTROID_CAP:                            # capped non-tail
                T.append(txt); Y1.append(ex.get("typelabel","")); Y2.append(sub)
                pernt[sub] = pernt.get(sub, 0) + 1
    return T, Y1, Y2

for split in ["train", "val", "test"]:
    print(f"Encoding {split} ...")
    if split == "train":
        T, Y1, Y2 = collect_train_for_centroids()
    else:
        T, Y1, Y2 = [], [], []
        for ex in ds[split]:
            pt = ex.get("producttype")
            if not pt: continue
            txt = get_text(ex)
            if not txt.strip(): continue
            T.append(txt); Y1.append(ex.get("typelabel", "")); Y2.append(canonical(pt[0]))
    X = np.zeros((len(T), body.get_sentence_embedding_dimension()), dtype=np.float16)
    for s in range(0, len(T), 20000):
        X[s:s+20000] = encode(T[s:s+20000])
    out = os.path.join(EMB_DIR, f"{split}_emb_setfit.npz")
    tmp = out + ".tmp.npz"
    np.savez(tmp, X=X, y_l1=np.array(Y1), y_l2=np.array(Y2))
    os.replace(tmp, out)
    print(f"  saved {out}  shape={X.shape}")
    del X; gc.collect()

print("Done. Now run the tail eval on the SetFit embeddings:")
print("  python eval_knn_tail_whiten.py --emb-dir embeddings_setfit "
      "--emb-name emb_setfit --emb full --configs none")
