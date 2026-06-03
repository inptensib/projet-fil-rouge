#!/usr/bin/env python3
"""
Hybrid E2E Evaluation v3 — SetFit+gates + Gen probe when L1 uncertain + checkpointing.

Architecture:
  1. CamemBERT L2 flat → pred + confidence (ALL samples)
  2. IF cam_conf ≥ 0.5 → accept (route: camembert_accept)
  3. IF cam_conf < 0.5 AND l1_conf ≥ 0.7 → confidence-gated specialists:
        SetFit → XGBoost → LoRA, all gated by specialist_conf > cam_conf
  4. IF cam_conf < 0.5 AND l1_conf < 0.7 → run BOTH CamemBERT AND Generative
        (use top-2 L1 predictions' L2 union as candidate set)
        Track both predictions independently to measure Gen's standalone potential
        Final pick: CamemBERT (safer default), but Gen tracked separately

Reports:
  - Per-route accuracy/F1 + sample breakdown
  - Override stats per specialist (tried vs won)
  - GEN-vs-CAM head-to-head on l1_uncertain branch:
      * Cam-only acc, Gen-only acc, Oracle (best of both)
      * Cam-wins / Gen-wins / both-right / both-wrong counts
  - 50 error samples per route

Checkpointing:
  - State saved to .hybrid_v3_checkpoint.pkl every CHECKPOINT_EVERY samples
  - On startup, resumes from last saved index unless --restart given

Usage:
    python eval_hybrid_confidence_v3.py             # resumes if ckpt exists
    python eval_hybrid_confidence_v3.py --debug     # 10K sample
    python eval_hybrid_confidence_v3.py --restart   # ignore checkpoint
"""
import os
import sys
import json
import time
import gc
import pickle
import unicodedata
import numpy as np
import torch
from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DEBUG = "--debug" in sys.argv
RESTART = "--restart" in sys.argv

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

L1_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/l1_classifier")
L2_CAMEMBERT_DIR = os.path.join(BASE_DIR, "checkpoints/l2_camembert_run2")
XGBOOST_DIR = os.path.join(BASE_DIR, "checkpoints/l2_xgboost")
SETFIT_DIR = os.path.join(BASE_DIR, "checkpoints/setfit_models")
LORA_DIR = os.path.join(BASE_DIR, "checkpoints/lora_models")
GEN_MODEL_NAME = os.environ.get("GEN_MODEL", "croissantllm/CroissantLLMChat-v0.1")

CAMEMBERT_ACCEPT_THRESHOLD = 0.5
GENERATIVE_FALLBACK_THRESHOLD = 0.3
L1_CONFIDENCE_THRESHOLD = 0.7
GEN_MAX_CANDIDATES = 30
GEN_TOPK_L1 = 2
BATCH_SIZE = 64
MAX_LEN_L1 = 128
MAX_LEN_L2 = 256
MAX_ERRORS_PER_ROUTE = 50
CHECKPOINT_EVERY = 50000
CHECKPOINT_FILE = os.path.join(BASE_DIR, ".hybrid_v3_checkpoint.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def find_best_ckpt(ckpt_dir):
    best = os.path.join(ckpt_dir, "best")
    if os.path.isdir(best):
        return best
    ckpts = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir


# ─── Load mappings & models ──────────────────────────
print("Loading L1→L2 mapping...")
with open(os.path.join(ANALYSIS_DIR, "l1_to_l2_mapping.json")) as f:
    l1_to_l2_mapping = json.load(f)
print(f"  {len(l1_to_l2_mapping)} L1 classes")

print("Loading L1 model...")
l1_model_dir = find_best_ckpt(L1_CKPT_DIR)
try:
    l1_tokenizer = AutoTokenizer.from_pretrained(l1_model_dir)
except OSError:
    l1_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
l1_model = AutoModelForSequenceClassification.from_pretrained(l1_model_dir).to(device)
l1_model.eval()
l1_id2label = {int(k): v for k, v in l1_model.config.id2label.items()}
l1_encoder = AutoModel.from_pretrained(l1_model_dir).to(device)
l1_encoder.eval()

print("Loading CamemBERT L2 flat model...")
l2_cam_dir = find_best_ckpt(L2_CAMEMBERT_DIR)
try:
    l2_cam_tokenizer = AutoTokenizer.from_pretrained(l2_cam_dir)
except OSError:
    l2_cam_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
l2_cam_model = AutoModelForSequenceClassification.from_pretrained(l2_cam_dir).to(device)
l2_cam_model.eval()
l2_cam_id2label = {int(k): v for k, v in l2_cam_model.config.id2label.items()}

print("Loading XGBoost models...")
with open(os.path.join(XGBOOST_DIR, "xgb_models.pkl"), "rb") as f:
    xgb_models = pickle.load(f)
with open(os.path.join(XGBOOST_DIR, "label_maps.json")) as f:
    xgb_label_maps = json.load(f)
xgb_available = set(xgb_models.keys())
print(f"  XGBoost: {len(xgb_available)} L1 classes")

print("Indexing SetFit models...")
from setfit import SetFitModel
setfit_model_paths, setfit_label_maps = {}, {}
for d in os.listdir(SETFIT_DIR):
    model_path = os.path.join(SETFIT_DIR, d)
    if not os.path.isdir(model_path):
        continue
    label_file = os.path.join(model_path, "label_mapping.json")
    if not os.path.isfile(label_file):
        continue
    with open(label_file) as f:
        lm = json.load(f)
    l1_class = lm.get("l1_class", d.replace("_", " "))
    setfit_model_paths[l1_class] = model_path
    setfit_label_maps[l1_class] = lm
print(f"  SetFit: {len(setfit_model_paths)} L1 classes")

_setfit_cache = {}
def get_setfit_model(l1_class):
    if l1_class in _setfit_cache:
        return _setfit_cache[l1_class]
    if len(_setfit_cache) >= 2:
        old = next(iter(_setfit_cache))
        del _setfit_cache[old]
        gc.collect(); torch.cuda.empty_cache()
    m = SetFitModel.from_pretrained(setfit_model_paths[l1_class]).to(device)
    _setfit_cache[l1_class] = m
    return m

print("Indexing LoRA models...")
from peft import PeftModel as PeftModelClass
from transformers import CamembertForSequenceClassification, CamembertTokenizer
lora_model_paths, lora_label_maps = {}, {}
lora_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
for d in os.listdir(LORA_DIR):
    model_path = os.path.join(LORA_DIR, d)
    if not os.path.isdir(model_path):
        continue
    label_file = os.path.join(model_path, "label_mapping.json")
    if not os.path.isfile(label_file):
        continue
    with open(label_file) as f:
        lm = json.load(f)
    l1_class = lm.get("l1_class", d.replace("_", " "))
    lora_label_maps[l1_class] = lm
    lora_model_paths[l1_class] = model_path
print(f"  LoRA: {len(lora_model_paths)} L1 classes")

_lora_cache = {}
def get_lora_model(l1_class):
    if l1_class in _lora_cache:
        return _lora_cache[l1_class]
    if len(_lora_cache) >= 3:
        old = next(iter(_lora_cache))
        del _lora_cache[old]
        gc.collect(); torch.cuda.empty_cache()
    n_labels = lora_label_maps[l1_class]["n_subcats"]
    base = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=n_labels)
    m = PeftModelClass.from_pretrained(base, lora_model_paths[l1_class]).to(device)
    m.eval()
    _lora_cache[l1_class] = m
    return m

# ─── Generative ──────────────────────────────────────
_gen_model, _gen_tokenizer = None, None
def get_generative_model():
    global _gen_model, _gen_tokenizer
    if _gen_model is not None:
        return _gen_model, _gen_tokenizer
    print("  Loading generative model (CroissantLLM)...")
    from transformers import AutoModelForCausalLM
    _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    _setfit_cache.clear(); _lora_cache.clear()
    gc.collect(); torch.cuda.empty_cache()
    _gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    _gen_model.eval()
    if _gen_tokenizer.pad_token is None:
        _gen_tokenizer.pad_token = _gen_tokenizer.eos_token
    return _gen_model, _gen_tokenizer


def generative_predict(text, candidates, max_text_len=300):
    gen_model, gen_tok = get_generative_model()
    text = text[:max_text_len]
    scores = []
    for candidate in candidates:
        prompt = (
            f"Classifie ce produit dans une des catégories.\n"
            f"Produit: {text}\n"
            f"Catégorie: {candidate}"
        )
        inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
        candidate_tokens = gen_tok(candidate, add_special_tokens=False)["input_ids"]
        n_ct = len(candidate_tokens)
        with torch.no_grad():
            outputs = gen_model(**inputs)
            log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)
        seq_len = inputs["input_ids"].shape[1]
        total = 0.0
        for j, tid in enumerate(candidate_tokens):
            pos = seq_len - n_ct + j
            if pos > 0:
                total += log_probs[pos - 1, tid].item()
        scores.append(total / max(n_ct, 1))
    return candidates[int(np.argmax(scores))]


# ─── Specialist helpers ──────────────────────────────
def predict_setfit(text, l1_class, cam_pred):
    sf_model = get_setfit_model(l1_class)
    sf_id2label = setfit_label_maps[l1_class]["id2label"]
    if len(sf_id2label) <= 1:
        return sf_id2label.get("0", cam_pred), 1.0
    try:
        probs = sf_model.predict_proba([text])
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        probs = np.asarray(probs)
        if probs.ndim == 2:
            probs = probs[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
    except Exception:
        pred = sf_model.predict([text])
        pred_idx = int(pred[0]) if hasattr(pred[0], "__int__") else int(pred[0].item())
        conf = 0.5
    return sf_id2label.get(str(pred_idx), cam_pred), conf


def predict_lora(text, l1_class, cam_pred):
    lora_model = get_lora_model(l1_class)
    lora_lm = lora_label_maps[l1_class]
    inputs = lora_tokenizer([text], padding=True, truncation=True,
                            max_length=MAX_LEN_L2, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = lora_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().cpu())
        conf = float(probs[pred_idx].cpu())
    return lora_lm["id2label"].get(str(pred_idx), cam_pred), conf


def predict_xgboost(text, l1_class, cam_pred):
    xgb_lm = xgb_label_maps.get(l1_class, {})
    if "single" in xgb_lm:
        return (xgb_lm["single"] or cam_pred), 1.0
    emb = extract_embedding(text)
    proba = xgb_models[l1_class].predict_proba(emb)
    pred_idx = int(np.argmax(proba, axis=-1)[0])
    conf = float(np.max(proba))
    id2sub = xgb_lm.get("id2sub", {})
    return id2sub.get(str(pred_idx), "UNKNOWN"), conf


# ─── Data + batch helpers ────────────────────────────
print("Loading test data...")
ds = load_dataset("json", data_files={"test": os.path.join(DATA_DIR, "test_split.jsonl")})
test_ds = ds["test"]
if DEBUG:
    test_ds = test_ds.select(range(min(10000, len(test_ds))))
print(f"  Test samples: {len(test_ds)}")


@torch.no_grad()
def predict_l1_batch(texts, topk=2):
    inputs = l1_tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN_L1, return_tensors="pt").to(device)
    logits = l1_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    top_confs, top_preds = probs.topk(topk, dim=-1)
    return top_preds.cpu().numpy(), top_confs.cpu().numpy()


@torch.no_grad()
def predict_l2_camembert_batch(texts):
    inputs = l2_cam_tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN_L2, return_tensors="pt").to(device)
    logits = l2_cam_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    return preds.cpu().numpy(), confs.cpu().numpy()


@torch.no_grad()
def extract_embedding(text):
    inputs = l1_tokenizer([text], padding=True, truncation=True, max_length=MAX_LEN_L1, return_tensors="pt").to(device)
    outputs = l1_encoder(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)


# ─── Checkpoint state ────────────────────────────────
def save_checkpoint(state):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, CHECKPOINT_FILE)


def load_checkpoint():
    if not os.path.isfile(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Failed to load checkpoint: {e}")
        return None


state = None if RESTART else load_checkpoint()
if state is not None:
    print(f"\n>>> Resuming from checkpoint at sample {state['next_idx']:,}/{len(test_ds):,}")
    all_preds = state["all_preds"]
    all_true = state["all_true"]
    all_l1_preds = state["all_l1_preds"]
    all_l1_true = state["all_l1_true"]
    route_preds = defaultdict(list, state["route_preds"])
    route_counts = Counter(state["route_counts"])
    errors_per_route = defaultdict(list, state["errors_per_route"])
    override_stats = Counter(state["override_stats"])
    gen_probe = state["gen_probe"]
    start_idx = state["next_idx"]
else:
    print("\n>>> Starting fresh evaluation")
    all_preds, all_true = [], []
    all_l1_preds, all_l1_true = [], []
    route_preds = defaultdict(list)
    route_counts = Counter()
    errors_per_route = defaultdict(list)
    override_stats = Counter()
    gen_probe = []
    start_idx = 0


# ─── Main loop ───────────────────────────────────────
print(f"\n{'='*60}")
print("Hybrid Evaluation v3 — SetFit+gates + Gen probe + checkpoints")
print(f"  Cam accept threshold:    {CAMEMBERT_ACCEPT_THRESHOLD}")
print(f"  L1 confidence threshold: {L1_CONFIDENCE_THRESHOLD}")
print(f"  Gen candidates cap:      {GEN_MAX_CANDIDATES} (union of top-{GEN_TOPK_L1} L1)")
print(f"  Checkpoint every:        {CHECKPOINT_EVERY:,} samples")
print(f"{'='*60}\n")

start = time.time()
n = len(test_ds)
samples_processed = start_idx
last_ckpt_at = start_idx

for batch_start in tqdm(range(start_idx, n, BATCH_SIZE), desc="Evaluating", initial=start_idx // BATCH_SIZE, total=(n + BATCH_SIZE - 1) // BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n)
    batch = test_ds[batch_start:batch_end]
    bsz = batch_end - batch_start

    texts_l1 = [batch["designation"][i] or "" for i in range(bsz)]
    texts_l2 = []
    for i in range(bsz):
        title = batch["designation"][i] or ""
        desc = batch.get("description", [""] * bsz)[i] or ""
        texts_l2.append(f"{title} {desc}"[:1024] if desc else title)

    true_l1 = [batch["typelabel"][i] for i in range(bsz)]
    true_l2 = [batch["producttype"][i][0] if batch["producttype"][i] else "" for i in range(bsz)]

    l1_top_idx, l1_top_conf = predict_l1_batch(texts_l1, topk=GEN_TOPK_L1)
    l2_preds_idx, l2_confs = predict_l2_camembert_batch(texts_l2)
    l2_pred_labels = [l2_cam_id2label.get(int(idx), "UNKNOWN") for idx in l2_preds_idx]

    for i in range(bsz):
        l1_pred = l1_id2label.get(int(l1_top_idx[i, 0]), "UNKNOWN")
        l1_conf = float(l1_top_conf[i, 0])
        l2_cam_pred = l2_pred_labels[i]
        l2_cam_conf = float(l2_confs[i])
        true_subcat = true_l2[i]

        all_l1_preds.append(l1_pred)
        all_l1_true.append(true_l1[i])
        all_true.append(true_subcat)

        # Gate 1: cam confident
        if l2_cam_conf >= CAMEMBERT_ACCEPT_THRESHOLD:
            final_pred = l2_cam_pred
            route_name = "camembert_accept"

        # Gate 2: cam unsure AND L1 unsure → probe Generative alongside CamemBERT
        elif l1_conf < L1_CONFIDENCE_THRESHOLD:
            # Build candidate set: union of top-K L1's L2 subcats
            cands = []
            seen = set()
            for k in range(GEN_TOPK_L1):
                l1_k = l1_id2label.get(int(l1_top_idx[i, k]), "")
                for c in l1_to_l2_mapping.get(l1_k, []):
                    if c not in seen:
                        cands.append(c)
                        seen.add(c)
                    if len(cands) >= GEN_MAX_CANDIDATES:
                        break
                if len(cands) >= GEN_MAX_CANDIDATES:
                    break

            gen_pred_val = None
            if len(cands) >= 2:
                try:
                    gen_pred_val = generative_predict(texts_l2[i], cands)
                except Exception:
                    gen_pred_val = None

            # Default to CamemBERT; track Gen separately
            final_pred = l2_cam_pred
            route_name = "camembert_l1_uncertain_gen_probed"
            gen_probe.append({
                "cam_pred": l2_cam_pred,
                "gen_pred": gen_pred_val if gen_pred_val is not None else l2_cam_pred,
                "true": true_subcat,
                "l1_pred": l1_pred,
                "true_l1": true_l1[i],
                "cam_conf": round(l2_cam_conf, 3),
                "l1_conf": round(l1_conf, 3),
                "n_cands": len(cands),
            })

        # Gate 3: cam unsure, L1 sure → specialists with confidence gates
        else:
            final_pred = l2_cam_pred
            route_name = "camembert_no_override"
            best_conf = l2_cam_conf

            if l1_pred in setfit_model_paths:
                sf_pred, sf_conf = predict_setfit(texts_l2[i], l1_pred, l2_cam_pred)
                override_stats["setfit_tried"] += 1
                if sf_conf > best_conf:
                    final_pred = sf_pred
                    best_conf = sf_conf
                    route_name = "setfit_override"
                    override_stats["setfit_won"] += 1

            elif l1_pred in xgb_available:
                xgb_pred, xgb_conf = predict_xgboost(texts_l1[i], l1_pred, l2_cam_pred)
                override_stats["xgb_tried"] += 1
                if xgb_conf > best_conf:
                    final_pred = xgb_pred
                    best_conf = xgb_conf
                    route_name = "xgboost_override"
                    override_stats["xgb_won"] += 1

            elif l1_pred in lora_model_paths:
                lora_pred, lora_conf = predict_lora(texts_l2[i], l1_pred, l2_cam_pred)
                override_stats["lora_tried"] += 1
                if lora_conf > best_conf:
                    final_pred = lora_pred
                    best_conf = lora_conf
                    route_name = "lora_override"
                    override_stats["lora_won"] += 1

        all_preds.append(final_pred)
        route_counts[route_name] += 1
        route_preds[route_name].append((final_pred, true_subcat))

        if final_pred != true_subcat and len(errors_per_route[route_name]) < MAX_ERRORS_PER_ROUTE:
            errors_per_route[route_name].append({
                "text": texts_l2[i][:200],
                "true_l1": true_l1[i],
                "pred_l1": l1_pred,
                "l1_conf": round(l1_conf, 3),
                "true_l2": true_subcat,
                "pred_l2": final_pred,
                "cam_conf": round(l2_cam_conf, 3),
                "route": route_name,
            })

        samples_processed += 1

    # Periodic checkpoint
    if samples_processed - last_ckpt_at >= CHECKPOINT_EVERY:
        save_checkpoint({
            "next_idx": batch_end,
            "all_preds": all_preds,
            "all_true": all_true,
            "all_l1_preds": all_l1_preds,
            "all_l1_true": all_l1_true,
            "route_preds": dict(route_preds),
            "route_counts": dict(route_counts),
            "errors_per_route": dict(errors_per_route),
            "override_stats": dict(override_stats),
            "gen_probe": gen_probe,
        })
        last_ckpt_at = samples_processed
        elapsed_min = (time.time() - start) / 60
        print(f"\n  [checkpoint] saved at {samples_processed:,}/{n:,} (elapsed {elapsed_min:.1f} min)")


# ─── Reports ──────────────────────────────────────────
elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"RESULTS v3 (eval time: {elapsed/60:.1f} min)")
print(f"{'='*60}")

l1_acc = accuracy_score(all_l1_true, all_l1_preds)
print(f"\nL1 Accuracy: {l1_acc:.4f}")

l2_acc = accuracy_score(all_true, all_preds)
l2_f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
l2_f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)

print(f"\nOverall L2 E2E:")
print(f"  Accuracy:    {l2_acc:.4f}")
print(f"  F1-macro:    {l2_f1_macro:.4f}")
print(f"  F1-weighted: {l2_f1_weighted:.4f}")

print(f"\n{'─'*60}")
print(f"COMPARISON")
print(f"{'─'*60}")
print(f"  CamemBERT L2 alone:           acc=0.820  f1_macro=0.573  f1_wt=0.815")
print(f"  Conf-only v1 (XGB first):     acc=0.832  f1_macro=0.621  f1_wt=0.828")
print(f"  v3 (SetFit+gates+gen-probe):  acc={l2_acc:.3f}  f1_macro={l2_f1_macro:.3f}  f1_wt={l2_f1_weighted:.3f}")

print(f"\n{'─'*60}")
print("Override statistics (specialist_conf > cam_conf):")
for spec in ["setfit", "xgb", "lora"]:
    tried = override_stats.get(f"{spec}_tried", 0)
    won = override_stats.get(f"{spec}_won", 0)
    if tried > 0:
        print(f"  {spec:10s}: {won:>8,}/{tried:>8,} = {100*won/tried:5.1f}% override rate")

print(f"\n{'─'*60}")
print(f"{'Route':<35} {'N':>10} {'%':>6} {'Accuracy':>10} {'F1-macro':>10}")
print(f"{'─'*60}")
route_results = {}
for route_name in sorted(route_counts.keys(), key=lambda x: -route_counts[x]):
    count = route_counts[route_name]
    pct = 100.0 * count / n
    preds_true = route_preds[route_name]
    r_preds = [p for p, t in preds_true]
    r_true = [t for p, t in preds_true]
    r_acc = accuracy_score(r_true, r_preds)
    r_f1 = f1_score(r_true, r_preds, average="macro", zero_division=0)
    print(f"  {route_name:<33} {count:>10,} {pct:>5.1f}% {r_acc:>10.4f} {r_f1:>10.4f}")
    route_results[route_name] = {
        "n_samples": count, "pct": round(pct, 2),
        "accuracy": round(r_acc, 4), "f1_macro": round(r_f1, 4),
    }

# Gen-vs-Cam head-to-head
print(f"\n{'─'*60}")
print("GEN-vs-CAM head-to-head (l1_uncertain samples):")
print(f"{'─'*60}")
gen_probe_summary = None
if len(gen_probe) > 0:
    cam_only = [p["cam_pred"] for p in gen_probe]
    gen_only = [p["gen_pred"] for p in gen_probe]
    probe_true = [p["true"] for p in gen_probe]
    cam_acc = accuracy_score(probe_true, cam_only)
    gen_acc = accuracy_score(probe_true, gen_only)
    oracle = [c if c == t else (g if g == t else c) for c, g, t in zip(cam_only, gen_only, probe_true)]
    oracle_acc = accuracy_score(probe_true, oracle)
    cam_wins = sum(1 for c, g, t in zip(cam_only, gen_only, probe_true) if c == t and g != t)
    gen_wins = sum(1 for c, g, t in zip(cam_only, gen_only, probe_true) if g == t and c != t)
    both_right = sum(1 for c, g, t in zip(cam_only, gen_only, probe_true) if c == t and g == t)
    both_wrong = sum(1 for c, g, t in zip(cam_only, gen_only, probe_true) if c != t and g != t)
    print(f"  N samples in probe:       {len(gen_probe):,}")
    print(f"  CamemBERT-only accuracy:  {cam_acc:.4f}")
    print(f"  Generative-only accuracy: {gen_acc:.4f}")
    print(f"  Oracle (best of both):    {oracle_acc:.4f}")
    print(f"  Cam ✓ Gen ✗:              {cam_wins:,}")
    print(f"  Gen ✓ Cam ✗:              {gen_wins:,}")
    print(f"  Both ✓:                   {both_right:,}")
    print(f"  Both ✗:                   {both_wrong:,}")
    gen_probe_summary = {
        "n_samples": len(gen_probe),
        "cam_only_accuracy": float(cam_acc),
        "gen_only_accuracy": float(gen_acc),
        "oracle_accuracy": float(oracle_acc),
        "cam_wins": cam_wins, "gen_wins": gen_wins,
        "both_right": both_right, "both_wrong": both_wrong,
        "cam_only_f1_macro": float(f1_score(probe_true, cam_only, average="macro", zero_division=0)),
        "gen_only_f1_macro": float(f1_score(probe_true, gen_only, average="macro", zero_division=0)),
    }
else:
    print("  (no samples hit the l1_uncertain branch)")

# Error samples
print(f"\n{'─'*60}")
print("Error Analysis (sample per route):")
print(f"{'─'*60}")
for route_name in sorted(errors_per_route.keys()):
    errors = errors_per_route[route_name]
    if not errors:
        continue
    print(f"\n  [{route_name}] — {len(errors)} error samples:")
    for err in errors[:5]:
        print(f"    Text: {err['text'][:80]}...")
        print(f"    True: {err['true_l2']} | Pred: {err['pred_l2']} | cam_conf: {err['cam_conf']} | l1_conf: {err['l1_conf']}")
        print()

# Save
results = {
    "method": "confidence_v3_setfit_gated_gen_probe",
    "priority": "SetFit > XGBoost > LoRA (all gated); Gen probe on l1_uncertain",
    "l1_accuracy": l1_acc,
    "l2_e2e_accuracy": l2_acc,
    "l2_e2e_f1_macro": l2_f1_macro,
    "l2_e2e_f1_weighted": l2_f1_weighted,
    "thresholds": {
        "camembert_accept": CAMEMBERT_ACCEPT_THRESHOLD,
        "l1_confidence_min": L1_CONFIDENCE_THRESHOLD,
        "generative_fallback": GENERATIVE_FALLBACK_THRESHOLD,
    },
    "override_stats": dict(override_stats),
    "route_breakdown": route_results,
    "gen_probe_summary": gen_probe_summary,
    "eval_time_minutes": elapsed / 60,
    "n_samples": n,
}

output_file = os.path.join(BASE_DIR, "hybrid_results_confidence_v3.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

errors_file = os.path.join(BASE_DIR, "hybrid_errors_confidence_v3.json")
with open(errors_file, "w") as f:
    json.dump(dict(errors_per_route), f, indent=2, ensure_ascii=False)

gen_probe_file = os.path.join(BASE_DIR, "hybrid_gen_probe_v3.json")
with open(gen_probe_file, "w") as f:
    json.dump(gen_probe[:5000], f, indent=2, ensure_ascii=False)

if os.path.isfile(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"\n  Checkpoint cleared (eval completed successfully)")

print(f"\nResults saved to {output_file}")
print(f"Errors  saved to {errors_file}")
print(f"Gen probe saved to {gen_probe_file}")
print("Done!")
