#!/usr/bin/env python3
"""
Hybrid E2E Evaluation v2 — CamemBERT L2 first, specialists as correction.

Architecture:
  1. CamemBERT L2 flat → pred + confidence (for ALL samples)
  2. IF confidence > 0.5 → accept
  3. ELSE:
     a. L1 predict → determine route
     b. XGBoost class AND xgb confidence > cam → use XGBoost
     c. SetFit/LoRA class → use specialist
     d. No specialist AND confidence < 0.3 → generative (CroissantLLM)
     e. ELSE → keep CamemBERT L2 prediction

Reports:
  - Full per-model accuracy/F1 breakdown
  - Hybrid sub-route distribution
  - Error samples for analysis

Usage:
    python eval_hybrid.py                  # full eval
    python eval_hybrid.py --debug          # 10K sample
"""
import os
import sys
import json
import time
import gc
import pickle
import unicodedata
import random
import numpy as np
import torch
from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DEBUG = "--debug" in sys.argv

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

CAMEMBERT_ACCEPT_THRESHOLD = 0.5   # accept CamemBERT L2 if above this
GENERATIVE_FALLBACK_THRESHOLD = 0.3  # use generative if below this AND no specialist
BATCH_SIZE = 64
MAX_LEN_L1 = 128
MAX_LEN_L2 = 256
MAX_ERRORS_PER_ROUTE = 50  # sample errors for analysis

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ─── Helpers ──────────────────────────────────────────
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


# ─── Load gating config ──────────────────────────────
print("Loading gating config...")
with open(os.path.join(ANALYSIS_DIR, "gating_config.json")) as f:
    gating = json.load(f)

routing = gating["routing"]
subcat_routing = gating.get("subcat_routing", {})

rare_subcats_set = {}
for l1_class, info in subcat_routing.items():
    rare_subcats_set[l1_class] = set(normalize(s) for s in info["rare_subcats"])

xgb_classes = set(c for c, r in routing.items() if r == "xgboost")

# Load full L1 → L2 mapping (ALL valid subcats per L1 class)
with open(os.path.join(ANALYSIS_DIR, "l1_to_l2_mapping.json")) as f:
    l1_to_l2_mapping = json.load(f)
print(f"  L1→L2 mapping: {len(l1_to_l2_mapping)} L1 classes")


# ─── Load L1 model ───────────────────────────────────
print("Loading L1 model...")
l1_model_dir = find_best_ckpt(L1_CKPT_DIR)
try:
    l1_tokenizer = AutoTokenizer.from_pretrained(l1_model_dir)
except OSError:
    l1_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
l1_model = AutoModelForSequenceClassification.from_pretrained(l1_model_dir).to(device)
l1_model.eval()
l1_id2label = {int(k): v for k, v in l1_model.config.id2label.items()}

# L1 encoder for XGBoost embeddings
l1_encoder = AutoModel.from_pretrained(l1_model_dir).to(device)
l1_encoder.eval()


# ─── Load CamemBERT L2 model ─────────────────────────
print("Loading CamemBERT L2 flat model...")
l2_cam_dir = find_best_ckpt(L2_CAMEMBERT_DIR)
try:
    l2_cam_tokenizer = AutoTokenizer.from_pretrained(l2_cam_dir)
except OSError:
    l2_cam_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
l2_cam_model = AutoModelForSequenceClassification.from_pretrained(l2_cam_dir).to(device)
l2_cam_model.eval()
l2_cam_id2label = {int(k): v for k, v in l2_cam_model.config.id2label.items()}


# ─── Load XGBoost models ─────────────────────────────
print("Loading XGBoost models...")
with open(os.path.join(XGBOOST_DIR, "xgb_models.pkl"), "rb") as f:
    xgb_models = pickle.load(f)
with open(os.path.join(XGBOOST_DIR, "label_maps.json")) as f:
    xgb_label_maps = json.load(f)


# ─── Index SetFit models (lazy load) ─────────────────
print("Indexing SetFit models...")
from setfit import SetFitModel

setfit_model_paths = {}
setfit_label_maps = {}
for l1_class in routing:
    if routing[l1_class] != "camembert_setfit":
        continue
    model_dir_name = l1_class.replace("/", "_").replace(" ", "_")
    model_path = os.path.join(SETFIT_DIR, model_dir_name)
    if not os.path.isdir(model_path):
        continue
    label_file = os.path.join(model_path, "label_mapping.json")
    if not os.path.isfile(label_file):
        continue
    setfit_model_paths[l1_class] = model_path
    with open(label_file) as f:
        setfit_label_maps[l1_class] = json.load(f)
print(f"  Found {len(setfit_model_paths)} SetFit models")

_setfit_cache = {}
def get_setfit_model(l1_class):
    global _setfit_cache
    if l1_class in _setfit_cache:
        return _setfit_cache[l1_class]
    if len(_setfit_cache) >= 2:
        old_key = next(iter(_setfit_cache))
        del _setfit_cache[old_key]
        gc.collect()
        torch.cuda.empty_cache()
    model = SetFitModel.from_pretrained(setfit_model_paths[l1_class]).to(device)
    _setfit_cache[l1_class] = model
    return model


# ─── Index LoRA models (lazy load) ───────────────────
print("Indexing LoRA models...")
from peft import PeftModel as PeftModelClass
from transformers import CamembertForSequenceClassification, CamembertTokenizer

lora_model_paths = {}
lora_label_maps = {}
lora_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

for l1_class in routing:
    if routing[l1_class] != "camembert_setfit":
        continue
    if l1_class in setfit_model_paths:
        continue
    model_dir_name = l1_class.replace("/", "_").replace(" ", "_")
    model_path = os.path.join(LORA_DIR, model_dir_name)
    if not os.path.isdir(model_path):
        continue
    label_file = os.path.join(model_path, "label_mapping.json")
    if not os.path.isfile(label_file):
        continue
    with open(label_file) as f:
        lora_label_maps[l1_class] = json.load(f)
    lora_model_paths[l1_class] = model_path
print(f"  Found {len(lora_model_paths)} LoRA models")

_lora_cache = {}
def get_lora_model(l1_class):
    global _lora_cache
    if l1_class in _lora_cache:
        return _lora_cache[l1_class]
    if len(_lora_cache) >= 3:
        old_key = next(iter(_lora_cache))
        del _lora_cache[old_key]
        gc.collect()
        torch.cuda.empty_cache()
    n_labels = lora_label_maps[l1_class]["n_subcats"]
    base = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=n_labels)
    model = PeftModelClass.from_pretrained(base, lora_model_paths[l1_class]).to(device)
    model.eval()
    _lora_cache[l1_class] = model
    return model


# ─── Generative model (lazy load) ────────────────────
_gen_model = None
_gen_tokenizer = None

def get_generative_model():
    global _gen_model, _gen_tokenizer
    if _gen_model is not None:
        return _gen_model, _gen_tokenizer
    print("  Loading generative model (CroissantLLM)...")
    from transformers import AutoModelForCausalLM
    _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    # Evict some cache to make room
    global _setfit_cache, _lora_cache
    _setfit_cache.clear()
    _lora_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    _gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    _gen_model.eval()
    if _gen_tokenizer.pad_token is None:
        _gen_tokenizer.pad_token = _gen_tokenizer.eos_token
    return _gen_model, _gen_tokenizer


def generative_predict(text, candidates, max_text_len=300):
    """Constrained log-prob scoring: pick the candidate with highest P(label|prompt)."""
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
        n_candidate_tokens = len(candidate_tokens)
        with torch.no_grad():
            outputs = gen_model(**inputs)
            log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)
        seq_len = inputs["input_ids"].shape[1]
        total_log_prob = 0.0
        for j, token_id in enumerate(candidate_tokens):
            pos = seq_len - n_candidate_tokens + j
            if pos > 0:
                total_log_prob += log_probs[pos - 1, token_id].item()
        scores.append(total_log_prob / max(n_candidate_tokens, 1))
    return candidates[int(np.argmax(scores))]


# ─── Load test data ──────────────────────────────────
print("Loading test data...")
ds = load_dataset("json", data_files={"test": os.path.join(DATA_DIR, "test_split.jsonl")})
test_ds = ds["test"]
if DEBUG:
    test_ds = test_ds.select(range(min(10000, len(test_ds))))
print(f"  Test samples: {len(test_ds)}")


# ─── Batch prediction helpers ────────────────────────
@torch.no_grad()
def predict_l1_batch(texts):
    inputs = l1_tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN_L1, return_tensors="pt").to(device)
    logits = l1_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    return preds.cpu().numpy(), confs.cpu().numpy()


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


# ─── Main evaluation loop ────────────────────────────
print(f"\n{'='*60}")
print("Hybrid E2E Evaluation v2")
print(f"  CamemBERT accept threshold: {CAMEMBERT_ACCEPT_THRESHOLD}")
print(f"  Generative fallback threshold: {GENERATIVE_FALLBACK_THRESHOLD}")
print(f"{'='*60}\n")

start = time.time()

# Collectors
all_preds = []
all_true = []
all_l1_preds = []
all_l1_true = []

# Per-route tracking
route_preds = defaultdict(list)   # route → list of (pred, true)
route_counts = Counter()

# Error samples
errors_per_route = defaultdict(list)  # route → list of error dicts

n = len(test_ds)
for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="Evaluating"):
    batch_end = min(batch_start + BATCH_SIZE, n)
    batch = test_ds[batch_start:batch_end]
    bsz = batch_end - batch_start

    # Get texts
    texts_l1 = [batch["designation"][i] or "" for i in range(bsz)]
    texts_l2 = []
    for i in range(bsz):
        title = batch["designation"][i] or ""
        desc = batch.get("description", [""] * bsz)[i] or ""
        texts_l2.append(f"{title} {desc}"[:1024] if desc else title)

    # Ground truth
    true_l1 = [batch["typelabel"][i] for i in range(bsz)]
    true_l2 = [batch["producttype"][i][0] if batch["producttype"][i] else "" for i in range(bsz)]

    # Step 1: L1 prediction (for routing only)
    l1_preds_idx, l1_confs = predict_l1_batch(texts_l1)
    l1_pred_labels = [l1_id2label.get(int(idx), "UNKNOWN") for idx in l1_preds_idx]

    # Step 2: CamemBERT L2 flat prediction (for ALL samples)
    l2_preds_idx, l2_confs = predict_l2_camembert_batch(texts_l2)
    l2_pred_labels = [l2_cam_id2label.get(int(idx), "UNKNOWN") for idx in l2_preds_idx]

    for i in range(bsz):
        l1_pred = l1_pred_labels[i]
        l1_conf = float(l1_confs[i])
        l2_cam_pred = l2_pred_labels[i]
        l2_cam_conf = float(l2_confs[i])
        true_subcat = true_l2[i]

        all_l1_preds.append(l1_pred)
        all_l1_true.append(true_l1[i])
        all_true.append(true_subcat)

        # ─── Decision logic ───────────────────────────────
        # Gate 1: CamemBERT L2 confident → accept
        if l2_cam_conf >= CAMEMBERT_ACCEPT_THRESHOLD:
            final_pred = l2_cam_pred
            route_name = "camembert_l2_accept"

        # Gate 2: Low confidence → try specialists
        else:
            route = routing.get(l1_pred, "camembert_setfit")

            # 2a: XGBoost class
            if route == "xgboost" and l1_pred in xgb_models:
                xgb_lm = xgb_label_maps.get(l1_pred, {})
                if "single" in xgb_lm:
                    final_pred = xgb_lm["single"] or l2_cam_pred
                else:
                    emb = extract_embedding(texts_l1[i])
                    proba = xgb_models[l1_pred].predict_proba(emb)
                    pred_idx = int(np.argmax(proba, axis=-1)[0])
                    xgb_conf = float(np.max(proba))
                    id2sub = xgb_lm.get("id2sub", {})
                    xgb_pred = id2sub.get(str(pred_idx), "UNKNOWN")
                    # Only override if XGBoost is more confident
                    final_pred = xgb_pred if xgb_conf > l2_cam_conf else l2_cam_pred
                route_name = "xgboost_override"

            # 2b: SetFit class (rare subcats)
            elif l1_pred in setfit_model_paths:
                sf_model = get_setfit_model(l1_pred)
                sf_pred = sf_model.predict([texts_l2[i]])
                sf_pred_idx = int(sf_pred[0]) if hasattr(sf_pred[0], "__int__") else int(sf_pred[0].item())
                sf_id2label = setfit_label_maps[l1_pred]["id2label"]
                final_pred = sf_id2label.get(str(sf_pred_idx), l2_cam_pred)
                route_name = "setfit_fallback"

            # 2c: LoRA class (rare subcats)
            elif l1_pred in lora_model_paths:
                lora_model = get_lora_model(l1_pred)
                lora_lm = lora_label_maps[l1_pred]
                inputs = lora_tokenizer(
                    [texts_l2[i]], padding=True, truncation=True,
                    max_length=MAX_LEN_L2, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = lora_model(**inputs).logits
                    lora_pred_idx = int(logits.argmax(dim=-1)[0].cpu())
                final_pred = lora_lm["id2label"].get(str(lora_pred_idx), l2_cam_pred)
                route_name = "lora_fallback"

            # 2d: No specialist AND very low confidence → generative
            elif l2_cam_conf < GENERATIVE_FALLBACK_THRESHOLD:
                # Get ALL valid L2 subcats for this L1 class (constrained output)
                l1_for_gen = l1_pred
                candidates = l1_to_l2_mapping.get(l1_for_gen, [])
                if len(candidates) >= 2:
                    final_pred = generative_predict(texts_l2[i], candidates)
                    route_name = "generative_fallback"
                else:
                    final_pred = l2_cam_pred
                    route_name = "camembert_l2_lowconf"

            # 2e: No specialist, confidence between 0.3-0.5 → keep CamemBERT
            else:
                final_pred = l2_cam_pred
                route_name = "camembert_l2_midconf"

        all_preds.append(final_pred)
        route_counts[route_name] += 1
        route_preds[route_name].append((final_pred, true_subcat))

        # Collect errors
        if final_pred != true_subcat and len(errors_per_route[route_name]) < MAX_ERRORS_PER_ROUTE:
            errors_per_route[route_name].append({
                "text": texts_l2[i][:200],
                "true_l1": true_l1[i],
                "pred_l1": l1_pred,
                "true_l2": true_subcat,
                "pred_l2": final_pred,
                "cam_conf": round(l2_cam_conf, 3),
                "route": route_name,
            })


# ─── Compute & Report Metrics ─────────────────────────
elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"RESULTS (eval time: {elapsed/60:.1f} min)")
print(f"{'='*60}")

# L1
l1_acc = accuracy_score(all_l1_true, all_l1_preds)
print(f"\nL1 Accuracy: {l1_acc:.4f}")

# Overall
l2_acc = accuracy_score(all_true, all_preds)
l2_f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
l2_f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)

print(f"\nOverall L2 E2E:")
print(f"  Accuracy:    {l2_acc:.4f}")
print(f"  F1-macro:    {l2_f1_macro:.4f}")
print(f"  F1-weighted: {l2_f1_weighted:.4f}")

# Comparison
print(f"\n{'─'*60}")
print(f"COMPARISON")
print(f"{'─'*60}")
print(f"  CamemBERT L2 alone:  acc=0.820  f1_macro=0.573  f1_wt=0.815")
print(f"  Previous hybrid v1:  acc=0.785  f1_macro=0.542  f1_wt=0.815")
print(f"  This hybrid v2:      acc={l2_acc:.3f}  f1_macro={l2_f1_macro:.3f}  f1_wt={l2_f1_weighted:.3f}")

# Per-route breakdown
print(f"\n{'─'*60}")
print(f"{'Route':<30} {'N':>10} {'%':>6} {'Accuracy':>10} {'F1-macro':>10}")
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
    print(f"  {route_name:<28} {count:>10,} {pct:>5.1f}% {r_acc:>10.4f} {r_f1:>10.4f}")
    route_results[route_name] = {
        "n_samples": count,
        "pct": round(pct, 2),
        "accuracy": round(r_acc, 4),
        "f1_macro": round(r_f1, 4),
    }

# Error analysis summary
print(f"\n{'─'*60}")
print("Error Analysis (sample of misclassifications per route):")
print(f"{'─'*60}")
for route_name in sorted(errors_per_route.keys()):
    errors = errors_per_route[route_name]
    if not errors:
        continue
    print(f"\n  [{route_name}] — {len(errors)} error samples:")
    for err in errors[:5]:
        print(f"    Text: {err['text'][:80]}...")
        print(f"    True: {err['true_l2']} | Pred: {err['pred_l2']} | Conf: {err['cam_conf']}")
        print(f"    L1: true={err['true_l1']} pred={err['pred_l1']}")
        print()

# ─── Save results ────────────────────────────────────
results = {
    "l1_accuracy": l1_acc,
    "l2_e2e_accuracy": l2_acc,
    "l2_e2e_f1_macro": l2_f1_macro,
    "l2_e2e_f1_weighted": l2_f1_weighted,
    "thresholds": {
        "camembert_accept": CAMEMBERT_ACCEPT_THRESHOLD,
        "generative_fallback": GENERATIVE_FALLBACK_THRESHOLD,
    },
    "route_breakdown": route_results,
    "eval_time_minutes": elapsed / 60,
    "n_samples": n,
}

output_file = os.path.join(BASE_DIR, "hybrid_results_v2.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save error samples
errors_file = os.path.join(BASE_DIR, "hybrid_errors_v2.json")
with open(errors_file, "w") as f:
    json.dump(dict(errors_per_route), f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_file}")
print(f"Error samples saved to {errors_file}")
print("Done!")
