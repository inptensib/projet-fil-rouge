#!/usr/bin/env python3
"""
Generative fallback for hardest L2 subcategory classification.

Uses constrained log-probability scoring:
- For each product, compute P(label | prompt) for every valid subcat
- Pick the label with highest log-prob
- Guaranteed to output only valid labels (no hallucination)

Targets classes where LoRA F1 < threshold (discriminative approach fails).
Uses CroissantLLM (1.3B, French-capable) or any causal LM.

Usage:
    python generative_fallback.py                  # evaluate worst classes
    python generative_fallback.py --debug          # test with 1 class
    python generative_fallback.py --threshold 0.4  # adjust F1 cutoff
"""
import os
import sys
import json
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import unicodedata

DEBUG = "--debug" in sys.argv

# Parse threshold from args
F1_THRESHOLD = 0.35
for i, arg in enumerate(sys.argv):
    if arg == "--threshold" and i + 1 < len(sys.argv):
        F1_THRESHOLD = float(sys.argv[i + 1])

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
LORA_DIR = os.path.join(BASE_DIR, "checkpoints/lora_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints/generative_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model config — CroissantLLM is 1.3B params, French-capable, fits in 24GB easily
GENERATIVE_MODEL = os.environ.get("GEN_MODEL", "croissantllm/CroissantLLMChat-v0.1")
MAX_SEQ_LEN = 512

# ─── Load gating config ──────────────────────────────
with open(os.path.join(ANALYSIS_DIR, "gating_config.json")) as f:
    gating = json.load(f)

subcat_routing = gating["subcat_routing"]


# ─── Identify worst-performing classes from LoRA results ──
lora_results_path = os.path.join(LORA_DIR, "results_per_class.json")
if not os.path.isfile(lora_results_path):
    print("ERROR: No LoRA results found. Run LoRA training first.")
    sys.exit(1)

with open(lora_results_path) as f:
    lora_data = json.load(f)

# Classes that need generative fallback: low F1 OR skipped by LoRA
generative_classes = []
for l1_class, metrics in lora_data["results"].items():
    if metrics["f1_macro"] < F1_THRESHOLD:
        generative_classes.append(l1_class)
        
# Also include classes that LoRA skipped (too few subcats) but have data
for entry in lora_data.get("failed", []):
    if entry.get("reason") == "too_few_subcats" and entry.get("n_valid", 0) == 1:
        # Single-subcat classes don't need classification
        pass

print(f"F1 threshold: {F1_THRESHOLD}")
print(f"Classes below threshold (need generative fallback): {len(generative_classes)}")
for c in generative_classes:
    f1 = lora_data["results"][c]["f1_macro"]
    print(f"  {c}: F1={f1:.3f}")

if DEBUG:
    generative_classes = generative_classes[:1]
    print(f"\nDEBUG: only evaluating {len(generative_classes)} classes")


# ─── Encoding normalization ──────────────────────────
def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


# ─── Load dataset ────────────────────────────────────
print("\nLoading dataset...")
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train_split.jsonl"),
    "val": os.path.join(DATA_DIR, "val_split.jsonl"),
})


def get_text(example):
    title = example.get("designation", "") or ""
    desc = example.get("description", "") or ""
    if desc:
        return f"{title} {desc}"[:512]
    return title


def is_rare_subcat(l1_class, subcat):
    if l1_class not in subcat_routing:
        return False
    norm = normalize(subcat)
    for rare_s in subcat_routing[l1_class]["rare_subcats"]:
        if normalize(rare_s) == norm:
            return True
    return False


# ─── Group val samples by L1 class → rare subcats ────
print("Grouping val samples...")
val_data_per_class = defaultdict(lambda: defaultdict(list))

for example in ds["val"]:
    l1_class = example.get("typelabel", "")
    if l1_class not in generative_classes:
        continue
    subcat = example["producttype"][0] if example.get("producttype") else ""
    if not subcat:
        continue
    if is_rare_subcat(l1_class, subcat):
        text = get_text(example)
        if text.strip():
            val_data_per_class[l1_class][subcat].append(text)


# ─── Load generative model ───────────────────────────
print(f"\nLoading generative model: {GENERATIVE_MODEL}")
from transformers import AutoTokenizer, AutoModelForCausalLM

gen_tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    GENERATIVE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
gen_model.eval()

# Ensure pad token
if gen_tokenizer.pad_token is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token


# ─── Constrained log-prob scoring ─────────────────────
def score_candidates(text, candidates, max_text_len=300):
    """
    For each candidate label, compute log P(label | prompt).
    Returns index of highest-scoring candidate.
    
    This is CONSTRAINED: output is guaranteed to be one of the candidates.
    """
    # Truncate text to avoid exceeding context window
    text = text[:max_text_len]
    
    scores = []
    
    for candidate in candidates:
        # Build prompt with candidate appended
        prompt = (
            f"Classifie ce produit dans une des catégories.\n"
            f"Produit: {text}\n"
            f"Catégorie: {candidate}"
        )
        
        # Tokenize
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
        
        # Tokenize just the candidate to know which tokens to score
        candidate_tokens = gen_tokenizer(candidate, add_special_tokens=False)["input_ids"]
        n_candidate_tokens = len(candidate_tokens)
        
        with torch.no_grad():
            outputs = gen_model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Compute log-prob of the candidate tokens (last n tokens of the sequence)
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        
        # Score: sum of log-probs for each candidate token
        # Token at position i predicts position i+1
        total_log_prob = 0.0
        seq_len = inputs["input_ids"].shape[1]
        
        for j, token_id in enumerate(candidate_tokens):
            # Position in sequence where this token appears
            pos = seq_len - n_candidate_tokens + j
            if pos > 0:
                # Log-prob of this token given all previous tokens
                total_log_prob += log_probs[pos - 1, token_id].item()
        
        # Normalize by number of tokens (avoid bias toward short labels)
        avg_log_prob = total_log_prob / max(n_candidate_tokens, 1)
        scores.append(avg_log_prob)
    
    return int(np.argmax(scores)), scores


# ─── Evaluate per class ──────────────────────────────
print(f"\n{'='*60}")
print(f"Generative Evaluation ({len(generative_classes)} classes)")
print(f"{'='*60}\n")

results = {}
start_total = time.time()

for i, l1_class in enumerate(generative_classes):
    start_class = time.time()
    
    subcats_data = val_data_per_class[l1_class]
    
    # Get valid subcats (at least 1 val sample)
    valid_subcats = {s: texts for s, texts in subcats_data.items() if len(texts) >= 1}
    
    if len(valid_subcats) < 2:
        print(f"[{i+1}/{len(generative_classes)}] SKIP {l1_class}: only {len(valid_subcats)} val subcats")
        continue
    
    candidates = sorted(valid_subcats.keys())
    print(f"[{i+1}/{len(generative_classes)}] {l1_class} ({len(candidates)} subcats)")
    
    # Build val set
    val_texts = []
    val_labels = []
    label2id = {name: idx for idx, name in enumerate(candidates)}
    
    for subcat, texts in valid_subcats.items():
        for t in texts[:20]:  # cap at 20 per subcat for speed
            val_texts.append(t)
            val_labels.append(label2id[subcat])
    
    print(f"  Evaluating {len(val_texts)} samples against {len(candidates)} candidates...")
    
    # Score each sample
    preds = []
    for j, text in enumerate(val_texts):
        pred_idx, _ = score_candidates(text, candidates)
        preds.append(pred_idx)
        
        # Progress
        if (j + 1) % 10 == 0:
            print(f"    {j+1}/{len(val_texts)} done...")
    
    # Metrics
    val_labels_arr = np.array(val_labels)
    preds_arr = np.array(preds)
    acc = accuracy_score(val_labels_arr, preds_arr)
    f1 = f1_score(val_labels_arr, preds_arr, average="macro", zero_division=0)
    
    elapsed = time.time() - start_class
    
    # Compare with LoRA
    lora_f1 = lora_data["results"][l1_class]["f1_macro"]
    improvement = f1 - lora_f1
    
    print(f"  → acc={acc:.3f}, f1_macro={f1:.3f} (LoRA was {lora_f1:.3f}, Δ={improvement:+.3f}), time={elapsed:.1f}s")
    
    results[l1_class] = {
        "n_subcats": len(candidates),
        "n_val": len(val_texts),
        "accuracy": acc,
        "f1_macro": f1,
        "lora_f1": lora_f1,
        "improvement": improvement,
        "time_seconds": elapsed,
        "candidates": candidates,
    }
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


# ─── Summary ─────────────────────────────────────────
elapsed_total = time.time() - start_total
print(f"\n{'='*60}")
print("Generative Fallback Evaluation Complete")
print(f"{'='*60}")
print(f"Total time: {elapsed_total/60:.1f} minutes")
print(f"Evaluated: {len(results)} classes")

if results:
    gen_f1s = [r["f1_macro"] for r in results.values()]
    lora_f1s = [r["lora_f1"] for r in results.values()]
    improvements = [r["improvement"] for r in results.values()]
    
    print(f"\nGenerative mean F1: {np.mean(gen_f1s):.3f}")
    print(f"LoRA mean F1 (same classes): {np.mean(lora_f1s):.3f}")
    print(f"Mean improvement: {np.mean(improvements):+.3f}")
    
    print("\nPer-class comparison:")
    for cls, r in sorted(results.items(), key=lambda x: x[1]["improvement"], reverse=True):
        print(f"  {cls}: Gen={r['f1_macro']:.3f} vs LoRA={r['lora_f1']:.3f} (Δ={r['improvement']:+.3f})")

# ─── Save results & routing decision ─────────────────
# Determine which classes should use generative at inference time
routing_decision = {}
for cls, r in results.items():
    if r["improvement"] > 0:
        routing_decision[cls] = "generative"  # generative is better
    else:
        routing_decision[cls] = "lora"  # keep LoRA

output = {
    "model": GENERATIVE_MODEL,
    "f1_threshold": F1_THRESHOLD,
    "results": results,
    "routing_decision": routing_decision,
    "total_time": elapsed_total,
}

with open(os.path.join(OUTPUT_DIR, "generative_results.json"), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Save the list of classes that should use generative fallback
gen_classes_final = [cls for cls, route in routing_decision.items() if route == "generative"]
with open(os.path.join(OUTPUT_DIR, "generative_classes.json"), "w") as f:
    json.dump({"classes": gen_classes_final, "model": GENERATIVE_MODEL}, f, indent=2, ensure_ascii=False)

print(f"\nClasses that benefit from generative: {len(gen_classes_final)}")
print(f"Results saved to: {OUTPUT_DIR}/generative_results.json")
print("\nDone!")
