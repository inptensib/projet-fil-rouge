#!/usr/bin/env python3
"""
Tail diagnostic — is a dedicated rare-class expert worth building?

Free go/no-go (CPU-only). Reuses the per-sample prediction cache from the
v3_balanced combined sweep and the per-L2 train counts. Answers:

  1. Where is F1-macro actually lost?  (frequency-stratified F1 + sample share)
  2. How much is recoverable?           (decision-recoverable vs needs-new-expert)
  3. What is the realistic ceiling?      (oracle band-fix upper bounds)

Run on the cluster:
    .venv/bin/python diagnose_tail.py
    # optional: --cache predictions_cache_combined_v3_balanced.pkl
    #           --counts analysis/l2_subcategory_counts.json
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ── args ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--cache",  default="predictions_cache_combined_v3_balanced.pkl")
ap.add_argument("--counts", default="analysis/l2_subcategory_counts.json")
args = ap.parse_args()

if not os.path.isfile(args.cache):
    sys.exit(f"ERROR: cache not found: {args.cache}\n"
             "Run the v3_balanced sweep first (it writes this pkl), or pass --cache.")

# ── load ──────────────────────────────────────────────────────────────────
print(f"Loading cache: {args.cache}")
with open(args.cache, "rb") as f:
    c = pickle.load(f)
cam_preds = np.asarray(c["cam_preds"])
cam_confs = np.asarray(c["cam_confs"], dtype=np.float32)
xgb_preds = np.asarray(c["xgb_preds"])
xgb_confs = np.asarray(c["xgb_confs"], dtype=np.float32)
true      = np.asarray(c["true_labels"])
n = len(true)
print(f"  {n:,} test samples")

counts = json.load(open(args.counts))          # L2 name -> train count

# ── reconstruct the max_conf standard prediction ──────────────────────────
xgb_has_model = xgb_confs > 0.0
final = cam_preds.copy()
override = xgb_has_model & (xgb_confs > cam_confs)
final[override] = xgb_preds[override]
print(f"  max_conf overall accuracy: {accuracy_score(true, final):.4f}  "
      f"(XGB used on {100*override.mean():.1f}% of samples)")

# ── frequency band of the TRUE label ──────────────────────────────────────
BANDS = [("ultra-rare (<10)", 0, 10),
         ("rare (10-100)",   10, 100),
         ("torso (100-1k)",  100, 1000),
         ("head (>=1k)",     1000, 10**12)]

def band_of(label):
    cnt = counts.get(label, 0)
    for name, lo, hi in BANDS:
        if lo <= cnt < hi:
            return name
    return "ultra-rare (<10)"   # count 0 / unseen -> ultra

true_band = np.array([band_of(t) for t in true])

# ── per-class F1 once, then group by band ─────────────────────────────────
labels_all = np.array(sorted(set(true) | set(final)))
_, _, f1_per_class, support = precision_recall_fscore_support(
    true, final, labels=labels_all, average=None, zero_division=0)
class_band = np.array([band_of(l) for l in labels_all])
overall_macro = f1_per_class.mean()
print(f"  overall F1-macro (check): {overall_macro:.4f}")

# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("1) WHERE F1-MACRO IS LOST  (stratified by TRUE-label frequency band)")
print("=" * 78)
print(f"{'band':<20}{'#cls':>7}{'#test':>11}{'samp%':>8}{'acc':>8}"
      f"{'F1mac':>8}{'→overall':>10}")
print("-" * 78)
N_cls_total = len(labels_all)
for name, lo, hi in BANDS:
    cls_mask = class_band == name
    smp_mask = true_band == name
    n_cls = int(cls_mask.sum())
    n_smp = int(smp_mask.sum())
    acc_b = accuracy_score(true[smp_mask], final[smp_mask]) if n_smp else 0.0
    f1_b  = f1_per_class[cls_mask].mean() if n_cls else 0.0
    contrib = f1_per_class[cls_mask].sum() / N_cls_total   # share of overall macro
    print(f"{name:<20}{n_cls:>7}{n_smp:>11,}{100*n_smp/n:>7.1f}%"
          f"{acc_b:>8.3f}{f1_b:>8.3f}{contrib:>10.4f}")
print("-" * 78)
print(f"{'TOTAL':<20}{N_cls_total:>7}{n:>11,}{100.0:>7.1f}%"
      f"{accuracy_score(true,final):>8.3f}{overall_macro:>8.3f}{overall_macro:>10.4f}")

# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("2) RECOVERABILITY  (of currently-WRONG samples, by TRUE-label band)")
print("   decision = one expert already had it (free via decision rule)")
print("   new-expert = BOTH experts wrong (only a new model could fix)")
print("   →as-head = wrong tail sample predicted as a head(>=1k) class")
print("=" * 78)
cam_right = cam_preds == true
xgb_right = (xgb_preds == true) & xgb_has_model
final_right = final == true
pred_band = np.array([band_of(p) for p in final])
print(f"{'band':<20}{'#wrong':>9}{'decision':>10}{'new-expert':>12}{'→as-head':>10}")
print("-" * 78)
for name, lo, hi in BANDS:
    smp = (true_band == name)
    wrong = smp & ~final_right
    nw = int(wrong.sum())
    if nw == 0:
        print(f"{name:<20}{0:>9}"); continue
    dec = int((wrong & (cam_right | xgb_right)).sum())     # recoverable by decision
    new = int((wrong & ~cam_right & ~xgb_right).sum())     # both wrong
    as_head = int((wrong & (pred_band == "head (>=1k)")).sum())
    print(f"{name:<20}{nw:>9,}{dec:>9,}{100*dec/nw:>4.0f}%"
          f"{new:>8,}{100*new/nw:>4.0f}%{100*as_head/nw:>9.0f}%")

# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("3) CEILING  (oracle: perfectly fix every sample whose TRUE label is in band)")
print("   upper bound on what ANY tail expert could deliver")
print("=" * 78)
base_acc, base_mac = accuracy_score(true, final), overall_macro
print(f"{'fixed bands':<28}{'acc':>9}{'Δacc':>9}{'F1mac':>9}{'Δmac':>9}")
print("-" * 78)
print(f"{'(current standard)':<28}{base_acc:>9.4f}{'—':>9}{base_mac:>9.4f}{'—':>9}")
for upto in range(1, len(BANDS)):                # ultra; ultra+rare; +torso
    fix_bands = {BANDS[i][0] for i in range(upto)}
    oracle = final.copy()
    fix = np.isin(true_band, list(fix_bands))
    oracle[fix] = true[fix]
    _, _, f1c, _ = precision_recall_fscore_support(
        true, oracle, labels=labels_all, average=None, zero_division=0)
    a, m = accuracy_score(true, oracle), f1c.mean()
    tag = "fix " + "+".join(b.split()[0] for b in
                            [BANDS[i][0] for i in range(upto)])
    print(f"{tag:<28}{a:>9.4f}{a-base_acc:>+9.4f}{m:>9.4f}{m-base_mac:>+9.4f}")

print("\nGuide: 'new-expert%' in (2) and 'Δmac' for fix-ultra/fix-ultra+rare in (3)")
print("are the go/no-go. Big Δmac with low accuracy cost = a tail expert is worth it.")
