#!/usr/bin/env python3
"""
Meta-learner: train a stacking classifier on cached predictions.

Uses the predictions_cache.pkl from the threshold sweep to train a logistic
regression that learns WHEN to trust XGBoost vs CamemBERT, conditioned on:
  - cam_conf (CamemBERT confidence)
  - xgb_conf (XGBoost confidence)
  - conf_diff = xgb_conf - cam_conf
  - conf_ratio = xgb_conf / (cam_conf + 1e-8)
  - agree (whether cam and xgb predict the same class)

Trains on val split, evaluates on test split.
Can also be used in production as the final decision layer.

Expected improvement: +0.3-1% accuracy over simple max_conf rule.
"""
import os
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

# ─── Config ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CACHE_FILE  = os.path.join(PROJECT_DIR, "predictions_cache.pkl")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "checkpoints/meta_learner")
DATA_DIR    = os.path.join(PROJECT_DIR, "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load cached predictions ─────────────────────────
print("=" * 60)
print("Meta-Learner Training")
print("=" * 60)

if not os.path.exists(CACHE_FILE):
    print(f"ERROR: {CACHE_FILE} not found.")
    print("Run eval_threshold_sweep.py first to generate predictions.")
    exit(1)

print(f"Loading predictions from {CACHE_FILE}...")
with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

cam_preds   = np.array(cache["cam_preds"])
cam_confs   = np.array(cache["cam_confs"], dtype=np.float32)
xgb_preds   = np.array(cache["xgb_preds"])
xgb_confs   = np.array(cache["xgb_confs"], dtype=np.float32)
true_labels = np.array(cache["true_labels"])
n           = cache["n_samples"]

print(f"  Loaded {n:,} predictions")

# ─── Build features ──────────────────────────────────
# XGBoost was only run where cam_conf < 0.9
xgb_available = xgb_confs > 0.0
print(f"  XGBoost available for {xgb_available.sum():,} samples ({100*xgb_available.mean():.1f}%)")

# For the meta-learner, we only need to decide for samples where XGBoost was run
# (for the rest, CamemBERT is accepted at >0.9 confidence)
mask = xgb_available
print(f"  Training meta-learner on {mask.sum():,} contested samples")

# Features
cam_c = cam_confs[mask]
xgb_c = xgb_confs[mask]
agree = (cam_preds[mask] == xgb_preds[mask]).astype(np.float32)
conf_diff = xgb_c - cam_c
conf_ratio = xgb_c / (cam_c + 1e-8)
# Clip ratio to avoid extreme values
conf_ratio = np.clip(conf_ratio, 0, 20)

X = np.column_stack([cam_c, xgb_c, conf_diff, conf_ratio, agree])
feature_names = ["cam_conf", "xgb_conf", "conf_diff", "conf_ratio", "agree"]

# Target: 1 = XGBoost is correct, 0 = CamemBERT is correct (or neither)
cam_correct = cam_preds[mask] == true_labels[mask]
xgb_correct = xgb_preds[mask] == true_labels[mask]

# Decision label: which model to trust
# 0 = use CamemBERT, 1 = use XGBoost
# If both correct or both wrong, prefer CamemBERT (it's the default)
y = np.zeros(mask.sum(), dtype=np.int32)
y[xgb_correct & ~cam_correct] = 1   # XGBoost right, CamemBERT wrong → use XGBoost
# y stays 0 when: both right, both wrong, or only CamemBERT right

print(f"  Target distribution: use_cam={100*(y==0).mean():.1f}%, use_xgb={100*(y==1).mean():.1f}%")
print(f"  CamemBERT correct: {100*cam_correct.mean():.1f}%")
print(f"  XGBoost correct:   {100*xgb_correct.mean():.1f}%")

# ─── Train/val split ─────────────────────────────────
# Use first 80% as train, last 20% as held-out test for the meta-learner
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Also keep predictions for evaluation
cam_preds_contested = cam_preds[mask]
xgb_preds_contested = xgb_preds[mask]
true_contested = true_labels[mask]

# ─── Train meta-learner ───────────────────────────────
print("\nTraining Logistic Regression meta-learner...")
lr_model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
lr_model.fit(X_train, y_train)

# Cross-val on training set
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Test accuracy
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"  Test accuracy (decision correctness): {lr_acc:.4f}")

print("\nTraining Gradient Boosting meta-learner...")
gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"  Test accuracy (decision correctness): {gb_acc:.4f}")

# ─── Evaluate end-to-end on contested samples ─────────
print("\n" + "=" * 60)
print("End-to-end evaluation on contested samples (meta-learner test split)")
print("=" * 60)

# Baseline: max_conf rule
maxconf_choices = (xgb_c[split_idx:] > cam_c[split_idx:]).astype(int)
maxconf_preds = np.where(maxconf_choices == 1, xgb_preds_contested[split_idx:], cam_preds_contested[split_idx:])
maxconf_acc = accuracy_score(true_contested[split_idx:], maxconf_preds)

# LR meta-learner
lr_choices = lr_model.predict(X_test)
lr_final_preds = np.where(lr_choices == 1, xgb_preds_contested[split_idx:], cam_preds_contested[split_idx:])
lr_e2e_acc = accuracy_score(true_contested[split_idx:], lr_final_preds)

# GB meta-learner
gb_choices = gb_model.predict(X_test)
gb_final_preds = np.where(gb_choices == 1, xgb_preds_contested[split_idx:], cam_preds_contested[split_idx:])
gb_e2e_acc = accuracy_score(true_contested[split_idx:], gb_final_preds)

# CamemBERT-only on contested
cam_only_acc = accuracy_score(true_contested[split_idx:], cam_preds_contested[split_idx:])
# XGBoost-only on contested
xgb_only_acc = accuracy_score(true_contested[split_idx:], xgb_preds_contested[split_idx:])

print(f"\n  {'Strategy':<25} {'Accuracy':>10}")
print(f"  {'─'*40}")
print(f"  {'CamemBERT only':<25} {cam_only_acc:>10.4f}")
print(f"  {'XGBoost only':<25} {xgb_only_acc:>10.4f}")
print(f"  {'max_conf rule':<25} {maxconf_acc:>10.4f}")
print(f"  {'LR meta-learner':<25} {lr_e2e_acc:>10.4f}")
print(f"  {'GB meta-learner':<25} {gb_e2e_acc:>10.4f}")

# Pick best
best_model_name = "logistic_regression" if lr_e2e_acc >= gb_e2e_acc else "gradient_boosting"
best_model = lr_model if lr_e2e_acc >= gb_e2e_acc else gb_model
best_acc = max(lr_e2e_acc, gb_e2e_acc)

print(f"\n  Best meta-learner: {best_model_name} (acc={best_acc:.4f})")
print(f"  Δ vs max_conf: {100*(best_acc - maxconf_acc):+.2f}%")

# ─── Feature importance ───────────────────────────────
print(f"\n  Feature importances ({best_model_name}):")
if best_model_name == "logistic_regression":
    importances = np.abs(best_model.coef_[0])
else:
    importances = best_model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"    {name:<15} {imp:.4f}")

# ─── Full dataset evaluation ─────────────────────────
print(f"\n{'='*60}")
print("Full dataset evaluation (all 1.19M samples)")
print(f"{'='*60}")

# Apply meta-learner to ALL contested samples
X_all = np.column_stack([
    cam_confs[mask],
    xgb_confs[mask],
    xgb_confs[mask] - cam_confs[mask],
    np.clip(xgb_confs[mask] / (cam_confs[mask] + 1e-8), 0, 20),
    (cam_preds[mask] == xgb_preds[mask]).astype(np.float32),
])
meta_decisions = best_model.predict(X_all)

# Build final predictions for all samples
final_preds = cam_preds.copy()  # start with CamemBERT for all
# Override contested samples where meta-learner says "use XGBoost"
contested_indices = np.where(mask)[0]
for i, idx in enumerate(contested_indices):
    if meta_decisions[i] == 1:
        final_preds[idx] = xgb_preds[idx]

meta_full_acc = accuracy_score(true_labels, final_preds)
meta_full_f1m = f1_score(true_labels, final_preds, average="macro", zero_division=0)
meta_full_f1w = f1_score(true_labels, final_preds, average="weighted", zero_division=0)

# Compare to baselines
# max_conf baseline (full dataset)
maxconf_full_preds = cam_preds.copy()
maxconf_full_preds[mask & (xgb_confs > cam_confs)] = xgb_preds[mask & (xgb_confs > cam_confs)]
maxconf_full_acc = accuracy_score(true_labels, maxconf_full_preds)
maxconf_full_f1m = f1_score(true_labels, maxconf_full_preds, average="macro", zero_division=0)
maxconf_full_f1w = f1_score(true_labels, maxconf_full_preds, average="weighted", zero_division=0)

cam_full_acc = accuracy_score(true_labels, cam_preds)
cam_full_f1m = f1_score(true_labels, cam_preds, average="macro", zero_division=0)

print(f"\n  {'System':<25} {'Accuracy':>10} {'F1-macro':>10} {'F1-wt':>10}")
print(f"  {'─'*60}")
print(f"  {'CamemBERT only':<25} {cam_full_acc:>10.4f} {cam_full_f1m:>10.4f} {'':>10}")
print(f"  {'max_conf hybrid':<25} {maxconf_full_acc:>10.4f} {maxconf_full_f1m:>10.4f} {maxconf_full_f1w:>10.4f}")
print(f"  {'Meta-learner hybrid':<25} {meta_full_acc:>10.4f} {meta_full_f1m:>10.4f} {meta_full_f1w:>10.4f}")
print(f"\n  Δ meta vs max_conf: {100*(meta_full_acc - maxconf_full_acc):+.3f}% acc, "
      f"{100*(meta_full_f1m - maxconf_full_f1m):+.3f}% F1m")

# ─── Save ─────────────────────────────────────────────
print(f"\nSaving meta-learner to {OUTPUT_DIR}...")
with open(os.path.join(OUTPUT_DIR, "meta_learner.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({
        "model_type": best_model_name,
        "features": feature_names,
        "contested_samples": int(mask.sum()),
        "decision_accuracy": float(max(lr_acc, gb_acc)),
        "end_to_end": {
            "cam_only_acc": float(cam_full_acc),
            "max_conf_acc": float(maxconf_full_acc),
            "max_conf_f1_macro": float(maxconf_full_f1m),
            "meta_learner_acc": float(meta_full_acc),
            "meta_learner_f1_macro": float(meta_full_f1m),
            "meta_learner_f1_weighted": float(meta_full_f1w),
            "delta_acc_vs_maxconf": float(meta_full_acc - maxconf_full_acc),
            "delta_f1m_vs_maxconf": float(meta_full_f1m - maxconf_full_f1m),
        },
        "contested_evaluation": {
            "cam_only": float(cam_only_acc),
            "xgb_only": float(xgb_only_acc),
            "max_conf": float(maxconf_acc),
            "lr_meta": float(lr_e2e_acc),
            "gb_meta": float(gb_e2e_acc),
        },
    }, f, indent=2)
print("Done!")
