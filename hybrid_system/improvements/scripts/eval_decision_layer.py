#!/usr/bin/env python3
"""
Decision-layer comparison for the Focal-L2 + XGBoost-v2 hybrid.

Takes the per-sample predictions/confidences already produced by
eval_threshold_sweep_combined.py and compares FOUR ways of combining the two
experts (CamemBERT focal L2 vs XGBoost v2), all on top of the IDENTICAL component
models — only the decision rule changes:

  (0) max_conf                — current standard: pick whichever raw conf is higher
  (1) calibrated max_conf     — isotonic-calibrate each conf -> P(correct) on VAL,
                                THEN pick the higher calibrated confidence  (#1)
  (2) per-L1 gate             — learn a per-L1-class accept threshold on VAL,
                                route to XGBoost only for the hard L1 groups  (#2)
  (3) learned gate            — logistic regression over calibrated features,
                                fit on VAL, evaluated on TEST                 (#6)

HONEST PROTOCOL
---------------
Every calibrator / threshold / gate is FIT on the VAL prediction cache and only
EVALUATED on the TEST prediction cache. Nothing is tuned on the test set.

Inputs (produced by the combined sweep):
  predictions_cache_combined_val.pkl   ← FIT set   (run with --split val)
  predictions_cache_combined.pkl       ← EVAL set  (test split, already exists)

Each cache holds aligned arrays: cam_preds, cam_confs, xgb_preds, xgb_confs,
true_labels (and n_samples). XGBoost conf is 0 where it was not run (cam_conf>=0.9).

Output:
  decision_layer_results.json

Usage:
    python eval_decision_layer.py
    python eval_decision_layer.py --val-cache PATH --test-cache PATH
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────
def load_cache(path):
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: cache not found: {path}\n"
            "Run eval_threshold_sweep_combined.py (test) and "
            "eval_threshold_sweep_combined.py --split val first."
        )
    with open(path, "rb") as f:
        c = pickle.load(f)
    out = {
        "cam_preds":   np.asarray(c["cam_preds"]),
        "cam_confs":   np.asarray(c["cam_confs"], dtype=np.float32),
        "xgb_preds":   np.asarray(c["xgb_preds"]),
        "xgb_confs":   np.asarray(c["xgb_confs"], dtype=np.float32),
        "true_labels": np.asarray(c["true_labels"]),
    }
    out["n"] = len(out["true_labels"])
    return out


def build_l2_to_l1(mapping_path):
    """Reverse the L1 -> [L2] mapping into L2 -> L1 (first L1 wins on ties)."""
    with open(mapping_path) as f:
        l1_to_l2 = json.load(f)
    l2_to_l1 = {}
    for l1, l2_list in l1_to_l2.items():
        for l2 in l2_list:
            l2_to_l1.setdefault(l2, l1)
    return l2_to_l1


# ─────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────
def score(true, pred):
    return {
        "accuracy":    round(float(accuracy_score(true, pred)), 5),
        "f1_macro":    round(float(f1_score(true, pred, average="macro",    zero_division=0)), 5),
        "f1_weighted": round(float(f1_score(true, pred, average="weighted", zero_division=0)), 5),
    }


def fmt(name, m, pct_xgb=None):
    extra = f"{pct_xgb:>7.1f}%" if pct_xgb is not None else " " * 8
    return (f"{name:<26} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} "
            f"{m['f1_weighted']:>9.4f} {extra}")


# ─────────────────────────────────────────────────────────
# Decision rules
# ─────────────────────────────────────────────────────────
def rule_max_conf(d):
    """(0) Pick XGBoost wherever it ran and its raw conf beats CamemBERT's."""
    preds = d["cam_preds"].copy()
    has = d["xgb_confs"] > 0.0
    override = has & (d["xgb_confs"] > d["cam_confs"])
    preds[override] = d["xgb_preds"][override]
    return preds, override


def fit_isotonic(conf, correct):
    """Map a raw top-1 confidence to P(correct) via isotonic regression."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(conf, correct.astype(np.float64))
    return iso


def rule_calibrated_max_conf(fit, ev):
    """(1) Calibrate each conf -> P(correct) on VAL, then compare calibrated."""
    # Fit calibrators on the VAL cache
    cam_correct = (fit["cam_preds"] == fit["true_labels"]).astype(np.float64)
    iso_cam = fit_isotonic(fit["cam_confs"], cam_correct)

    has_fit = fit["xgb_confs"] > 0.0
    xgb_correct = (fit["xgb_preds"][has_fit] == fit["true_labels"][has_fit]).astype(np.float64)
    iso_xgb = fit_isotonic(fit["xgb_confs"][has_fit], xgb_correct)

    # Apply on the EVAL cache
    cam_cal = iso_cam.predict(ev["cam_confs"])
    xgb_cal = np.zeros(ev["n"], dtype=np.float64)
    has = ev["xgb_confs"] > 0.0
    xgb_cal[has] = iso_xgb.predict(ev["xgb_confs"][has])

    preds = ev["cam_preds"].copy()
    override = has & (xgb_cal > cam_cal)
    preds[override] = ev["xgb_preds"][override]
    return preds, override, (iso_cam, iso_xgb)


def rule_per_l1_gate(fit, ev, l2_to_l1):
    """(2) Learn a per-L1 accept threshold on VAL; route hard L1 groups to XGB.

    For each L1 group (derived from the CamemBERT-predicted L2), pick the
    cam-conf threshold on VAL that maximizes accuracy when overriding with
    XGBoost below it (max_conf-style tie-break). Apply those thresholds on EVAL.
    """
    cand_thr = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])

    def l1_of(cam_preds):
        return np.array([l2_to_l1.get(p, "__UNK__") for p in cam_preds])

    fit_l1 = l1_of(fit["cam_preds"])
    ev_l1  = l1_of(ev["cam_preds"])
    fit_has = fit["xgb_confs"] > 0.0

    # Default threshold (global best) as fallback for L1 groups unseen in val
    def acc_for_threshold(mask_idx, thr):
        cam_p = fit["cam_preds"][mask_idx]
        cam_c = fit["cam_confs"][mask_idx]
        xgb_p = fit["xgb_preds"][mask_idx]
        xgb_c = fit["xgb_confs"][mask_idx]
        has   = fit_has[mask_idx]
        true  = fit["true_labels"][mask_idx]
        pred  = cam_p.copy()
        override = has & (cam_c < thr) & (xgb_c > cam_c)
        pred[override] = xgb_p[override]
        return (pred == true).mean()

    all_idx = np.arange(fit["n"])
    global_thr = max(cand_thr, key=lambda t: acc_for_threshold(all_idx, t))

    per_l1_thr = {}
    for l1 in np.unique(fit_l1):
        idx = all_idx[fit_l1 == l1]
        if len(idx) < 50:                      # too few to trust → use global
            per_l1_thr[l1] = float(global_thr)
            continue
        per_l1_thr[l1] = float(max(cand_thr, key=lambda t: acc_for_threshold(idx, t)))

    # Apply on EVAL
    thr_vec = np.array([per_l1_thr.get(l1, global_thr) for l1 in ev_l1])
    has = ev["xgb_confs"] > 0.0
    preds = ev["cam_preds"].copy()
    override = has & (ev["cam_confs"] < thr_vec) & (ev["xgb_confs"] > ev["cam_confs"])
    preds[override] = ev["xgb_preds"][override]
    return preds, override, {"global_thr": float(global_thr), "n_l1_groups": len(per_l1_thr)}


def _gate_features(d, iso_cam, iso_xgb):
    """Calibrated features for the learned gate (only meaningful where xgb ran)."""
    cam_cal = iso_cam.predict(d["cam_confs"])
    xgb_cal = np.zeros(d["n"], dtype=np.float64)
    has = d["xgb_confs"] > 0.0
    xgb_cal[has] = iso_xgb.predict(d["xgb_confs"][has])
    agree = (d["cam_preds"] == d["xgb_preds"]).astype(np.float64)
    margin = xgb_cal - cam_cal
    X = np.column_stack([cam_cal, xgb_cal, margin, agree])
    return X, has


def rule_learned_gate(fit, ev, calibrators):
    """(3) Logistic-regression gate over calibrated features. Fit VAL, eval TEST."""
    iso_cam, iso_xgb = calibrators
    Xf, has_f = _gate_features(fit, iso_cam, iso_xgb)
    # Train only on contested samples (where XGBoost actually ran)
    cam_ok = fit["cam_preds"] == fit["true_labels"]
    xgb_ok = fit["xgb_preds"] == fit["true_labels"]
    y = np.zeros(fit["n"], dtype=np.int32)
    y[has_f & xgb_ok & ~cam_ok] = 1            # use XGBoost only when it fixes a CAM error
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(Xf[has_f], y[has_f])

    Xe, has_e = _gate_features(ev, iso_cam, iso_xgb)
    use_xgb = np.zeros(ev["n"], dtype=bool)
    use_xgb[has_e] = clf.predict(Xe[has_e]).astype(bool)
    preds = ev["cam_preds"].copy()
    preds[use_xgb] = ev["xgb_preds"][use_xgb]
    return preds, use_xgb, clf


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cache",  default=os.path.join(BASE_DIR, "predictions_cache_combined_val.pkl"))
    ap.add_argument("--test-cache", default=os.path.join(BASE_DIR, "predictions_cache_combined.pkl"))
    ap.add_argument("--mapping",    default=os.path.join(BASE_DIR, "analysis/l1_to_l2_mapping.json"))
    ap.add_argument("--out",        default=os.path.join(BASE_DIR, "decision_layer_results.json"))
    args = ap.parse_args()

    print("=" * 72)
    print("Decision-layer comparison  (fit on VAL, evaluate on TEST)")
    print("=" * 72)
    fit = load_cache(args.val_cache)
    ev  = load_cache(args.test_cache)
    print(f"  VAL  (fit):  {fit['n']:,} samples")
    print(f"  TEST (eval): {ev['n']:,} samples")
    l2_to_l1 = build_l2_to_l1(args.mapping)
    print(f"  L2->L1 reverse map: {len(l2_to_l1):,} L2 labels")

    true = ev["true_labels"]
    n = ev["n"]
    results = {}

    print(f"\n{'rule':<26} {'accuracy':>9} {'f1_macro':>9} {'f1_wt':>9} {'%XGB':>8}")
    print("─" * 72)

    # Component baselines on EVAL (for reference)
    cam_only = score(true, ev["cam_preds"])
    xgb_only_preds = np.where(ev["xgb_confs"] > 0.0, ev["xgb_preds"], ev["cam_preds"])
    xgb_only = score(true, xgb_only_preds)
    results["camembert_only"] = cam_only
    results["xgboost_only"]   = xgb_only
    print(fmt("CamemBERT-only (focal)", cam_only))
    print(fmt("XGBoost-v2-only", xgb_only))
    print("─" * 72)

    # (0) max_conf
    p0, ov0 = rule_max_conf(ev)
    m0 = score(true, p0)
    results["max_conf"] = {**m0, "pct_xgb": round(100 * ov0.sum() / n, 2)}
    print(fmt("(0) max_conf [standard]", m0, 100 * ov0.sum() / n))

    # (1) calibrated max_conf
    p1, ov1, calibrators = rule_calibrated_max_conf(fit, ev)
    m1 = score(true, p1)
    results["calibrated_max_conf"] = {**m1, "pct_xgb": round(100 * ov1.sum() / n, 2)}
    print(fmt("(1) calibrated max_conf", m1, 100 * ov1.sum() / n))

    # (2) per-L1 gate
    p2, ov2, info2 = rule_per_l1_gate(fit, ev, l2_to_l1)
    m2 = score(true, p2)
    results["per_l1_gate"] = {**m2, "pct_xgb": round(100 * ov2.sum() / n, 2), **info2}
    print(fmt("(2) per-L1 gate", m2, 100 * ov2.sum() / n))

    # (3) learned gate
    p3, ov3, _clf = rule_learned_gate(fit, ev, calibrators)
    m3 = score(true, p3)
    results["learned_gate"] = {**m3, "pct_xgb": round(100 * ov3.sum() / n, 2)}
    print(fmt("(3) learned gate", m3, 100 * ov3.sum() / n))

    # Winner
    rules = {k: results[k] for k in ["max_conf", "calibrated_max_conf", "per_l1_gate", "learned_gate"]}
    best_acc = max(rules, key=lambda k: rules[k]["accuracy"])
    best_f1m = max(rules, key=lambda k: rules[k]["f1_macro"])
    print("─" * 72)
    print(f"BEST accuracy : {best_acc}  ({rules[best_acc]['accuracy']:.4f})")
    print(f"BEST f1_macro : {best_f1m}  ({rules[best_f1m]['f1_macro']:.4f})")
    d_acc = rules[best_acc]["accuracy"] - results["max_conf"]["accuracy"]
    d_f1m = rules[best_f1m]["f1_macro"] - results["max_conf"]["f1_macro"]
    print(f"Δ vs max_conf : acc {d_acc:+.4f}   f1_macro {d_f1m:+.4f}")

    out = {
        "n_val":  fit["n"],
        "n_test": ev["n"],
        "rules":  results,
        "best_by_accuracy": best_acc,
        "best_by_f1_macro": best_f1m,
        "standard_baseline": "max_conf (Focal L2 + XGBoost v2)",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
