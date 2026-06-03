#!/usr/bin/env python3
"""
Analyze L1→L2 gating thresholds for the hybrid routing strategy.
No GPU needed — reads only the JSON analysis files.

Outputs a table showing how many L1 classes fall into each bucket
(SetFit / CamemBERT / XGBoost) for different threshold combinations.

Usage:
    python analyze_gating.py
"""
import json
import unicodedata
from collections import defaultdict

BASE_DIR = "."

# ─── Load data ────────────────────────────────────────
with open(f"{BASE_DIR}/analysis/l1_to_l2_mapping.json") as f:
    l1_to_l2 = json.load(f)

with open(f"{BASE_DIR}/analysis/l2_subcategory_counts.json") as f:
    raw_counts = json.load(f)

with open(f"{BASE_DIR}/analysis/l1_class_difficulty.json") as f:
    l1_difficulty = json.load(f)

# ─── Normalize encoding duplicates ───────────────────
def normalize(s):
    """Collapse accented + corrupted variants to same key."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


# Merge counts for encoding duplicates
canonical_counts = defaultdict(int)
canonical_name = {}

for name, count in raw_counts.items():
    norm = normalize(name)
    canonical_counts[norm] += count
    if norm not in canonical_name:
        canonical_name[norm] = name


def get_count(subcat_name):
    return canonical_counts.get(normalize(subcat_name), 0)


# ─── Per-L1 class stats ──────────────────────────────
print(f"{'L1 class':<40} {'#subcats':>8} {'min_cnt':>8} {'median':>8} {'max_cnt':>8} {'L1_acc':>7}")
print("=" * 85)

class_stats = {}
for l1_class, subcats in sorted(l1_to_l2.items()):
    # Deduplicate encoding variants within this L1 class
    seen_norms = set()
    unique_subcats = []
    for s in subcats:
        norm = normalize(s)
        if norm not in seen_norms:
            seen_norms.add(norm)
            unique_subcats.append(s)

    counts = [get_count(s) for s in unique_subcats]
    n_subcats = len(unique_subcats)
    min_c = min(counts) if counts else 0
    max_c = max(counts) if counts else 0
    median_c = sorted(counts)[len(counts) // 2] if counts else 0
    l1_acc = l1_difficulty.get(l1_class, -1)

    class_stats[l1_class] = {
        "n_subcats": n_subcats,
        "min": min_c,
        "max": max_c,
        "median": median_c,
        "l1_acc": l1_acc,
        "counts": counts,
    }
    print(f"{l1_class:<40} {n_subcats:>8} {min_c:>8} {median_c:>8} {max_c:>8} {l1_acc:>7.3f}")

# ─── Encoding duplicates report ──────────────────────
print("\n\n── ENCODING DUPLICATES DETECTED ──")
dup_groups = defaultdict(list)
for name in raw_counts:
    norm = normalize(name)
    dup_groups[norm].append(name)

n_dupes = 0
for norm, variants in sorted(dup_groups.items()):
    if len(variants) > 1:
        n_dupes += 1
        counts_str = ", ".join(f'"{v}" ({raw_counts[v]})' for v in variants)
        print(f"  {counts_str}")

print(f"\nTotal duplicate groups: {n_dupes}")

# ─── Gating bucket analysis ──────────────────────────
print("\n\n── GATING BUCKET ANALYSIS ──")
print(f"Thresholds: T_rare = min sample count to be 'rare', T_card = subcat count to be 'high cardinality'")
print(f"Confidence gate for L1 classes with accuracy < 0.10 → generative fallback\n")

T_rare_values = [20, 50, 100, 200, 500]
T_card_values = [20, 30, 50, 80]

header = f"{'T_rare':<8} {'T_card':<8} {'generative':>11} {'setfit':>8} {'camembert':>11} {'xgboost':>9} {'total':>7}"
print(header)
print("-" * len(header))

for T_rare in T_rare_values:
    for T_card in T_card_values:
        generative = setfit = camembert = xgboost = 0
        for l1_class, stats in class_stats.items():
            # Gate 1: L1 routing broken → generative fallback
            if stats["l1_acc"] < 0.10:
                generative += 1
                continue
            # Gate 2: has rare subcategories → SetFit
            has_rare = stats["min"] < T_rare
            if has_rare:
                setfit += 1
                continue
            # Gate 3: high cardinality → CamemBERT
            if stats["n_subcats"] > T_card:
                camembert += 1
            else:
                xgboost += 1

        total = generative + setfit + camembert + xgboost
        print(f"{T_rare:<8} {T_card:<8} {generative:>11} {setfit:>8} {camembert:>11} {xgboost:>9} {total:>7}")

# ─── Detailed routing for best candidate thresholds ──
print("\n\n── DETAILED ROUTING (T_rare=50) ──")
T_rare = 50

# ─── L1-level routing ────────────────────────────────
# generative: L1 acc < 10%
# xgboost: all subcats >= T_rare AND low cardinality (already well-served)
# camembert+setfit: mixed classes — CamemBERT handles common subcats,
#                   SetFit handles rare subcats within same L1 class

buckets = {"generative": [], "camembert_setfit": [], "xgboost": []}

for l1_class, stats in sorted(class_stats.items()):
    if stats["l1_acc"] < 0.10:
        buckets["generative"].append(l1_class)
    elif stats["min"] >= T_rare:
        buckets["xgboost"].append(l1_class)
    else:
        buckets["camembert_setfit"].append(l1_class)

for bucket, classes in buckets.items():
    print(f"\n[{bucket.upper()}] ({len(classes)} classes):")
    for c in classes:
        s = class_stats[c]
        print(f"  {c:<40} subcats={s['n_subcats']:>3}  min={s['min']:>5}  L1_acc={s['l1_acc']:.3f}")

# ─── Per-subcat routing within mixed classes ─────────
print("\n\n── SUBCAT-LEVEL ROUTING (within camembert_setfit classes) ──")

# For each L1 class in the mixed bucket, identify which subcats are rare
rare_subcats_per_class = {}  # l1_class → list of rare subcat names
common_subcats_per_class = {}  # l1_class → list of common subcat names

total_rare = 0
total_common = 0

for l1_class in buckets["camembert_setfit"]:
    subcats = l1_to_l2[l1_class]
    # Deduplicate
    seen_norms = set()
    unique_subcats = []
    for s in subcats:
        norm = normalize(s)
        if norm not in seen_norms:
            seen_norms.add(norm)
            unique_subcats.append(s)

    rare = [s for s in unique_subcats if get_count(s) < T_rare]
    common = [s for s in unique_subcats if get_count(s) >= T_rare]
    rare_subcats_per_class[l1_class] = rare
    common_subcats_per_class[l1_class] = common
    total_rare += len(rare)
    total_common += len(common)

print(f"\nTotal rare subcats (< {T_rare} samples): {total_rare}")
print(f"Total common subcats (≥ {T_rare} samples): {total_common}")
print(f"Ratio: {total_rare}/{total_rare+total_common} = {total_rare/(total_rare+total_common)*100:.1f}% of subcats are rare")

# Show top 10 classes by number of rare subcats
print(f"\nTop 15 classes by rare subcat count:")
sorted_by_rare = sorted(rare_subcats_per_class.items(), key=lambda x: len(x[1]), reverse=True)
for l1_class, rare_list in sorted_by_rare[:15]:
    n_common = len(common_subcats_per_class[l1_class])
    print(f"  {l1_class:<40} rare={len(rare_list):>4}  common={n_common:>4}")

# Show examples of rare subcats
print(f"\nExample rare subcats (from top classes):")
for l1_class, rare_list in sorted_by_rare[:5]:
    print(f"\n  [{l1_class}] ({len(rare_list)} rare subcats):")
    for s in rare_list[:8]:
        print(f"    {s:<40} count={get_count(s):>5}")
    if len(rare_list) > 8:
        print(f"    ... and {len(rare_list)-8} more")

# ─── Save routing config ─────────────────────────────
routing_table = {}
subcat_routing = {}  # l1_class → {subcat: "camembert"|"setfit"}

for l1_class in class_stats:
    stats = class_stats[l1_class]
    if stats["l1_acc"] < 0.10:
        routing_table[l1_class] = "generative"
    elif stats["min"] >= T_rare:
        routing_table[l1_class] = "xgboost"
    else:
        routing_table[l1_class] = "camembert_setfit"
        subcat_routing[l1_class] = {
            "rare_subcats": rare_subcats_per_class[l1_class],
            "common_subcats": common_subcats_per_class[l1_class],
        }

output = {
    "thresholds": {"T_rare": T_rare, "L1_acc_dead": 0.10},
    "architecture": {
        "description": "L1 CamemBERT routes to L1 class. Then: "
                       "xgboost classes use existing XGBoost models. "
                       "camembert_setfit classes use CamemBERT L2 flat for common subcats, "
                       "SetFit fallback for rare subcats (when CamemBERT predicts rare or low confidence). "
                       "generative classes use LLM fallback.",
        "inference_flow": [
            "1. L1 CamemBERT → predicted L1 class + confidence",
            "2. If L1 confidence < threshold → generative fallback",
            "3. If L1 class in xgboost bucket → XGBoost predict",
            "4. If L1 class in camembert_setfit bucket → CamemBERT L2 predict",
            "   4a. If CamemBERT predicts a common subcat with high confidence → accept",
            "   4b. If CamemBERT predicts a rare subcat OR low confidence → SetFit predict over rare subcats",
        ],
    },
    "routing": routing_table,
    "subcat_routing": subcat_routing,
    "summary": {
        "generative": len(buckets["generative"]),
        "camembert_setfit": len(buckets["camembert_setfit"]),
        "xgboost": len(buckets["xgboost"]),
        "total_rare_subcats": total_rare,
        "total_common_subcats": total_common,
    },
}

with open(f"{BASE_DIR}/analysis/gating_config.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n\nSaved routing config to analysis/gating_config.json")
