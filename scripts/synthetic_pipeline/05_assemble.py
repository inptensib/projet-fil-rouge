#!/usr/bin/env python3
"""
STEP 05 - Assemble + validate the final synthetic set (pure Python, no LLM).

Composes the blocks from steps 03 (SYNTH-GEN, SYNTH-ORPH) and 04 (SYNTH-OVS,
SYNTH-OVL) into exactly enough rows to bring every L1 and L2 class to TARGET,
without overshooting:

  per L2 class: take generated rows first, then top up with duplicated rows.
  per L1 class: after L2 top-ups, add duplicated rows only for any residual gap.

Then it validates the result: every class reaches TARGET, all ids are unique,
and no emitted row collides with val/test (exact or normalised hash).

Outputs (in $RAKUTEN_OUT):
  synthetic.jsonl       all rows to append to train_split.jsonl
  synthetic_ids.txt     manifest: <productid>\t<kind>
"""
import os, json, pickle, collections
try:
    import orjson  # fast path
except ImportError:  # stdlib fallback (orjson is only a speed optimisation)
    import json as _json

    class _OrjsonShim:
        @staticmethod
        def loads(b):
            return _json.loads(b)

        @staticmethod
        def dumps(o):
            return _json.dumps(o, ensure_ascii=False).encode("utf-8")

    orjson = _OrjsonShim()

import config as C


def load_block(name):
    path = os.path.join(C.OUT, name)
    rows = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            for raw in f:
                if raw.strip():
                    rows.append(orjson.loads(raw))
    return rows


def kind_of(pid):
    return pid.split("-")[1] if pid.startswith("SYNTH-") else "REAL"


def main():
    counts = pickle.load(open(os.path.join(C.OUT, "counts.pkl"), "rb"))
    real_l1, real_l2 = counts["l1"], counts["l2"]
    under = json.load(open(os.path.join(C.OUT, "under_target.json"), encoding="utf-8"))

    gen  = load_block("synth_gen.jsonl")
    orph = load_block("synth_orph.jsonl")
    ovs  = load_block("synth_ovs.jsonl")
    ovl  = load_block("synth_ovl.jsonl")

    gen_by = collections.defaultdict(list)
    for o in gen:
        gen_by[o["producttype"][0]].append(o)
    ovs_by = collections.defaultdict(list)
    for o in ovs:
        ovs_by[o["producttype"][0]].append(o)

    final = []
    comp = collections.Counter()
    l1_added = collections.Counter()
    short = []

    # ---- L2: generated first, duplicated to top up --------------------------
    for cls in sorted(under):
        need = under[cls]["need"]
        picked = gen_by.get(cls, [])[:need]
        if len(picked) < need:
            picked = picked + ovs_by.get(cls, [])[:need - len(picked)]
        if len(picked) < need:
            short.append(("L2", cls, need - len(picked)))
        for o in picked:
            final.append(o); comp[kind_of(o["productid"])] += 1
            l1_added[o["typelabel"]] += 1

    # ---- ORPH classes (no real rows; each block already sized to TARGET) -----
    for o in orph:
        final.append(o); comp["ORPH"] += 1
        l1_added[o["typelabel"]] += 1

    # ---- L1: add duplicated rows only for any residual deficit ---------------
    under1 = json.load(open(os.path.join(C.OUT, "under_target_l1.json"), encoding="utf-8"))
    ovl_by = collections.defaultdict(list)
    for o in ovl:
        ovl_by[o["typelabel"]].append(o)
    for lab in sorted(under1):
        have = real_l1.get(lab, 0) + l1_added.get(lab, 0)
        residual = C.TARGET - have
        if residual > 0:
            picked = ovl_by.get(lab, [])[:residual]
            if len(picked) < residual:
                short.append(("L1", lab, residual - len(picked)))
            for o in picked:
                final.append(o); comp["OVL"] += 1

    # ---- write --------------------------------------------------------------
    with open(os.path.join(C.OUT, "synthetic.jsonl"), "wb") as f:
        for o in final:
            f.write(orjson.dumps(o) + b"\n")
    with open(os.path.join(C.OUT, "synthetic_ids.txt"), "w", encoding="utf-8") as f:
        f.write("# Synthetic rows to append to train_split.jsonl (val/test frozen).\n")
        f.write("# Identifiable by the non-numeric 'SYNTH-' prefix.\n")
        for o in final:
            f.write(f"{o['productid']}\t{kind_of(o['productid'])}\n")

    # ---- validate -----------------------------------------------------------
    print("STEP 05 - composition:")
    for k in ("OVS", "GEN", "ORPH", "OVL"):
        print(f"    SYNTH-{k}: {comp.get(k, 0):,}")
    print(f"    TOTAL   : {len(final):,}")

    ids = [o["productid"] for o in final]
    assert len(ids) == len(set(ids)), "duplicate synthetic ids!"

    ev = pickle.load(open(os.path.join(C.OUT, "ev_sets.pkl"), "rb"))
    ev_ex, ev_no = ev["ev_ex"], ev["ev_no"]
    leak = 0
    final_l2 = collections.Counter()
    for o in final:
        final_l2[o["producttype"][0]] += 1
        d, de = o["designation"] or "", o["description"] or ""
        if C.hd(d + C.SEP + de) in ev_ex:
            leak += 1; continue
        nd, nde = C.norm(d), C.norm(de)
        if len(nd) + len(nde) >= 3 and C.hd(nd + C.SEP + nde) in ev_no:
            leak += 1

    bad_l2 = [(c, real_l2.get(c, 0) + final_l2.get(c, 0))
              for c in under if real_l2.get(c, 0) + final_l2.get(c, 0) < C.TARGET]

    print(f"  unique ids: OK ({len(ids):,})")
    print(f"  val/test leakage among emitted rows: {leak}")
    print(f"  L2 classes still under target: {len(bad_l2)}")
    if short:
        print(f"  ! deficits (insufficient material): {short[:10]}")
    if leak == 0 and not bad_l2 and not short:
        print("  VALIDATION PASSED")
    print(f"  -> wrote synthetic.jsonl ({len(final):,} rows) + synthetic_ids.txt in {C.OUT}")


if __name__ == "__main__":
    main()
