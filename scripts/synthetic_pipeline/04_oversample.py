#!/usr/bin/env python3
"""
STEP 04 - Oversampling by verbatim duplication (pure Python, no LLM).

The cheapest way to lift an under-target class to TARGET: copy its own real,
non-leaking rows (cycling through them) and re-id the copies. The text is left
untouched, so the label is guaranteed correct and nothing new can leak (the
reservoir rows were already filtered against val/test in step 00).

This is the volume workhorse. In the historical v2 build, duplication produced
the large majority of synthetic rows (SYNTH-OVS for L2, SYNTH-OVL for L1); the
LLM-generation track (steps 01-03) produced the rest. Step 05 decides, per class,
how many duplicated vs generated rows actually make it into the final set.

Outputs:
  $RAKUTEN_OUT/synth_ovs.jsonl   productid = SYNTH-OVS-<7 digits>   (L2)
  $RAKUTEN_OUT/synth_ovl.jsonl   productid = SYNTH-OVL-<7 digits>   (L1)
"""
import os, json, pickle
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


def dup_to(real_rows, need, id_fmt, seq, producttype=None):
    """Emit `need` verbatim copies of real_rows (cycled), with fresh synthetic ids.

    producttype=None  -> keep each source row's own producttype (used for L1 top-ups);
    producttype=[cls] -> force the class (used for L2 top-ups, where all rows share it).
    """
    out = []
    if not real_rows:
        return out, seq
    for i in range(need):
        src = real_rows[i % len(real_rows)]
        out.append({"productid": id_fmt % seq,
                    "designation": src["designation"],
                    "description": src["description"],
                    "prdtypecode": src["prdtypecode"],
                    "typelabel": src["typelabel"],
                    "producttype": producttype if producttype is not None
                                   else src.get("producttype", [""])})
        seq += 1
    return out, seq


def main():
    # ---- L2 oversampling (SYNTH-OVS) ----------------------------------------
    reservoirs = pickle.load(open(os.path.join(C.OUT, "reservoirs.pkl"), "rb"))
    gen = json.load(open(os.path.join(C.OUT, "under_target.json"), encoding="utf-8"))
    need_l2 = {cls: gen[cls]["need"] for cls in gen}

    seq = 1
    total = 0
    with open(os.path.join(C.OUT, "synth_ovs.jsonl"), "wb") as f:
        for cls in sorted(need_l2):
            rows, seq = dup_to(reservoirs.get(cls, []), need_l2[cls],
                               "SYNTH-OVS-%07d", seq, producttype=[cls])
            for o in rows:
                f.write(orjson.dumps(o) + b"\n")
            total += len(rows)
    print(f"STEP 04 - SYNTH-OVS: {total} duplicated rows for {len(need_l2)} L2 classes")

    # ---- L1 oversampling (SYNTH-OVL) ----------------------------------------
    res1 = pickle.load(open(os.path.join(C.OUT, "l1_reservoirs.pkl"), "rb"))
    under1 = json.load(open(os.path.join(C.OUT, "under_target_l1.json"), encoding="utf-8"))
    seq = 1
    total1 = 0
    with open(os.path.join(C.OUT, "synth_ovl.jsonl"), "wb") as f:
        for lab in sorted(under1):
            rows, seq = dup_to(res1.get(lab, []), under1[lab],
                               "SYNTH-OVL-%07d", seq, producttype=None)
            for o in rows:
                f.write(orjson.dumps(o) + b"\n")
            total1 += len(rows)
    print(f"STEP 04 - SYNTH-OVL: {total1} duplicated rows for {len(under1)} L1 classes")


if __name__ == "__main__":
    main()
