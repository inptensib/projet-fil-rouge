#!/usr/bin/env python3
"""
STEP 00 - Build the anti-leak hash sets, class counts, and per-class reservoirs.

This is the only step that reads the (large) real splits. It produces small
pickle artifacts that every later step consumes, so the heavy I/O happens once.

Outputs (in $RAKUTEN_OUT):
  ev_sets.pkl     {ev_ex, ev_no}  -> hashes of every val/test row (exact + normalised)
  counts.pkl      {l1, l2, meta}  -> per-class counts + (typelabel, prdtypecode) per L2
  reservoirs.pkl  {cls: [rows]}   -> up to RES_CAP surviving REAL rows per under-target L2 class
  under_target.json               -> human-readable list of L2 classes needing SYNTH-GEN

Orphan classes (zero real rows in train) cannot be discovered here - by
definition they are absent from the data. Supply them in orphan_classes.json
(see the template shipped in this folder); step 02/03 read that file directly.

Real vs synthetic rows are told apart by the productid: real ids are purely
numeric, synthetic ids carry a non-numeric 'SYNTH-' prefix.
"""
import json, pickle, os, sys, time
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


def iter_jsonl(path):
    with open(path, "rb", buffering=32 * 1024 * 1024) as f:
        for raw in f:
            if raw.strip():
                yield orjson.loads(raw)


def build_leak_sets():
    """Hash every val + test row so generated rows can be rejected if they collide."""
    ev_ex, ev_no = set(), set()
    for split in (C.VAL, C.TEST):
        if not os.path.exists(split):
            print(f"  ! missing {split} - skipping", file=sys.stderr)
            continue
        n = 0
        for o in iter_jsonl(split):
            d  = o.get("designation") or ""
            de = o.get("description") or ""
            ev_ex.add(C.hd(d + C.SEP + de))
            nd, nde = C.norm(d), C.norm(de)
            if len(nd) + len(nde) >= 3:
                ev_no.add(C.hd(nd + C.SEP + nde))
            n += 1
            if n % 200000 == 0:
                print(f"  {os.path.basename(split)}: {n:,} rows hashed")
        print(f"  {os.path.basename(split)}: {n:,} rows total")
    return ev_ex, ev_no


def scan_train(ev_ex, ev_no):
    """One streaming pass: count classes and collect reservoirs of surviving real rows."""
    l1, l2, meta = {}, {}, {}
    reservoirs = {}      # per-L2 surviving real rows (for GEN context + OVS duplication)
    res1 = {}            # per-L1 surviving real rows (for OVL duplication)
    n = leaks = 0
    t0 = time.time()
    for o in iter_jsonl(C.TRAIN):
        pid = str(o.get("productid"))
        a   = o.get("typelabel") or ""
        b   = (o.get("producttype") or [""])[0]
        l1[a] = l1.get(a, 0) + 1
        l2[b] = l2.get(b, 0) + 1
        if b and b not in meta:
            meta[b] = (a, o.get("prdtypecode"))

        # Only real (numeric id), non-leaking rows are eligible for the reservoir.
        if pid.isdigit():
            d  = o.get("designation") or ""
            de = o.get("description") or ""
            leak = C.hd(d + C.SEP + de) in ev_ex
            if not leak:
                nd, nde = C.norm(d), C.norm(de)
                if len(nd) + len(nde) >= 3 and C.hd(nd + C.SEP + nde) in ev_no:
                    leak = True
            if leak:
                leaks += 1
            else:
                row = {"productid": pid, "designation": d, "description": de,
                       "typelabel": a, "prdtypecode": o.get("prdtypecode"),
                       "producttype": [b]}
                r = reservoirs.setdefault(b, [])
                if len(r) < C.RES_CAP:
                    r.append(row)
                r1 = res1.setdefault(a, [])
                if len(r1) < C.RES_CAP:
                    r1.append(row)
        n += 1
        if n % 500000 == 0:
            print(f"  train: {n:,} rows ({time.time()-t0:.0f}s)")
    print(f"  train: {n:,} rows total, {leaks:,} real rows skipped as leak-colliding")
    return l1, l2, meta, reservoirs, res1


def main():
    print("STEP 00 - building leak sets ...")
    ev_ex, ev_no = build_leak_sets()
    with open(os.path.join(C.OUT, "ev_sets.pkl"), "wb") as f:
        pickle.dump({"ev_ex": ev_ex, "ev_no": ev_no}, f, protocol=4)
    print(f"  -> {len(ev_ex):,} exact + {len(ev_no):,} normalised hashes\n")

    print("STEP 00 - scanning train ...")
    l1, l2, meta, reservoirs, res1 = scan_train(ev_ex, ev_no)
    with open(os.path.join(C.OUT, "counts.pkl"), "wb") as f:
        pickle.dump({"l1": l1, "l2": l2, "meta": meta}, f, protocol=4)

    # Keep reservoirs only for the L2 classes that are actually under target.
    under = {cls: C.TARGET - cnt for cls, cnt in l2.items() if cnt < C.TARGET}
    reservoirs = {cls: rows for cls, rows in reservoirs.items() if cls in under}
    with open(os.path.join(C.OUT, "reservoirs.pkl"), "wb") as f:
        pickle.dump(reservoirs, f, protocol=4)

    # L1 reservoirs + deficits (for the small SYNTH-OVL oversampling track).
    under1 = {lab: C.TARGET - cnt for lab, cnt in l1.items() if cnt < C.TARGET}
    res1 = {lab: rows for lab, rows in res1.items() if lab in under1}
    with open(os.path.join(C.OUT, "l1_reservoirs.pkl"), "wb") as f:
        pickle.dump(res1, f, protocol=4)
    json.dump({lab: under1[lab] for lab in sorted(under1)},
              open(os.path.join(C.OUT, "under_target_l1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    gen = {cls: {"need": under[cls], "have_real": len(reservoirs.get(cls, [])),
                 "typelabel": meta.get(cls, ("", None))[0],
                 "prdtypecode": meta.get(cls, ("", None))[1]}
           for cls in sorted(under) if reservoirs.get(cls)}
    with open(os.path.join(C.OUT, "under_target.json"), "w", encoding="utf-8") as f:
        json.dump(gen, f, ensure_ascii=False, indent=2)

    l1_under = {k: C.TARGET - v for k, v in l1.items() if v < C.TARGET}
    print(f"\nSummary:")
    print(f"  L1 classes: {len(l1)}  (under target: {len(l1_under)})")
    print(f"  L2 classes: {len(l2)}  (under target: {len(under)})")
    print(f"  L2 needing SYNTH-GEN (have real rows): {len(gen)}")
    print(f"  L2 with no usable reservoir (treat as orphan): {len(under) - len(gen)}")
    print(f"  -> wrote ev_sets.pkl, counts.pkl, reservoirs.pkl, under_target.json in {C.OUT}")


if __name__ == "__main__":
    main()
