#!/usr/bin/env python3
"""
STEP 03 - Deterministic template filling (pure Python, no LLM).

Takes the LLM keywords (step 01/02) and the LLM templates (step 02) and produces
the actual synthetic rows by randomly slotting keywords into templates with a
seeded RNG. Because the seed is fixed, this step is fully reproducible: same
keyword/template files in -> identical rows out.

Each generated row is checked against the val/test leak sets and against rows
already produced (and the real reservoir designations) before being accepted.

Outputs:
  $RAKUTEN_OUT/synth_gen.jsonl    productid = SYNTH-GEN-<6 digits>
  $RAKUTEN_OUT/synth_orph.jsonl   productid = SYNTH-ORPH-<7 digits>
"""
import os, re, json, pickle, random
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

PLACE = re.compile(r"\{(\w+)\}")
WS = re.compile(r"\s{2,}")
DANGLE = re.compile(r"(^[\s,;:\-–—]+)|([\s,;:\-–—]+$)")
GAP = re.compile(r"\s+([,.;:])")           # space before punctuation
EMPTY_PAREN = re.compile(r"\(\s*\)|\[\s*\]")

KW_DIR  = os.path.join(C.OUT, "keywords")
TPL_DIR = os.path.join(C.OUT, "templates")


def tidy(s):
    s = EMPTY_PAREN.sub(" ", s)
    s = GAP.sub(r"\1", s)
    s = WS.sub(" ", s).strip()
    s = DANGLE.sub("", s).strip()
    s = re.sub(r"([,.;:\-–—])\1+", r"\1", s)   # collapse doubled punctuation
    # sentence-case: capitalise the first letter after . ! ? (templates may start a clause with a keyword)
    s = re.sub(r"([.!?]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s[:1].upper() + s[1:] if s else s


def make_filler(cls, kw, rng):
    pools = {
        "cls": [cls],
        "brand": kw.get("brands", []),
        "attr": kw.get("attributes", []),
        "material": kw.get("materials", []),
        "term": kw.get("terms", []),
        "dim": kw.get("dims", []),
    }

    def fill(template):
        used_terms = []

        def repl(m):
            name = m.group(1)
            if name in ("term2", "term3"):
                choices = [t for t in pools["term"] if t not in used_terms]
                if not choices:
                    return ""
                v = rng.choice(choices); used_terms.append(v); return v
            if name == "qty":
                return "Lot de %d" % rng.choice([2, 3, 4, 5, 6, 10])
            pool = pools.get(name)
            if not pool:
                return ""
            v = rng.choice(pool)
            if name == "term":
                used_terms.append(v)
            return v

        return tidy(PLACE.sub(repl, template))

    return fill


def gen_class(cls, code, typelabel, need, ev_ex, ev_no, used, reservoir_norms,
              rng, id_fmt, seq):
    kwp = os.path.join(KW_DIR, f"{C.slug(cls)}.json")
    tpp = os.path.join(TPL_DIR, f"{C.slug(cls)}.json")
    if not (os.path.exists(kwp) and os.path.exists(tpp)):
        return [], seq, need
    kw = json.load(open(kwp, encoding="utf-8"))
    tpl = json.load(open(tpp, encoding="utf-8"))
    des_t, dsc_t = tpl["designation"], tpl["description"]
    if not des_t or not dsc_t:
        return [], seq, need
    fill = make_filler(cls, kw, rng)

    out, tries, limit = [], 0, need * 80
    while len(out) < need and tries < limit:
        tries += 1
        desig = fill(rng.choice(des_t))
        desc = " ".join(fill(rng.choice(dsc_t)) for _ in range(rng.randint(1, 2)))
        desc = tidy(desc)
        if len(desig) < 3 or len(desc) < 3:
            continue
        ex = C.hd(desig + C.SEP + desc)
        nd, nde = C.norm(desig), C.norm(desc)
        no = C.hd(nd + C.SEP + nde)
        if ex in ev_ex or no in ev_no or ex in used or nd in reservoir_norms:
            continue
        used.add(ex)
        out.append({"productid": id_fmt % seq, "designation": desig,
                    "description": desc, "prdtypecode": code,
                    "typelabel": typelabel, "producttype": [cls]})
        seq += 1
    return out, seq, need - len(out)


def main():
    ev = pickle.load(open(os.path.join(C.OUT, "ev_sets.pkl"), "rb"))
    ev_ex, ev_no = ev["ev_ex"], ev["ev_no"]
    reservoirs = pickle.load(open(os.path.join(C.OUT, "reservoirs.pkl"), "rb"))
    used = set()
    deficits = []

    # ---- SYNTH-GEN -----------------------------------------------------------
    gen = json.load(open(os.path.join(C.OUT, "under_target.json"), encoding="utf-8"))
    rng = random.Random(C.SEED)
    seq = 1
    total_gen = 0
    with open(os.path.join(C.OUT, "synth_gen.jsonl"), "wb") as f:
        for cls in sorted(gen, key=lambda c: gen[c]["prdtypecode"] or 0):
            code = gen[cls]["prdtypecode"]
            need = gen[cls]["need"]
            rnorms = {C.norm(o["designation"] or "") for o in reservoirs.get(cls, [])}
            rows, seq, miss = gen_class(cls, code, gen[cls]["typelabel"], need,
                                        ev_ex, ev_no, used, rnorms, rng,
                                        "SYNTH-GEN-%06d", seq)
            for o in rows:
                f.write(orjson.dumps(o) + b"\n")
            total_gen += len(rows)
            if miss > 0:
                deficits.append(("GEN", cls, miss))
    print(f"STEP 03 - SYNTH-GEN: {total_gen} rows for {len(gen)} classes")

    # ---- SYNTH-ORPH ----------------------------------------------------------
    opath = C.ORPHANS
    total_orph = 0
    if os.path.exists(opath):
        orph = json.load(open(opath, encoding="utf-8"))
        rng = random.Random(C.ORPH_SEED)
        seq = 1
        with open(os.path.join(C.OUT, "synth_orph.jsonl"), "wb") as f:
            for rec in orph:
                cls, code = rec["producttype"], rec["prdtypecode"]
                rows, seq, miss = gen_class(cls, code, rec["typelabel"], C.TARGET,
                                            ev_ex, ev_no, used, set(), rng,
                                            "SYNTH-ORPH-%07d", seq)
                for o in rows:
                    f.write(orjson.dumps(o) + b"\n")
                total_orph += len(rows)
                if miss > 0:
                    deficits.append(("ORPH", cls, miss))
        print(f"STEP 03 - SYNTH-ORPH: {total_orph} rows for {len(orph)} classes")
    else:
        print("STEP 03 - no orphan_classes.json, skipping SYNTH-ORPH")

    if deficits:
        print(f"  ! {len(deficits)} classes under target (template space too small):")
        for kind, cls, miss in deficits[:10]:
            print(f"    {kind}: {cls} short by {miss}")
    else:
        print("  all classes reached their target")


if __name__ == "__main__":
    main()
