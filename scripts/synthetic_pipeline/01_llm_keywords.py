#!/usr/bin/env python3
"""
STEP 01 - LLM keyword extraction.

For every under-target L2 class that still has real surviving rows, we show the
LLM a sample of those real products and ask it to distil the vocabulary that is
*characteristic* of the class: brands, attributes/adjectives, materials, typical
nouns/terms, and dimension/quantity tokens.

The LLM is the component that "reads" the class and decides which keywords
matter. Its output is a compact JSON vocabulary, cached per class, that step 03
later recombines with the templates from step 02.

Output: $RAKUTEN_OUT/keywords/<class>.json  (one per class)
"""
import os, json, pickle
import config as C
from llm import call_json

KW_DIR = os.path.join(C.OUT, "keywords")
os.makedirs(KW_DIR, exist_ok=True)

SYSTEM = (
    "You extract product vocabulary for an e-commerce catalogue (Rakuten France). "
    "You are given real product listings that all belong to ONE category. "
    "Return ONLY a JSON object with these string-array fields: "
    '"brands", "attributes", "materials", "terms", "dims". '
    "Rules: keep items short (1-3 words), in the SAME language as the listings "
    "(mostly French), deduplicated, lowercase except brands/proper nouns. "
    "'terms' = characteristic nouns a buyer would search. "
    "'dims' = sizes/quantities/units actually seen (e.g. '50 cm', '500 g', 'lot de 3'). "
    "Do NOT invent brands that are not plausibly present. "
    "Aim for up to 25 terms, 15 attributes, 12 brands, 10 materials, 12 dims."
)


def build_user(cls, rows):
    lines = []
    for o in rows[:60]:
        d = (o.get("designation") or "").strip().replace("\n", " ")
        de = (o.get("description") or "").strip().replace("\n", " ")
        if len(de) > 240:
            de = de[:240] + "..."
        lines.append(f"- {d}" + (f"  |  {de}" if de else ""))
    listing = "\n".join(lines)
    return (f'Category name: "{cls}"\n\n'
            f"Real listings in this category:\n{listing}\n\n"
            "Extract the JSON vocabulary now.")


def main():
    gen = json.load(open(os.path.join(C.OUT, "under_target.json"), encoding="utf-8"))
    reservoirs = pickle.load(open(os.path.join(C.OUT, "reservoirs.pkl"), "rb"))

    classes = sorted(gen, key=lambda c: gen[c]["prdtypecode"] if gen[c]["prdtypecode"] is not None else 0)
    print(f"STEP 01 - extracting keywords for {len(classes)} classes ...")
    done = 0
    for cls in classes:
        code = gen[cls]["prdtypecode"]
        out = os.path.join(KW_DIR, f"{C.slug(cls)}.json")
        if os.path.exists(out):
            done += 1
            continue
        rows = reservoirs.get(cls) or []
        if not rows:
            continue
        kw = call_json(SYSTEM, build_user(cls, rows))
        # normalise shape so step 03 can rely on the fields existing
        kw = {k: [str(x).strip() for x in kw.get(k, []) if str(x).strip()]
              for k in ("brands", "attributes", "materials", "terms", "dims")}
        kw["_class"] = cls
        kw["_prdtypecode"] = code
        kw["_typelabel"] = gen[cls]["typelabel"]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False, indent=2)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(classes)} classes")
    print(f"  -> {done} keyword files in {KW_DIR}")


if __name__ == "__main__":
    main()
