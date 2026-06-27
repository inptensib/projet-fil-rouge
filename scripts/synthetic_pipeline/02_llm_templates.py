#!/usr/bin/env python3
"""
STEP 02 - LLM template authoring.

The LLM writes the *shape* of synthetic listings for each class: a set of
designation templates and description templates, written with placeholders that
step 03 fills in with the keywords. The LLM never writes the final rows - it
writes reusable molds. This keeps generation cheap (one call per class) while the
actual volume is produced deterministically in Python.

Handles two kinds of class:
  - GEN  (from under_target.json): real rows exist; keywords came from step 01,
          and are passed to the LLM as context so the templates fit the class.
  - ORPH (from orphan_classes.json): NO real rows exist; the LLM invents a small
          plausible vocabulary from the class name alone and also writes it to
          keywords/<class>.json so step 03 has something to fill the molds with.

Allowed placeholders (anything else is ignored by step 03):
  {cls} {brand} {attr} {material} {term} {term2} {term3} {dim} {qty}

Outputs:
  $RAKUTEN_OUT/templates/<class>.json
  $RAKUTEN_OUT/keywords/<class>.json   (only created here for ORPH classes)
"""
import os, json
import config as C
from llm import call_json

TPL_DIR = os.path.join(C.OUT, "templates")
KW_DIR  = os.path.join(C.OUT, "keywords")
os.makedirs(TPL_DIR, exist_ok=True)
os.makedirs(KW_DIR, exist_ok=True)

PLACEHOLDERS = "{cls} {brand} {attr} {material} {term} {term2} {term3} {dim} {qty}"

SYSTEM_TPL = (
    "You design listing templates for an e-commerce catalogue (Rakuten France). "
    "Return ONLY a JSON object with two string-array fields: "
    '"designation" (8-12 short title templates) and '
    '"description" (8-12 one-to-three sentence description templates). '
    f"Use ONLY these placeholders, each may appear at most once per template: {PLACEHOLDERS}. "
    "Every designation template MUST contain {cls}. Write in French. Keep titles "
    "realistic and varied (some with brand, some with dimension, some with a 'lot de' "
    "quantity). Descriptions should read like real seller copy, not keyword spam."
)

SYSTEM_ORPH = (
    "You are bootstrapping an e-commerce category that has NO example listings. "
    "Given only the category name, return ONLY a JSON object with fields: "
    '"brands","attributes","materials","terms","dims" (short string arrays, French, '
    "plausible for this category; brands only if obviously generic, otherwise leave empty), "
    'plus "designation" (8-12 title templates) and "description" (8-12 description templates). '
    f"Templates use ONLY these placeholders, max once each: {PLACEHOLDERS}. "
    "Every designation template MUST contain {cls}. Write in French, realistic seller copy."
)


def gen_user(cls, kw):
    vocab = {k: kw.get(k, []) for k in ("brands", "attributes", "materials", "terms", "dims")}
    return (f'Category: "{cls}"\n'
            f"Characteristic vocabulary (for context):\n{json.dumps(vocab, ensure_ascii=False)}\n\n"
            "Write the designation + description templates now.")


def clean_templates(data):
    des = [str(t).strip() for t in data.get("designation", []) if "{cls}" in str(t)]
    dsc = [str(t).strip() for t in data.get("description", []) if str(t).strip()]
    return {"designation": des or ["{cls} {term} {attr}", "{cls} - {dim}", "{cls} {brand}"],
            "description": dsc or ["{cls} {attr}. {term} {term2}.", "{cls} de qualité, {material}."]}


def do_gen():
    if not os.path.exists(os.path.join(C.OUT, "under_target.json")):
        return 0
    gen = json.load(open(os.path.join(C.OUT, "under_target.json"), encoding="utf-8"))
    n = 0
    for cls in sorted(gen, key=lambda c: gen[c]["prdtypecode"] or 0):
        code = gen[cls]["prdtypecode"]
        out = os.path.join(TPL_DIR, f"{C.slug(cls)}.json")
        if os.path.exists(out):
            n += 1
            continue
        kwpath = os.path.join(KW_DIR, f"{C.slug(cls)}.json")
        kw = json.load(open(kwpath, encoding="utf-8")) if os.path.exists(kwpath) else {}
        data = clean_templates(call_json(SYSTEM_TPL, gen_user(cls, kw)))
        json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n += 1
        if n % 25 == 0:
            print(f"  GEN templates {n}/{len(gen)}")
    return n


def do_orph():
    path = C.ORPHANS
    if not os.path.exists(path):
        print("  (no orphan_classes.json - skipping ORPH templates)")
        return 0
    orph = json.load(open(path, encoding="utf-8"))
    n = 0
    for rec in orph:
        cls = rec["producttype"]
        code = rec["prdtypecode"]
        tpl_out = os.path.join(TPL_DIR, f"{C.slug(cls)}.json")
        kw_out  = os.path.join(KW_DIR, f"{C.slug(cls)}.json")
        if os.path.exists(tpl_out) and os.path.exists(kw_out):
            n += 1
            continue
        data = call_json(SYSTEM_ORPH, f'Category name: "{cls}". Produce the JSON now.')
        kw = {k: [str(x).strip() for x in data.get(k, []) if str(x).strip()]
              for k in ("brands", "attributes", "materials", "terms", "dims")}
        kw.update(_class=cls, _prdtypecode=code, _typelabel=rec["typelabel"])
        json.dump(kw, open(kw_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(clean_templates(data), open(tpl_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        n += 1
    return n


def main():
    print("STEP 02 - authoring templates ...")
    g = do_gen()
    o = do_orph()
    print(f"  -> templates for {g} GEN + {o} ORPH classes in {TPL_DIR}")


if __name__ == "__main__":
    main()
