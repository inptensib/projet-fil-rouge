# Worked example — one GEN class, end to end

This traces a single under-target L2 class (`Tournevis`) through the whole
pipeline. Everything below is the **actual output of the offline demo**, so it is
reproducible right now with no API key:

```bash
python3 sample/make_sample.py sample/demo_ds                 # write a mini datasets_finaux/
export RAKUTEN_LLM_OFFLINE=1
export RAKUTEN_DS=sample/demo_ds
export RAKUTEN_OUT=sample/out
export RAKUTEN_ORPHANS=sample/demo_orphan.json
bash run_all.sh
```

> In offline mode the vocabulary/templates are **generic**. With a real
> `ANTHROPIC_API_KEY` (and `RAKUTEN_LLM_OFFLINE` unset), steps 01–02 anchor the
> vocabulary in the class's *real* listings, so the text is higher quality — but
> the data flow, the IDs, and the validation are exactly the ones shown here.

The mini dataset has three L2 classes plus one orphan:

| class | real rows in train | situation |
|-------|-------------------:|-----------|
| `Tournevis` | 25 | under target → **GEN + OVS** |
| `Arrosoir`  | 8 (1 dropped as a val leak) | under target → **GEN + OVS** |
| `Marteau`   | 130 | over target → **left untouched** |
| `Niveau laser` | 0 (orphan) | no real rows → **ORPH** |

---

## Step 00 — scan, deficits, reservoirs, leak sets

One streaming pass over the real splits produces:

```
train: 163 rows total, 1 real rows skipped as leak-colliding
L1 classes: 2  (under target: 1)
L2 classes: 3  (under target: 2)
L2 needing SYNTH-GEN (have real rows): 2
```

For `Tournevis`: count = 25 < 100 → **deficit = 75**, and up to 100 surviving
real rows are kept as a *reservoir* (used both as LLM context in step 01 and as
the duplication source in step 04). The `1 real row skipped` is the `Arrosoir`
row we deliberately duplicated into `val` — the anti-leak filter removed it from
the reservoir, so it can never be copied back into train.

## Step 01 — LLM keyword extraction (one call per class)

The reservoir rows are shown to the LLM, which returns the class vocabulary
(`out/keywords/<Tournevis>.json`):

```json
{
  "brands":     ["GenericPro", "MaxiLine", "EcoBasic", "Standard"],
  "attributes": ["robuste", "compact", "léger", "pratique", "résistant", "premium", "classique", "durable"],
  "materials":  ["plastique", "acier", "aluminium", "bois", "inox"],
  "terms":      ["tournevis", "tournevis pro", "tournevis standard", "modèle tournevis", "kit tournevis", "accessoire", "pièce de rechange", "équipement"],
  "dims":       ["50 cm", "30 x 20 cm", "500 g", "2 kg", "diamètre 10 cm", "1 m"]
}
```

## Step 02 — LLM template authoring (one call per class)

The LLM writes reusable **molds** with placeholders, not finished rows
(`out/templates/<Tournevis>.json`):

```json
{
  "designation": [
    "{cls} {attr} {brand}", "{cls} - {dim}", "{cls} {material} {attr}",
    "{cls} {term} {dim}", "{qty} {cls} {attr}", "{cls} {brand} {term}",
    "{cls} {attr} {material}", "{cls} {term2} {dim}"
  ],
  "description": [
    "{cls} {attr}, idéal pour un usage quotidien. {term} de qualité.",
    "Ce {cls} en {material} est {attr}. Parfait comme {term}.",
    "{cls} {brand} : {term}, {dim}.", "..."
  ]
}
```

Allowed placeholders: `{cls} {brand} {attr} {material} {term} {term2} {term3} {dim} {qty}`.
Every designation template must contain `{cls}`.

## Step 03 — deterministic fill (pure Python, seeded)

A seeded RNG (`SEED=42`) slots random keywords into a random template, **once per
row**. For one `Tournevis` row:

```
designation template : "{cls} {attr} {brand}"
  {cls}=Tournevis  {attr}=léger  {brand}=MaxiLine
  → "Tournevis léger MaxiLine"

description templates : "{cls} {brand} : {term}, {dim}." + "Découvrez le {cls} {term} : {attr}, {dim}."
  → "Tournevis GenericPro : tournevis standard, 50 cm. Découvrez le Tournevis accessoire : robuste, 30 x 20 cm."

→ {"productid": "SYNTH-GEN-000093", "designation": "...", "description": "...",
   "prdtypecode": 1620, "typelabel": "Bricolage", "producttype": ["Tournevis"]}
```

Before a row is accepted it must pass: not colliding with val/test (exact **or**
normalised hash), not already produced, and not equal to a real reservoir
designation. The label is always the source class — generated rows stay
**in-category**.

## Step 04 — oversampling by verbatim duplication

The cheapest top-up: copy the class's own real, non-leaking reservoir rows
**unchanged** and give the copies fresh `SYNTH-OVS-…` ids. The text is identical
to a real row, so the label is guaranteed correct and nothing new can leak. This
is *not* synthesis — it is resampling.

## Step 05 — compose + validate

Per class: take **generated rows first**, then top up with **duplicated rows** to
exactly the target — never overshooting. Then validate. The demo emits:

```
SYNTH-GEN : 167      (Tournevis 75 + Arrosoir 92)
SYNTH-ORPH: 100      (Niveau laser)
SYNTH-OVS : 0
TOTAL     : 267
unique ids: OK (267)
val/test leakage among emitted rows: 0
L2 classes still under target: 0
VALIDATION PASSED
```

`Marteau` (over target) gets **0** synthetic rows — the pipeline only touches
under-target classes.

## Why GEN here, OVS in production

In this demo, generation covers the whole deficit (`SYNTH-OVS: 0`) because the
generic stub vocabulary is large enough to produce that many *unique* rows. In
the real v2 build the per-class real vocabulary is narrower, so step 03 exhausts
the distinct-row space sooner and step 05 falls back to duplication — which is
why the historical composition is **≈92 % duplication / ≈8 % templates**
(119 697 vs 9 944 of 129 641 rows). This is the *"more vocabulary → more
templates"* relationship made concrete: richer vocabulary ⇒ more generated rows
⇒ less oversampling.

## Provenance of each row

Every synthetic `productid` carries a non-numeric prefix, so the four tracks stay
auditable forever:

| prefix | track | text origin |
|--------|-------|-------------|
| `SYNTH-GEN-`  | generated, class had real rows | LLM templates, real-vocab-anchored |
| `SYNTH-ORPH-` | generated, class had no real rows | LLM templates, from class name |
| `SYNTH-OVS-`  | oversampling (L2) | verbatim copy of a real row |
| `SYNTH-OVL-`  | oversampling (L1) | verbatim copy of a real row |

Real product ids are purely numeric.
