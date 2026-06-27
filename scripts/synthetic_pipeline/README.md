# Synthetic-data pipeline (LLM-native) — Rakuten France rebalancing

This pipeline builds the **synthetic portion** of the rebalanced training set: the
extra rows that bring every L1 (`typelabel`, 157 classes) and every L2
(`producttype`, 5 777 classes) up to **at least 100 products in `train` only**.
Validation and test are **frozen** — no synthetic row is ever added to them, and
every generated row is checked against them so nothing leaks.

The design puts the **LLM in the loop at generation time**: the LLM reads each
class and decides its vocabulary (keywords) and its listing shapes (templates).
Python then does the mechanical work — filling those templates with those
keywords, duplicating real rows for raw volume, and validating the result.

```
LLM  ──► keywords per class      (step 01)
LLM  ──► templates per class     (step 02)
Python ─► fill templates at random with the keywords   (step 03)  → SYNTH-GEN / SYNTH-ORPH
Python ─► duplicate real rows for volume                (step 04)  → SYNTH-OVS  / SYNTH-OVL
Python ─► compose to exactly 100/class + validate       (step 05)  → synthetic.jsonl
```

## Files & execution order

| Step | File | LLM? | What it does |
|------|------|------|--------------|
| 00 | `00_build_reservoirs.py` | no | Single pass over the splits. Hashes every val/test row (exact + normalised) into anti-leak sets; counts every L1/L2 class; collects up to 100 surviving real rows per under-target class as a *reservoir*. |
| 01 | `01_llm_keywords.py` | **yes** | For each under-target class **with** real rows, sends a sample of them to the LLM and gets back a JSON vocabulary: `brands, attributes, materials, terms, dims`. |
| 02 | `02_llm_templates.py` | **yes** | For each class, the LLM writes 8–12 designation templates and 8–12 description templates using placeholders. Orphan classes (no real rows) get both a vocabulary **and** templates invented from the class name alone. |
| 03 | `03_fill_templates.py` | no | Seeded RNG fills the templates with the keywords to produce the rows. Rejects any row colliding with val/test or already produced. → `SYNTH-GEN`, `SYNTH-ORPH`. |
| 04 | `04_oversample.py` | no | Verbatim duplication of a class's own real rows (re-id'd) to reach volume cheaply. → `SYNTH-OVS` (L2), `SYNTH-OVL` (L1). |
| 05 | `05_assemble.py` | no | Per class: generated rows first, duplicated rows to top up, to **exactly** 100. Validates: every class ≥ 100, unique ids, zero val/test leakage. → `synthetic.jsonl` + `synthetic_ids.txt`. |

Config in `config.py`; orchestrate with `bash run_all.sh`. The orphan-class list
path can be overridden with `RAKUTEN_ORPHANS` (defaults to `orphan_classes.json`
next to the scripts).

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export RAKUTEN_DS=/abs/path/to/datasets_finaux     # train/val/test _split.jsonl
export RAKUTEN_OUT=./out
export ANTHROPIC_MODEL=claude-opus-4-8             # the LLM used in steps 01–02
bash run_all.sh
```

## Running without an LLM (offline / CI / smoke test)

Steps 00, 03, 04, 05 are pure Python and never touch the network. Steps 01–02
call the LLM, but the pipeline ships a deterministic **offline stub**, so the
whole chain runs end-to-end with **no API key**:

```bash
export RAKUTEN_LLM_OFFLINE=1     # steps 01–02 use a generic stub vocab/templates
bash run_all.sh
```

In offline mode the vocabulary and templates are *generic* (not anchored in the
class's real listings), so the generated **text** is lower quality — but the
correctness of steps 03–05 (anti-leak, ≥100/class, dedup, validation) is
**identical** to a real run. Use it for tests, demos, and to verify the plumbing
before spending tokens. `orjson` is optional too: a stdlib `json` fallback kicks
in automatically if it is not installed.

A tiny, fully reproducible end-to-end demo lives in `sample/` — run
`sample/make_sample.py` to write a mini `datasets_finaux/`, then run the pipeline
offline against it. See `WORKED_EXAMPLE.md` for a step-by-step trace.

## The two generation strategies

**Generation (LLM, the centrepiece).** Steps 01–03. The LLM is the component that
"detects the keywords" and "writes the templates". The keyword extraction is a
real LLM call over the class's surviving real listings (step 01); the template
authoring is a real LLM call per class (step 02). Python only *assembles* — it
slots a random brand / attribute / term / dimension into a random template with a
fixed seed (step 03). This produces fresh, non-duplicated rows and is what fills:

- **`SYNTH-GEN`** — under-target classes that still have real rows (templates are
  anchored in that class's real vocabulary).
- **`SYNTH-ORPH`** — the 29 L2 classes with **zero** real rows in train
  (`orphan_classes.json`). Here the LLM works from the class name only.

**Oversampling (duplication, the volume workhorse).** Step 04. To lift a class to
target cheaply and with a guaranteed-correct label, copy its own real,
non-leaking rows and give the copies new ids. The text is unchanged. This fills:

- **`SYNTH-OVS`** — L2 top-ups by duplication.
- **`SYNTH-OVL`** — the few L1 top-ups by duplication.

Step 05 composes the two: for each class it takes generated rows first, then uses
duplicated rows only to top up to exactly 100, so the two strategies never
overshoot. Choosing how much of each you want is just a matter of which blocks you
run (generation-only, duplication-only, or both).

## Anti-leak (val/test stay frozen)

`config.py` defines the exact normalisation used by the original audit: lowercase,
strip HTML, collapse punctuation, then an 8-byte `blake2b` digest. Step 00 hashes
every val/test row two ways — exact `(designation, description)` and normalised —
into `ev_ex` / `ev_no`. Every generated row (and every reservoir row used for
duplication) is rejected if either hash is present in those sets. Step 05
re-checks the whole emitted set and asserts **zero** leakage before writing.

## Reproducibility

The Python steps (00, 03, 04, 05) are fully deterministic: seeds `42` (SYNTH-GEN
fill) and `4242` (SYNTH-ORPH fill) are fixed in `config.py`, so the *same keyword
and template files in produce the same rows out*.

The LLM steps (01, 02) are stochastic by nature. To make a run pinnable, every LLM
response is cached on disk (`out/llm_cache/`, keyed by a hash of model + prompt):
re-running reuses the cache, so the pipeline is resumable and a fixed cache pins a
build exactly. A *fresh* run (cleared cache, or a different model) will produce
semantically equivalent but not byte-identical templates — keep the `out/` cache
with the dataset if you need to reproduce a specific build verbatim.

## Composition (target)

The rebalanced v2 training set adds **129 641** synthetic rows on top of the real
data, distributed as:

| kind | rows | how |
|------|------|-----|
| `SYNTH-OVS` | 119 510 | duplication (L2) |
| `SYNTH-GEN` | 7 044 | LLM templates, real-vocab-anchored |
| `SYNTH-ORPH` | 2 900 | LLM templates, 29 orphan classes × 100 |
| `SYNTH-OVL` | 187 | duplication (L1) |
| **total** | **129 641** | |

Synthetic rows are identifiable by the non-numeric `SYNTH-` prefix on
`productid`; all real product ids are purely numeric.

## What is and isn't the LLM

To be precise about provenance: in this pipeline the LLM does the **judgement**
work — which keywords characterise a class, and what a realistic listing for it
looks like (steps 01–02, real API calls). Python does the **mechanical** work —
random slotting, duplication, dedup, leak-checking, validation (steps 03–05, no
network). Steps 03–05 contain no LLM call by design, which is exactly what makes
the volume cheap and the output reproducible from the cached LLM artifacts.
