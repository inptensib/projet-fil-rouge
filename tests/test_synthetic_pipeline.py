"""End-to-end test of the synthetic-data pipeline in OFFLINE mode (no LLM, no network).

Proves the pipeline is correct and runnable without an Anthropic API key: the LLM
only affects template *quality*; the deterministic steps (anti-leak, >=100/class,
dedup, validation) need no LLM at all. The whole chain runs here via the
`RAKUTEN_LLM_OFFLINE` stub on a tiny temporary fixture dataset.
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_pipeline"
TARGET = 40
STEPS = [
    "00_build_reservoirs.py",
    "01_llm_keywords.py",
    "02_llm_templates.py",
    "03_fill_templates.py",
    "04_oversample.py",
    "05_assemble.py",
]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _l2(o):
    pt = o["producttype"]
    return pt[0] if isinstance(pt, list) else pt


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the full pipeline once, offline, into a temp dir; return its artefacts."""
    if not PIPE.is_dir():
        pytest.skip(f"pipeline dir not found: {PIPE}")
    work = tmp_path_factory.mktemp("synthpipe")
    ds = work / "ds"
    out = work / "out"

    env = dict(os.environ)
    env.update(
        RAKUTEN_LLM_OFFLINE="1",
        RAKUTEN_DS=str(ds),
        RAKUTEN_OUT=str(out),
        RAKUTEN_ORPHANS=str(PIPE / "sample" / "demo_orphan.json"),
        RAKUTEN_TARGET=str(TARGET),
    )

    # 1) write a mini datasets_finaux/ with the shipped sample generator
    gen = subprocess.run(
        [sys.executable, str(PIPE / "sample" / "make_sample.py"), str(ds)],
        capture_output=True, text=True, env=env,
    )
    assert gen.returncode == 0, gen.stderr

    # 2) run the six steps in order
    for step in STEPS:
        r = subprocess.run(
            [sys.executable, str(PIPE / step)],
            cwd=str(PIPE), capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, f"{step} failed:\n{r.stdout}\n{r.stderr}"

    return {
        "synthetic": _read_jsonl(out / "synthetic.jsonl"),
        "train": _read_jsonl(ds / "train_split.jsonl"),
        "frozen": _read_jsonl(ds / "val_split.jsonl") + _read_jsonl(ds / "test_split.jsonl"),
        "out": out,
    }


def test_offline_pipeline_produces_rows(pipeline_run):
    assert pipeline_run["synthetic"], "pipeline produced no synthetic rows"


def test_all_ids_unique_and_marked(pipeline_run):
    ids = [o["productid"] for o in pipeline_run["synthetic"]]
    assert len(ids) == len(set(ids)), "duplicate synthetic ids"
    assert all(str(i).startswith("SYNTH-") for i in ids), "a synthetic id is not SYNTH-prefixed"


def test_every_under_target_class_reaches_target(pipeline_run):
    real = Counter(_l2(o) for o in pipeline_run["train"])
    synth = Counter(_l2(o) for o in pipeline_run["synthetic"])
    for cls in ("Tournevis", "Arrosoir", "Niveau laser"):
        assert real[cls] + synth[cls] >= TARGET, (
            f"{cls}: {real[cls]} real + {synth[cls]} synth < {TARGET}")


def test_over_target_class_untouched(pipeline_run):
    # Marteau has 130 real rows (>= target) -> must receive zero synthetic rows
    extra = [o for o in pipeline_run["synthetic"] if _l2(o) == "Marteau"]
    assert extra == [], f"over-target class got {len(extra)} synthetic rows"


def test_no_val_test_leakage(pipeline_run):
    frozen = {(o.get("designation") or "", o.get("description") or "")
              for o in pipeline_run["frozen"]}
    for o in pipeline_run["synthetic"]:
        pair = (o.get("designation") or "", o.get("description") or "")
        assert pair not in frozen, f"synthetic row leaks into val/test: {o['productid']}"


def test_rows_stay_in_category(pipeline_run):
    for o in pipeline_run["synthetic"]:
        assert isinstance(o["producttype"], list) and o["producttype"], "producttype not a non-empty list"
        assert o.get("typelabel"), "missing typelabel (label not preserved)"


def test_manifest_matches_rows(pipeline_run):
    manifest = pipeline_run["out"] / "synthetic_ids.txt"
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    assert len(lines) == len(pipeline_run["synthetic"]), "manifest/rows length mismatch"
