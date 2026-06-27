#!/usr/bin/env python3
"""
Central configuration for the LLM-native synthetic-data pipeline.

Everything that depends on the environment (paths, model, API key) is read from
environment variables so the pipeline is portable. Defaults are provided for a
local run.

    export ANTHROPIC_API_KEY=sk-ant-...
    export RAKUTEN_DS=/path/to/dataset/datasets_finaux   # holds the 3 splits
    export RAKUTEN_OUT=./out                              # work / cache dir
    export ANTHROPIC_MODEL=claude-opus-4-8                # the LLM used
    export RAKUTEN_LLM_OFFLINE=1                          # optional: run with no API key
"""
import os

# --- Paths -------------------------------------------------------------------
DS  = os.environ.get("RAKUTEN_DS",  os.path.join(os.path.dirname(__file__), "..", "datasets_finaux"))
OUT = os.environ.get("RAKUTEN_OUT", os.path.join(os.path.dirname(__file__), "out"))

TRAIN = os.path.join(DS, "train_split.jsonl")
VAL   = os.path.join(DS, "val_split.jsonl")
TEST  = os.path.join(DS, "test_split.jsonl")

# Orphan classes (zero real rows in train). Override to point at a custom list.
ORPHANS = os.environ.get("RAKUTEN_ORPHANS",
                         os.path.join(os.path.dirname(__file__), "orphan_classes.json"))

# --- LLM ---------------------------------------------------------------------
MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
MAX_TOKENS  = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.7"))
LLM_RETRIES = 4          # retries on transient API / JSON errors

# --- Rebalancing targets -----------------------------------------------------
TARGET   = int(os.environ.get("RAKUTEN_TARGET", "100"))   # min products per L1/L2 class
RES_CAP  = int(os.environ.get("RAKUTEN_RES_CAP", "100"))   # max real rows kept per class

# --- Determinism (Python template filling only) ------------------------------
SEED      = 42           # SYNTH-GEN  fill
ORPH_SEED = 4242         # SYNTH-ORPH fill

# --- Anti-leak normalisation (identical to the audit/purge code) -------------
import re, html, hashlib
_TAG  = re.compile(r"<[^>]+>")
_KEEP = re.compile(r"[^0-9a-zÀ-ſ]+")

def norm(s: str) -> str:
    """Lower-cased, tag-stripped, punctuation-collapsed form used for fuzzy leak checks."""
    if not s:
        return ""
    if "&" in s:
        s = html.unescape(s)
    if "<" in s:
        s = _TAG.sub(" ", s)
    return _KEEP.sub(" ", s.lower()).strip()

def hd(s: str) -> bytes:
    """8-byte blake2b digest (same parameters as the leak audit)."""
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()

def slug(cls: str) -> str:
    """Filesystem-safe, collision-free key for an L2 class name (used for cache files).
    Keyed by class, NOT prdtypecode: many L2 classes share one prdtypecode."""
    return hashlib.blake2b(cls.encode("utf-8"), digest_size=8).hexdigest()

# Field separator used when hashing a (designation, description) pair.
SEP = "\x1f"

os.makedirs(OUT, exist_ok=True)
