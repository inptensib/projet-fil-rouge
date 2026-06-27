#!/usr/bin/env python3
"""
Thin wrapper around the Anthropic Messages API used by steps 01 and 02.

Responsibilities:
  - one place that constructs the client (reads ANTHROPIC_API_KEY)
  - force-JSON responses with defensive parsing (handles ```json fences / prose)
  - automatic retries on transient API or JSON errors
  - per-call disk caching keyed by a hash of (model, system, user), so the whole
    pipeline is resumable and re-runs cost nothing.

The LLM is called here and ONLY here. Steps 03-05 are pure Python and never
touch the network.
"""
import os, json, time, hashlib, sys
import config as C

try:
    import anthropic
except ImportError:
    anthropic = None

_CACHE_DIR = os.path.join(C.OUT, "llm_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

_client = None


def _get_client():
    global _client
    if _client is None:
        if anthropic is None:
            raise RuntimeError("pip install anthropic  (see requirements.txt)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic()
    return _client


def _cache_key(model, system, user):
    h = hashlib.sha256()
    for part in (model, system, user):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _extract_json(text):
    """Pull the first JSON object/array out of a model reply, tolerating fences/prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # fall back: span from first opening bracket to its matching last one
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if 0 <= i < j:
            return json.loads(t[i:j + 1])
    raise ValueError("no JSON found in model reply")


def _offline_stub(system, user):
    """Deterministic offline stand-in for the LLM (steps 01/02).

    Enabled by RAKUTEN_LLM_OFFLINE=1 so the whole pipeline runs end-to-end with
    NO API key (CI, demos, tests). The vocabulary/templates are generic — a real
    run (key set, flag unset) produces class-anchored vocabulary instead. The
    correctness of steps 03-05 (anti-leak, >=100/class, dedup) does not depend on
    the quality of this text, only on its SHAPE, which matches the real prompts.
    """
    import re
    m = re.search(r'[Cc]ategory(?:\s+name)?:\s*"([^"]+)"', user)
    cls = (m.group(1) if m else "Produit").strip()
    base = cls.lower()
    vocab = {
        "brands": ["GenericPro", "MaxiLine", "EcoBasic", "Standard"],
        "attributes": ["robuste", "compact", "léger", "pratique", "résistant",
                       "premium", "classique", "durable"],
        "materials": ["plastique", "acier", "aluminium", "bois", "inox"],
        "terms": [base, f"{base} pro", f"{base} standard", f"modèle {base}",
                  f"kit {base}", "accessoire", "pièce de rechange", "équipement"],
        "dims": ["50 cm", "30 x 20 cm", "500 g", "2 kg", "diamètre 10 cm", "1 m"],
    }
    templates = {
        "designation": [
            "{cls} {attr} {brand}", "{cls} - {dim}", "{cls} {material} {attr}",
            "{cls} {term} {dim}", "{qty} {cls} {attr}", "{cls} {brand} {term}",
            "{cls} {attr} {material}", "{cls} {term2} {dim}",
        ],
        "description": [
            "{cls} {attr}, idéal pour un usage quotidien. {term} de qualité.",
            "Ce {cls} en {material} est {attr}. Parfait comme {term}.",
            "{cls} {brand} : {term}, {dim}.",
            "Profitez de ce {cls} {attr}. Disponible en {dim}.",
            "{cls} {material}, très {attr}. Convient comme {term}.",
            "Découvrez le {cls} {term} : {attr}, {dim}.",
            "{cls} {attr}, fabriqué en {material}.",
            "{qty} de {cls} {attr}, {dim}.",
        ],
    }
    s = system.lower()
    if "vocabulary" in s:          # step 01 - keyword extraction
        return vocab
    if "bootstrapping" in s:       # step 02 - ORPH: vocabulary + templates
        return {**vocab, **templates}
    return templates               # step 02 - GEN templates


def call_json(system, user, model=None, max_tokens=None, temperature=None):
    """Send a prompt, return parsed JSON. Cached on disk; retried on failure."""
    if os.environ.get("RAKUTEN_LLM_OFFLINE", "").lower() in ("1", "true", "yes"):
        return _offline_stub(system, user)
    model = model or C.MODEL
    max_tokens = max_tokens or C.MAX_TOKENS
    temperature = C.TEMPERATURE if temperature is None else temperature

    key = _cache_key(model, system, user)
    cpath = os.path.join(_CACHE_DIR, key + ".json")
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)

    last = None
    for attempt in range(C.LLM_RETRIES):
        try:
            msg = _get_client().messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            data = _extract_json(text)
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            return data
        except Exception as e:  # noqa: BLE001 - API + JSON errors both retried
            last = e
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{C.LLM_RETRIES} after error: {e} (sleep {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {C.LLM_RETRIES} attempts: {last}")
