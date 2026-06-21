"""Generate offline L2 category definitions using an LLM.

Each definition captures:
- What the category includes
- What it excludes (especially siblings under the same L1)
- Positive lexical signals
- Negative signals / traps
- Confidence level

Output: category_definitions.jsonl (one JSON per line)

Usage (dry-run, no LLM):
    python scripts/generate_category_definitions.py \\
        --split-dir /path/to/dataset \\
        --output data/category_definitions/category_definitions_v1.jsonl \\
        --dry-run

Usage with local LLM:
    python scripts/generate_category_definitions.py \\
        --split-dir /path/to/dataset \\
        --output data/category_definitions/category_definitions_v1.jsonl \\
        --provider local \\
        --api-base-url http://localhost:8080/v1 \\
        --model mixtral-8x7b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _ROOT / "01_baselines"

for p in (_ROOT, _BASELINES_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.data_loader import prepare_hierarchical_data_from_splits  # noqa: E402
from src.dual_expert.llm_reranker import parse_llm_json                # noqa: E402


def _sample_examples(
    texts: list[str],
    n: int = 20,
    seed: int = 42,
    representative: bool = False,
    confidences: list[float] | None = None,
) -> list[str]:
    """Sample up to n product texts.

    When representative=True, uses length-stratified sampling: divides texts
    into thirds by character length (short / medium / long) and samples equally
    from each stratum to capture the breadth of product descriptions in the
    category.  Falls back to random sampling if a stratum is empty.

    When confidences is provided, selects the n products with the lowest model
    confidence (hardest cases), ignoring representative and seed.
    """
    import random
    if len(texts) <= n:
        return texts
    if confidences is not None:
        # Least confident first — best examples for discriminative definitions
        sorted_pairs = sorted(zip(confidences, texts), key=lambda x: x[0])
        return [t for _, t in sorted_pairs[:n]]
    if not representative:
        return random.Random(seed).sample(texts, n)

    # Length-stratified sampling for category coverage diversity
    sorted_by_len = sorted(texts, key=len)
    third = max(1, len(sorted_by_len) // 3)
    strata = [
        sorted_by_len[:third],
        sorted_by_len[third: 2 * third],
        sorted_by_len[2 * third:],
    ]
    rng = random.Random(seed)
    per_stratum = max(1, n // 3)
    sampled: list[str] = []
    for stratum in strata:
        k = min(per_stratum, len(stratum))
        sampled.extend(rng.sample(stratum, k))
    # Fill any remaining slots from the full pool
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [t for t in texts if t not in set(sampled)]
        sampled.extend(rng.sample(pool, min(remaining, len(pool))))
    return sampled[:n]


def _compute_confidence_scores(
    texts: list[str],
    checkpoint_dir: str | Path,
    num_classes_l1: int,
    batch_size: int = 64,
) -> list[float]:
    """Run FlatHydra inference and return max L2 confidence per sample.

    Args:
        texts: Product texts (designation [+ description]).
        checkpoint_dir: FlatHydra checkpoint directory.
        num_classes_l1: Number of L1 classes (to slice unified logits).
        batch_size: Tokenization + inference batch size.

    Returns:
        List of floats in [0, 1], one per text. Lower = model was less confident.
    """
    import json
    import sys

    import torch
    import torch.nn.functional as F
    from pathlib import Path as _P
    from transformers import AutoTokenizer

    d = _P(checkpoint_dir)
    proj_root = _P(__file__).resolve().parents[1]
    baselines_root = proj_root / "01_baselines"
    for p in (proj_root, baselines_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from src.flat_model import FlatHydra  # noqa: E402

    with open(d / "config.json") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
    model = FlatHydra(
        encoder_name=cfg.get("encoder", "camembert-base"),
        num_classes_l1=cfg["num_classes_l1"],
        num_classes_l2=cfg["num_classes_l2"],
        pooling=cfg.get("pooling", "cls"),
        project_embedding=cfg.get("project_embedding", True),
        dropout=0.0,
    )
    ckpt = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    max_length = cfg.get("max_length", 256)

    confidences: list[float] = []
    n_total = len(texts)
    for i in range(0, n_total, batch_size):
        batch = [str(t) for t in texts[i: i + batch_size]]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            _, unified_logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            logits_l2 = unified_logits[:, num_classes_l1:]
            probs_l2 = F.softmax(logits_l2, dim=-1)
            max_conf = probs_l2.max(dim=-1).values
            confidences.extend(max_conf.cpu().tolist())
        if (i // batch_size + 1) % 50 == 0:
            print(f"  [Confidence] {min(i + batch_size, n_total):,}/{n_total:,}")

    return confidences


def _build_generation_prompt(
    l1_name: str,
    l2_name: str,
    sampled_products: list[str],
    sibling_names: list[str],
    prompt_template: str,
) -> str:
    products_block = "\n".join(f"- {t[:300]}" for t in sampled_products[:20])
    siblings_block = ", ".join(sibling_names[:20]) or "(aucune)"
    return prompt_template.format(
        l1_name=l1_name,
        l2_name=l2_name,
        siblings_l2_names=siblings_block,
        sampled_products=products_block,
    )


def _make_dry_run_definition(
    l2_id: int,
    l1_id: int,
    l1_name: str,
    l2_name: str,
    sampled_products: list[str],
    sibling_names: list[str],
    version: str = "v1",
) -> dict:
    """Generate a placeholder definition without calling an LLM."""
    return {
        "l2_id": l2_id,
        "l1_id": l1_id,
        "l1_name": l1_name,
        "l2_name": l2_name,
        "path": f"{l1_name} > {l2_name}",
        "definition_fr": (
            f"Catégorie {l2_name} appartenant à {l1_name}. "
            f"Regroupe des produits de type : {l2_name.lower()}. "
            f"[Définition générée automatiquement — dry-run]"
        ),
        "included_products": [l2_name],
        "excluded_products": sibling_names[:5],
        "positive_signals": [l2_name.lower()],
        "negative_signals": [],
        "ambiguities": [],
        "confidence": "low",
        "num_examples_used": len(sampled_products),
        "generated_by_model": "dry-run",
        "generation_prompt_version": version,
        "definition_version": version,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def _call_llm(
    prompt: str,
    provider: str,
    api_base_url: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
) -> str:
    """Call LLM API and return raw text response."""
    import requests
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{api_base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _validate_definition(defn: dict) -> tuple[bool, str]:
    """Return (is_valid, reason_if_invalid)."""
    if not defn.get("definition_fr"):
        return False, "empty definition_fr"
    if len(defn["definition_fr"]) < 80:
        return False, "definition_fr too short"
    if len(defn["definition_fr"]) > 800:
        return False, "definition_fr too long"
    if not defn.get("positive_signals"):
        return False, "empty positive_signals"
    if defn.get("confidence") == "low":
        return False, "low confidence"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline L2 category definitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split-dir", required=True, metavar="DIR",
                        help="Directory with train/val/test_split.jsonl")
    parser.add_argument("--output", required=True, metavar="JSONL",
                        help="Output JSONL file path")
    parser.add_argument("--prompt-path",
                        default=str(_ROOT / "prompts" / "generate_category_definition_fr.txt"),
                        help="Path to generation prompt template")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit total samples loaded (for testing)")
    parser.add_argument("--examples-per-category", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, generate placeholder definitions")
    parser.add_argument("--provider", choices=["local", "openai_compatible", "mistral_api"],
                        default="local")
    parser.add_argument("--api-base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--model", default="mixtral-8x7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip L2s already present in output file")
    parser.add_argument("--representative-sampling", action="store_true",
                        help=(
                            "Use length-stratified sampling to capture diversity "
                            "within each category (short / medium / long descriptions). "
                            "Default is random sampling."
                        ))
    parser.add_argument("--confidence-sampling", action="store_true",
                        help=(
                            "Select the N products per category where the model was "
                            "least confident (lowest max L2 probability). Produces "
                            "more discriminative definitions for ambiguous categories. "
                            "Requires --checkpoint."
                        ))
    parser.add_argument("--checkpoint", default=None, metavar="DIR",
                        help="FlatHydra checkpoint directory (required for --confidence-sampling)")
    parser.add_argument("--confidence-batch-size", type=int, default=64,
                        help="Batch size for confidence scoring inference")
    args = parser.parse_args()

    if args.confidence_sampling and not args.checkpoint:
        import argparse as _ap
        raise _ap.ArgumentTypeError("--confidence-sampling requires --checkpoint")

    # Load data
    split_dir = Path(args.split_dir)
    print(f"[GenDefs] Loading data from {split_dir}...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        n_samples=args.n_samples,
        seed=args.seed,
    )

    l1_enc = data["l1_encoder"]
    l2_enc = data["l2_encoder"]
    l2_to_l1 = data["l2_to_l1_mapping"]
    num_l2 = data["num_classes_l2"]
    num_l1 = data["num_classes_l1"]

    # Group training texts by L2
    print("[GenDefs] Grouping training examples by L2 class...")
    from collections import defaultdict
    l2_texts: dict[int, list[str]] = defaultdict(list)
    for text, l2 in zip(data["X_train"], data["y_l2_train"]):
        l2_texts[int(l2)].append(str(text))

    # Confidence scores (optional)
    l2_confidences: dict[int, list[float]] | None = None
    if args.confidence_sampling:
        print(f"[GenDefs] Computing confidence scores from {args.checkpoint}...")
        all_texts = [str(t) for t in data["X_train"]]
        all_confs = _compute_confidence_scores(
            all_texts,
            args.checkpoint,
            num_classes_l1=num_l1,
            batch_size=args.confidence_batch_size,
        )
        l2_confidences = defaultdict(list)
        for conf, l2 in zip(all_confs, data["y_l2_train"]):
            l2_confidences[int(l2)].append(conf)
        print(f"[GenDefs] Confidence scores computed for {len(all_texts):,} samples.")

    # Build L1 → siblings map
    l1_to_l2s: dict[int, list[int]] = defaultdict(list)
    for l2, l1 in l2_to_l1.items():
        l1_to_l2s[int(l1)].append(int(l2))

    # Load prompt template
    prompt_template = ""
    if not args.dry_run:
        p = Path(args.prompt_path)
        if p.exists():
            prompt_template = p.read_text(encoding="utf-8")
        else:
            print(f"[GenDefs] Prompt not found: {p} — using built-in default.")

    # Resume: load already generated l2_ids
    already_done: set[int] = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_done.add(int(rec["l2_id"]))
                except Exception:
                    pass
        print(f"[GenDefs] Resuming — {len(already_done)} L2s already done.")

    n_ok = 0
    n_fail = 0
    n_skip = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for l2_id in range(num_l2):
            if l2_id in already_done:
                n_skip += 1
                continue

            l1_id = int(l2_to_l1.get(l2_id, 0))
            l2_name = str(l2_enc.classes_[l2_id]) if l2_id < len(l2_enc.classes_) else f"L2_{l2_id}"
            l1_name = str(l1_enc.classes_[l1_id]) if l1_id < len(l1_enc.classes_) else f"L1_{l1_id}"

            texts = l2_texts.get(l2_id, [])
            confs = l2_confidences.get(l2_id) if l2_confidences is not None else None
            sampled = _sample_examples(
                texts,
                n=args.examples_per_category,
                seed=args.seed + l2_id,
                representative=args.representative_sampling,
                confidences=confs,
            )

            siblings = [
                str(l2_enc.classes_[s_l2])
                for s_l2 in l1_to_l2s.get(l1_id, [])
                if s_l2 != l2_id and s_l2 < len(l2_enc.classes_)
            ][:20]

            if args.dry_run or not prompt_template:
                defn = _make_dry_run_definition(
                    l2_id, l1_id, l1_name, l2_name, sampled, siblings, args.version
                )
            else:
                prompt = _build_generation_prompt(
                    l1_name, l2_name, sampled, siblings, prompt_template
                )
                try:
                    raw = _call_llm(
                        prompt=prompt,
                        provider=args.provider,
                        api_base_url=args.api_base_url,
                        api_key=args.api_key,
                        model_name=args.model,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        timeout=args.timeout,
                    )
                    parsed = parse_llm_json(raw)
                    valid, reason = _validate_definition(parsed)
                    defn = {
                        "l2_id": l2_id,
                        "l1_id": l1_id,
                        "l1_name": l1_name,
                        "l2_name": l2_name,
                        "path": f"{l1_name} > {l2_name}",
                        "definition_fr": parsed.get("definition_fr", ""),
                        "included_products": parsed.get("included_products", []),
                        "excluded_products": parsed.get("excluded_products", []),
                        "positive_signals": parsed.get("positive_signals", []),
                        "negative_signals": parsed.get("negative_signals", []),
                        "ambiguities": parsed.get("ambiguities", []),
                        "confidence": parsed.get("confidence", "medium"),
                        "num_examples_used": len(sampled),
                        "needs_review": not valid,
                        "review_reason": reason,
                        "generated_by_model": args.model,
                        "generation_prompt_version": args.version,
                        "definition_version": args.version,
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d"),
                    }
                    if valid:
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"  [Warn] L2={l2_id} ({l2_name}): {reason}")
                except Exception as e:
                    print(f"  [Error] L2={l2_id} ({l2_name}): {e}")
                    defn = _make_dry_run_definition(
                        l2_id, l1_id, l1_name, l2_name, sampled, siblings, args.version
                    )
                    defn["_error"] = str(e)
                    n_fail += 1

                time.sleep(0.1)  # polite rate-limiting

            out_f.write(json.dumps(defn, ensure_ascii=False) + "\n")

            if (l2_id + 1) % 100 == 0:
                print(f"  [{l2_id + 1}/{num_l2}] ok={n_ok} fail={n_fail} skip={n_skip}")

    print(f"\n[GenDefs] Done. ok={n_ok}, fail={n_fail}, skip={n_skip}")
    print(f"[GenDefs] Definitions saved to {output_path}")


if __name__ == "__main__":
    main()
