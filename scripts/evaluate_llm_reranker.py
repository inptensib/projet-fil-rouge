"""Reproducible evaluation of the LLM General Expert Reranker on hard cases.

Loads a DualExpertProductClassifier checkpoint, runs inference on the test split,
collects the samples that the difficulty router would send to the LLM, then:
  1. Runs the LLM reranker on those hard cases.
  2. Computes before/after L2 accuracy and F1 on the hard subset.
  3. Generates llm_routing_report.json with full routing statistics.

Usage (dry-run, no LLM):
    python scripts/evaluate_llm_reranker.py \\
        --checkpoint models/flat_hydra/ \\
        --split-dir /path/to/dataset \\
        --output results/llm_reranker_eval/ \\
        --provider none

Usage with a local LLM:
    python scripts/evaluate_llm_reranker.py \\
        --checkpoint models/flat_hydra/ \\
        --split-dir /path/to/dataset \\
        --output results/llm_reranker_eval/ \\
        --provider local \\
        --api-base-url http://localhost:8080/v1 \\
        --model mixtral-8x7b-instruct \\
        --category-definitions data/category_definitions/category_definitions_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_ROOT = _ROOT / "01_baselines"
for p in (_ROOT, _BASELINES_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _compute_accuracy(preds: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum(p == g for p, g in zip(preds, labels)) / len(labels)


def _compute_f1_macro(preds: list[int], labels: list[int], num_classes: int) -> float:
    """Macro-F1 averaged over classes present in ground-truth labels only.

    Averaging over all num_classes with a small eval set inflates the
    denominator with thousands of zero-F1 absent classes, making the
    metric meaningless.  Restricting to observed classes matches sklearn's
    default behaviour and gives an interpretable number.
    """
    from collections import defaultdict
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)
    for p, g in zip(preds, labels):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    observed_classes = set(labels)
    f1s = []
    for c in observed_classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM reranker on difficulty-routed hard cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="DIR",
                        help="FlatHydra checkpoint directory")
    parser.add_argument("--split-dir", required=True, metavar="DIR",
                        help="Directory with train/val/test_split.jsonl")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Output directory for report and predictions")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="Which split to evaluate on")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit number of test samples (for fast smoke runs)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of candidates passed to the LLM")
    parser.add_argument("--provider",
                        choices=["none", "local", "openai_compatible", "mistral_api", "hf_local"],
                        default="none")
    parser.add_argument("--api-base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--model", default="mixtral-8x7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--hf-load-in-4bit", action="store_true",
                        help="Quantize hf_local model to 4-bit (bitsandbytes)")
    parser.add_argument("--confidence-threshold", type=float, default=0.75,
                        help="Route to LLM when L2 confidence < threshold (default 0.75)")
    parser.add_argument("--min-l2-margin", type=float, default=0.15,
                        help="Route to LLM when top-1 − top-2 margin < threshold (default 0.15)")
    parser.add_argument("--min-parent-l1-confidence", type=float, default=0.50,
                        help="Route to LLM when parent L1 confidence < threshold (default 0.50)")
    parser.add_argument("--rare-class-threshold", type=int, default=100,
                        help="L2 classes with fewer train samples than this are considered rare (default 100)")
    parser.add_argument("--rare-class-min-confidence", type=float, default=0.90,
                        help="Rare classes must exceed this confidence to skip LLM (default 0.90)")
    parser.add_argument("--sibling-confusion-threshold", type=int, default=5,
                        help="Min siblings in top-K to flag sibling confusion (default 5)")
    # Per-rule on/off switches
    parser.add_argument("--no-hierarchy-check", action="store_true",
                        help="Disable hierarchy inconsistency routing rule")
    parser.add_argument("--no-confidence-check", action="store_true",
                        help="Disable low L2 confidence routing rule")
    parser.add_argument("--no-margin-check", action="store_true",
                        help="Disable low L2 margin routing rule")
    parser.add_argument("--no-rare-class-check", action="store_true",
                        help="Disable rare class routing rule")
    parser.add_argument("--enable-short-text-check", action="store_true",
                        help="Enable short text routing rule (off by default)")
    parser.add_argument("--no-sibling-confusion", action="store_true",
                        help="Disable sibling confusion routing rule")
    parser.add_argument("--prompt-path",
                        default=str(_ROOT / "prompts" / "llm_reranker_fr.txt"),
                        help="Path to LLM reranker prompt template")
    parser.add_argument("--category-definitions", default=None, metavar="JSONL",
                        help="Path to category_definitions.jsonl for candidate enrichment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from shared.data_loader import prepare_hierarchical_data_from_splits
    from src.dual_expert.candidate_generator import CandidateGenerationConfig
    from src.dual_expert.category_definitions import CategoryDefinitionStore
    from src.dual_expert.difficulty_router import DifficultyRouterConfig
    from src.dual_expert.dual_expert_classifier import DualExpertProductClassifier
    from src.dual_expert.llm_reranker import LLMRerankerConfig

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] Loading data from {split_dir}...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        n_samples=None,
        seed=args.seed,
    )

    if args.split == "test":
        texts = data["X_test"]
        labels_l2 = [int(l) for l in data["y_l2_test"]]
    else:
        texts = data["X_val"]
        labels_l2 = [int(l) for l in data["y_l2_val"]]

    if args.n_samples is not None and args.n_samples < len(texts):
        import random as _random
        rng = _random.Random(args.seed)
        indices = rng.sample(range(len(texts)), args.n_samples)
        texts = [texts[i] for i in indices]
        labels_l2 = [labels_l2[i] for i in indices]
        print(f"[Eval] Subsampled to {len(texts)} samples from {args.split} split.")

    num_classes_l2 = int(data["num_classes_l2"])

    definition_store = None
    if args.category_definitions:
        p = Path(args.category_definitions)
        if p.exists():
            definition_store = CategoryDefinitionStore(str(p))
            print(f"[Eval] Loaded {len(definition_store)} category definitions.")
        else:
            print(f"[Eval] Warning: category definitions not found at {p} — skipping.")

    reranker_cfg = LLMRerankerConfig(
        enabled=args.provider != "none",
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        prompt_path=args.prompt_path,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        fallback_to_domain_expert=True,
        hf_load_in_4bit=getattr(args, "hf_load_in_4bit", False),
    )
    from src.dual_expert.difficulty_router import RoutingRules, RoutingThresholds
    router_cfg = DifficultyRouterConfig(
        enabled=True,
        max_llm_pass_through_rate=1.0,  # no cap during evaluation
        thresholds=RoutingThresholds(
            min_l2_confidence=args.confidence_threshold,
            min_l2_margin=args.min_l2_margin,
            min_parent_l1_confidence=args.min_parent_l1_confidence,
            rare_class_threshold=args.rare_class_threshold,
            rare_class_min_confidence=args.rare_class_min_confidence,
            sibling_confusion_min_count=args.sibling_confusion_threshold,
        ),
        rules=RoutingRules(
            send_if_hierarchy_inconsistent=not args.no_hierarchy_check,
            send_if_low_l2_confidence=not args.no_confidence_check,
            send_if_low_l2_margin=not args.no_margin_check,
            send_if_rare_and_not_very_confident=not args.no_rare_class_check,
            send_if_short_text=args.enable_short_text_check,
            send_if_sibling_confusion=not args.no_sibling_confusion,
        ),
    )
    gen_cfg = CandidateGenerationConfig(top_k=args.top_k)

    print(f"[Eval] Loading classifier from {args.checkpoint}...")
    clf = DualExpertProductClassifier.from_checkpoint(
        args.checkpoint,
        generation_config=gen_cfg,
        router_config=router_cfg,
        reranker_config=reranker_cfg,
        definition_store=definition_store,
        l2_counts={int(k): int(v) for k, v in data.get("l2_counts_train", {}).items()},
    )

    # Two-pass evaluation: first collect routing decisions, then optionally LLM
    print(f"[Eval] Running inference on {len(texts)} {args.split} samples...")

    routed_indices: list[int] = []
    routed_labels: list[int] = []
    routed_preds_before: list[int] = []   # domain expert top-1 for routed samples
    routed_preds_after: list[int] = []    # LLM output for routed samples

    all_preds_l2: list[int] = []
    all_results: list = []

    for i, (text, label) in enumerate(zip(texts, labels_l2)):
        designation = str(text)
        product = {"designation": designation, "description": "", "product_id": str(i)}
        result = clf.predict(product)

        all_preds_l2.append(result.pred_l2_id)
        all_results.append(result)

        if result.llm_used:
            routed_indices.append(i)
            routed_labels.append(label)
            domain_top1 = getattr(result, "domain_top1_l2_id", result.pred_l2_id)
            routed_preds_before.append(domain_top1)
            routed_preds_after.append(result.pred_l2_id)

            reasons = getattr(result, "routing_reasons", [])
            correct_before = domain_top1 == label
            correct_after  = result.pred_l2_id == label
            outcome = ("✓→✓" if correct_before and correct_after else
                       "✗→✓" if not correct_before and correct_after else
                       "✓→✗" if correct_before and not correct_after else
                       "✗→✗")
            print(
                f"  [Router] #{i:>5} | {outcome} | reasons: {', '.join(reasons)} | "
                f"conf_l2={result.confidence_l2:.3f} | "
                f"text: {designation[:60]!r}"
            )

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(texts)}] routed so far: {len(routed_indices)}")

    # Global accuracy and F1
    overall_acc = _compute_accuracy(all_preds_l2, labels_l2)
    overall_f1 = _compute_f1_macro(all_preds_l2, labels_l2, num_classes_l2)

    print(f"\n[Eval] Overall accuracy: {overall_acc:.4f}")
    print(f"[Eval] Overall macro-F1 (L2): {overall_f1:.4f}")
    print(f"[Eval] Routed to LLM: {len(routed_indices)} / {len(texts)}")

    # Routing reason breakdown
    if routed_indices:
        from collections import Counter
        reason_counts: Counter = Counter()
        for result in all_results:
            if result.llm_used:
                for r in getattr(result, "routing_reasons", []):
                    reason_counts[r] += 1
        print("[Router] Reason breakdown (a sample can have multiple reasons):")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<40} {count:>4} ({count / len(routed_indices):.0%})")

    # Build routing report
    report = clf.router.build_routing_report(
        routed_labels_l2=routed_labels if routed_labels else None,
        routed_preds_before=routed_preds_before if routed_preds_before else None,
        routed_preds_after=routed_preds_after if routed_preds_after else None,
        output_path=output_dir / "llm_routing_report.json",
    )

    report["overall_accuracy"] = overall_acc
    report["overall_f1_l2_macro"] = overall_f1
    report["split"] = args.split
    report["n_evaluated"] = len(texts)
    report["provider"] = args.provider
    report["top_k"] = args.top_k

    # Re-save with overall metrics included
    with open(output_dir / "llm_routing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save per-sample predictions for further analysis
    routed_set = set(routed_indices)
    n_fallback = 0
    predictions_path = output_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for i, (pred, label, result) in enumerate(zip(all_preds_l2, labels_l2, all_results)):
            is_routed = i in routed_set
            fallback = getattr(result, "llm_fallback_used", False) if is_routed else None
            if fallback:
                n_fallback += 1
            f.write(json.dumps({
                "sample_idx": i,
                "pred_l2": pred,
                "label_l2": label,
                "correct": pred == label,
                "routed": is_routed,
                "llm_fallback_used": fallback,
                "llm_reason": getattr(result, "llm_reason", None) if is_routed else None,
                "routing_reasons": getattr(result, "routing_reasons", None) if is_routed else None,
            }) + "\n")

    if routed_indices:
        print(f"[Eval] LLM fallback rate: {n_fallback}/{len(routed_indices)} ({n_fallback/len(routed_indices):.1%})")

    print(f"\n[Eval] Results saved to {output_dir}/")
    print(f"  llm_routing_report.json")
    print(f"  predictions.jsonl")

    if routed_labels:
        acc_b = _compute_accuracy(routed_preds_before, routed_labels)
        acc_a = _compute_accuracy(routed_preds_after, routed_labels)
        print(f"\n[Eval] Hard-case subset ({len(routed_labels)} samples):")
        print(f"  Accuracy before LLM: {acc_b:.4f}")
        print(f"  Accuracy after  LLM: {acc_a:.4f}")
        print(f"  Delta:               {acc_a - acc_b:+.4f}")


if __name__ == "__main__":
    main()
