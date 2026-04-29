"""Évalue les modèles HYDRA avec et sans conditioning L2 sur L1 prédit.

Compare les métriques (F1 L1, F1 L2, HCR, constrained F1) pour chaque
variante (local, nested, global) en mode standard vs hiérarchique.

Usage:
    python scripts/eval_hierarchy.py --split-dir /path/to/splits
    python scripts/eval_hierarchy.py --split-dir /path/to/splits --variant local
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / "02_hydra"))

from src.trainer import HYDRATrainer
from shared.data_loader import prepare_hierarchical_data_from_splits


VARIANTS = ["local", "nested", "global"]


def _map_config(cfg: dict) -> dict:
    """Map YAML config keys to HYDRATrainer constructor kwargs."""
    kwargs = {}
    for key in ("batch_size", "epochs", "lr", "weight_decay", "max_length", "seed"):
        if key in cfg:
            kwargs[key] = cfg[key]
    if "encoder" in cfg:
        encoder_map = {
            "camembert-base": "CamemBERT",
            "xlm-roberta-base": "XLM-RoBERTa",
            "bert-base-multilingual-cased": "mBERT",
        }
        kwargs["encoder_name"] = encoder_map.get(cfg["encoder"], cfg["encoder"])
    if "gradient_clip" in cfg:
        kwargs["max_grad_norm"] = cfg["gradient_clip"]
    if "level_weights" in cfg and len(cfg["level_weights"]) >= 1:
        kwargs["level_weight_l1"] = cfg["level_weights"][0]
    if "loss" in cfg:
        kwargs["use_focal"] = cfg["loss"] == "focal"
    return kwargs


def evaluate_variant(
    variant: str,
    data: dict,
    cfg_kwargs: dict,
    eval_batch_size: int = 1024,
    model_dir: str = "models",
    model_suffix: str = "",
) -> dict:
    """Évalue une variante avec et sans conditioning hiérarchique."""
    model_path = Path(_ROOT) / f"{model_dir}/HYDRA_{variant}_CamemBERT{model_suffix}/best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} introuvable")
        return {}

    # Créer le trainer (check_hierarchy=False d'abord)
    cfg_kwargs.pop("epochs", None)
    trainer = HYDRATrainer(
        data=data, variant=variant, check_hierarchy=False,
        epochs=1, **cfg_kwargs,
    )

    # Charger les poids du meilleur modèle
    ckpt = torch.load(model_path, map_location=trainer.device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    trainer._base_model.load_state_dict(state_dict)
    print(f"  Modèle chargé: {model_path.name}")

    # Override batch_size pour l'éval : pas de gradient → VRAM disponible bien supérieure
    trainer.batch_size = eval_batch_size
    print(f"  Eval batch_size: {eval_batch_size} (DataParallel: {trainer.device})")

    test_loader = trainer._create_dataloader(
        data["X_test"], data["y_l1_test"], data["y_l2_test"], shuffle=False,
    )

    # Évaluation SANS conditioning
    print("  Évaluation SANS conditioning L2...")
    t0 = time.time()
    trainer.check_hierarchy = False
    metrics_base = trainer.evaluate(test_loader)
    t_base = time.time() - t0

    # Évaluation AVEC conditioning
    print("  Évaluation AVEC conditioning L2 sur L1 prédit...")
    t0 = time.time()
    trainer.check_hierarchy = True
    metrics_hier = trainer.evaluate(test_loader)
    t_hier = time.time() - t0

    # Nettoyer les clés internes
    for m in (metrics_base, metrics_hier):
        for k in list(m.keys()):
            if k.startswith("_"):
                del m[k]

    return {
        "base": metrics_base,
        "hier": metrics_hier,
        "time_base": t_base,
        "time_hier": t_hier,
    }


def print_results(all_results: dict):
    """Affiche le tableau comparatif."""
    keys = ["f1_l1_macro", "f1_l2_macro", "hcr", "constrained_f1_macro"]
    labels = ["F1 L1", "F1 L2", "HCR", "Constr. F1"]

    header = f"{'Variante':<16} {'Mode':<10}"
    for lbl in labels:
        header += f" {lbl:>12}"
    header += f" {'Temps':>8}"

    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  Comparaison avec/sans conditioning L2 sur L1 prédit")
    print(sep)
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        if variant not in all_results:
            continue
        r = all_results[variant]
        for mode, label, t_key in [("base", "Sans", "time_base"), ("hier", "Avec", "time_hier")]:
            m = r[mode]
            line = f"HYDRA {variant.capitalize():<10} {label:<10}"
            for k in keys:
                v = m.get(k)
                line += f" {v*100:>11.2f}%" if v is not None else f" {'--':>12}"
            line += f" {r[t_key]:>7.0f}s"
            print(line)
        # Deltas
        delta_line = f"{'':16} {'Δ':<10}"
        for k in keys:
            vb = r["base"].get(k)
            vh = r["hier"].get(k)
            if vb is not None and vh is not None:
                d = (vh - vb) * 100
                sign = "+" if d >= 0 else ""
                delta_line += f" {sign}{d:>10.2f}%"
            else:
                delta_line += f" {'--':>12}"
        delta_line += f" {'':>8}"
        print(delta_line)
        print("-" * len(header))

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Évaluation HYDRA avec/sans conditioning hiérarchique")
    parser.add_argument("--split-dir", required=True, help="Répertoire des splits train/val/test")
    parser.add_argument("--config", default="02_hydra/configs/default.yaml")
    parser.add_argument("--variant", default=None, choices=VARIANTS, help="Une seule variante (défaut: toutes)")
    parser.add_argument(
        "--eval-batch-size", type=int, default=2048,
        help="Batch size pour l'évaluation (plus grand = plus rapide, pas de gradient).",
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Répertoire racine des modèles (ex: 'models' ou 'models_hierarchy').",
    )
    parser.add_argument(
        "--model-suffix", default="",
        help="Suffixe ajouté au nom du dossier modèle : HYDRA_{variant}_CamemBERT{suffix}",
    )
    parser.add_argument(
        "--no-description", action="store_true",
        help="Évaluer en mode title-only (use_description=False).",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Si fourni, sérialise les métriques (mode hier) dans ce fichier JSON.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_kwargs = _map_config(cfg)

    split_dir = Path(args.split_dir)
    print("Chargement des données...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        seed=cfg.get("seed", 42),
        use_description=not args.no_description,
    )
    print(f"  Test: {len(data['X_test']):,} samples, L1: {data['num_classes_l1']}, L2: {data['num_classes_l2']}")

    variants = [args.variant] if args.variant else VARIANTS
    all_results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  HYDRA {variant.capitalize()}")
        print(f"{'='*60}")
        result = evaluate_variant(
            variant, data, cfg_kwargs,
            eval_batch_size=args.eval_batch_size,
            model_dir=args.model_dir,
            model_suffix=args.model_suffix,
        )
        if result:
            all_results[variant] = result

    print_results(all_results)

    if args.output_json and all_results:
        import json
        output = {}
        for variant, r in all_results.items():
            m = r.get("hier", {})
            output[variant] = {
                "f1_l1_macro": m.get("f1_l1_macro"),
                "f1_l2_macro": m.get("f1_l2_macro"),
                "hcr": m.get("hcr"),
                "constrained_f1_macro": m.get("constrained_f1_macro"),
                "eval_time_s": round(r.get("time_hier", 0), 1),
            }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n[Saved] {args.output_json}")


if __name__ == "__main__":
    main()
