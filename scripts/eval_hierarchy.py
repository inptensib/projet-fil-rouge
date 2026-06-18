"""Évalue les modèles HYDRA avec et sans conditioning L2 sur L1 prédit.

Compare les métriques (F1 L1, F1 L2, HCR, constrained F1) pour chaque
variante (local, nested, global) en mode standard vs hiérarchique.

ATTENTION — deux modes d'évaluation NON comparables entre eux :
- "base" : sans masque hiérarchique (HCR ~0.86) — le mode des modèles standards
- "hier" : avec masque (HCR ~0.997 par construction) — le mode des modèles
  entraînés avec condition_l2_on_l1/check_hierarchy

Le JSON de sortie (--output-json) suit --output-mode ; le défaut "auto"
choisit selon la config pour éviter de mélanger les deux familles de chiffres
(cf. docs/METRICS.md, incident mBERT).

Usage:
    python scripts/eval_hierarchy.py --split-dir /path/to/splits
    python scripts/eval_hierarchy.py --split-dir /path/to/splits --variant local
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import yaml

from rakuten.models.hydra.config import map_config
from rakuten.models.hydra.trainer import HYDRATrainer
from rakuten.common.data_loader import prepare_hierarchical_data_from_splits

_ROOT = Path(__file__).resolve().parents[1]


VARIANTS = ["local", "nested", "global"]

# Clés pilotées explicitement par evaluate_variant() ou sans effet en évaluation
_EVAL_IRRELEVANT_KEYS = (
    "variant",
    "check_hierarchy",
    "epochs",
    "save_dir",
    "experiment_suffix",
    "condition_l2_on_l1",
    "class_weighting",
    "logit_adjustment_tau",
)


def _resolve_output_mode(args_mode: str, cfg: dict) -> str:
    """Détermine le mode des métriques sérialisées dans le JSON.

    "auto" suit la config : un modèle entraîné pour la hiérarchie
    (check_hierarchy ou condition_l2_on_l1) est publié en mode "hier",
    un modèle standard en mode "base". Évite de publier des métriques
    masquées pour un modèle standard (incident mBERT).
    """
    if args_mode != "auto":
        return args_mode
    if cfg.get("check_hierarchy", False) or cfg.get("condition_l2_on_l1", False):
        return "hier"
    return "base"


def _serialize_results(all_results: dict, mode: str, config_path: str) -> dict:
    """Construit le JSON plat {variant: {métriques}} + métadonnées préfixées _."""
    time_key = f"time_{mode}"
    output = {
        "_mode": mode,
        "_config": config_path,
        "_date": datetime.date.today().isoformat(),
    }
    for variant, r in all_results.items():
        m = r.get(mode, {})
        output[variant] = {
            "f1_l1_macro": m.get("f1_l1_macro"),
            "f1_l2_macro": m.get("f1_l2_macro"),
            "hcr": m.get("hcr"),
            "constrained_f1_macro": m.get("constrained_f1_macro"),
            "eval_time_s": round(r.get(time_key, 0), 1),
        }
    return output


def evaluate_variant(
    variant: str,
    data: dict,
    cfg_kwargs: dict,
    eval_batch_size: int = 1024,
    model_dir: str = "models",
    model_suffix: str = "",
) -> dict:
    """Évalue une variante avec et sans conditioning hiérarchique."""
    encoder_name = cfg_kwargs.get("encoder_name", "CamemBERT")
    model_path = Path(_ROOT) / f"{model_dir}/HYDRA_{variant}_{encoder_name}{model_suffix}/best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} introuvable")
        return {}

    # Créer le trainer (check_hierarchy=False d'abord)
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
    parser.add_argument("--config", default="configs/hydra/default.yaml")
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
        help="Si fourni, sérialise les métriques dans ce fichier JSON (voir --output-mode).",
    )
    parser.add_argument(
        "--output-mode", default="auto", choices=["auto", "base", "hier"],
        help="Mode des métriques sérialisées : 'base' (sans masque hiérarchique), "
             "'hier' (avec masque), 'auto' (défaut : 'hier' si la config a "
             "check_hierarchy/condition_l2_on_l1, sinon 'base').",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_kwargs = map_config(cfg)
    for key in _EVAL_IRRELEVANT_KEYS:
        cfg_kwargs.pop(key, None)

    split_dir = Path(args.split_dir)
    print("Chargement des données...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        seed=cfg.get("seed", 42),
        use_description=not args.no_description,
        normalize=cfg.get("normalize_text", False),
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
        mode = _resolve_output_mode(args.output_mode, cfg)
        output = _serialize_results(all_results, mode, args.config)
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        mode_desc = (
            "AVEC masque hiérarchique" if mode == "hier"
            else "SANS masque hiérarchique"
        )
        print(f"\n[Saved] {args.output_json} (mode={mode} — métriques {mode_desc})")


if __name__ == "__main__":
    main()
