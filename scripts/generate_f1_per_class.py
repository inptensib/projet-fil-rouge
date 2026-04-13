"""Génère les fichiers f1_per_class_l2.json pour les modèles HYDRA entraînés.

Charge chaque best_model.pt, évalue sur le test set, et sauvegarde
le F1 per-class L1 et L2 en JSON dans results/hydra-{variant}-camembert/.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_f1_per_class.py \
        --split-dir /path/to/splits
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_f1_per_class.py \
        --split-dir /path/to/splits --variant local
"""
from __future__ import annotations

import argparse
import json
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
from shared.metrics import compute_per_class_f1

VARIANTS = ["local", "nested", "global"]
RESULTS_DIR = Path(_ROOT) / "results"


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


def evaluate_and_save(
    variant: str,
    data: dict,
    cfg_kwargs: dict,
    model_dir: str = "models",
    results_tag: str = "",
    check_hierarchy: bool = False,
):
    """Évalue un modèle et sauvegarde les F1 per-class en JSON."""
    model_path = Path(_ROOT) / f"{model_dir}/HYDRA_{variant}_CamemBERT/best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} introuvable")
        return

    cfg_kwargs.pop("epochs", None)
    trainer = HYDRATrainer(
        data=data, variant=variant, check_hierarchy=check_hierarchy,
        epochs=1, **cfg_kwargs,
    )

    ckpt = torch.load(model_path, map_location=trainer.device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    trainer._base_model.load_state_dict(state_dict)
    print(f"  Modèle chargé: {model_path}")

    test_loader = trainer._create_dataloader(
        data["X_test"], data["y_l1_test"], data["y_l2_test"], shuffle=False,
    )

    print("  Évaluation sur le test set...")
    t0 = time.time()
    metrics = trainer.evaluate(test_loader)
    elapsed = time.time() - t0

    labels_l1 = metrics["_labels_l1"]
    preds_l1 = metrics["_preds_l1"]
    labels_l2 = metrics["_labels_l2"]
    preds_l2 = metrics["_preds_l2"]

    print(f"  F1 L1: {metrics['f1_l1_macro']:.4f} | F1 L2: {metrics['f1_l2_macro']:.4f} "
          f"| HCR: {metrics.get('hcr', 0):.4f} | Temps: {elapsed:.0f}s")

    # Calcul per-class F1
    f1_l1_pc = compute_per_class_f1(labels_l1, preds_l1, trainer.l1_encoder)
    f1_l2_pc = compute_per_class_f1(labels_l2, preds_l2, trainer.l2_encoder)

    # Sauvegarde
    suffix = f"-{results_tag}" if results_tag else ""
    out_dir = RESULTS_DIR / f"hydra-{variant}{suffix}-camembert"
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_path = out_dir / "f1_per_class_l1.json"
    l2_path = out_dir / "f1_per_class_l2.json"

    l1_path.write_text(json.dumps(f1_l1_pc, ensure_ascii=False, indent=2))
    l2_path.write_text(json.dumps(f1_l2_pc, ensure_ascii=False, indent=2))

    print(f"  → {l1_path} ({len(f1_l1_pc)} classes)")
    print(f"  → {l2_path} ({len(f1_l2_pc)} classes)")


def main():
    parser = argparse.ArgumentParser(description="Génère f1_per_class L1/L2 en JSON")
    parser.add_argument("--split-dir", required=True, help="Répertoire des splits")
    parser.add_argument("--config", default="02_hydra/configs/default.yaml")
    parser.add_argument("--variant", default=None, choices=VARIANTS)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size pour l'évaluation (défaut: config YAML)")
    parser.add_argument("--model-dir", default="models",
                        help="Répertoire racine des modèles (ex: 'models_hierarchy')")
    parser.add_argument("--results-tag", default="",
                        help="Suffixe dossier résultats (ex: 'hierarchy' → hydra-local-hierarchy-camembert/)")
    parser.add_argument("--check-hierarchy", action="store_true",
                        help="Activer le masque hiérarchique à l'inférence")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_kwargs = _map_config(cfg)
    if args.batch_size:
        cfg_kwargs["batch_size"] = args.batch_size

    split_dir = Path(args.split_dir)
    print("Chargement des données...")
    data = prepare_hierarchical_data_from_splits(
        train_path=str(split_dir / "train_split.jsonl"),
        val_path=str(split_dir / "val_split.jsonl"),
        test_path=str(split_dir / "test_split.jsonl"),
        seed=cfg.get("seed", 42),
    )
    print(f"  Test: {len(data['X_test']):,} samples, L1: {data['num_classes_l1']}, L2: {data['num_classes_l2']}")

    variants = [args.variant] if args.variant else VARIANTS

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  HYDRA {variant.capitalize()}")
        print(f"{'='*60}")
        evaluate_and_save(variant, data, dict(cfg_kwargs),
                          model_dir=args.model_dir,
                          results_tag=args.results_tag,
                          check_hierarchy=args.check_hierarchy)


if __name__ == "__main__":
    main()
