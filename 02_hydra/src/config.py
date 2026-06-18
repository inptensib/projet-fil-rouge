"""Mapping YAML config → kwargs du constructeur HYDRATrainer.

Source unique partagée par scripts/run_train.py, scripts/eval_hierarchy.py
et 03_distributed/src/main.py. Toute clé YAML supportée par le trainer doit
être mappée ici — sinon elle est silencieusement ignorée.

Les clés de CHARGEMENT (normalize_text, synthetic_policy, use_description)
ne sont pas mappées : elles sont lues directement depuis le YAML par les
scripts et passées au data_loader, pas au trainer.
"""
from __future__ import annotations

# Nom HuggingFace → clé interne (shared.utils.ENCODERS)
ENCODER_MAP = {
    "camembert-base": "CamemBERT",
    "xlm-roberta-base": "XLM-RoBERTa",
    "bert-base-multilingual-cased": "mBERT",
}

# Clés YAML transmises telles quelles au constructeur HYDRATrainer
_DIRECT_KEYS = (
    "variant",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "max_length",
    "seed",
    "condition_l2_on_l1",
    "check_hierarchy",
    "save_dir",
    "use_focal",
    "focal_gamma",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "fp16",
    "experiment_suffix",
    "max_grad_norm",
    "dropout",
    "hidden_dim",
    "pooling",
    "project_embedding",
    "freeze_encoder",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "class_weighting",
    "logit_adjustment_tau",
)


def map_config(cfg: dict) -> dict:
    """Convertit une config YAML en kwargs pour HYDRATrainer.

    Args:
        cfg: Dictionnaire issu de yaml.safe_load().

    Returns:
        Kwargs prêts pour HYDRATrainer(**kwargs).
    """
    kwargs = {}

    for key in _DIRECT_KEYS:
        if key in cfg:
            kwargs[key] = cfg[key]

    if "encoder" in cfg:
        kwargs["encoder_name"] = ENCODER_MAP.get(cfg["encoder"], cfg["encoder"])

    # Alias historique : gradient_clip (anciennes configs) == max_grad_norm
    if "gradient_clip" in cfg:
        kwargs["max_grad_norm"] = cfg["gradient_clip"]

    if "level_weights" in cfg and len(cfg["level_weights"]) >= 1:
        kwargs["level_weight_l1"] = cfg["level_weights"][0]

    # Alias historique : loss: focal (anciennes configs) == use_focal: true
    if "loss" in cfg and "use_focal" not in cfg:
        kwargs["use_focal"] = cfg["loss"] == "focal"

    return kwargs
