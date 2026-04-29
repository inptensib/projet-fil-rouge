#!/usr/bin/env python3
"""Génère la page HTML de comparaison de tous les modèles.

Usage:
    python scripts/generate_comparison.py

Lit les données depuis MLflow (mlflow.db) et results/, génère results/comparison.html.
Pour ajouter un modèle : relancer le script après que les artefacts sont dans results/.
"""

import base64
import json
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
OUTPUT_HTML = RESULTS_DIR / "comparison.html"

HYDRA_EXPERIMENTS = {
    "1": {"variant": "local", "name": "HYDRA Local", "color": "#3498db"},
    "2": {"variant": "nested", "name": "HYDRA Nested", "color": "#2ecc71"},
    "3": {"variant": "global", "name": "HYDRA Global", "color": "#9b59b6"},
}

METRIC_KEYS = [
    "train_loss", "val_f1_l1_macro", "val_f1_l2_macro", "val_hcr",
    "test_f1_l1_macro", "test_f1_l2_macro", "test_hcr",
]

HISTORY_KEYS = ["train_loss", "val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"]

IMAGE_FILES = [
    "training_curves.png",
    "confusion_matrix_l1.png",
    "confusion_matrix_l2_top50.png",
    "f1_per_class_l1.png",
    "f1_per_class_l2.png",
]

SKIP_PARAMS = {"gpu_type", "num_classes_l1", "num_classes_l2"}

EVAL_TEST_METRICS = RESULTS_DIR / "hydra_eval_test_metrics.json"
HYDRA_HIERARCHY_METRICS = RESULTS_DIR / "hydra_eval_hierarchy_metrics.json"

HYDRA_HIERARCHY_VARIANTS = {
    "local":  {"name": "HYDRA Local + Hierarchy",  "color": "#1a5276"},
    "nested": {"name": "HYDRA Nested + Hierarchy", "color": "#1d8348"},
    "global": {"name": "HYDRA Global + Hierarchy", "color": "#6c3483"},
}

# Timestamp de séparation : runs standard (avant) vs runs hierarchy (après)
# April 11, 2026 00:00:00 UTC = 1775865600 s = 1775865600000 ms
HIERARCHY_START_MS = 1775865600000

VARIANT_TO_EXP_ID = {info["variant"]: eid for eid, info in HYDRA_EXPERIMENTS.items()}

HYDRA_TITLE_VARIANTS = {
    "local":  {"name": "HYDRA Local (Title-only)",  "color": "#5dade2"},
    "nested": {"name": "HYDRA Nested (Title-only)", "color": "#58d68d"},
    "global": {"name": "HYDRA Global (Title-only)", "color": "#bb8fce"},
}

HYDRA_TITLE_EXPERIMENT_NAMES = {
    variant: f"HYDRA_{variant}_CamemBERT_title"
    for variant in HYDRA_TITLE_VARIANTS
}

HYDRA_TITLE_HIERARCHY_VARIANTS = {
    "local":  {"name": "HYDRA Local + Hierarchy (Title-only)",  "color": "#1f618d"},
    "nested": {"name": "HYDRA Nested + Hierarchy (Title-only)", "color": "#239b56"},
    "global": {"name": "HYDRA Global + Hierarchy (Title-only)", "color": "#7d3c98"},
}

FLAT_HYDRA_TITLE_RESULTS_DIR = RESULTS_DIR / "flat_hydra_title"
FLAT_HYDRA_TITLE_MODELS_DIR = PROJECT_ROOT / "models" / "flat_hydra_title"


def _get_merged_history(
    conn: sqlite3.Connection,
    exp_id: str,
    *,
    before_ms: int | None = None,
    since_ms: int | None = None,
) -> dict:
    """Fusionne l'historique de plusieurs runs en un dict epochs/values par clé de métrique.

    Priorité par step : FINISHED non-nul > RUNNING non-nul > toute valeur (earliest start_time).
    Permet de reconstruire epoch 1-10 depuis plusieurs jobs SLURM fragmentés, en ignorant
    les runs de redémarrage qui ont loggé des zéros avant d'être tués.
    """
    query = "SELECT run_uuid, start_time, status FROM runs WHERE experiment_id=?"
    params: list = [exp_id]
    if before_ms is not None:
        query += " AND start_time < ?"
        params.append(before_ms)
    if since_ms is not None:
        query += " AND start_time >= ?"
        params.append(since_ms)
    query += " ORDER BY start_time ASC"

    runs = conn.execute(query, params).fetchall()
    if not runs:
        return {}

    # step → liste de (value, start_time, status)
    candidates: dict[str, dict[int, list]] = {key: {} for key in HISTORY_KEYS}
    for run_uuid, start_time, status in runs:
        for key in HISTORY_KEYS:
            rows = conn.execute(
                "SELECT step, value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
                (run_uuid, key),
            ).fetchall()
            for step, value in rows:
                candidates[key].setdefault(step, []).append((value, start_time, status))

    def _pick_best(items: list) -> float:
        """FINISHED non-nul > RUNNING non-nul > tout, à start_time égal préfère le plus ancien."""
        for status_filter in ("FINISHED", "RUNNING", None):
            subset = [
                (v, t) for v, t, s in items
                if (status_filter is None or s == status_filter) and v != 0.0
            ]
            if subset:
                return min(subset, key=lambda x: x[1])[0]
        # fallback : valeur du run le plus ancien
        return min(items, key=lambda x: x[1])[0]

    history = {}
    for key, step_data in candidates.items():
        if step_data:
            sorted_steps = sorted(step_data)
            history[key] = {
                "epochs": sorted_steps,
                "values": [_pick_best(step_data[s]) for s in sorted_steps],
            }
    return history


def encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def discover_hydra_models() -> list[dict]:
    conn = sqlite3.connect(str(MLFLOW_DB))
    models = []

    # Overrides de métriques test (eval officielle sur test_split.jsonl)
    eval_overrides = {}
    if EVAL_TEST_METRICS.exists():
        raw = json.loads(EVAL_TEST_METRICS.read_text())
        eval_overrides = {k: v for k, v in raw.items() if not k.startswith("_")}

    for exp_id, info in HYDRA_EXPERIMENTS.items():
        variant = info["variant"]
        results_dir = RESULTS_DIR / f"hydra-{variant}-camembert"

        # Meilleur run FINISHED
        row = conn.execute(
            "SELECT run_uuid FROM runs "
            "WHERE experiment_id=? AND status='FINISHED' AND start_time < ? "
            "ORDER BY end_time DESC LIMIT 1",
            (exp_id, HIERARCHY_START_MS),
        ).fetchone()

        if row is None:
            # Vérifier si en cours
            running = conn.execute(
                "SELECT run_uuid FROM runs "
                "WHERE experiment_id=? AND status='RUNNING' AND start_time < ? "
                "ORDER BY start_time DESC LIMIT 1",
                (exp_id, HIERARCHY_START_MS),
            ).fetchone()
            if running:
                run_id = running[0]
                metrics_rows = conn.execute(
                    "SELECT key, value FROM latest_metrics WHERE run_uuid=?",
                    (run_id,),
                ).fetchall()
                m = dict(metrics_rows)
                model_id = f"hydra-{variant}-camembert"
                if model_id in eval_overrides:
                    m.update(eval_overrides[model_id])
                models.append({
                    "id": f"hydra-{variant}-camembert",
                    "name": info["name"],
                    "family": "hydra",
                    "subtitle": f"CamemBERT — variante {variant}",
                    "status": "running",
                    "color": info["color"],
                    "metrics": {
                        "test_f1_l1_macro": m.get("test_f1_l1_macro"),
                        "test_f1_l2_macro": m.get("test_f1_l2_macro"),
                        "test_hcr": m.get("test_hcr"),
                    },
                    "val_metrics": {
                        "val_f1_l1_macro": m.get("val_f1_l1_macro"),
                        "val_f1_l2_macro": m.get("val_f1_l2_macro"),
                        "val_hcr": m.get("val_hcr"),
                    },
                    "training": {"params": {}},
                    "history": None,
                    "f1_per_class_l1": None,
                    "f1_per_class_l2": None,
                    "images": {},
                })
            continue

        run_id = row[0]

        # Métriques
        metrics_rows = conn.execute(
            "SELECT key, value FROM latest_metrics WHERE run_uuid=?", (run_id,),
        ).fetchall()
        m = dict(metrics_rows)
        model_id = f"hydra-{variant}-camembert"
        if model_id in eval_overrides:
            m.update(eval_overrides[model_id])

        # Paramètres
        params_rows = conn.execute(
            "SELECT key, value FROM params WHERE run_uuid=?", (run_id,),
        ).fetchall()
        p = dict(params_rows)

        # Historique par epoch (fusion de tous les runs standard pour reconstruire epoch 1-10)
        history = _get_merged_history(conn, exp_id, before_ms=HIERARCHY_START_MS)

        # F1 par classe (results/ puis fallback MLflow artifacts)
        artifacts_dir = PROJECT_ROOT / "mlruns" / exp_id / run_id / "artifacts"
        f1_json_path = results_dir / "f1_per_class_l1.json"
        if not f1_json_path.exists():
            f1_json_path = artifacts_dir / "f1_per_class_l1.json"
        f1_per_class = json.loads(f1_json_path.read_text()) if f1_json_path.exists() else None
        f1_l2_json_path = results_dir / "f1_per_class_l2.json"
        if not f1_l2_json_path.exists():
            f1_l2_json_path = artifacts_dir / "f1_per_class_l2.json"
        f1_per_class_l2 = json.loads(f1_l2_json_path.read_text()) if f1_l2_json_path.exists() else None

        # Images
        images = {}
        for fname in IMAGE_FILES:
            key = fname.replace(".png", "")
            encoded = encode_image(results_dir / fname)
            if encoded:
                images[key] = encoded

        params_clean = {k: v for k, v in p.items() if k not in SKIP_PARAMS}

        models.append({
            "id": f"hydra-{variant}-camembert",
            "name": info["name"],
            "family": "hydra",
            "subtitle": f"CamemBERT — variante {variant}",
            "status": "finished",
            "color": info["color"],
            "metrics": {
                "test_f1_l1_macro": m.get("test_f1_l1_macro"),
                "test_f1_l2_macro": m.get("test_f1_l2_macro"),
                "test_hcr": m.get("test_hcr"),
            },
            "val_metrics": {
                "val_f1_l1_macro": m.get("val_f1_l1_macro"),
                "val_f1_l2_macro": m.get("val_f1_l2_macro"),
                "val_hcr": m.get("val_hcr"),
            },
            "training": {
                "params": params_clean,
            },
            "history": history if history else None,
            "f1_per_class_l1": f1_per_class,
            "f1_per_class_l2": f1_per_class_l2,
            "images": images,
        })

    conn.close()
    return models


def discover_hydra_hierarchy_models() -> list[dict]:
    """Découvre les modèles HYDRA hierarchy depuis un JSON statique + per-class JSONs + MLflow."""
    if not HYDRA_HIERARCHY_METRICS.exists():
        return []

    raw = json.loads(HYDRA_HIERARCHY_METRICS.read_text())
    overrides = {k: v for k, v in raw.items() if not k.startswith("_")}

    conn = sqlite3.connect(str(MLFLOW_DB)) if MLFLOW_DB.exists() else None

    models = []
    for variant, info in HYDRA_HIERARCHY_VARIANTS.items():
        model_id = f"hydra-{variant}-hierarchy-camembert"
        results_dir = RESULTS_DIR / f"hydra-{variant}-hierarchy-camembert"
        m = overrides.get(model_id, {})

        # Historique MLflow (runs hierarchy = après HIERARCHY_START_MS)
        history = None
        val_metrics: dict = {}
        if conn is not None:
            exp_id = VARIANT_TO_EXP_ID.get(variant)
            if exp_id:
                history_raw = _get_merged_history(conn, exp_id, since_ms=HIERARCHY_START_MS)
                if history_raw:
                    history = history_raw
                    for mk in ("val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"):
                        if mk in history_raw and history_raw[mk]["values"]:
                            val_metrics[mk] = history_raw[mk]["values"][-1]

        f1_per_class_l1, f1_per_class_l2 = None, None
        if results_dir.exists():
            p1 = results_dir / "f1_per_class_l1.json"
            p2 = results_dir / "f1_per_class_l2.json"
            if p1.exists():
                f1_per_class_l1 = json.loads(p1.read_text())
            if p2.exists():
                f1_per_class_l2 = json.loads(p2.read_text())

        images = {}
        if results_dir.exists():
            for fname in IMAGE_FILES:
                key = fname.replace(".png", "")
                encoded = encode_image(results_dir / fname)
                if encoded:
                    images[key] = encoded

        models.append({
            "id": model_id,
            "name": info["name"],
            "family": "hydra",
            "subtitle": f"CamemBERT — variante {variant} + masque hiérarchique",
            "status": "finished",
            "color": info["color"],
            "metrics": {
                "test_f1_l1_macro": m.get("test_f1_l1_macro"),
                "test_f1_l2_macro": m.get("test_f1_l2_macro"),
                "test_hcr": m.get("test_hcr"),
            },
            "val_metrics": val_metrics,
            "training": {"params": {
                "check_hierarchy": "true",
                "condition_l2_on_l1": "true",
                "encoder": "camembert-base",
                "lr": "5.0e-5",
                "batch_size": "128",
                "epochs": "10",
            }},
            "history": history,
            "f1_per_class_l1": f1_per_class_l1,
            "f1_per_class_l2": f1_per_class_l2,
            "images": images,
        })

    if conn is not None:
        conn.close()
    return models


def _load_baseline_per_class(embedding: str, level: str) -> dict | None:
    """Load per-class F1 JSON for a baseline (e.g. per_class_f1_l1_tfidf.json)."""
    path = RESULTS_DIR / "baselines" / f"per_class_f1_{level}_{embedding}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _discover_baseline_models_from_file(json_path: Path, family: str) -> list[dict]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text())
    models = []
    colors = {
        "tfidf": "#e67e22", "word2vec": "#e74c3c",
        "fasttext": "#1abc9c", "fasttext-ova": "#1abc9c", "fasttext-softmax": "#148f77",
        "tfidf-title": "#d4820a", "word2vec-title": "#c0392b",
        "fasttext-softmax-title": "#0e6655",
    }
    name_map = {
        "tfidf": "TF-IDF", "word2vec": "Word2Vec",
        "fasttext": "FastText OVA", "fasttext-softmax": "FastText Softmax",
        "tfidf-title": "TF-IDF (Title-only)", "word2vec-title": "Word2Vec (Title-only)",
        "fasttext-softmax-title": "FastText Softmax (Title-only)",
    }
    for entry in data:
        emb = entry["embedding"]
        clf = entry["classifier"]
        models.append({
            "id": f"{emb}-{clf.lower().replace('/', '-').replace(' ', '')}",
            "name": name_map.get(emb, f"{emb.upper()} + {clf}"),
            "family": family,
            "subtitle": entry.get("subtitle", f"{emb} + {clf}"),
            "status": entry.get("status", "finished"),
            "color": colors.get(emb, "#95a5a6"),
            "metrics": entry.get("metrics", {}),
            "val_metrics": entry.get("val_metrics", {}),
            "training": {"params": entry.get("params", {})},
            "history": None,
            "f1_per_class_l1": _load_baseline_per_class(emb, "l1") or entry.get("f1_per_class_l1"),
            "f1_per_class_l2": _load_baseline_per_class(emb, "l2") or entry.get("f1_per_class_l2"),
            "images": {},
        })
    return models


def discover_baseline_models() -> list[dict]:
    return _discover_baseline_models_from_file(
        RESULTS_DIR / "baselines" / "baseline_results.json", "baseline"
    )


def discover_baseline_title_models() -> list[dict]:
    return _discover_baseline_models_from_file(
        RESULTS_DIR / "baselines" / "baseline_results_title.json", "baseline-title"
    )


FLAT_HYDRA_RESULTS_DIR = RESULTS_DIR / "flat_hydra"
FLAT_HYDRA_MODELS_DIR = PROJECT_ROOT / "models" / "flat_hydra"


def discover_flat_hydra_model() -> list[dict]:
    """Découvre le modèle FlatHydra depuis results/flat_hydra/ et models/flat_hydra/."""
    test_metrics_path = FLAT_HYDRA_RESULTS_DIR / "test_metrics.json"
    if not test_metrics_path.exists():
        return []

    test_m = json.loads(test_metrics_path.read_text())

    # Historique d'entraînement (models/flat_hydra/metrics.json)
    history: dict | None = None
    val_metrics: dict = {}
    history_path = FLAT_HYDRA_MODELS_DIR / "metrics.json"
    if history_path.exists():
        h = json.loads(history_path.read_text())
        epochs = h.get("epoch", [])
        history = {}
        for key in ("train_loss", "val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"):
            if key in h:
                history[key] = {"epochs": epochs, "values": h[key]}
        for mk in ("val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"):
            if mk in h and h[mk]:
                val_metrics[mk] = h[mk][-1]

    # Paramètres de config
    params: dict = {}
    config_path = FLAT_HYDRA_MODELS_DIR / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        skip = {"save_dir", "experiment_name", "num_workers", "seed", "num_classes_l1", "num_classes_l2"}
        params = {k: str(v) for k, v in cfg.items() if k not in skip}

    # Images
    images = {}
    for fname in IMAGE_FILES:
        key = fname.replace(".png", "")
        encoded = encode_image(FLAT_HYDRA_RESULTS_DIR / fname)
        if encoded:
            images[key] = encoded

    return [{
        "id": "flat-hydra-camembert",
        "name": "FlatHydra",
        "family": "baseline",
        "subtitle": "CamemBERT — classification plate (1 tête, CrossEntropy)",
        "status": "finished",
        "color": "#f39c12",
        "metrics": {
            "test_f1_l1_macro": test_m.get("f1_l1_macro"),
            "test_f1_l2_macro": test_m.get("f1_l2_macro"),
            "test_hcr": test_m.get("hcr"),
            "test_constrained_f1_macro": test_m.get("constrained_f1_macro"),
        },
        "val_metrics": val_metrics,
        "training": {"params": params},
        "history": history,
        "f1_per_class_l1": None,
        "f1_per_class_l2": None,
        "images": images,
    }]


def discover_hydra_title_models() -> list[dict]:
    """Découvre les modèles HYDRA Title-only (designation seule) depuis MLflow."""
    if not MLFLOW_DB.exists():
        return []
    conn = sqlite3.connect(str(MLFLOW_DB))
    models = []

    for variant, info in HYDRA_TITLE_VARIANTS.items():
        exp_name = HYDRA_TITLE_EXPERIMENT_NAMES[variant]
        row = conn.execute(
            "SELECT experiment_id FROM experiments WHERE name=?", (exp_name,)
        ).fetchone()
        if row is None:
            continue
        exp_id = str(row[0])

        best = conn.execute(
            "SELECT run_uuid FROM runs "
            "WHERE experiment_id=? AND status='FINISHED' "
            "ORDER BY end_time DESC LIMIT 1",
            (exp_id,),
        ).fetchone()
        if best is None:
            continue
        run_id = best[0]

        m = dict(conn.execute(
            "SELECT key, value FROM latest_metrics WHERE run_uuid=?", (run_id,),
        ).fetchall())
        p = dict(conn.execute(
            "SELECT key, value FROM params WHERE run_uuid=?", (run_id,),
        ).fetchall())
        params_clean = {k: v for k, v in p.items() if k not in SKIP_PARAMS}

        history = _get_merged_history(conn, exp_id)

        models.append({
            "id": f"hydra-{variant}-title-camembert",
            "name": info["name"],
            "family": "hydra-title",
            "subtitle": f"CamemBERT — variante {variant} — Title-only (designation seule)",
            "status": "finished",
            "color": info["color"],
            "metrics": {
                "test_f1_l1_macro": m.get("test_f1_l1_macro"),
                "test_f1_l2_macro": m.get("test_f1_l2_macro"),
                "test_hcr": m.get("test_hcr"),
            },
            "val_metrics": {
                "val_f1_l1_macro": m.get("val_f1_l1_macro"),
                "val_f1_l2_macro": m.get("val_f1_l2_macro"),
                "val_hcr": m.get("val_hcr"),
            },
            "training": {"params": params_clean},
            "history": history if history else None,
            "f1_per_class_l1": None,
            "f1_per_class_l2": None,
            "images": {},
        })

    conn.close()
    return models


def discover_hydra_title_hierarchy_models() -> list[dict]:
    """Découvre les modèles HYDRA Hierarchy Title-only depuis models_title_hierarchy/ et results/."""
    import re
    metrics_path = RESULTS_DIR / "hydra_eval_title_hierarchy_metrics.json"
    test_metrics: dict = {}
    if metrics_path.exists():
        test_metrics = json.loads(metrics_path.read_text())

    models_root = PROJECT_ROOT / "models_title_hierarchy"
    models = []

    for variant, info in HYDRA_TITLE_HIERARCHY_VARIANTS.items():
        model_dir = models_root / f"HYDRA_{variant}_CamemBERT_title_hierarchy"
        complete_txt = model_dir / "training_complete.txt"
        ckpt_latest = model_dir / "checkpoint_latest.pt"

        if not model_dir.exists():
            continue

        if complete_txt.exists():
            status = "finished"
        elif ckpt_latest.exists():
            status = "running"
        else:
            status = "planned"

        val_f1_l2: float | None = None
        if complete_txt.exists():
            m = re.search(r"Best F1 L2:\s*([\d.]+)", complete_txt.read_text())
            if m:
                val_f1_l2 = float(m.group(1))

        tm = test_metrics.get(variant, {})
        metrics = {
            "test_f1_l1_macro": tm.get("f1_l1_macro"),
            "test_f1_l2_macro": tm.get("f1_l2_macro"),
            "test_hcr": tm.get("hcr"),
            "test_constrained_f1_macro": tm.get("constrained_f1_macro"),
        } if tm else {}

        val_metrics = {
            "val_f1_l2_macro": val_f1_l2,
        } if val_f1_l2 is not None else {}

        params: dict = {}
        cfg_path = PROJECT_ROOT / "02_hydra" / "configs" / "title_only_hierarchy.yaml"
        if cfg_path.exists():
            import yaml as _yaml
            cfg = _yaml.safe_load(cfg_path.read_text())
            skip = {"save_dir", "experiment_suffix", "seed", "num_workers"}
            params = {k: str(v) for k, v in cfg.items() if k not in skip}
        params["variant"] = variant
        params["use_description"] = "False"

        models.append({
            "id": f"hydra-{variant}-title-hier-camembert",
            "name": info["name"],
            "family": "hydra-title",
            "subtitle": f"CamemBERT — variante {variant} — Hierarchy + Title-only",
            "status": status,
            "color": info["color"],
            "metrics": metrics,
            "val_metrics": val_metrics,
            "training": {"params": params},
            "history": None,
            "f1_per_class_l1": None,
            "f1_per_class_l2": None,
            "images": {},
        })

    return models


def discover_flat_hydra_title_model() -> list[dict]:
    """Découvre le modèle FlatHydra Title-only depuis models/flat_hydra_title/."""
    history_path = FLAT_HYDRA_TITLE_MODELS_DIR / "metrics.json"
    if not history_path.exists():
        return []

    h = json.loads(history_path.read_text())
    epochs = h.get("epoch", [])
    history: dict = {}
    val_metrics: dict = {}
    for key in ("train_loss", "val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"):
        if key in h:
            history[key] = {"epochs": epochs, "values": h[key]}
    for mk in ("val_f1_l1_macro", "val_f1_l2_macro", "val_hcr"):
        if mk in h and h[mk]:
            val_metrics[mk] = h[mk][-1]

    params: dict = {}
    config_path = FLAT_HYDRA_TITLE_MODELS_DIR / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        skip = {"save_dir", "experiment_name", "num_workers", "seed", "num_classes_l1", "num_classes_l2"}
        params = {k: str(v) for k, v in cfg.items() if k not in skip}

    images = {}
    for fname in IMAGE_FILES:
        key = fname.replace(".png", "")
        encoded = encode_image(FLAT_HYDRA_TITLE_RESULTS_DIR / fname)
        if encoded:
            images[key] = encoded

    test_m: dict = {}
    test_metrics_path = FLAT_HYDRA_TITLE_RESULTS_DIR / "test_metrics.json"
    if test_metrics_path.exists():
        tm = json.loads(test_metrics_path.read_text())
        test_m = {
            "test_f1_l1_macro": tm.get("f1_l1_macro"),
            "test_f1_l2_macro": tm.get("f1_l2_macro"),
            "test_hcr": tm.get("hcr"),
            "test_constrained_f1_macro": tm.get("constrained_f1_macro"),
        }

    return [{
        "id": "flat-hydra-title-camembert",
        "name": "FlatHydra (Title-only)",
        "family": "baseline-title",
        "subtitle": "CamemBERT — classification plate (1 tête) — Title-only (designation seule)",
        "status": "finished",
        "color": "#f5b041",
        "metrics": test_m,
        "val_metrics": val_metrics,
        "training": {"params": params},
        "history": history if history else None,
        "f1_per_class_l1": None,
        "f1_per_class_l2": None,
        "images": images,
    }]


def render_html(models_data: dict) -> str:
    data_json = json.dumps(models_data, ensure_ascii=False, indent=2)
    return HTML_TEMPLATE.replace("/* __MODELS_DATA__ */", data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Comparaison des modèles — Rakuten Classification Hiérarchique</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 20px 30px;
    color: #333; background: #fff; line-height: 1.6;
}
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 12px; }
h2 { color: #2c3e50; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }
nav { background: #f8f9fa; padding: 16px; border-radius: 8px; margin: 20px 0; }
nav a { margin-right: 16px; color: #3498db; text-decoration: none; font-weight: 500; }
nav a:hover { text-decoration: underline; }

/* Tables */
.table-container { overflow-x: auto; margin: 16px 0; }
table { border-collapse: collapse; width: 100%; }
th { background: #f1f3f5; padding: 10px 12px; text-align: left; font-size: 0.85em; color: #495057; }
td { padding: 8px 12px; border-bottom: 1px solid #f1f3f5; }
.best-cell { background: #d4edda; font-weight: 700; }
.worst-cell { background: #f8d7da; }
.metric-main { font-weight: 700; font-size: 1.05em; }

/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
.badge-finished { background: #d4edda; color: #155724; }
.badge-running { background: #fff3cd; color: #856404; }
.badge-planned { background: #e2e3e5; color: #383d41; }

/* Charts */
.chart-row { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }
.chart-container { flex: 1; min-width: 400px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 16px; }
.chart-container.full-width { flex-basis: 100%; }
.chart-container canvas { max-height: 400px; }

/* Per-class controls */
.controls-bar { display: flex; gap: 12px; margin: 12px 0; align-items: center; flex-wrap: wrap; }
.controls-bar input, .controls-bar select {
    padding: 8px 12px; border: 1px solid #dee2e6; border-radius: 6px;
    font-size: 0.9em; font-family: inherit;
}
.controls-bar input { flex: 1; min-width: 200px; }

/* Accordions */
details { margin: 12px 0; border: 1px solid #e9ecef; border-radius: 8px; }
details summary {
    padding: 12px 16px; cursor: pointer; background: #f8f9fa; border-radius: 8px;
    font-weight: 600; color: #2c3e50; list-style: none;
}
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "▶ "; font-size: 0.8em; }
details[open] summary::before { content: "▼ "; }
details[open] summary { border-radius: 8px 8px 0 0; }
.detail-content { padding: 16px; }
.img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.img-container {
    text-align: center; border: 1px solid #e9ecef; border-radius: 8px;
    padding: 12px; background: #fafafa;
}
.img-container img {
    max-width: 100%; height: auto; cursor: pointer;
    border-radius: 4px; transition: transform 0.2s;
}
.img-container img:hover { transform: scale(1.02); }
.img-caption { font-size: 0.85em; color: #6c757d; margin-top: 8px; font-style: italic; }

/* Overlay zoom */
.overlay {
    display: none; position: fixed; top: 0; left: 0;
    width: 100%; height: 100%; background: rgba(0,0,0,0.92);
    z-index: 1000; overflow: hidden;
}
.overlay.active { display: block; }
.overlay img { position: absolute; cursor: grab; transform-origin: 0 0; transition: none; }
.overlay img.grabbing { cursor: grabbing; }
.overlay .controls {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 10px; z-index: 1001;
}
.overlay .controls button {
    background: rgba(255,255,255,0.15); color: #fff; border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px; padding: 8px 16px; font-size: 16px; cursor: pointer;
    backdrop-filter: blur(4px);
}
.overlay .controls button:hover { background: rgba(255,255,255,0.3); }
.overlay .zoom-level { position: fixed; top: 20px; right: 20px; color: rgba(255,255,255,0.7); font-size: 14px; z-index: 1001; font-family: monospace; }
.overlay .hint { position: fixed; top: 20px; left: 20px; color: rgba(255,255,255,0.5); font-size: 13px; z-index: 1001; }

/* Color helpers */
.f1-high { color: #27ae60; }
.f1-mid { color: #e67e22; }
.f1-low { color: #e74c3c; }

@media (max-width: 768px) {
    .chart-row { flex-direction: column; }
    .chart-container { min-width: 100%; }
}
</style>
</head>
<body>

<h1>Comparaison des modèles — Classification Hiérarchique Rakuten</h1>
<p style="color:#6c757d">
    157 catégories L1 · 6 126 catégories L2 · 8M+ produits<br>
    Télécom Paris × Rakuten France · Généré le <span id="gen-date"></span>
</p>

<nav>
    <a href="#recap">Récapitulatif</a>
    <a href="#charts">Graphiques</a>
    <a href="#per-class">F1 par classe L1</a>
    <a href="#per-class-l2">F1 par classe L2</a>
    <a href="#details">Détail par modèle</a>
    <a href="#config">Configuration</a>
</nav>

<!-- Overlay zoom -->
<div class="overlay" id="overlay">
    <img id="overlay-img" src="" alt="Plein écran">
    <div class="hint">Molette = zoom · Clic-glisser = déplacer · Double-clic = reset · Echap = fermer</div>
    <div class="zoom-level" id="zoom-level">100%</div>
    <div class="controls">
        <button onclick="zoomBy(1.5)">+ Zoom</button>
        <button onclick="zoomBy(1/1.5)">&minus; Zoom</button>
        <button onclick="resetView()">Reset</button>
        <button onclick="closeOverlay()">Fermer (Echap)</button>
    </div>
</div>

<section id="recap">
    <h2>Tableau récapitulatif</h2>
    <h3 style="color:#5dade2;border-left:4px solid #5dade2;padding-left:10px">Title-only (<code>designation</code> seule)</h3>
    <p style="color:#6c757d;font-size:0.9em">Modèles entraînés sur le titre du produit uniquement (benchmark sans description).</p>
    <div class="table-container"><table id="recap-table-title-only"></table></div>
    <h3 style="color:#3498db;border-left:4px solid #3498db;padding-left:10px;margin-top:30px">Title + Description (<code>designation + description</code>)</h3>
    <p style="color:#6c757d;font-size:0.9em">Modèles entraînés sur la concaténation titre + description (régime principal).</p>
    <div class="table-container"><table id="recap-table-full-text"></table></div>
</section>
<section id="charts">
    <h2>Graphiques comparatifs</h2>
    <div class="chart-row">
        <div class="chart-container full-width"><canvas id="chart-metrics"></canvas></div>
    </div>
    <div class="chart-row">
        <div class="chart-container full-width" style="min-height:420px"><canvas id="chart-convergence"></canvas></div>
    </div>
    <div class="chart-row">
        <div class="chart-container full-width" style="min-height:420px"><canvas id="chart-convergence-l1"></canvas></div>
    </div>
    <div class="chart-row">
        <div class="chart-container full-width" style="min-height:420px"><canvas id="chart-convergence-loss"></canvas></div>
    </div>
</section>
<section id="per-class">
    <h2>Comparaison par classe L1 (157 classes)</h2>
    <div class="controls-bar">
        <input type="text" id="class-search" placeholder="Filtrer par nom de classe...">
        <select id="class-sort">
            <option value="best-first">Meilleur F1 d'abord</option>
            <option value="worst-first">Pire F1 d'abord</option>
            <option value="alpha">Alphabétique</option>
            <option value="diff">Plus grand écart</option>
        </select>
    </div>
    <div class="table-container"><table id="per-class-table"></table></div>
</section>
<section id="per-class-l2">
    <h2>Comparaison par classe L2 (6 126 classes)</h2>
    <div class="controls-bar">
        <input type="text" id="class-l2-search" placeholder="Filtrer par nom de classe L2...">
        <select id="class-l2-sort">
            <option value="best-first">Meilleur F1 d'abord</option>
            <option value="worst-first">Pire F1 d'abord</option>
            <option value="alpha">Alphabétique</option>
            <option value="diff">Plus grand écart</option>
        </select>
        <span id="class-l2-count" style="color:#6c757d;font-size:0.85em"></span>
    </div>
    <div class="table-container"><table id="per-class-l2-table"></table></div>
</section>
<section id="details"><h2>Détail par modèle</h2><div id="accordions"></div></section>
<section id="config">
    <h2>Hyperparamètres</h2>
    <h3 style="color:#5dade2;border-left:4px solid #5dade2;padding-left:10px">Title-only</h3>
    <div class="table-container"><table id="config-table-title-only"></table></div>
    <h3 style="color:#3498db;border-left:4px solid #3498db;padding-left:10px;margin-top:30px">Title + Description</h3>
    <div class="table-container"><table id="config-table-full-text"></table></div>
</section>

<script>
const MODELS_DATA = /* __MODELS_DATA__ */;

// --- Utils ---
const fmt = (v, pct=true) => v != null ? (pct ? (v*100).toFixed(2)+'%' : v.toFixed(4)) : '--';
const f1Color = (v) => v == null ? '' : v >= 0.8 ? 'f1-high' : v >= 0.5 ? 'f1-mid' : 'f1-low';
const badgeHtml = (status) => {
    const cls = {'finished':'badge-finished','running':'badge-running','planned':'badge-planned'}[status] || '';
    const txt = {'finished':'Terminé','running':'En cours','planned':'Planifié'}[status] || status;
    return `<span class="badge ${cls}">${txt}</span>`;
};
const finished = MODELS_DATA.models.filter(m => m.status !== 'planned');

// --- 1. Recap Table ---
function renderRecapTable(modelList, tableId) {
    const t = document.getElementById(tableId);
    if (!t) return;
    if (modelList.length === 0) {
        t.innerHTML = '<tbody><tr><td style="text-align:center;padding:20px;color:#6c757d">Aucun modèle dans cette catégorie.</td></tr></tbody>';
        return;
    }
    const metrics = [
        {key: 'test_f1_l1_macro', label: 'Test F1 L1 macro', main: false},
        {key: 'test_f1_l2_macro', label: 'Test F1 L2 macro', main: true},
        {key: 'test_hcr', label: 'HCR (test)', main: false},
        {key: 'val_f1_l1_macro', label: 'Val F1 L1 macro', main: false, src: 'val_metrics'},
        {key: 'val_f1_l2_macro', label: 'Val F1 L2 macro', main: false, src: 'val_metrics'},
    ];
    let html = '<thead><tr><th></th>';
    for (const m of modelList) {
        html += `<th style="text-align:center;min-width:120px">
            <span style="color:${m.color}">●</span> ${m.name}<br>${badgeHtml(m.status)}</th>`;
    }
    html += '</tr></thead><tbody>';
    html += '<tr><td style="font-weight:600">Famille</td>';
    for (const m of modelList) html += `<td style="text-align:center">${m.family.startsWith('hydra') ? 'HYDRA' : 'Baseline'}</td>`;
    html += '</tr>';
    for (const mdef of metrics) {
        const vals = modelList.map(m => {
            const src = mdef.src === 'val_metrics' ? m.val_metrics : m.metrics;
            return src ? src[mdef.key] : null;
        });
        const validVals = vals.filter(v => v != null);
        const best = validVals.length ? Math.max(...validVals) : null;
        const worst = validVals.length > 1 ? Math.min(...validVals) : null;

        html += `<tr><td style="font-weight:600">${mdef.label}</td>`;
        for (let i = 0; i < modelList.length; i++) {
            const v = vals[i];
            let cls = mdef.main ? 'metric-main' : '';
            if (v != null && v === best && validVals.length > 1) cls += ' best-cell';
            else if (v != null && v === worst && validVals.length > 1) cls += ' worst-cell';
            html += `<td class="${cls}" style="text-align:center">
                <span class="${f1Color(v)}">${fmt(v)}</span></td>`;
        }
        html += '</tr>';
    }
    html += '</tbody>';
    t.innerHTML = html;
}

// --- 2. Charts ---
function renderCharts() {
    // Grouped bar: F1 L1, F1 L2, HCR
    const labels = finished.map(m => m.name);
    const colors = finished.map(m => m.color);
    new Chart(document.getElementById('chart-metrics'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {label: 'F1 L1 macro', data: finished.map(m => m.metrics?.test_f1_l1_macro), backgroundColor: 'rgba(39,174,96,0.7)', borderColor: '#27ae60', borderWidth: 1},
                {label: 'F1 L2 macro', data: finished.map(m => m.metrics?.test_f1_l2_macro), backgroundColor: 'rgba(41,128,185,0.7)', borderColor: '#2980b9', borderWidth: 1},
                {label: 'HCR', data: finished.map(m => m.metrics?.test_hcr), backgroundColor: 'rgba(230,126,34,0.7)', borderColor: '#e67e22', borderWidth: 1},
            ],
        },
        options: {
            responsive: true,
            plugins: {title: {display: true, text: 'Métriques de test (F1 macro + HCR)'}},
            scales: {
                x: {ticks: {maxRotation: 45, minRotation: 30}},
                y: {beginAtZero: true, min: 0, max: 1.0, ticks: {callback: v => (v*100).toFixed(0)+'%'}},
            },
        },
    });

    // Courbes de convergence (HYDRA uniquement — modèles avec historique MLflow)
    const hydraWithHistory = finished.filter(m => m.family === 'hydra' && m.history);

    function makeConvergenceChart(canvasId, metricKey, title, yScaleOpts, modelsPool, subtitleText) {
        const pool = modelsPool ?? hydraWithHistory;
        const withMetric = pool.filter(m => m.history[metricKey]);
        if (withMetric.length === 0) return;
        const datasets = withMetric.map(m => ({
            label: m.name,
            data: m.history[metricKey].values.map((v, i) => ({x: m.history[metricKey].epochs[i], y: v})),
            borderColor: m.color, backgroundColor: m.color + '33',
            fill: false, tension: 0.3, pointRadius: 4,
        }));
        const pluginsOpts = {title: {display: true, text: title}};
        if (subtitleText) pluginsOpts.subtitle = {display: true, text: subtitleText, color: '#888', font: {size: 11}};
        new Chart(document.getElementById(canvasId), {
            type: 'line',
            data: {datasets},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: pluginsOpts,
                scales: {
                    x: {type: 'linear', min: 1, max: 10, title: {display: true, text: 'Epoch'}, ticks: {stepSize: 1}},
                    y: yScaleOpts,
                },
            },
        });
    }

    const pctF2 = v => (v * 100).toFixed(1) + '%';
    makeConvergenceChart('chart-convergence', 'val_f1_l2_macro',
        'Convergence — Val F1 L2 macro (epoch 1→10)',
        {min: 0.3, title: {display: true, text: 'F1 L2 macro'}, ticks: {callback: pctF2}});
    makeConvergenceChart('chart-convergence-l1', 'val_f1_l1_macro',
        'Convergence — Val F1 L1 macro (epoch 1→10)',
        {min: 0.84, title: {display: true, text: 'F1 L1 macro'}, ticks: {callback: pctF2}});
    // Train Loss : double axe Y — standard (gauche, log) vs hierarchy (droite, log, somme brute)
    const lossStd  = hydraWithHistory.filter(m => !m.id.includes('hierarchy') && m.history?.train_loss);
    const lossHier = hydraWithHistory.filter(m =>  m.id.includes('hierarchy') && m.history?.train_loss);
    if (lossStd.length > 0 || lossHier.length > 0) {
        const lossDatasets = [
            ...lossStd.map(m => ({
                label: m.name,
                data: m.history.train_loss.values.map((v, i) => ({x: m.history.train_loss.epochs[i], y: v})),
                borderColor: m.color, backgroundColor: m.color + '33',
                fill: false, tension: 0.3, pointRadius: 4, yAxisID: 'y',
            })),
            ...lossHier.map(m => ({
                label: m.name + ' ▶',
                data: m.history.train_loss.values.map((v, i) => ({x: m.history.train_loss.epochs[i], y: v})),
                borderColor: m.color, backgroundColor: m.color + '33',
                borderDash: [6, 3], fill: false, tension: 0.3, pointRadius: 4, yAxisID: 'y2',
            })),
        ];
        new Chart(document.getElementById('chart-convergence-loss'), {
            type: 'line',
            data: {datasets: lossDatasets},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {display: true, text: 'Train Loss (epoch 1→10)'},
                    subtitle: {display: true,
                        text: 'Trait plein = standard (axe gauche) · Pointillé ▶ = hierarchy (axe droit, somme brute)',
                        color: '#888', font: {size: 11}},
                },
                scales: {
                    x: {type: 'linear', min: 1, max: 10, title: {display: true, text: 'Epoch'}, ticks: {stepSize: 1}},
                    y:  {type: 'logarithmic', position: 'left',
                         title: {display: true, text: 'Standard (log)'},
                         ticks: {callback: v => v >= 1 ? v.toExponential(1) : v.toFixed(3)}},
                    y2: {type: 'logarithmic', position: 'right',
                         title: {display: true, text: 'Hierarchy (log)'},
                         ticks: {callback: v => v.toExponential(2)},
                         grid: {drawOnChartArea: false}},
                },
            },
        });
    }
}

// --- 3. Per-class comparison (generic for L1 and L2) ---
function renderPerClassGeneric(opts) {
    const {dataKey, tableId, searchId, sortId, countId, maxRows} = opts;
    const modelsWithData = finished.filter(m => m[dataKey]);
    if (modelsWithData.length === 0) {
        if (countId) document.getElementById(countId).textContent = 'Aucune donnée disponible';
        return;
    }

    const allClasses = [...new Set(modelsWithData.flatMap(m => Object.keys(m[dataKey])))];
    let tableData = allClasses.map(cls => {
        const row = {name: cls};
        for (const m of finished) row[m.id] = m[dataKey]?.[cls] ?? null;
        const vals = finished.map(m => m[dataKey]?.[cls]).filter(v => v != null);
        row._avg = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
        row._diff = vals.length > 1 ? Math.max(...vals) - Math.min(...vals) : 0;
        return row;
    });

    function renderTable(data) {
        const shown = maxRows ? data.slice(0, maxRows) : data;
        if (countId) {
            const msg = maxRows && data.length > maxRows
                ? `${shown.length} / ${data.length} classes affichées`
                : `${data.length} classes`;
            document.getElementById(countId).textContent = msg;
        }
        let html = '<thead><tr><th>Classe</th>';
        for (const m of finished) html += `<th style="text-align:center"><span style="color:${m.color}">●</span> ${m.name}</th>`;
        if (finished.length > 1) html += '<th style="text-align:center">Écart</th>';
        html += '</tr></thead><tbody>';
        for (const row of shown) {
            html += `<tr><td style="white-space:nowrap;font-size:0.85em">${row.name}</td>`;
            for (const m of finished) {
                const v = row[m.id];
                const cls = f1Color(v);
                html += `<td style="text-align:center" class="${cls}">${v != null ? v.toFixed(4) : '--'}</td>`;
            }
            if (finished.length > 1) {
                const vals = finished.map(m => row[m.id]).filter(v => v != null);
                const diff = vals.length > 1 ? Math.max(...vals) - Math.min(...vals) : 0;
                html += `<td style="text-align:center;font-size:0.85em">${diff.toFixed(4)}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody>';
        document.getElementById(tableId).innerHTML = html;
    }

    function sortAndFilter() {
        const search = document.getElementById(searchId).value.toLowerCase();
        const sort = document.getElementById(sortId).value;
        let data = tableData.filter(r => r.name.toLowerCase().includes(search));
        if (sort === 'alpha') data.sort((a,b) => a.name.localeCompare(b.name));
        else if (sort === 'best-first') data.sort((a,b) => b._avg - a._avg);
        else if (sort === 'worst-first') data.sort((a,b) => a._avg - b._avg);
        else if (sort === 'diff') data.sort((a,b) => b._diff - a._diff);
        renderTable(data);
    }

    document.getElementById(searchId).addEventListener('input', sortAndFilter);
    document.getElementById(sortId).addEventListener('change', sortAndFilter);
    sortAndFilter();
}

function renderPerClassComparison() {
    renderPerClassGeneric({
        dataKey: 'f1_per_class_l1', tableId: 'per-class-table',
        searchId: 'class-search', sortId: 'class-sort', countId: null, maxRows: null,
    });
}

function renderPerClassL2Comparison() {
    renderPerClassGeneric({
        dataKey: 'f1_per_class_l2', tableId: 'per-class-l2-table',
        searchId: 'class-l2-search', sortId: 'class-l2-sort', countId: 'class-l2-count', maxRows: 100,
    });
}

// --- 4. Accordions ---
function renderAccordionBlock(modelList) {
    let html = '';
    for (const m of modelList) {
        const metricsLine = [
            m.metrics?.test_f1_l1_macro != null ? `F1 L1: ${fmt(m.metrics.test_f1_l1_macro)}` : null,
            m.metrics?.test_f1_l2_macro != null ? `F1 L2: ${fmt(m.metrics.test_f1_l2_macro)}` : null,
            m.metrics?.test_hcr != null ? `HCR: ${fmt(m.metrics.test_hcr)}` : null,
        ].filter(Boolean).join(' · ');

        html += `<details><summary>
            <span style="color:${m.color}">●</span> ${m.name}
            <span style="font-weight:400;color:#6c757d;margin-left:12px">${metricsLine}</span>
            ${badgeHtml(m.status)}
        </summary><div class="detail-content">`;
        html += `<p style="color:#6c757d">${m.subtitle || ''}</p>`;

        if (m.images && Object.keys(m.images).length > 0) {
            const imgLabels = {
                training_curves: 'Courbes d\'entraînement',
                confusion_matrix_l1: 'Matrice de confusion L1',
                confusion_matrix_l2_top50: 'Matrice de confusion L2 (top 50)',
                f1_per_class_l1: 'F1 par classe L1',
                f1_per_class_l2: 'F1 par classe L2',
            };
            html += '<div class="img-grid">';
            for (const [key, src] of Object.entries(m.images)) {
                html += `<div class="img-container">
                    <img src="${src}" alt="${imgLabels[key] || key}" onclick="openOverlay(this)">
                    <div class="img-caption">${imgLabels[key] || key}</div>
                </div>`;
            }
            html += '</div>';
            const reportPath = `hydra-${m.id.replace('hydra-','').replace('-camembert','')}-camembert/${m.id}.html`;
            html += `<p style="margin-top:12px"><a href="${reportPath}">Voir le rapport individuel →</a></p>`;
        } else {
            html += '<table class="config-table"><thead><tr><th>Paramètre</th><th>Valeur</th></tr></thead><tbody>';
            for (const [k,v] of Object.entries(m.training.params || {})) {
                html += `<tr><td>${k}</td><td>${v}</td></tr>`;
            }
            html += '</tbody></table>';
        }
        html += '</div></details>';
    }
    return html;
}

function renderAccordions() {
    const container = document.getElementById('accordions');
    const titleOnly = finished.filter(m => m.family.endsWith('-title'));
    const fullText  = finished.filter(m => !m.family.endsWith('-title'));
    let html = '';
    if (titleOnly.length > 0) {
        html += '<h3 style="color:#5dade2;border-left:4px solid #5dade2;padding-left:10px">Title-only</h3>';
        html += renderAccordionBlock(titleOnly);
    }
    if (fullText.length > 0) {
        html += '<h3 style="color:#3498db;border-left:4px solid #3498db;padding-left:10px;margin-top:24px">Title + Description</h3>';
        html += renderAccordionBlock(fullText);
    }
    container.innerHTML = html;
}

// --- 5. Config Table ---
function renderConfigTable(modelList, tableId, allKeysGlobal) {
    const tbl = document.getElementById(tableId);
    if (!tbl) return;
    if (modelList.length === 0) {
        tbl.innerHTML = '<tbody><tr><td style="text-align:center;padding:20px;color:#6c757d">Aucun modèle dans cette catégorie.</td></tr></tbody>';
        return;
    }
    const important = ['variant','encoder','epochs','lr','batch_size','max_length','dropout','hidden_dim','level_weight_l1','warmup_ratio','weight_decay','fp16'];
    const sorted = important.filter(k => allKeysGlobal.includes(k)).concat(allKeysGlobal.filter(k => !important.includes(k)));

    let html = '<thead><tr><th>Paramètre</th>';
    for (const m of modelList) html += `<th style="text-align:center"><span style="color:${m.color}">●</span> ${m.name}</th>`;
    html += '</tr></thead><tbody>';
    html += '<tr><td style="font-weight:600">Famille</td>';
    for (const m of modelList) html += `<td style="text-align:center">${m.family.startsWith('hydra') ? 'HYDRA' : 'Baseline'}</td>`;
    html += '</tr>';
    for (const key of sorted) {
        html += `<tr><td style="font-weight:600">${key}</td>`;
        for (const m of modelList) html += `<td style="text-align:center">${m.training.params?.[key] ?? '--'}</td>`;
        html += '</tr>';
    }
    html += '</tbody>';
    tbl.innerHTML = html;
}

// --- Overlay Zoom (verbatim from individual reports) ---
let scale = 1, panX = 0, panY = 0, dragging = false, startX, startY;
const overlay = document.getElementById('overlay');
const oImg = document.getElementById('overlay-img');

function openOverlay(img) {
    oImg.src = img.src;
    overlay.classList.add('active');
    resetView();
}
function closeOverlay() { overlay.classList.remove('active'); }
function resetView() {
    scale = 1; panX = 0; panY = 0;
    oImg.style.left = '50%'; oImg.style.top = '50%';
    oImg.style.transform = 'translate(-50%, -50%) scale(1)';
    document.getElementById('zoom-level').textContent = '100%';
}
function applyTransform() {
    oImg.style.left = '50%'; oImg.style.top = '50%';
    oImg.style.transform = `translate(calc(-50% + ${panX}px), calc(-50% + ${panY}px)) scale(${scale})`;
    document.getElementById('zoom-level').textContent = Math.round(scale*100) + '%';
}
function zoomBy(factor) { scale *= factor; applyTransform(); }
overlay.addEventListener('wheel', e => { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.15 : 1/1.15); });
oImg.addEventListener('mousedown', e => { dragging = true; startX = e.clientX - panX; startY = e.clientY - panY; oImg.classList.add('grabbing'); });
window.addEventListener('mousemove', e => { if (dragging) { panX = e.clientX - startX; panY = e.clientY - startY; applyTransform(); } });
window.addEventListener('mouseup', () => { dragging = false; oImg.classList.remove('grabbing'); });
oImg.addEventListener('dblclick', resetView);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeOverlay(); });
overlay.addEventListener('click', e => { if (e.target === overlay) closeOverlay(); });

// --- Init ---
document.getElementById('gen-date').textContent = MODELS_DATA.meta.generated;
const titleOnlyModels = finished.filter(m => m.family.endsWith('-title'));
const fullTextModels  = finished.filter(m => !m.family.endsWith('-title'));
renderRecapTable(titleOnlyModels, 'recap-table-title-only');
renderRecapTable(fullTextModels,  'recap-table-full-text');
renderCharts();
renderPerClassComparison();
renderPerClassL2Comparison();
renderAccordions();
const allConfigKeys = [...new Set(finished.flatMap(m => Object.keys(m.training.params || {})))];
renderConfigTable(titleOnlyModels, 'config-table-title-only', allConfigKeys);
renderConfigTable(fullTextModels,  'config-table-full-text',  allConfigKeys);
</script>
</body>
</html>"""


def main():
    print("Découverte des modèles HYDRA...")
    hydra_models = discover_hydra_models()
    print(f"  → {len(hydra_models)} variante(s) HYDRA standard")

    print("Découverte des modèles HYDRA hierarchy...")
    hydra_hier_models = discover_hydra_hierarchy_models()
    print(f"  → {len(hydra_hier_models)} variante(s) HYDRA hierarchy")

    print("Découverte des baselines...")
    baseline_models = discover_baseline_models()
    print(f"  → {len(baseline_models)} baseline(s)")

    print("Découverte de FlatHydra...")
    flat_hydra_models = discover_flat_hydra_model()
    print(f"  → {len(flat_hydra_models)} modèle(s) FlatHydra")

    print("Découverte des modèles HYDRA Title-only...")
    hydra_title_models = discover_hydra_title_models()
    print(f"  → {len(hydra_title_models)} variante(s) HYDRA Title-only")

    print("Découverte des modèles HYDRA Hierarchy Title-only...")
    hydra_title_hier_models = discover_hydra_title_hierarchy_models()
    print(f"  → {len(hydra_title_hier_models)} variante(s) HYDRA Hierarchy Title-only")

    print("Découverte de FlatHydra Title-only...")
    flat_hydra_title_models = discover_flat_hydra_title_model()
    print(f"  → {len(flat_hydra_title_models)} modèle(s) FlatHydra Title-only")

    print("Découverte des baselines Title-only...")
    baseline_title_models = discover_baseline_title_models()
    print(f"  → {len(baseline_title_models)} baseline(s) Title-only")

    models_data = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "project": "Rakuten Classification Hiérarchique",
            "num_classes_l1": 157,
            "num_classes_l2": 6126,
        },
        "models": (
            hydra_models
            + hydra_hier_models
            + flat_hydra_models
            + hydra_title_models
            + hydra_title_hier_models
            + flat_hydra_title_models
            + baseline_title_models
            + baseline_models
        ),
    }

    print("Génération du HTML...")
    html = render_html(models_data)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1024 / 1024
    print(f"  → {OUTPUT_HTML} ({size_mb:.1f} Mo)")
    print(f"  → {len(models_data['models'])} modèle(s) dans la comparaison")


if __name__ == "__main__":
    main()
