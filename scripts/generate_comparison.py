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

import yaml

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
    "metrics_global.png",
    "l2_accuracy_distribution.png",
]

SKIP_PARAMS = {"gpu_type", "num_classes_l1", "num_classes_l2"}

EVAL_TEST_METRICS = RESULTS_DIR / "hydra_eval_test_metrics.json"
HYDRA_HIERARCHY_METRICS = RESULTS_DIR / "hydra_eval_hierarchy_metrics.json"
HYDRA_MBERT_METRICS = RESULTS_DIR / "hydra_eval_test_metrics_mbert.json"
HLPD_RESULTS_DIR = RESULTS_DIR / "hlpd"

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

    flat_f1_l1 = json.loads((FLAT_HYDRA_RESULTS_DIR / "f1_per_class_l1.json").read_text()) \
        if (FLAT_HYDRA_RESULTS_DIR / "f1_per_class_l1.json").exists() else None
    flat_f1_l2 = json.loads((FLAT_HYDRA_RESULTS_DIR / "f1_per_class_l2.json").read_text()) \
        if (FLAT_HYDRA_RESULTS_DIR / "f1_per_class_l2.json").exists() else None

    return [{
        "id": "flat-hydra-camembert",
        "name": "CamemBERT Flat",
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
        "f1_per_class_l1": flat_f1_l1,
        "f1_per_class_l2": flat_f1_l2,
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

        title_results_dir = RESULTS_DIR / f"hydra-{variant}-title-camembert"
        f1_pc_l1, f1_pc_l2 = None, None
        if title_results_dir.exists():
            p1 = title_results_dir / "f1_per_class_l1.json"
            p2 = title_results_dir / "f1_per_class_l2.json"
            if p1.exists():
                f1_pc_l1 = json.loads(p1.read_text())
            if p2.exists():
                f1_pc_l2 = json.loads(p2.read_text())

        images = {}
        if title_results_dir.exists():
            for fname in IMAGE_FILES:
                key = fname.replace(".png", "")
                encoded = encode_image(title_results_dir / fname)
                if encoded:
                    images[key] = encoded

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
            "f1_per_class_l1": f1_pc_l1,
            "f1_per_class_l2": f1_pc_l2,
            "images": images,
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
        has_eval = bool(test_metrics.get(variant))

        # Poids éventuellement archivés sur HF (rakuten-models) : les métriques
        # d'éval suffisent à inclure le modèle même sans dossier local.
        if not model_dir.exists() and not has_eval:
            continue

        if complete_txt.exists() or (not model_dir.exists() and has_eval):
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

        hier_results_dir = RESULTS_DIR / f"hydra-{variant}-title-hier-camembert"
        f1_pc_l1, f1_pc_l2 = None, None
        if hier_results_dir.exists():
            p1 = hier_results_dir / "f1_per_class_l1.json"
            p2 = hier_results_dir / "f1_per_class_l2.json"
            if p1.exists():
                f1_pc_l1 = json.loads(p1.read_text())
            if p2.exists():
                f1_pc_l2 = json.loads(p2.read_text())

        hier_images = {}
        if hier_results_dir.exists():
            for fname in IMAGE_FILES:
                key = fname.replace(".png", "")
                encoded = encode_image(hier_results_dir / fname)
                if encoded:
                    hier_images[key] = encoded

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
            "f1_per_class_l1": f1_pc_l1,
            "f1_per_class_l2": f1_pc_l2,
            "images": hier_images,
        })

    return models


def discover_flat_hydra_title_model() -> list[dict]:
    """Découvre le modèle FlatHydra Title-only depuis models/flat_hydra_title/."""
    history_path = FLAT_HYDRA_TITLE_MODELS_DIR / "metrics.json"
    # Poids/historique éventuellement archivés sur HF (rakuten-models) : les
    # métriques d'éval de results/ suffisent à inclure le modèle.
    if not history_path.exists() and not (FLAT_HYDRA_TITLE_RESULTS_DIR / "test_metrics.json").exists():
        return []

    history: dict = {}
    val_metrics: dict = {}
    if history_path.exists():
        h = json.loads(history_path.read_text())
        epochs = h.get("epoch", [])
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

    title_f1_l1 = json.loads((FLAT_HYDRA_TITLE_RESULTS_DIR / "f1_per_class_l1.json").read_text()) \
        if (FLAT_HYDRA_TITLE_RESULTS_DIR / "f1_per_class_l1.json").exists() else None
    title_f1_l2 = json.loads((FLAT_HYDRA_TITLE_RESULTS_DIR / "f1_per_class_l2.json").read_text()) \
        if (FLAT_HYDRA_TITLE_RESULTS_DIR / "f1_per_class_l2.json").exists() else None

    return [{
        "id": "flat-hydra-title-camembert",
        "name": "CamemBERT Flat (Title-only)",
        "family": "baseline-title",
        "subtitle": "CamemBERT — classification plate (1 tête) — Title-only (designation seule)",
        "status": "finished",
        "color": "#f5b041",
        "metrics": test_m,
        "val_metrics": val_metrics,
        "training": {"params": params},
        "history": history if history else None,
        "f1_per_class_l1": title_f1_l1,
        "f1_per_class_l2": title_f1_l2,
        "images": images,
    }]


def discover_hydra_mbert_models() -> list[dict]:
    """Découvre HYDRA Local mBERT depuis le JSON produit par eval_hierarchy.py."""
    if not HYDRA_MBERT_METRICS.exists():
        return []

    raw = json.loads(HYDRA_MBERT_METRICS.read_text())
    m = raw.get("local", {})
    if not m:
        return []

    results_dir = RESULTS_DIR / "hydra-local-mbert"

    f1_per_class_l1, f1_per_class_l2 = None, None
    if results_dir.exists():
        p1 = results_dir / "f1_per_class_l1.json"
        p2 = results_dir / "f1_per_class_l2.json"
        if p1.exists():
            f1_per_class_l1 = json.loads(p1.read_text())
        if p2.exists():
            f1_per_class_l2 = json.loads(p2.read_text())

    images = {}
    for fname in IMAGE_FILES:
        key = fname.replace(".png", "")
        encoded = encode_image(results_dir / fname)
        if encoded:
            images[key] = encoded

    return [{
        "id": "hydra-local-mbert",
        "name": "HYDRA Local (mBERT)",
        "family": "hydra",
        "subtitle": "mBERT — variante local — bert-base-multilingual-cased",
        "status": "finished",
        "color": "#e74c3c",
        "metrics": {
            "test_f1_l1_macro": m.get("f1_l1_macro"),
            "test_f1_l2_macro": m.get("f1_l2_macro"),
            "test_hcr": m.get("hcr"),
            "test_constrained_f1_macro": m.get("constrained_f1_macro"),
        },
        "val_metrics": {},
        "training": {"params": {"encoder": "bert-base-multilingual-cased", "variant": "local"}},
        "history": None,
        "f1_per_class_l1": f1_per_class_l1,
        "f1_per_class_l2": f1_per_class_l2,
        "images": images,
    }]


def _hlpd_config_to_params(cfg: dict) -> dict:
    """Extrait les hyperparamètres HLPD pertinents pour la table #config."""
    keys = (
        "encoder", "num_decoder_layers", "nhead", "ffn_dim",
        "batch_size", "epochs", "lr", "weight_decay", "max_length",
        "precision", "gradient_accumulation_steps", "seed",
    )
    return {k: cfg[k] for k in keys if k in cfg}


def discover_hlpd_model() -> list[dict]:
    """Découvre HLPD-CamemBERT depuis results/hlpd/test_metrics.json."""
    test_path = HLPD_RESULTS_DIR / "test_metrics.json"
    if not test_path.exists():
        return []

    test_m = json.loads(test_path.read_text())
    val_path = HLPD_RESULTS_DIR / "val_metrics.json"
    val_m = json.loads(val_path.read_text()) if val_path.exists() else {}

    cfg_path = Path("configs/hlpd/hlpd_3090.yaml")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    images = {}
    for fname in IMAGE_FILES:
        key = fname.replace(".png", "")
        encoded = encode_image(HLPD_RESULTS_DIR / fname)
        if encoded:
            images[key] = encoded

    return [{
        "id": "hlpd-camembert",
        "name": "HLPD-CamemBERT",
        "family": "hlpd",
        "subtitle": "Décodeur multi-label hiérarchique — ZLPR + biais hiérarchique (ECIR 2025)",
        "status": "finished",
        "color": "#9b59b6",
        "metrics": {
            "test_f1_l1_macro": test_m.get("f1_l1_macro"),
            "test_f1_l2_macro": test_m.get("f1_l2_macro"),
            "test_hcr": test_m.get("hcr"),
        },
        "val_metrics": {
            "f1_l1_macro": val_m.get("f1_l1_macro"),
            "f1_l2_macro": val_m.get("f1_l2_macro"),
            "hcr": val_m.get("hcr"),
        },
        "training": {"params": _hlpd_config_to_params(cfg)},
        "history": None,
        "f1_per_class_l1": None,
        "f1_per_class_l2": None,
        "images": images,
    }]


CLASS_METADATA_PATH = RESULTS_DIR / "class_metadata.json"


def render_html(models_data: dict) -> str:
    data_json = json.dumps(models_data, ensure_ascii=False, indent=2)
    class_meta = {}
    if CLASS_METADATA_PATH.exists():
        class_meta = json.loads(CLASS_METADATA_PATH.read_text(encoding="utf-8"))
    class_meta_json = json.dumps(class_meta, ensure_ascii=False)
    return (
        HTML_TEMPLATE
        .replace("/* __MODELS_DATA__ */", data_json)
        .replace("/* __CLASS_META__ */", class_meta_json)
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Comparaison des modèles — Rakuten Classification Hiérarchique</title>
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
@media print {
    .table-container { overflow: visible; }
    nav, .view-toggle { display: none; }
}
.recap-compact td, .recap-compact th { padding: 6px 8px; font-size: 0.88em; }
.recap-compact th { white-space: nowrap; }
.view-toggle {
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    padding: 4px 10px; font-size: 0.85em; cursor: pointer; margin-bottom: 4px;
}
.view-toggle:hover { background: #e9ecef; }
</style>
</head>
<body>

<h1>Comparaison des modèles — Classification Hiérarchique Rakuten</h1>
<p style="color:#6c757d">
    157 catégories L1 · 6 126 catégories L2 · 8M+ produits<br>
    Télécom Paris × Rakuten France · Généré le <span id="gen-date"></span>
</p>

<div style="background:#fff3cd;border:1px solid #ffc107;border-left:5px solid #ffc107;
    border-radius:8px;padding:14px 18px;margin:16px 0;color:#664d03">
    <strong>⚠ Benchmarks v1 — fuite de données identifiée (audit du 12/06/2026).</strong>
    11,2&nbsp;% des lignes de val/test avaient un jumeau exact dans le train
    (98&nbsp;% avec le même label) : les scores absolus de cette page sont
    <strong>optimistes de +1,5 à +3 points</strong>. Les comparaisons
    <em>entre</em> modèles restent valides (même fuite pour tous). Le dataset
    redressé (v2, fuite purgée) servira aux prochains benchmarks — leurs scores,
    plus bas, ne seront pas comparables à ceux-ci : la baisse correspond à la
    disparition de la triche, pas à une régression. Détails :
    <code>docs/METRICS.md</code>.
</div>

<nav>
    <a href="#synthese">Synthèse</a>
    <a href="#recap">Récapitulatif</a>
    <a href="#impact-description">Apport de la description</a>
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

<section id="synthese" style="background:linear-gradient(135deg,#f8fff9 0%,#f0f8ff 100%);border:1px solid #d4edda;border-radius:10px;padding:20px;margin:20px 0">
    <h2 style="margin-top:0;border:none;padding:0;color:#27ae60">Synthèse — modèles champions</h2>
    <p style="color:#6c757d;font-size:0.88em;margin:6px 0 14px">
        Meilleur modèle par catégorie sur le test set. <em>F1 = métrique principale (macro, toutes classes).</em>
    </p>
    <div id="champions-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px"></div>
</section>

<section id="recap">
    <h2>Tableau récapitulatif</h2>
    <h3 style="color:#5dade2;border-left:4px solid #5dade2;padding-left:10px">Title-only (<code>designation</code> seule)</h3>
    <p style="color:#6c757d;font-size:0.9em">Modèles entraînés sur le titre du produit uniquement (benchmark sans description).</p>
    <div style="display:flex;justify-content:flex-end;margin:4px 0">
        <button class="view-toggle" id="toggle-title-only">Vue large ▶</button>
    </div>
    <div class="table-container"><table id="recap-table-title-only"></table></div>
    <h3 style="color:#3498db;border-left:4px solid #3498db;padding-left:10px;margin-top:30px">Title + Description (<code>designation + description</code>)</h3>
    <p style="color:#6c757d;font-size:0.9em">Modèles entraînés sur la concaténation titre + description (régime principal).</p>
    <div style="display:flex;justify-content:flex-end;margin:4px 0">
        <button class="view-toggle" id="toggle-full-text">Vue large ▶</button>
    </div>
    <div class="table-container"><table id="recap-table-full-text"></table></div>
</section>

<section id="impact-description">
    <h2>Apport de la description</h2>
    <p style="color:#6c757d;font-size:0.95em">
        Que change l'ajout de la <code>description</code> au <code>title</code> ? Pour chaque modèle existant
        dans les deux régimes, on affiche les scores Title-only, Title + Description, et l'écart
        en points absolus + variation relative (style cours de bourse).
    </p>
    <p style="color:#6c757d;font-size:0.85em;margin-top:-8px">
        Δ &gt; 0 (vert) = la description aide · Δ &lt; 0 (rouge) = la description dégrade ·
        cellules vides = pas de variante Title-only entraînée. Tri par Δ F1 L2 macro décroissant.
    </p>
    <div class="table-container"><table id="impact-table"></table></div>
</section>

<section id="per-class">
    <h2>Comparaison par classe L1 (157 classes)</h2>
    <div class="controls-bar">
        <input type="text" id="class-search" placeholder="Filtrer par nom de classe...">
        <select id="class-sort">
            <option value="worst-first">Pire F1 d'abord (défaut)</option>
            <option value="best-first">Meilleur F1 d'abord</option>
            <option value="alpha">Alphabétique</option>
            <option value="diff">Plus grand écart</option>
            <option value="support">Support décroissant</option>
        </select>
    </div>
    <div class="table-container"><table id="per-class-table"></table></div>
</section>
<section id="per-class-l2">
    <h2>Comparaison par classe L2 (6 126 classes)</h2>
    <div class="controls-bar">
        <input type="text" id="class-l2-search" placeholder="Filtrer par nom ou catégorie L1...">
        <select id="class-l2-sort">
            <option value="worst-first">Pire F1 d'abord (défaut)</option>
            <option value="best-first">Meilleur F1 d'abord</option>
            <option value="alpha">Alphabétique</option>
            <option value="diff">Plus grand écart</option>
            <option value="support">Support décroissant</option>
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
const CLASS_META = /* __CLASS_META__ */;

// --- Utils ---
const fmt = (v, pct=true) => v != null ? (pct ? (v*100).toFixed(2)+'%' : v.toFixed(4)) : '--';
const f1Color = (v) => v == null ? '' : v >= 0.8 ? 'f1-high' : v >= 0.5 ? 'f1-mid' : 'f1-low';
const badgeHtml = (status) => {
    const cls = {'finished':'badge-finished','running':'badge-running','planned':'badge-planned'}[status] || '';
    const txt = {'finished':'Terminé','running':'En cours','planned':'Planifié'}[status] || status;
    return `<span class="badge ${cls}">${txt}</span>`;
};
const finished = MODELS_DATA.models.filter(m => m.status !== 'planned');
const familyLabel = (family) => {
    if (family.startsWith('hydra')) return 'HYDRA';
    if (family === 'hlpd') return 'HLPD';
    return 'Baseline';
};

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
    for (const m of modelList) html += `<td style="text-align:center">${familyLabel(m.family)}</td>`;
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

// --- 0. Champions ---
function _championCard(label, model, metricVal, metricLabel) {
    if (!model) return '';
    return `<div style="background:#fff;border:1px solid #e9ecef;border-radius:8px;
        padding:14px 16px;border-left:4px solid ${model.color}">
        <div style="font-size:0.75em;color:#6c757d;font-weight:700;
            text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px">${label}</div>
        <div style="font-size:1em;font-weight:600;color:#2c3e50;margin-bottom:4px">
            <span style="color:${model.color}">●</span> ${model.name}</div>
        <div style="font-size:1.55em;font-weight:700;color:#27ae60;line-height:1.2">
            ${(metricVal*100).toFixed(2)}<span style="font-size:0.6em">%</span></div>
        <div style="font-size:0.78em;color:#6c757d;margin-top:2px">${metricLabel}</div>
    </div>`;
}

function renderChampions() {
    const grid = document.getElementById('champions-grid');
    if (!grid) return;
    const best = (arr, getter) => arr.reduce((a, b) => (getter(b) ?? -1) > (getter(a) ?? -1) ? b : a, arr[0]);
    const get = (m, key) => m.metrics?.[key] ?? null;
    const fullText  = finished.filter(m => !m.family.endsWith('-title'));
    const titleOnly = finished.filter(m => m.family.endsWith('-title'));
    const baselines = finished.filter(m => m.family.startsWith('baseline'));
    const cards = [
        ['Champion F1 L2 (test)',
            best(fullText, m => get(m, 'test_f1_l2_macro')),
            m => get(m, 'test_f1_l2_macro'), 'F1 L2 macro — test set'],
        ['Champion HCR (test)',
            best(fullText, m => get(m, 'test_hcr')),
            m => get(m, 'test_hcr'), 'Hiérarchical Consistency Rate — test set'],
        ['Champion Title-only F1 L2',
            titleOnly.length ? best(titleOnly, m => get(m, 'test_f1_l2_macro')) : null,
            m => get(m, 'test_f1_l2_macro'), 'F1 L2 macro — title seule, test set'],
        ['Meilleure Baseline F1 L2',
            baselines.length ? best(baselines, m => get(m, 'test_f1_l2_macro')) : null,
            m => get(m, 'test_f1_l2_macro'), 'F1 L2 macro — baseline, test set'],
    ];
    grid.innerHTML = cards.map(([label, model, getter, metricLabel]) =>
        model ? _championCard(label, model, getter(model), metricLabel) : ''
    ).join('');
}

// --- 1bis. Impact Description (Title-only vs Title+Description) ---
const MODEL_PAIRS = [
    // [fullTextId, titleOnlyId|null]
    ['hydra-local-mbert', null],
    ['hlpd-camembert', null],
    ['hydra-local-hierarchy-camembert',  'hydra-local-title-hier-camembert'],
    ['hydra-nested-hierarchy-camembert', 'hydra-nested-title-hier-camembert'],
    ['hydra-global-hierarchy-camembert', 'hydra-global-title-hier-camembert'],
    ['hydra-local-camembert',  'hydra-local-title-camembert'],
    ['hydra-nested-camembert', 'hydra-nested-title-camembert'],
    ['hydra-global-camembert', 'hydra-global-title-camembert'],
    ['flat-hydra-camembert',   'flat-hydra-title-camembert'],
    ['tfidf-linearsvc',           'tfidf-title-linearsvc'],
    ['word2vec-linearsvc',        'word2vec-title-linearsvc'],
    ['fasttext-softmax-softmax',  'fasttext-softmax-title-softmax'],
    ['fasttext-supervised', null],
];

function _deltaCell(titleVal, fullVal) {
    if (titleVal == null || fullVal == null) return '<td style="text-align:center;color:#adb5bd">--</td>';
    const deltaPts = (fullVal - titleVal) * 100;     // points
    const deltaRel = titleVal !== 0 ? (fullVal - titleVal) / titleVal * 100 : 0;
    const arrow = deltaPts > 0 ? '▲' : (deltaPts < 0 ? '▼' : '–');
    const sign  = deltaPts > 0 ? '+' : '';
    const color = deltaPts > 0 ? '#27ae60' : (deltaPts < 0 ? '#e74c3c' : '#7f8c8d');
    // Intensité fond ∝ |Δ pts|, capée à 5 pts → alpha max 0.28
    const alpha = Math.min(Math.abs(deltaPts) / 5, 1) * 0.28;
    const bg = deltaPts > 0 ? `rgba(39,174,96,${alpha.toFixed(3)})`
            : deltaPts < 0 ? `rgba(231,76,60,${alpha.toFixed(3)})`
            : 'transparent';
    return `<td style="text-align:center;background:${bg};font-weight:600;color:${color};white-space:nowrap">
        ${arrow} ${sign}${deltaPts.toFixed(2)} pts<br>
        <span style="font-size:0.82em;font-weight:500">(${sign}${deltaRel.toFixed(1)}%)</span>
    </td>`;
}

function renderImpactTable() {
    const t = document.getElementById('impact-table');
    if (!t) return;
    const byId = Object.fromEntries(finished.map(m => [m.id, m]));
    const rows = [];
    for (const [fullId, titleId] of MODEL_PAIRS) {
        const full  = byId[fullId];
        if (!full) continue;
        const title = titleId ? byId[titleId] : null;
        rows.push({ full, title });
    }
    // Tri par Δ F1 L2 décroissant ; les modèles sans pendant title-only restent en haut
    rows.sort((a, b) => {
        const da = (a.title && a.full.metrics?.test_f1_l2_macro != null && a.title.metrics?.test_f1_l2_macro != null)
            ? a.full.metrics.test_f1_l2_macro - a.title.metrics.test_f1_l2_macro : Number.POSITIVE_INFINITY;
        const db = (b.title && b.full.metrics?.test_f1_l2_macro != null && b.title.metrics?.test_f1_l2_macro != null)
            ? b.full.metrics.test_f1_l2_macro - b.title.metrics.test_f1_l2_macro : Number.POSITIVE_INFINITY;
        return db - da;
    });

    let html = `<thead>
        <tr>
            <th rowspan="2" style="vertical-align:middle">Modèle</th>
            <th rowspan="2" style="vertical-align:middle;text-align:center">Famille</th>
            <th colspan="3" style="text-align:center;background:#eaf3fb">F1 L1 macro (test)</th>
            <th colspan="3" style="text-align:center;background:#fff5e6">F1 L2 macro (test)</th>
            <th colspan="3" style="text-align:center;background:#f0f0f0">HCR (test)</th>
        </tr>
        <tr>
            <th style="text-align:center;font-size:0.78em;background:#eaf3fb">Title</th>
            <th style="text-align:center;font-size:0.78em;background:#eaf3fb">Title+Desc</th>
            <th style="text-align:center;font-size:0.78em;background:#eaf3fb">Δ</th>
            <th style="text-align:center;font-size:0.78em;background:#fff5e6">Title</th>
            <th style="text-align:center;font-size:0.78em;background:#fff5e6">Title+Desc</th>
            <th style="text-align:center;font-size:0.78em;background:#fff5e6">Δ</th>
            <th style="text-align:center;font-size:0.78em;background:#f0f0f0">Title</th>
            <th style="text-align:center;font-size:0.78em;background:#f0f0f0">Title+Desc</th>
            <th style="text-align:center;font-size:0.78em;background:#f0f0f0">Δ</th>
        </tr>
    </thead><tbody>`;

    for (const { full, title } of rows) {
        const fM = full.metrics  || {};
        const tM = title ? (title.metrics || {}) : {};
        html += `<tr>
            <td><span style="color:${full.color}">●</span> ${full.name}</td>
            <td style="text-align:center">${familyLabel(full.family)}</td>`;
        for (const key of ['test_f1_l1_macro', 'test_f1_l2_macro', 'test_hcr']) {
            const tVal = title ? tM[key] : null;
            const fVal = fM[key];
            html += `<td style="text-align:center"><span class="${f1Color(tVal)}">${fmt(tVal)}</span></td>`;
            html += `<td style="text-align:center"><span class="${f1Color(fVal)}">${fmt(fVal)}</span></td>`;
            html += _deltaCell(tVal, fVal);
        }
        html += '</tr>';
    }
    html += '</tbody>';
    t.innerHTML = html;
    t.className = 'recap-compact';
}

function renderRecapTableCompact(modelList, tableId) {
    const t = document.getElementById(tableId);
    if (!t) return;
    if (modelList.length === 0) {
        t.innerHTML = '<tbody><tr><td style="text-align:center;padding:20px;color:#6c757d">Aucun modèle dans cette catégorie.</td></tr></tbody>';
        t.className = '';
        return;
    }
    const metrics = [
        {key: 'test_f1_l1_macro', label: 'F1 L1 test', main: false},
        {key: 'test_f1_l2_macro', label: 'F1 L2 test', main: true},
        {key: 'test_hcr', label: 'HCR test', main: false},
        {key: 'val_f1_l1_macro', label: 'F1 L1 val', src: 'val_metrics'},
        {key: 'val_f1_l2_macro', label: 'F1 L2 val', src: 'val_metrics'},
    ];
    const colVals = metrics.map(mdef => modelList.map(m => {
        const src = mdef.src === 'val_metrics' ? m.val_metrics : m.metrics;
        return src ? src[mdef.key] : null;
    }));
    const colBest  = colVals.map(vals => { const v = vals.filter(x => x != null); return v.length > 1 ? Math.max(...v) : null; });
    const colWorst = colVals.map(vals => { const v = vals.filter(x => x != null); return v.length > 1 ? Math.min(...v) : null; });

    let html = '<thead><tr><th>Modèle</th><th>Famille</th>';
    for (const mdef of metrics) html += `<th style="text-align:center">${mdef.label}</th>`;
    html += '</tr></thead><tbody>';
    for (let i = 0; i < modelList.length; i++) {
        const m = modelList[i];
        html += `<tr><td><span style="color:${m.color}">●</span> ${m.name} ${badgeHtml(m.status)}</td>`;
        html += `<td style="text-align:center">${familyLabel(m.family)}</td>`;
        for (let j = 0; j < metrics.length; j++) {
            const v = colVals[j][i];
            let cls = metrics[j].main ? 'metric-main' : '';
            if (v != null && v === colBest[j])  cls += ' best-cell';
            else if (v != null && v === colWorst[j]) cls += ' worst-cell';
            html += `<td class="${cls}" style="text-align:center"><span class="${f1Color(v)}">${fmt(v)}</span></td>`;
        }
        html += '</tr>';
    }
    html += '</tbody>';
    t.innerHTML = html;
    t.className = 'recap-compact';
}

// --- 2. Per-class comparison (generic for L1 and L2) ---
function renderPerClassGeneric(opts) {
    const {dataKey, tableId, searchId, sortId, countId, pageSize, metaLevel, showParent} = opts;
    const modelsWithData = finished.filter(m => m[dataKey]);
    if (modelsWithData.length === 0) {
        if (countId) document.getElementById(countId).textContent = 'Aucune donnée disponible';
        return;
    }

    const metaMap = (CLASS_META && metaLevel) ? (CLASS_META[metaLevel] || {}) : {};

    const allClasses = [...new Set(modelsWithData.flatMap(m => Object.keys(m[dataKey])))];
    let tableData = allClasses.map(cls => {
        const row = {name: cls};
        for (const m of finished) row[m.id] = m[dataKey]?.[cls] ?? null;
        const vals = finished.map(m => m[dataKey]?.[cls]).filter(v => v != null);
        row._avg = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
        row._diff = vals.length > 1 ? Math.max(...vals) - Math.min(...vals) : 0;
        row._support = metaMap[cls]?.support_test ?? null;
        row._parent = metaMap[cls]?.parent_l1 ?? '';
        return row;
    });

    let currentData = [];
    let shownCount = 0;

    function renderTable(data, appendMode) {
        if (!appendMode) {
            shownCount = 0;
            currentData = data;
        }
        const batch = currentData.slice(shownCount, shownCount + pageSize);
        shownCount += batch.length;

        const total = currentData.length;
        if (countId) {
            document.getElementById(countId).textContent =
                shownCount < total
                    ? `${shownCount} / ${total} classes affichées`
                    : `${total} classe${total > 1 ? 's' : ''}`;
        }

        let html = '';
        if (!appendMode) {
            html += '<thead><tr><th>Classe</th>';
            if (showParent) html += '<th>Catégorie L1</th>';
            html += '<th style="text-align:right">Support test</th>';
            for (const m of finished) html += `<th style="text-align:center"><span style="color:${m.color}">●</span> ${m.name}</th>`;
            if (finished.length > 1) html += '<th style="text-align:center">Écart</th>';
            html += '</tr></thead><tbody id="' + tableId + '-body">';
        } else {
            html = '';
        }

        const tbody = appendMode ? document.getElementById(tableId + '-body') : null;
        let rowsHtml = '';
        for (const row of batch) {
            rowsHtml += `<tr><td style="white-space:nowrap;font-size:0.85em">${row.name}</td>`;
            if (showParent) rowsHtml += `<td style="font-size:0.8em;color:#6c757d">${row._parent || '--'}</td>`;
            rowsHtml += `<td style="text-align:right;font-size:0.82em;color:#495057">${row._support != null ? row._support.toLocaleString() : '--'}</td>`;
            for (const m of finished) {
                const v = row[m.id];
                const cls = f1Color(v);
                rowsHtml += `<td style="text-align:center" class="${cls}">${v != null ? (v * 100).toFixed(1) + '%' : '--'}</td>`;
            }
            if (finished.length > 1) {
                const vals = finished.map(m => row[m.id]).filter(v => v != null);
                const diff = vals.length > 1 ? Math.max(...vals) - Math.min(...vals) : 0;
                rowsHtml += `<td style="text-align:center;font-size:0.85em">${(diff * 100).toFixed(1)}%</td>`;
            }
            rowsHtml += '</tr>';
        }

        if (appendMode && tbody) {
            tbody.insertAdjacentHTML('beforeend', rowsHtml);
        } else {
            document.getElementById(tableId).innerHTML = html + rowsHtml + '</tbody>';
        }

        // Bouton "voir plus"
        const btnId = tableId + '-more-btn';
        const existing = document.getElementById(btnId);
        if (existing) existing.remove();
        if (shownCount < total) {
            const btn = document.createElement('button');
            btn.id = btnId;
            btn.textContent = `Afficher 50 de plus (${total - shownCount} restantes)`;
            btn.style.cssText = 'margin:12px 0;padding:8px 20px;border:1px solid #dee2e6;border-radius:6px;background:#f8f9fa;cursor:pointer;font-size:0.9em';
            btn.onclick = () => renderTable(null, true);
            document.getElementById(tableId).insertAdjacentElement('afterend', btn);
        }
    }

    function sortAndFilter() {
        const search = document.getElementById(searchId).value.toLowerCase();
        const sort = document.getElementById(sortId).value;
        let data = tableData.filter(r =>
            r.name.toLowerCase().includes(search) ||
            r._parent.toLowerCase().includes(search)
        );
        if (sort === 'alpha') data.sort((a,b) => a.name.localeCompare(b.name));
        else if (sort === 'best-first') data.sort((a,b) => b._avg - a._avg);
        else if (sort === 'worst-first') data.sort((a,b) => a._avg - b._avg);
        else if (sort === 'diff') data.sort((a,b) => b._diff - a._diff);
        else if (sort === 'support') data.sort((a,b) => (b._support ?? -1) - (a._support ?? -1));
        renderTable(data, false);
    }

    document.getElementById(searchId).addEventListener('input', sortAndFilter);
    document.getElementById(sortId).addEventListener('change', sortAndFilter);
    sortAndFilter();
}

function renderPerClassComparison() {
    renderPerClassGeneric({
        dataKey: 'f1_per_class_l1', tableId: 'per-class-table',
        searchId: 'class-search', sortId: 'class-sort', countId: null,
        pageSize: 157, metaLevel: 'l1', showParent: false,
    });
}

function renderPerClassL2Comparison() {
    renderPerClassGeneric({
        dataKey: 'f1_per_class_l2', tableId: 'per-class-l2-table',
        searchId: 'class-l2-search', sortId: 'class-l2-sort', countId: 'class-l2-count',
        pageSize: 50, metaLevel: 'l2', showParent: true,
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
    for (const m of modelList) html += `<td style="text-align:center">${familyLabel(m.family)}</td>`;
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

// Compact par défaut (optimisé capture PowerPoint)
const _recapState = {
    'recap-table-title-only': { compact: true, models: titleOnlyModels },
    'recap-table-full-text':  { compact: true, models: fullTextModels  },
};
function _renderRecap(tableId) {
    const s = _recapState[tableId];
    if (s.compact) renderRecapTableCompact(s.models, tableId);
    else           renderRecapTable(s.models, tableId);
}
_renderRecap('recap-table-title-only');
_renderRecap('recap-table-full-text');
renderChampions();

document.getElementById('toggle-title-only').addEventListener('click', function() {
    _recapState['recap-table-title-only'].compact = !_recapState['recap-table-title-only'].compact;
    _renderRecap('recap-table-title-only');
    this.textContent = _recapState['recap-table-title-only'].compact ? 'Vue large ▶' : '◀ Vue compacte';
});
document.getElementById('toggle-full-text').addEventListener('click', function() {
    _recapState['recap-table-full-text'].compact = !_recapState['recap-table-full-text'].compact;
    _renderRecap('recap-table-full-text');
    this.textContent = _recapState['recap-table-full-text'].compact ? 'Vue large ▶' : '◀ Vue compacte';
});

renderImpactTable();
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

    print("Découverte du modèle HYDRA Local mBERT...")
    hydra_mbert_models = discover_hydra_mbert_models()
    print(f"  → {len(hydra_mbert_models)} modèle(s) HYDRA mBERT")

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

    print("Découverte de HLPD-CamemBERT...")
    hlpd_models = discover_hlpd_model()
    print(f"  → {len(hlpd_models)} modèle(s) HLPD")

    models_data = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "project": "Rakuten Classification Hiérarchique",
            "num_classes_l1": 157,
            "num_classes_l2": 6126,
        },
        "models": (
            hlpd_models
            + hydra_models
            + hydra_hier_models
            + hydra_mbert_models
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
