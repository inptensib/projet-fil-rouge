#!/usr/bin/env python3
"""Génère le rapport HTML individuel d'une variante HYDRA.

Usage:
    python scripts/generate_report.py local
    python scripts/generate_report.py nested
    python scripts/generate_report.py global
"""

import argparse
import base64
import json
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

HYDRA_EXPERIMENTS = {
    "local": "1",
    "nested": "2",
    "global": "3",
}

IMAGE_FILES = [
    ("training_curves.png", "Courbes d'entraînement (loss, F1, HCR par epoch)"),
    ("confusion_matrix_l1.png", "Matrice de confusion L1 — top 30 classes (normalisée par ligne)"),
    ("confusion_matrix_l2_top50.png", "Matrice de confusion L2 — top 50 classes"),
    ("f1_per_class_l1.png", "F1 par classe L1 — top/bottom 20"),
    ("f1_per_class_l2.png", "F1 par classe L2 — top/bottom 20"),
]

SKIP_PARAMS = {"gpu_type", "num_classes_l1", "num_classes_l2"}


def encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def get_run_data(variant: str) -> dict:
    exp_id = HYDRA_EXPERIMENTS[variant]
    conn = sqlite3.connect(str(MLFLOW_DB))

    row = conn.execute(
        "SELECT run_uuid FROM runs WHERE experiment_id=? AND status='FINISHED' "
        "ORDER BY end_time DESC LIMIT 1", (exp_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise SystemExit(f"Aucun run FINISHED pour experiment_id={exp_id} (variant={variant})")

    run_id = row[0]
    metrics = dict(conn.execute(
        "SELECT key, value FROM latest_metrics WHERE run_uuid=?", (run_id,),
    ).fetchall())
    params = dict(conn.execute(
        "SELECT key, value FROM params WHERE run_uuid=?", (run_id,),
    ).fetchall())
    conn.close()

    return {"run_id": run_id, "metrics": metrics, "params": params}


def render_report(variant: str) -> str:
    results_dir = RESULTS_DIR / f"hydra-{variant}-camembert"
    run_data = get_run_data(variant)
    m = run_data["metrics"]
    p = {k: v for k, v in run_data["params"].items() if k not in SKIP_PARAMS}

    # Images par section
    def img_block(fname, caption):
        encoded = encode_image(results_dir / fname)
        if not encoded:
            return ""
        return (f'<div class="img-container">'
                f'<img src="{encoded}" alt="{caption}" onclick="openOverlay(this)">'
                f'<div class="img-caption">{caption}</div></div>')

    img_curves = img_block("training_curves.png", "Courbes d'entraînement (loss, F1, HCR par epoch)")
    img_cm_l1 = img_block("confusion_matrix_l1.png", "Matrice de confusion L1 — top 30 classes")
    img_cm_l2 = img_block("confusion_matrix_l2_top50.png", "Matrice de confusion L2 — top 50 classes")
    img_f1_l1 = img_block("f1_per_class_l1.png", "F1 par classe L1 — top/bottom 20")
    img_f1_l2 = img_block("f1_per_class_l2.png", "F1 par classe L2 — top/bottom 20")

    # F1 per class tables
    f1_json = results_dir / "f1_per_class_l1.json"
    f1_tables_html = ""
    if f1_json.exists():
        f1_data = json.loads(f1_json.read_text())
        sorted_f1 = sorted(f1_data.items(), key=lambda x: x[1])
        # Top 10
        f1_tables_html += '<h3>Top 10 classes L1 (meilleur F1)</h3><table><thead><tr><th>Classe</th><th>F1</th></tr></thead><tbody>'
        for name, score in sorted_f1[-10:][::-1]:
            color = "#27ae60" if score >= 0.8 else "#e67e22" if score >= 0.5 else "#e74c3c"
            f1_tables_html += f'<tr><td>{name}</td><td style="color:{color};font-weight:600">{score:.4f}</td></tr>'
        f1_tables_html += '</tbody></table>'
        # Bottom 10
        f1_tables_html += '<h3>Bottom 10 classes L1 (pire F1)</h3><table><thead><tr><th>Classe</th><th>F1</th></tr></thead><tbody>'
        for name, score in sorted_f1[:10]:
            color = "#27ae60" if score >= 0.8 else "#e67e22" if score >= 0.5 else "#e74c3c"
            f1_tables_html += f'<tr><td>{name}</td><td style="color:{color};font-weight:600">{score:.4f}</td></tr>'
        f1_tables_html += '</tbody></table>'

    # Explainability
    explain_file = results_dir / f"explainability-hydra-{variant}-camembert.html"
    explain_html = explain_file.read_text(encoding="utf-8") if explain_file.exists() else "<p>Non disponible</p>"

    # Config table
    important = ['variant', 'encoder', 'epochs', 'lr', 'batch_size', 'max_length',
                 'dropout', 'hidden_dim', 'level_weight_l1', 'warmup_ratio', 'weight_decay', 'fp16']
    sorted_keys = [k for k in important if k in p] + [k for k in p if k not in important]
    config_rows = ""
    for k in sorted_keys:
        config_rows += f"<tr><td>{k}</td><td>{p[k]}</td></tr>"

    # Metric helpers
    def fmt(key, pct=True):
        v = m.get(key)
        if v is None:
            return "--"
        return f"{v*100:.2f}%" if pct else f"{v:.4f}"

    variant_title = variant.capitalize()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>HYDRA {variant_title} — Rapport d'entraînement</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 20px 30px;
    color: #333; background: #fff; line-height: 1.6;
}}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 12px; }}
h2 {{ color: #2c3e50; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }}
.metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin: 20px 0;
}}
.metric-card {{
    background: #f8f9fa; border: 1px solid #e9ecef;
    border-radius: 8px; padding: 16px; text-align: center;
}}
.metric-card .value {{ font-size: 2em; font-weight: 700; color: #2c3e50; }}
.metric-card .label {{ font-size: 0.85em; color: #6c757d; margin-top: 4px; }}
.img-container {{
    text-align: center; margin: 20px 0;
    border: 1px solid #e9ecef; border-radius: 8px;
    padding: 12px; background: #fafafa;
}}
.img-container img {{
    max-width: 100%; height: auto; cursor: pointer;
    border-radius: 4px; transition: transform 0.2s;
}}
.img-container img:hover {{ transform: scale(1.02); }}
.img-caption {{ font-size: 0.85em; color: #6c757d; margin-top: 8px; font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th {{ background: #f1f3f5; padding: 8px; text-align: left; font-size: 0.85em; color: #495057; }}
td {{ border-bottom: 1px solid #f1f3f5; }}
.config-table td {{ padding: 6px 12px; }}
.config-table td:first-child {{ font-weight: 600; color: #495057; }}
nav {{ background: #f8f9fa; padding: 16px; border-radius: 8px; margin: 20px 0; }}
nav a {{ margin-right: 16px; color: #3498db; text-decoration: none; font-weight: 500; }}
nav a:hover {{ text-decoration: underline; }}

.overlay {{
    display: none; position: fixed; top: 0; left: 0;
    width: 100%; height: 100%; background: rgba(0,0,0,0.92);
    z-index: 1000; overflow: hidden;
}}
.overlay.active {{ display: block; }}
.overlay img {{ position: absolute; cursor: grab; transform-origin: 0 0; transition: none; }}
.overlay img.grabbing {{ cursor: grabbing; }}
.overlay .controls {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 10px; z-index: 1001;
}}
.overlay .controls button {{
    background: rgba(255,255,255,0.15); color: #fff; border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px; padding: 8px 16px; font-size: 16px; cursor: pointer;
    backdrop-filter: blur(4px);
}}
.overlay .controls button:hover {{ background: rgba(255,255,255,0.3); }}
.overlay .zoom-level {{ position: fixed; top: 20px; right: 20px; color: rgba(255,255,255,0.7); font-size: 14px; z-index: 1001; font-family: monospace; }}
.overlay .hint {{ position: fixed; top: 20px; left: 20px; color: rgba(255,255,255,0.5); font-size: 13px; z-index: 1001; }}
</style>
</head>
<body>

<h1>HYDRA {variant_title} — Rapport d'entraînement</h1>
<p style="color:#6c757d">
    CamemBERT — variante {variant} · 157 catégories L1 · 6 126 catégories L2 (dataset v1)<br>
    Télécom Paris × Rakuten France · Généré le {generated}
</p>

<nav>
    <a href="#metriques">Métriques</a>
    <a href="#courbes">Courbes</a>
    <a href="#confusion">Confusion</a>
    <a href="#f1-classes">F1 par classe</a>
    <a href="#explicabilite">Explicabilité</a>
    <a href="#config">Configuration</a>
    <a href="../comparison.html">← Comparaison</a>
</nav>

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

<h2 id="metriques">Métriques finales</h2>
<div class="metrics-grid">
    <div class="metric-card">
        <div class="value">{fmt("test_f1_l1_macro")}</div>
        <div class="label">Test F1 L1 macro</div>
    </div>
    <div class="metric-card">
        <div class="value">{fmt("test_f1_l2_macro")}</div>
        <div class="label">Test F1 L2 macro</div>
    </div>
    <div class="metric-card">
        <div class="value">{fmt("test_hcr")}</div>
        <div class="label">HCR (test)</div>
    </div>
    <div class="metric-card">
        <div class="value">{fmt("val_f1_l1_macro")}</div>
        <div class="label">Val F1 L1 macro</div>
    </div>
    <div class="metric-card">
        <div class="value">{fmt("val_f1_l2_macro")}</div>
        <div class="label">Val F1 L2 macro</div>
    </div>
</div>

<h2 id="courbes">Courbes d'entraînement</h2>
{img_curves or '<p>Non disponible</p>'}

<h2 id="confusion">Matrices de confusion</h2>
{img_cm_l1}
{img_cm_l2}

<h2 id="f1-classes">F1 par classe</h2>
{img_f1_l1}
{img_f1_l2}
{f1_tables_html}

<h2 id="explicabilite">Explicabilité (Gradient × Input)</h2>
{explain_html}

<h2 id="config">Configuration</h2>
<table class="config-table">
<thead><tr><th>Paramètre</th><th>Valeur</th></tr></thead>
<tbody>{config_rows}</tbody>
</table>

<script>
let scale = 1, panX = 0, panY = 0, dragging = false, startX, startY;
const overlay = document.getElementById('overlay');
const oImg = document.getElementById('overlay-img');
function openOverlay(img) {{ oImg.src = img.src; overlay.classList.add('active'); resetView(); }}
function closeOverlay() {{ overlay.classList.remove('active'); }}
function resetView() {{
    scale = 1; panX = 0; panY = 0;
    oImg.style.left = '50%'; oImg.style.top = '50%';
    oImg.style.transform = 'translate(-50%, -50%) scale(1)';
    document.getElementById('zoom-level').textContent = '100%';
}}
function applyTransform() {{
    oImg.style.left = '50%'; oImg.style.top = '50%';
    oImg.style.transform = `translate(calc(-50% + ${{panX}}px), calc(-50% + ${{panY}}px)) scale(${{scale}})`;
    document.getElementById('zoom-level').textContent = Math.round(scale*100) + '%';
}}
function zoomBy(factor) {{ scale *= factor; applyTransform(); }}
overlay.addEventListener('wheel', e => {{ e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.15 : 1/1.15); }});
oImg.addEventListener('mousedown', e => {{ dragging = true; startX = e.clientX - panX; startY = e.clientY - panY; oImg.classList.add('grabbing'); }});
window.addEventListener('mousemove', e => {{ if (dragging) {{ panX = e.clientX - startX; panY = e.clientY - startY; applyTransform(); }} }});
window.addEventListener('mouseup', () => {{ dragging = false; oImg.classList.remove('grabbing'); }});
oImg.addEventListener('dblclick', resetView);
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeOverlay(); }});
overlay.addEventListener('click', e => {{ if (e.target === overlay) closeOverlay(); }});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Génère le rapport HTML d'une variante HYDRA")
    parser.add_argument("variant", choices=["local", "nested", "global"])
    args = parser.parse_args()

    variant = args.variant
    results_dir = RESULTS_DIR / f"hydra-{variant}-camembert"
    if not results_dir.exists():
        raise SystemExit(f"Répertoire introuvable: {results_dir}")

    print(f"Génération du rapport HYDRA {variant.capitalize()}...")
    html = render_report(variant)
    output = results_dir / f"hydra-{variant}-camembert.html"
    output.write_text(html, encoding="utf-8")
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"  → {output} ({size_mb:.1f} Mo)")


if __name__ == "__main__":
    main()
