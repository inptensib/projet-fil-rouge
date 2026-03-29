"""MCP server exposing MLflow operations to Claude Code.

Requires: mlflow, mcp
Start MLflow first: docker compose up mlflow -d
"""

from __future__ import annotations

import json
import os

import mlflow
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

mcp = FastMCP("mlflow")
client: MlflowClient | None = None


def _client() -> MlflowClient:
    global client
    if client is None:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient(TRACKING_URI)
    return client


@mcp.tool()
def list_experiments() -> str:
    """List all MLflow experiments with their IDs and names."""
    experiments = _client().search_experiments()
    rows = [{"id": e.experiment_id, "name": e.name, "lifecycle": e.lifecycle_stage}
            for e in experiments]
    return json.dumps(rows, indent=2)


@mcp.tool()
def list_runs(experiment_name: str = "hydra", max_results: int = 20) -> str:
    """List recent runs for an experiment. Shows run_id, status, start_time, and key metrics."""
    exp = _client().get_experiment_by_name(experiment_name)
    if not exp:
        return f"Experiment '{experiment_name}' not found."
    runs = _client().search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id[:8],
            "run_name": r.info.run_name or "",
            "status": r.info.status,
            "duration_s": (r.info.end_time - r.info.start_time) / 1000 if r.info.end_time else None,
            "params": {k: v for k, v in list(r.data.params.items())[:10]},
            "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
        })
    return json.dumps(rows, indent=2)


@mcp.tool()
def get_run_details(run_id: str) -> str:
    """Get full details of a run: all params, metrics, and tags."""
    run = _client().get_run(run_id)
    return json.dumps({
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }, indent=2)


@mcp.tool()
def get_metric_history(run_id: str, metric_name: str) -> str:
    """Get the step-by-step history of a metric for a run (e.g. train_loss, f1_l1_macro)."""
    history = _client().get_metric_history(run_id, metric_name)
    points = [{"step": m.step, "value": round(m.value, 6), "timestamp": m.timestamp}
              for m in history]
    return json.dumps(points, indent=2)


@mcp.tool()
def compare_runs(run_ids: list[str], metric_names: list[str] | None = None) -> str:
    """Compare multiple runs side-by-side on key metrics.

    Args:
        run_ids: List of run IDs to compare.
        metric_names: Specific metrics to compare. If None, shows all.
    """
    if metric_names is None:
        metric_names = ["f1_l1_macro", "f1_l2_macro", "f1_l1_weighted", "f1_l2_weighted",
                        "constrained_f1", "hcr", "train_loss", "val_loss"]
    rows = []
    for rid in run_ids:
        run = _client().get_run(rid)
        row = {
            "run_id": rid[:8],
            "run_name": run.info.run_name or "",
            "variant": run.data.params.get("variant", "?"),
            "encoder": run.data.params.get("encoder", "?"),
        }
        for m in metric_names:
            row[m] = round(run.data.metrics.get(m, 0), 4)
        rows.append(row)
    return json.dumps(rows, indent=2)


@mcp.tool()
def search_best_runs(experiment_name: str = "hydra", metric: str = "f1_l2_macro",
                     top_k: int = 5) -> str:
    """Find the top K runs by a given metric.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to rank by (e.g. f1_l2_macro, f1_l1_weighted).
        top_k: Number of top runs to return.
    """
    exp = _client().get_experiment_by_name(experiment_name)
    if not exp:
        return f"Experiment '{experiment_name}' not found."
    runs = _client().search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=top_k,
        order_by=[f"metrics.{metric} DESC"],
        filter_string="status = 'FINISHED'",
    )
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id[:8],
            "run_name": r.info.run_name or "",
            "variant": r.data.params.get("variant", "?"),
            "encoder": r.data.params.get("encoder", "?"),
            metric: round(r.data.metrics.get(metric, 0), 4),
            "all_metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
        })
    return json.dumps(rows, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
