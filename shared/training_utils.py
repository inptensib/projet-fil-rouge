"""Common training utilities shared by the project trainers.

Provides:
- get_linear_schedule_with_warmup: linear warmup then linear decay scheduler.
- make_hierarchical_loader: HierarchicalProductDataset + DataLoader factory.
- EarlyStopping: patience-based early stopping state.
- MLflowSession: fault-tolerant MLflow logging wrapper (never fails a training).
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from shared.data_loader import HierarchicalProductDataset


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """Linear warmup then linear decay to 0."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def make_hierarchical_loader(
    texts,
    labels_l1,
    labels_l2,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader over a HierarchicalProductDataset.

    Args:
        texts: List or Series of product text descriptions.
        labels_l1: List or Series of encoded L1 labels.
        labels_l2: List or Series of encoded L2 labels.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
        batch_size: Batch size.
        shuffle: Shuffle samples between epochs.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin batches in page-locked memory.
        drop_last: Drop the last incomplete batch.
    """
    dataset = HierarchicalProductDataset(
        texts, labels_l1, labels_l2, tokenizer, max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class EarlyStopping:
    """Patience-based early stopping on a metric to maximise.

    The ``best`` and ``counter`` attributes are public and assignable so
    trainers can inject state restored from a checkpoint.

    Args:
        patience: Epochs without improvement before ``should_stop`` is True.
        min_delta: Minimum improvement over ``best`` to count as progress.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.counter = 0

    def update(self, value: float) -> bool:
        """Record a new metric value.

        Returns:
            True if ``value`` improves ``best`` by more than ``min_delta``
            (counter is reset), False otherwise (counter is incremented).
        """
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


class MLflowSession:
    """Fault-tolerant wrapper around the MLflow tracking boilerplate.

    Every method is a silent no-op when the session is disabled or not
    started, and swallows MLflow exceptions with a printed warning:
    MLflow must never make a training fail.

    Args:
        tracking_uri: MLflow tracking URI (e.g. ``sqlite:///.../mlflow.db``).
        experiment_name: MLflow experiment name.
        enabled: Disable all MLflow calls when False.
    """

    def __init__(self, tracking_uri: str, experiment_name: str, enabled: bool = True):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enabled = enabled
        self.active = False

    def start(self, run_name: str | None = None, params: dict | None = None) -> None:
        """Open the run and log hyperparameters."""
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(params)
            self.active = True
            print(f"[MLflow] Tracking active → {self.tracking_uri}")
        except Exception as e:
            print(f"[MLflow] Disabled ({e})")

    def _safe(self, method: str, *args, **kwargs) -> None:
        if not self.active:
            return
        try:
            import mlflow
            getattr(mlflow, method)(*args, **kwargs)
        except Exception as e:
            print(f"  [MLflow] {method} failed ({e})")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._safe("log_metrics", metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._safe("log_metric", key, value, step=step)

    def log_figure(self, figure, artifact_path: str) -> None:
        self._safe("log_figure", figure, artifact_path)

    def log_dict(self, dictionary: dict, artifact_path: str) -> None:
        self._safe("log_dict", dictionary, artifact_path)

    def log_text(self, text: str, artifact_path: str) -> None:
        self._safe("log_text", text, artifact_path)

    def log_table(self, data, artifact_path: str) -> None:
        self._safe("log_table", data, artifact_path)

    def log_artifact(self, path: str) -> None:
        self._safe("log_artifact", path)

    def end(self, status: str | None = None) -> None:
        """Close the run. Exceptions are swallowed silently."""
        if not self.active:
            return
        try:
            import mlflow
            if status is None:
                mlflow.end_run()
            else:
                mlflow.end_run(status=status)
        except Exception:
            pass
        self.active = False
