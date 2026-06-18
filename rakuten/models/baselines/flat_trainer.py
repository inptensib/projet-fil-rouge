"""Training orchestrator for the FlatHydra baseline.

Mirrors rakuten/models/hydra/trainer.py in structure, adapted for the Flat
architecture (single unified head, unified_logits sliced for L1/L2).

Public API:
    FlatHydraConfig  — typed hyperparameter dataclass.
    FlatHydraTrainer — training loop, evaluate, save, resume.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from rakuten.common.metrics import compute_all_metrics, compute_f1_by_tranche  # noqa: E402
from rakuten.common.training_utils import (  # noqa: E402
    EarlyStopping,
    MLflowSession,
    get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
    make_hierarchical_loader,
    resolve_rebalancing,
)

from rakuten.models.hydra.losses import HYDRALoss  # noqa: E402

from .flat_model import FlatHydra                          # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "flat_hydra.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FlatHydraConfig:
    """Typed hyperparameters for the FlatHydra baseline.

    Canonical values live in ``configs/baselines/flat_hydra.yaml``.

    Attributes:
        encoder: HuggingFace model name.
        max_length: Maximum tokenised sequence length.
        project_embedding: Project last_hidden_state before pooling.
        pooling: Pooling strategy ('cls', 'mean', 'max', 'last').
        dropout: Dropout in the unified classifier head (0.2 per paper).
        batch_size: Training batch size per GPU.
        epochs: Maximum training epochs.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps for linear warmup.
        max_grad_norm: Gradient clipping threshold.
        use_focal: Use FocalLoss instead of CrossEntropy.
        focal_gamma: Focal loss gamma (used if use_focal=True).
        level_weight_l1: Weight for L1 loss relative to L2.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Minimum F1 improvement to reset patience.
        seed: Random seed.
        fp16: Enable AMP mixed precision.
        num_workers: DataLoader workers.
        save_dir: Model artefacts directory.
        experiment_name: MLflow experiment name.
        freeze_encoder: Freeze encoder weights.
        use_description: Concatenate the description to the designation.
        normalize_text: Loader-side option: normalize designation/description.
        synthetic_policy: Loader-side option: 'keep_all' or 'drop_oversampled'.
        class_weighting: Per-class loss weights from the real (non synthetic)
            train counts: 'none', 'inv_sqrt' or 'inv'. Requires
            synthetic_policy='drop_oversampled'.
        logit_adjustment_tau: Temperature of the logit-adjusted loss
            (Menon et al., 2021). 0 disables. Same data requirement.
    """

    encoder: str = "camembert-base"
    max_length: int = 256
    project_embedding: bool = True
    pooling: str = "cls"
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_focal: bool = False
    focal_gamma: float = 2.0
    level_weight_l1: float = 0.5
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.001
    seed: int = 42
    fp16: bool = True
    num_workers: int = 4
    save_dir: str = "models/flat_hydra"
    experiment_name: str = "flat_hydra"
    freeze_encoder: bool = False
    use_description: bool = True
    normalize_text: bool = False
    synthetic_policy: str = "keep_all"
    class_weighting: str = "none"
    logit_adjustment_tau: float = 0.0

    def __post_init__(self) -> None:
        """Validate hyperparameter ranges, raising ValueError on invalid values."""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.pooling not in ("cls", "mean", "max", "last"):
            raise ValueError(f"Unknown pooling '{self.pooling}'")
        if self.class_weighting not in ("none", "inv_sqrt", "inv"):
            raise ValueError(f"Unknown class_weighting '{self.class_weighting}'")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "FlatHydraConfig":
        """Load config from YAML, then apply keyword overrides."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        known = {field for field in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        filtered.update(overrides)
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FlatHydraTrainer:
    """Training orchestrator for the FlatHydra baseline.

    Args:
        data: Dict from ``shared.data_loader.prepare_hierarchical_data()``.
        config: Hyperparameters.
        device: Target device. Auto-detected if None.
        resume_from: Path to ``checkpoint_latest.pt`` to resume training.
    """

    def __init__(
        self,
        data: dict,
        config: Optional[FlatHydraConfig] = None,
        device: Optional[torch.device] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.data = data
        self.config = config or FlatHydraConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.resume_from = resume_from

        torch.manual_seed(self.config.seed)

        self.num_classes_l1: int = data["num_classes_l1"]
        self.num_classes_l2: int = data["num_classes_l2"]
        self.l1_encoder = data["l1_encoder"]
        self.l2_encoder = data["l2_encoder"]
        self.l2_to_l1_mapping: dict = data["l2_to_l1_mapping"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder)

        self.model = FlatHydra(
            encoder_name=self.config.encoder,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            pooling=self.config.pooling,
            project_embedding=self.config.project_embedding,
            dropout=self.config.dropout,
            freeze_encoder=self.config.freeze_encoder,
        ).to(self.device)

        self._base_model = self.model
        if torch.cuda.device_count() > 1:
            print(f"[Trainer] DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Rééquilibrage des pertes (None partout quand les options sont off)
        rebalancing = resolve_rebalancing(
            data, self.num_classes_l1, self.num_classes_l2,
            class_weighting=self.config.class_weighting,
            logit_adjustment_tau=self.config.logit_adjustment_tau,
        )
        rebalancing = {
            key: tensor.to(self.device) if tensor is not None else None
            for key, tensor in rebalancing.items()
        }

        self.criterion = HYDRALoss(
            architecture="global",
            alpha=1.0,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            level_weight_l1=self.config.level_weight_l1,
            focal_gamma=self.config.focal_gamma,
            use_focal=self.config.use_focal,
            **rebalancing,
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.fp16 = self.config.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.fp16)

    def _make_loader(self, texts, labels_l1, labels_l2, shuffle: bool) -> DataLoader:
        """Build a DataLoader with config settings; drops the last batch when shuffling."""
        return make_hierarchical_loader(
            texts,
            labels_l1,
            labels_l2,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate the model on a dataloader.

        Slices unified_logits into L1 / L2 parts before computing metrics.

        Returns:
            Dict with f1_l1_macro, f1_l2_macro, constrained_f1_macro, hcr,
            plus raw _preds / _labels / _confs lists prefixed with underscore.
        """
        self.model.eval()
        all_preds_l1, all_preds_l2 = [], []
        all_labels_l1, all_labels_l2 = [], []
        all_confs_l1, all_confs_l2 = [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast("cuda", enabled=self.fp16):
                _, unified_logits = self.model(input_ids, attention_mask)

            # Slice unified output into per-level logits
            logits_l1 = unified_logits[:, :self.num_classes_l1]
            logits_l2 = unified_logits[:, self.num_classes_l1:]

            probs_l1 = torch.softmax(logits_l1, dim=-1)
            probs_l2 = torch.softmax(logits_l2, dim=-1)
            confs_l1, preds_l1 = probs_l1.max(dim=-1)
            confs_l2, preds_l2 = probs_l2.max(dim=-1)

            all_preds_l1.extend(preds_l1.cpu().tolist())
            all_preds_l2.extend(preds_l2.cpu().tolist())
            all_confs_l1.extend(confs_l1.cpu().tolist())
            all_confs_l2.extend(confs_l2.cpu().tolist())
            all_labels_l1.extend(batch["label_l1"].tolist())
            all_labels_l2.extend(batch["label_l2"].tolist())

        metrics = compute_all_metrics(
            all_labels_l1, all_preds_l1,
            all_labels_l2, all_preds_l2,
            self.l2_to_l1_mapping,
        )
        real_counts_l2 = self.data.get("real_class_counts_l2")
        if real_counts_l2 is not None:
            metrics["f1_l2_by_tranche"] = compute_f1_by_tranche(
                all_labels_l2, all_preds_l2, real_counts_l2
            )
        metrics["_preds_l1"] = all_preds_l1
        metrics["_preds_l2"] = all_preds_l2
        metrics["_labels_l1"] = all_labels_l1
        metrics["_labels_l2"] = all_labels_l2
        metrics["_confs_l1"] = all_confs_l1
        metrics["_confs_l2"] = all_confs_l2
        return metrics

    def _setup_mlflow(self) -> MLflowSession:
        """Open the optional MLflow run and log hyperparameters."""
        db_path = Path(__file__).resolve().parents[2] / "mlflow.db"
        session = MLflowSession(f"sqlite:///{db_path}", self.config.experiment_name)
        session.start(params={
            "encoder": self.config.encoder,
            "pooling": self.config.pooling,
            "project_embedding": self.config.project_embedding,
            "dropout": self.config.dropout,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "warmup_ratio": self.config.warmup_ratio,
            "use_focal": self.config.use_focal,
            "level_weight_l1": self.config.level_weight_l1,
            "num_classes_l1": self.num_classes_l1,
            "num_classes_l2": self.num_classes_l2,
            "class_weighting": self.config.class_weighting,
            "logit_adjustment_tau": self.config.logit_adjustment_tau,
            "fp16": self.fp16,
        })
        return session

    def _resume_from_checkpoint(self, history: dict) -> tuple[int, dict]:
        """Restore model/optimizer/scheduler/scaler state from checkpoint_latest.pt."""
        start_epoch = 0
        if self.resume_from and Path(self.resume_from).exists():
            print(f"[Resume] Loading {self.resume_from}")
            ckpt = torch.load(self.resume_from, map_location=self.device, weights_only=False)
            self._base_model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"]
            self._early_stopping.best = ckpt["best_val_f1"]
            self._early_stopping.counter = ckpt["epochs_no_improve"]
            history = ckpt.get("history", history)
            print(f"[Resume] Epoch {start_epoch}, best F1 L2={self._early_stopping.best:.4f}")
        return start_epoch, history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            l1 = batch["label_l1"].to(self.device)
            l2 = batch["label_l2"].to(self.device)

            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.fp16):
                level_outputs, unified_output = self.model(input_ids, mask)
                loss, _, _ = self.criterion(level_outputs, unified_output, l1, l2)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{epoch_loss / n_batches:.4f}"})

        return epoch_loss / max(n_batches, 1)

    def _log_epoch_mlflow(
        self, epoch: int, avg_loss: float, val_metrics: dict, history: dict
    ) -> None:
        """Log epoch metrics and figures to MLflow."""
        if not self._mlflow.active:
            return
        try:
            import matplotlib.pyplot as plt
            from rakuten.common.metrics import (
                compute_per_class_f1,
                plot_confusion_matrix,
                plot_per_class_f1,
                plot_training_curves,
            )
            self._mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_f1_l1_macro": val_metrics["f1_l1_macro"],
                "val_f1_l2_macro": val_metrics["f1_l2_macro"],
                "val_hcr": val_metrics.get("hcr", 0.0),
                "val_constrained_f1_macro": val_metrics.get("constrained_f1_macro", 0.0),
            }, step=epoch + 1)

            v_labels_l1 = val_metrics["_labels_l1"]
            v_preds_l1 = val_metrics["_preds_l1"]
            v_labels_l2 = val_metrics["_labels_l2"]
            v_preds_l2 = val_metrics["_preds_l2"]

            if v_labels_l1:
                fig = plot_confusion_matrix(
                    v_labels_l1, v_preds_l1, self.l1_encoder,
                    title=f"Confusion L1 — epoch {epoch + 1}",
                )
                self._mlflow.log_figure(fig, "confusion_matrix_l1.png")
                plt.close(fig)

                fig = plot_confusion_matrix(
                    v_labels_l2, v_preds_l2, self.l2_encoder,
                    title=f"Confusion L2 (top 50) — epoch {epoch + 1}",
                    top_n=50,
                )
                self._mlflow.log_figure(fig, "confusion_matrix_l2.png")
                plt.close(fig)

                f1_pc = compute_per_class_f1(v_labels_l1, v_preds_l1, self.l1_encoder)
                fig = plot_per_class_f1(f1_pc, f"F1 par classe L1 — epoch {epoch + 1}")
                self._mlflow.log_figure(fig, "per_class_f1_l1.png")
                plt.close(fig)

            fig = plot_training_curves(history)
            self._mlflow.log_figure(fig, "training_curves.png")
            plt.close(fig)
        except Exception as e:
            print(f"  [MLflow] Artefacts failed ({e})")

    def _evaluate_and_checkpoint(
        self,
        val_loader: DataLoader,
        epoch: int,
        avg_loss: float,
        history: dict,
        model_dir: Path,
        start_time: float,
    ) -> bool:
        """Validate, log, save best/resume checkpoints. Returns True to stop early."""
        val_metrics = self.evaluate(val_loader)
        val_f1_l2 = val_metrics["f1_l2_macro"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_f1_l1_macro"].append(val_metrics["f1_l1_macro"])
        history["val_f1_l2_macro"].append(val_f1_l2)
        history["val_hcr"].append(val_metrics.get("hcr", 0.0))
        history["val_constrained_f1_macro"].append(
            val_metrics.get("constrained_f1_macro", 0.0)
        )

        elapsed = time.time() - start_time
        print(
            f"  [{epoch + 1}/{self.config.epochs}] "
            f"loss={avg_loss:.4f} | "
            f"F1 L1={val_metrics['f1_l1_macro']:.4f} | "
            f"F1 L2={val_f1_l2:.4f} | "
            f"HCR={val_metrics.get('hcr', 0):.4f} | "
            f"{elapsed:.0f}s"
        )

        # Persist metrics locally (fallback sans MLflow)
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        self._log_epoch_mlflow(epoch, avg_loss, val_metrics, history)

        # Early stopping & checkpointing
        if self._early_stopping.update(val_f1_l2):
            torch.save(
                {
                    "model_state_dict": self._base_model.state_dict(),
                    "config": asdict(self.config),
                    "num_classes_l1": self.num_classes_l1,
                    "num_classes_l2": self.num_classes_l2,
                    "best_val_f1_l2": self._early_stopping.best,
                    "epoch": epoch + 1,
                },
                model_dir / "best_model.pt",
            )
            print(f"  -> Best model saved (F1 L2: {self._early_stopping.best:.4f})")
        elif self._early_stopping.should_stop:
            print(
                f"  [Early stopping] No improvement for "
                f"{self.config.early_stopping_patience} epochs. Stopping."
            )
            return True

        # Checkpoint for resume
        torch.save(
            {
                "model_state_dict": self._base_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch + 1,
                "best_val_f1": self._early_stopping.best,
                "epochs_no_improve": self._early_stopping.counter,
                "history": history,
            },
            model_dir / "checkpoint_latest.pt",
        )
        return False

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            history dict with per-epoch train_loss, val_f1_l1/l2, val_hcr.
        """
        d = self.data
        train_loader = self._make_loader(
            d["X_train"], d["y_l1_train"], d["y_l2_train"], shuffle=True
        )
        val_loader = self._make_loader(
            d["X_val"], d["y_l1_val"], d["y_l2_val"], shuffle=False
        )

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = _get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, max(total_steps, 1)
        )

        model_dir = Path(self.config.save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        history: dict = {
            "epoch": [],
            "train_loss": [],
            "val_f1_l1_macro": [],
            "val_f1_l2_macro": [],
            "val_hcr": [],
            "val_constrained_f1_macro": [],
        }
        self._early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

        self._mlflow = self._setup_mlflow()

        start_epoch, history = self._resume_from_checkpoint(history)

        print(f"\n{'=' * 60}")
        print(f"FlatHydra — {self.config.encoder}")
        print(f"L1: {self.num_classes_l1} classes | L2: {self.num_classes_l2} classes")
        print(f"{'=' * 60}")

        start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            avg_loss = self._train_epoch(train_loader, epoch)
            if self._evaluate_and_checkpoint(
                val_loader, epoch, avg_loss, history, model_dir, start_time
            ):
                break

        self._mlflow.end()

        best_val_f1 = self._early_stopping.best
        (model_dir / "training_complete.txt").write_text(
            f"best_val_f1_l2={best_val_f1:.4f}\n"
            f"epochs_trained={history['epoch'][-1] if history['epoch'] else 0}\n"
        )
        print(f"\n[Done] Best F1 L2: {best_val_f1:.4f} — artefacts in {model_dir}/")
        return history

    def save(self, save_dir: Optional[str] = None) -> None:
        """Persist model artefacts to disk.

        Layout::

            save_dir/
                tokenizer/             HuggingFace tokenizer
                best_model.pt          best checkpoint (written during train)
                checkpoint_latest.pt   resume checkpoint
                metrics.json           per-epoch metrics
                l1_label_encoder.pkl   sklearn LabelEncoder L1
                l2_label_encoder.pkl   sklearn LabelEncoder L2
                l2_to_l1_mapping.json  hierarchy map
                config.json            FlatHydraConfig + num_classes

        Args:
            save_dir: Destination directory. Defaults to ``config.save_dir``.
        """
        dest = Path(save_dir or self.config.save_dir)
        dest.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(str(dest / "tokenizer"))

        with open(dest / "l1_label_encoder.pkl", "wb") as f:
            pickle.dump(self.l1_encoder, f)
        with open(dest / "l2_label_encoder.pkl", "wb") as f:
            pickle.dump(self.l2_encoder, f)

        mapping = {str(k): int(v) for k, v in self.l2_to_l1_mapping.items()}
        with open(dest / "l2_to_l1_mapping.json", "w") as f:
            json.dump(mapping, f)

        config_dict = asdict(self.config)
        config_dict["num_classes_l1"] = self.num_classes_l1
        config_dict["num_classes_l2"] = self.num_classes_l2
        with open(dest / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"[FlatHydra] Artefacts saved to {dest}/")
