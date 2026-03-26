"""Training loop for HYDRA models (single GPU version).

This is the clean, single-GPU training loop. For distributed training
with DDP, see 03_production/src/trainer_ddp.py.

Features:
- Linear warmup scheduler (from YAML warmup_ratio).
- Mixed precision (AMP) when fp16=True.
- Early stopping on f1_l2_macro.
- Gradient accumulation with final flush.
- MLflow logging (hyperparams, per-epoch metrics, model artifact).
- Config loaded from configs/default.yaml.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data_loader import HierarchicalProductDataset, build_parent_child_matrix
from shared.metrics import compute_all_metrics
from shared.utils import set_seed, get_device, ENCODERS, enforce_hierarchy

from .models import create_hydra_model
from .losses import HYDRALoss

# Path to default config
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load YAML configuration."""
    path = Path(path) if path else _CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
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


class HYDRATrainer:
    """Training orchestrator for HYDRA models.

    Args:
        data: Dictionary from shared.data_loader.prepare_hierarchical_data().
        encoder_name: Key from ENCODERS dict (e.g., 'CamemBERT').
        variant: HYDRA variant ('local', 'global', 'nested').
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        max_length: Maximum token sequence length.
        loss_alpha: Weight for unified loss (global/nested variants).
        level_weight_l1: Weight for L1 loss relative to L2.
        focal_gamma: Focal Loss gamma parameter.
        use_focal: Use Focal Loss (True) or CrossEntropy (False).
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Minimum improvement to count as progress.
        project_embedding: Whether to project encoder embeddings.
        pooling: Pooling strategy.
        check_hierarchy: Whether to enforce hierarchy at inference.
        save_dir: Directory for saving model checkpoints.
        seed: Random seed.
        warmup_ratio: Fraction of total steps for linear warmup.
        fp16: Enable mixed precision training.
    """

    def __init__(
        self,
        data: dict,
        encoder_name: str = "CamemBERT",
        variant: str = "local",
        batch_size: int = 128,
        epochs: int = 10,
        lr: float = 3.5e-5,
        weight_decay: float = 0.01,
        max_length: int = 128,
        loss_alpha: float = 1.0,
        level_weight_l1: float = 0.5,
        focal_gamma: float = 2.0,
        use_focal: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 2,
        early_stopping_min_delta: float = 0.001,
        project_embedding: bool = True,
        pooling: str = "cls",
        check_hierarchy: bool = False,
        save_dir: str = "models",
        seed: int = 42,
        warmup_ratio: float = 0.1,
        fp16: bool = True,
        resume_from: str | None = None,
    ):
        set_seed(seed)
        self.resume_from = resume_from
        self.device = get_device()
        self.variant = variant
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.check_hierarchy = check_hierarchy
        self.save_dir = Path(save_dir)
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16 and self.device.type == "cuda"
        self.lr = lr

        # Data
        self.data = data
        self.l1_encoder = data["l1_encoder"]
        self.l2_encoder = data["l2_encoder"]
        self.l2_to_l1_mapping = data["l2_to_l1_mapping"]
        self.num_classes_l1 = data["num_classes_l1"]
        self.num_classes_l2 = data["num_classes_l2"]

        # Parent-child matrix for hierarchy enforcement
        self.parent_child_matrix = build_parent_child_matrix(
            self.l2_to_l1_mapping, self.num_classes_l1, self.num_classes_l2
        ).to(self.device)

        # Tokenizer
        encoder_path = ENCODERS[encoder_name]
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)

        # Model
        self.model = create_hydra_model(
            variant=variant,
            encoder_name=encoder_path,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            pooling=pooling,
            project_embedding=project_embedding,
        ).to(self.device)

        # DataParallel if multiple GPUs available
        self._base_model = self.model
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[Trainer] Using DataParallel on {n_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # Loss
        self.criterion = HYDRALoss(
            architecture=variant,
            alpha=loss_alpha,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            level_weight_l1=level_weight_l1,
            focal_gamma=focal_gamma,
            use_focal=use_focal,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # AMP scaler
        self.scaler = GradScaler("cuda", enabled=self.fp16)

    def _create_dataloader(self, texts, labels_l1, labels_l2, shuffle: bool = True) -> DataLoader:
        dataset = HierarchicalProductDataset(
            texts, labels_l1, labels_l2, self.tokenizer, self.max_length
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=4, pin_memory=True, drop_last=shuffle,
        )

    def _enforce_hierarchy(self, logits_l1: torch.Tensor, logits_l2: torch.Tensor) -> torch.Tensor:
        """Mask invalid L2 predictions based on predicted L1 parent."""
        return enforce_hierarchy(logits_l1, logits_l2, self.parent_child_matrix)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on a dataloader."""
        self.model.eval()
        all_preds_l1, all_preds_l2 = [], []
        all_labels_l1, all_labels_l2 = [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast("cuda", enabled=self.fp16):
                level_outputs, _ = self.model(input_ids, attention_mask)
            logits_l1, logits_l2 = level_outputs

            if self.check_hierarchy:
                logits_l2 = self._enforce_hierarchy(logits_l1, logits_l2)

            preds_l1 = torch.argmax(logits_l1, dim=-1).cpu().numpy()
            preds_l2 = torch.argmax(logits_l2, dim=-1).cpu().numpy()

            all_preds_l1.extend(preds_l1)
            all_preds_l2.extend(preds_l2)
            all_labels_l1.extend(batch["label_l1"].numpy())
            all_labels_l2.extend(batch["label_l2"].numpy())

        metrics = compute_all_metrics(
            all_labels_l1, all_preds_l1,
            all_labels_l2, all_preds_l2,
            self.l2_to_l1_mapping,
        )
        metrics["_preds_l1"] = all_preds_l1
        metrics["_preds_l2"] = all_preds_l2
        metrics["_labels_l1"] = all_labels_l1
        metrics["_labels_l2"] = all_labels_l2
        return metrics

    def train(self) -> dict:
        """Run the full training loop."""
        model_key = f"HYDRA_{self.variant}_{self.encoder_name}"
        print(f"\n{'='*60}")
        print(f"Training {model_key}")
        print(f"{'='*60}")

        # Create dataloaders
        train_loader = self._create_dataloader(
            self.data["X_train"], self.data["y_l1_train"], self.data["y_l2_train"],
        )
        val_loader = self._create_dataloader(
            self.data["X_val"], self.data["y_l1_val"], self.data["y_l2_val"],
            shuffle=False,
        )
        test_loader = self._create_dataloader(
            self.data["X_test"], self.data["y_l1_test"], self.data["y_l2_test"],
            shuffle=False,
        )

        # Warmup scheduler
        extra_steps = int((len(train_loader) % self.gradient_accumulation_steps) > 0)
        num_training_steps = (len(train_loader) // self.gradient_accumulation_steps + extra_steps) * self.epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        # MLflow logging
        mlflow_active = False
        try:
            import mlflow
            mlflow.set_experiment(model_key)
            mlflow.start_run()
            gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            mlflow.log_params({
                "variant": self.variant,
                "encoder": self.encoder_name,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "lr": self.lr,
                "warmup_ratio": self.warmup_ratio,
                "fp16": self.fp16,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "num_classes_l1": self.num_classes_l1,
                "num_classes_l2": self.num_classes_l2,
                "weight_decay": self.weight_decay,
                "max_length": self.max_length,
                "max_grad_norm": self.max_grad_norm,
                "level_weight_l1": self.level_weight_l1,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "gpu_type": gpu_type,
            })
            mlflow_active = True
        except Exception:
            pass

        # Training state
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        start_epoch = 0
        history = {
            "epoch": [], "train_loss": [],
            "val_f1_l1_macro": [], "val_f1_l2_macro": [],
            "val_hcr": [],
        }
        model_dir = self.save_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        global_step = 0

        # Resume from checkpoint if provided
        if self.resume_from:
            print(f"[Resume] Loading checkpoint from {self.resume_from}")
            ckpt = torch.load(self.resume_from, map_location=self.device, weights_only=False)
            self._base_model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            best_val_f1 = ckpt["best_val_f1"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            history = ckpt.get("history", history)
            print(
                f"[Resume] Reprise à epoch {start_epoch}, "
                f"step {global_step}, best F1={best_val_f1:.4f}"
            )

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                l1 = batch["label_l1"].to(self.device)
                l2 = batch["label_l2"].to(self.device)

                with autocast("cuda", enabled=self.fp16):
                    level_outputs, unified_output = self.model(input_ids, mask)
                    loss, _, _ = self.criterion(level_outputs, unified_output, l1, l2)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                pbar.set_postfix({"loss": f"{epoch_loss / num_batches:.4f}"})

            # Flush remaining accumulated gradients
            if num_batches % self.gradient_accumulation_steps != 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1

            # Epoch-level metrics
            avg_train_loss = epoch_loss / num_batches

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_f1_l2 = val_metrics["f1_l2_macro"]

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_f1_l1_macro"].append(val_metrics["f1_l1_macro"])
            history["val_f1_l2_macro"].append(val_f1_l2)
            history["val_hcr"].append(val_metrics.get("hcr", 0.0))

            print(
                f"  Loss: {avg_train_loss:.4f} | "
                f"Val F1 L1: {val_metrics['f1_l1_macro']:.4f} | "
                f"Val F1 L2: {val_f1_l2:.4f} | "
                f"HCR: {val_metrics.get('hcr', 0):.4f}"
            )

            # MLflow per-epoch logging
            if mlflow_active:
                try:
                    mlflow.log_metrics({
                        "train_loss": avg_train_loss,
                        "val_f1_l1_macro": val_metrics["f1_l1_macro"],
                        "val_f1_l2_macro": val_f1_l2,
                        "val_hcr": val_metrics.get("hcr", 0.0),
                    }, step=epoch + 1)
                except Exception:
                    pass

            # Early stopping & checkpointing on f1_l2_macro only
            if val_f1_l2 > best_val_f1 + self.early_stopping_min_delta:
                best_val_f1 = val_f1_l2
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": self._base_model.state_dict(),
                        "config": {
                            "variant": self.variant,
                            "encoder_name": self.encoder_name,
                            "num_classes_l1": self.num_classes_l1,
                            "num_classes_l2": self.num_classes_l2,
                            "pooling": self._base_model.pooling,
                            "project_embedding": self._base_model.project_embedding,
                            "dropout": self._base_model.level_1_head[3].p,
                            "hidden_dim": self._base_model.hidden_dim,
                        },
                        "best_val_f1_l2": best_val_f1,
                        "epoch": epoch + 1,
                    },
                    model_dir / "best_model.pt",
                )
                print(f"  -> New best model saved (F1 L2: {best_val_f1:.4f})")
            else:
                epochs_without_improvement += 1

            # Save full checkpoint for resume (every epoch, overwritten)
            torch.save(
                {
                    "model_state_dict": self._base_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_val_f1": best_val_f1,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history": history,
                    "config": {
                        "variant": self.variant,
                        "encoder_name": self.encoder_name,
                        "num_classes_l1": self.num_classes_l1,
                        "num_classes_l2": self.num_classes_l2,
                        "pooling": self._base_model.pooling,
                        "project_embedding": self._base_model.project_embedding,
                        "dropout": self._base_model.level_1_head[3].p,
                        "hidden_dim": self._base_model.hidden_dim,
                    },
                },
                model_dir / "checkpoint_latest.pt",
            )
            print(f"  -> Checkpoint saved (epoch {epoch + 1}, step {global_step})")

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"  Early stopping after {epoch + 1} epochs.")
                break

        # Final test evaluation
        test_metrics = self.evaluate(test_loader)
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Test F1 L1 Macro: {test_metrics['f1_l1_macro']:.4f}")
        print(f"Test F1 L2 Macro: {test_metrics['f1_l2_macro']:.4f}")
        print(f"Test HCR: {test_metrics.get('hcr', 0):.4f}")
        print(f"{'='*60}")

        # MLflow: log final metrics, audit artifacts, and model
        if mlflow_active:
            try:
                from shared.metrics import (
                    compute_per_class_f1,
                    plot_confusion_matrix,
                    plot_per_class_f1,
                    plot_training_curves,
                )
                import matplotlib.pyplot as plt

                mlflow.log_metrics({
                    "test_f1_l1_macro": test_metrics["f1_l1_macro"],
                    "test_f1_l2_macro": test_metrics["f1_l2_macro"],
                    "test_hcr": test_metrics.get("hcr", 0),
                    "total_time_s": total_time,
                })
                mlflow.log_artifact(str(model_dir / "best_model.pt"))

                # Audit visualizations (on validation set, not test)
                val_audit = self.evaluate(val_loader)
                preds_l1 = val_audit.get("_preds_l1", [])
                labels_l1 = val_audit.get("_labels_l1", [])
                preds_l2 = val_audit.get("_preds_l2", [])
                labels_l2 = val_audit.get("_labels_l2", [])

                if labels_l1:
                    fig = plot_confusion_matrix(
                        labels_l1, preds_l1, self.l1_encoder,
                        title="Matrice de Confusion — L1",
                    )
                    mlflow.log_figure(fig, "confusion_matrix_l1.png")
                    plt.close(fig)

                    fig = plot_confusion_matrix(
                        labels_l2, preds_l2, self.l2_encoder,
                        title="Matrice de Confusion — L2 (top 50)", top_n=50,
                    )
                    mlflow.log_figure(fig, "confusion_matrix_l2_top50.png")
                    plt.close(fig)

                    f1_l1 = compute_per_class_f1(labels_l1, preds_l1, self.l1_encoder)
                    fig = plot_per_class_f1(f1_l1, "F1 par classe — L1")
                    mlflow.log_figure(fig, "f1_per_class_l1.png")
                    plt.close(fig)
                    mlflow.log_dict(f1_l1, "f1_per_class_l1.json")

                    f1_l2 = compute_per_class_f1(labels_l2, preds_l2, self.l2_encoder)
                    fig = plot_per_class_f1(f1_l2, "F1 par classe — L2 (top/bottom 30)")
                    mlflow.log_figure(fig, "f1_per_class_l2.png")
                    plt.close(fig)

                if history["epoch"]:
                    fig = plot_training_curves(history)
                    mlflow.log_figure(fig, "training_curves.png")
                    plt.close(fig)

                # Explainability report (on validation set)
                try:
                    from shared.explainability import generate_explainability_report
                    val_texts = self.data["X_val"].tolist() if hasattr(self.data["X_val"], "tolist") else list(self.data["X_val"])
                    explain_html = generate_explainability_report(
                        model=self._base_model,
                        tokenizer=self.tokenizer,
                        texts=val_texts,
                        labels_l1=list(self.data["y_l1_val"]),
                        labels_l2=list(self.data["y_l2_val"]),
                        preds_l1=preds_l1,
                        preds_l2=preds_l2,
                        l1_encoder=self.l1_encoder,
                        l2_encoder=self.l2_encoder,
                        device=self.device,
                    )
                    mlflow.log_text(explain_html, "explainability_report.html")
                except Exception:
                    pass

                mlflow.end_run()
            except Exception:
                pass

        return {
            "history": history,
            "test_metrics": test_metrics,
            "best_val_f1_l2": best_val_f1,
            "total_time": total_time,
        }
