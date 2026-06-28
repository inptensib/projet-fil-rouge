"""Training loop for HYDRA models (single GPU version).

This is the clean, single-GPU training loop (DataParallel multi-GPU pris en
charge via CUDA_VISIBLE_DEVICES). Le pipeline DDP multi-nœuds exploré pendant
le projet a été écarté du livrable (cf. docs/ARCHITECTURE.md, archivé sous le
tag git archive/distributed-pipeline).

Features:
- Linear warmup scheduler (from YAML warmup_ratio).
- Mixed precision (AMP) when fp16=True.
- Early stopping on f1_l2_macro.
- Gradient accumulation with final flush.
- Config loaded from configs/hydra/default.yaml.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

# H100/3090 (Ampere+): TF32 (matmul/cudnn plus rapides, effet negligeable)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
from transformers import AutoTokenizer

from rakuten.common.data_loader import build_parent_child_matrix_from_labels
from rakuten.common.metrics import compute_all_metrics, compute_f1_by_tranche
from rakuten.common.training_utils import (
    EarlyStopping,
    get_linear_schedule_with_warmup,
    make_hierarchical_loader,
    resolve_rebalancing,
)
from rakuten.common.utils import ENCODERS, enforce_hierarchy, get_device, set_seed

from .losses import HYDRALoss
from .models import create_hydra_model

# Path to default config (configs/ vit à la racine du repo, hors package)
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "hydra" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load YAML configuration."""
    path = Path(path) if path else _CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


class HYDRATrainer:
    """Training orchestrator for HYDRA models.

    Args:
        data: Dictionary from rakuten.common.data_loader.prepare_hierarchical_data().
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
        dropout: Dropout rate of the classification heads.
        hidden_dim: Hidden dimension of the classification heads.
        freeze_encoder: Freeze encoder weights (train heads only).
        check_hierarchy: Whether to enforce hierarchy at inference.
        save_dir: Directory for saving model checkpoints.
        seed: Random seed.
        warmup_ratio: Fraction of total steps for linear warmup.
        fp16: Enable mixed precision training.
        class_weighting: Per-class loss weights from the real (non synthetic)
            train counts: 'none', 'inv_sqrt' or 'inv'. Requires data loaded
            with synthetic_policy='drop_oversampled'.
        logit_adjustment_tau: Temperature of the logit-adjusted loss
            (Menon et al., 2021). 0 disables. Same data requirement.
    """

    def __init__(
        self,
        data: dict,
        encoder_name: str = "CamemBERT",
        variant: str = "local",
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 3.5e-5,
        weight_decay: float = 0.01,
        max_length: int = 256,
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
        dropout: float = 0.3,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
        check_hierarchy: bool = False,
        save_dir: str = "models/v1",
        seed: int = 42,
        warmup_ratio: float = 0.1,
        fp16: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        resume_from: str | None = None,
        checkpoint_every_n_steps: int = 10_000,
        condition_l2_on_l1: bool = False,
        experiment_suffix: str = "",
        class_weighting: str = "none",
        logit_adjustment_tau: float = 0.0,
    ):
        set_seed(seed)
        self.resume_from = resume_from
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.condition_l2_on_l1 = condition_l2_on_l1
        self.experiment_suffix = experiment_suffix
        self.class_weighting = class_weighting
        self.logit_adjustment_tau = logit_adjustment_tau
        self.device = get_device()
        self.variant = variant
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.epochs = epochs
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.level_weight_l1 = level_weight_l1
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.check_hierarchy = check_hierarchy
        self.save_dir = Path(save_dir)
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16 and self.device.type == "cuda"
        self.lr = lr
        self.pooling = pooling
        self.project_embedding = project_embedding
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder

        # Data
        self.data = data
        self.l1_encoder = data["l1_encoder"]
        self.l2_encoder = data["l2_encoder"]
        self.l2_to_l1_mapping = data["l2_to_l1_mapping"]
        self.num_classes_l1 = data["num_classes_l1"]
        self.num_classes_l2 = data["num_classes_l2"]

        # Rééquilibrage des pertes (None partout quand les options sont off)
        rebalancing = resolve_rebalancing(
            data, self.num_classes_l1, self.num_classes_l2,
            class_weighting=class_weighting,
            logit_adjustment_tau=logit_adjustment_tau,
        )
        rebalancing = {
            key: tensor.to(self.device) if tensor is not None else None
            for key, tensor in rebalancing.items()
        }

        # Parent-child matrix for hierarchy enforcement
        _all_l1 = np.concatenate([data["y_l1_train"], data["y_l1_val"], data["y_l1_test"]])
        _all_l2 = np.concatenate([data["y_l2_train"], data["y_l2_val"], data["y_l2_test"]])
        self.parent_child_matrix = build_parent_child_matrix_from_labels(
            _all_l1, _all_l2, self.num_classes_l1, self.num_classes_l2
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
            dropout=dropout,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
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
            **rebalancing,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # AMP scaler
        self.scaler = GradScaler("cuda", enabled=self.fp16)

    def _create_dataloader(self, texts, labels_l1, labels_l2, shuffle: bool = True) -> DataLoader:
        """Build a DataLoader with trainer settings; drops the last batch when shuffling."""
        return make_hierarchical_loader(
            texts, labels_l1, labels_l2,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
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
        all_confs_l1, all_confs_l2 = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with autocast("cuda", enabled=self.fp16):
                    level_outputs, _ = self.model(input_ids, attention_mask)
                logits_l1, logits_l2 = level_outputs

                if self.check_hierarchy:
                    logits_l2 = self._enforce_hierarchy(logits_l1, logits_l2)

                probs_l1 = torch.softmax(logits_l1, dim=-1)
                probs_l2 = torch.softmax(logits_l2, dim=-1)
                confs_l1, preds_l1 = probs_l1.max(dim=-1)
                confs_l2, preds_l2 = probs_l2.max(dim=-1)

                all_preds_l1.extend(preds_l1.cpu().numpy())
                all_preds_l2.extend(preds_l2.cpu().numpy())
                all_confs_l1.extend(confs_l1.cpu().numpy())
                all_confs_l2.extend(confs_l2.cpu().numpy())
                all_labels_l1.extend(batch["label_l1"].numpy())
                all_labels_l2.extend(batch["label_l2"].numpy())

        metrics = compute_all_metrics(
            all_labels_l1, all_preds_l1,
            all_labels_l2, all_preds_l2,
            self.l2_to_l1_mapping,
        )
        # HCR multi-parents : mesuré sur les prédictions réelles via la matrice
        # multi-parents (pred_l2 est-il un enfant valide de pred_l1 sous TOUTES
        # les arêtes (L1,L2) observées, pas seulement le premier parent ?).
        if len(all_preds_l1):
            pcm = self.parent_child_matrix.detach().cpu().numpy()
            _pl1 = np.asarray(all_preds_l1)
            _pl2 = np.asarray(all_preds_l2)
            metrics["hcr_multiparent"] = float((pcm[_pl1, _pl2] > 0).mean())
        else:
            metrics["hcr_multiparent"] = 0.0
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

    def train(self) -> dict:
        """Run the full training loop."""
        model_key = f"HYDRA_{self.variant}_{self.encoder_name}{self.experiment_suffix}"
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
        steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps + extra_steps
        num_training_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        # Training state
        self._early_stopping = EarlyStopping(
            self.early_stopping_patience, self.early_stopping_min_delta
        )
        self._history = {
            "epoch": [], "train_loss": [],
            "val_f1_l1_macro": [], "val_f1_l2_macro": [],
            "val_hcr": [],
        }
        self._model_dir = self.save_dir / model_key
        self._model_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        self._global_step = 0
        self._start_epoch = 0
        self._start_step_in_epoch = 0

        # Resume from checkpoint if provided
        if self.resume_from:
            self._resume_from_checkpoint()

        for epoch in range(self._start_epoch, self.epochs):
            avg_train_loss = self._train_epoch(train_loader, epoch)
            if self._evaluate_and_checkpoint(val_loader, epoch, avg_train_loss):
                break

        return self._finalize(test_loader, epoch, start_time)

    def _resume_from_checkpoint(self) -> None:
        """Restore model, optimizer, scheduler, scaler and training state."""
        print(f"[Resume] Loading checkpoint from {self.resume_from}")
        ckpt = torch.load(self.resume_from, map_location=self.device, weights_only=False)
        self._base_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self._start_epoch = ckpt["epoch"]
        self._global_step = ckpt["global_step"]
        self._early_stopping.best = ckpt["best_val_f1"]
        self._early_stopping.counter = ckpt["epochs_without_improvement"]
        self._history = ckpt.get("history", self._history)
        self._start_step_in_epoch = ckpt.get("step_in_epoch", 0)
        print(
            f"[Resume] Reprise à epoch {self._start_epoch}, "
            f"step_in_epoch {self._start_step_in_epoch}, "
            f"global_step {self._global_step}, best F1={self._early_stopping.best:.4f}"
        )

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch and return the average train loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for step, batch in enumerate(pbar):
            # Fast-forward if resuming mid-epoch
            if step < self._start_step_in_epoch:
                continue

            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            l1 = batch["label_l1"].to(self.device)
            l2 = batch["label_l2"].to(self.device)

            with autocast("cuda", enabled=self.fp16):
                level_outputs, unified_output = self.model(input_ids, mask)

            # Cast fp32 + clamp : élimine les +inf dus à l'overflow fp16
            level_outputs_fp32 = [o.float().clamp(-1e4, 1e4) for o in level_outputs]
            unified_output_fp32 = (
                unified_output.float().clamp(-1e4, 1e4)
                if unified_output is not None else None
            )
            if self.condition_l2_on_l1:
                logits_l1, logits_l2 = level_outputs_fp32
                valid_mask = self.parent_child_matrix[l1]  # [B, num_l2]
                # has_valid : lignes non vides (L1 ayant au moins 1 enfant L2 connu)
                has_valid = valid_mask.any(dim=1, keepdim=True)  # [B, 1]
                # -1e9 au lieu de -inf : évite -inf - (-inf) = nan sur lignes vides
                logits_l2 = logits_l2.masked_fill(
                    (valid_mask == 0) & has_valid, -1e9
                )
                level_outputs_fp32 = [logits_l1, logits_l2]

            loss, _, _ = self.criterion(level_outputs_fp32, unified_output_fp32, l1, l2)
            loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()

                # Intra-epoch checkpoint every N steps
                if self._global_step % self.checkpoint_every_n_steps == 0:
                    self._save_latest_checkpoint(epoch, step + 1)
                    print(f"\n  [Checkpoint] Step {self._global_step} saved")

            epoch_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{epoch_loss / num_batches:.4f}"})

        # Reset start_step for subsequent epochs
        self._start_step_in_epoch = 0

        # Flush remaining accumulated gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            self._optimizer_step()

        return epoch_loss / num_batches

    def _optimizer_step(self) -> None:
        """Unscale, clip, step the optimizer and scheduler."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self._global_step += 1

    def _checkpoint_config(self) -> dict:
        """Model hyperparameters stored alongside checkpoint weights."""
        return {
            "variant": self.variant,
            "encoder_name": self.encoder_name,
            "num_classes_l1": self.num_classes_l1,
            "num_classes_l2": self.num_classes_l2,
            "pooling": self.pooling,
            "project_embedding": self.project_embedding,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
        }

    def _save_latest_checkpoint(self, epoch: int, step_in_epoch: int) -> None:
        """Save the full resume checkpoint (checkpoint_latest.pt)."""
        torch.save(
            {
                "model_state_dict": self._base_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": self._global_step,
                "best_val_f1": self._early_stopping.best,
                "epochs_without_improvement": self._early_stopping.counter,
                "history": self._history,
                "config": self._checkpoint_config(),
            },
            self._model_dir / "checkpoint_latest.pt",
        )

    def _evaluate_and_checkpoint(
        self, val_loader: DataLoader, epoch: int, avg_train_loss: float
    ) -> bool:
        """Validate, save best/latest checkpoints; return True to early stop."""
        val_metrics = self.evaluate(val_loader)
        val_f1_l2 = val_metrics["f1_l2_macro"]

        self._history["epoch"].append(epoch + 1)
        self._history["train_loss"].append(avg_train_loss)
        self._history["val_f1_l1_macro"].append(val_metrics["f1_l1_macro"])
        self._history["val_f1_l2_macro"].append(val_f1_l2)
        self._history["val_hcr"].append(val_metrics.get("hcr", 0.0))

        print(
            f"  Loss: {avg_train_loss:.4f} | "
            f"Val F1 L1: {val_metrics['f1_l1_macro']:.4f} | "
            f"Val F1 L2: {val_f1_l2:.4f} | "
            f"HCR: {val_metrics.get('hcr', 0):.4f}"
        )

        # Early stopping & checkpointing on f1_l2_macro only
        if self._early_stopping.update(val_f1_l2):
            torch.save(
                {
                    "model_state_dict": self._base_model.state_dict(),
                    "config": self._checkpoint_config(),
                    "best_val_f1_l2": self._early_stopping.best,
                    "epoch": epoch + 1,
                },
                self._model_dir / "best_model.pt",
            )
            print(f"  -> New best model saved (F1 L2: {self._early_stopping.best:.4f})")

        # Save full checkpoint for resume (end of epoch)
        self._save_latest_checkpoint(epoch + 1, 0)
        print(f"  -> Checkpoint saved (epoch {epoch + 1}, step {self._global_step})")

        if self._early_stopping.should_stop:
            print(f"  Early stopping after {epoch + 1} epochs.")
            return True
        return False

    def _finalize(self, test_loader: DataLoader, epoch: int, start_time: float) -> dict:
        """Mark training complete and run the final test evaluation."""
        (self._model_dir / "training_complete.txt").write_text(
            f"Training finished. Epochs: {epoch + 1}, Best F1 L2: {self._early_stopping.best:.4f}\n"
        )

        # Final test evaluation
        test_metrics = self.evaluate(test_loader)
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Test F1 L1 Macro: {test_metrics['f1_l1_macro']:.4f}")
        print(f"Test F1 L2 Macro: {test_metrics['f1_l2_macro']:.4f}")
        print(f"Test HCR: {test_metrics.get('hcr', 0):.4f}")
        print(f"{'='*60}")

        return {
            "history": self._history,
            "test_metrics": test_metrics,
            "best_val_f1_l2": self._early_stopping.best,
            "total_time": total_time,
        }
