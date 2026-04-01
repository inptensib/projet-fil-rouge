"""Distributed Data Parallel trainer for HYDRA hierarchical classification.

Based on the HYDRABenchmarkDDP class, refactored with:
- Gradient clipping (max_norm=1.0) -- previously missing
- Hierarchy-aware metrics from shared.metrics
- Clean logging without emoji
- Proper separation of concerns
- MLflow integration for experiment tracking
- Intra-epoch evaluation with dual-speed LR scheduling
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from contextlib import nullcontext
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Cross-directory imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.data_loader import (
    HierarchicalProductDataset,
    build_parent_child_matrix,
)
from shared.metrics import (
    compute_all_metrics,
    compute_per_class_f1,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
)
from shared.utils import ENCODERS, enforce_hierarchy

# Models and losses from 02_hydra
_HYDRA_ROOT = os.path.join(_PROJECT_ROOT, "02_hydra")
if _HYDRA_ROOT not in sys.path:
    sys.path.insert(0, _HYDRA_ROOT)

from src.losses import HYDRALoss
from src.models import HYDRAGlobal, HYDRALocal, HYDRANested

from .ddp_utils import cleanup_ddp, print_rank0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HYDRA_VARIANTS = ["local", "global", "nested"]

MODEL_CLASSES = {
    "local": HYDRALocal,
    "global": HYDRAGlobal,
    "nested": HYDRANested,
}

# Gradient clipping max norm -- this was MISSING from the original code
GRAD_CLIP_MAX_NORM = 1.0


class HYDRATrainerDDP:
    """Distributed Data Parallel trainer for HYDRA hierarchical models.

    Supports multi-node training with:
    - Per-rank data sharding
    - Gradient accumulation with DDP no_sync optimization
    - Gradient clipping for training stability
    - Intra-epoch validation with adaptive LR scheduling
    - MLflow experiment tracking (rank 0 only)
    - Error tracking for hard example analysis

    Args:
        shard_dir: Directory containing pre-computed data shards.
        rank: Global rank of this process.
        world_size: Total number of processes.
        device: CUDA device for this process.
        max_length: Maximum token sequence length.
        batch_size: Per-GPU batch size.
        epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        loss_alpha: Weight for the unified loss term.
        save_dir: Directory to save model checkpoints.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        project_embedding: Whether to project encoder embeddings.
        pooling: Pooling strategy ('cls', 'mean', 'max', 'last').
        use_unified_predictions: Use global/nested head for predictions.
        check_hierarchy_consistency: Enforce parent-child constraints at inference.
        early_stopping_patience: Patience for early stopping (in eval intervals).
        early_stopping_min_delta: Minimum F1 improvement for early stopping.
        level_weight_l1: Weight for L1 loss relative to L2.
        focal_gamma: Focal loss gamma parameter.
        grad_clip_max_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        shard_dir: str,
        rank: int,
        world_size: int,
        device: torch.device,
        max_length: int = 256,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 3.5e-5,
        weight_decay: float = 0.01,
        loss_alpha: float = 1.0,
        save_dir: str = "models",
        gradient_accumulation_steps: int = 1,
        project_embedding: bool = True,
        pooling: str = "cls",
        use_unified_predictions: bool = True,
        check_hierarchy_consistency: bool = False,
        early_stopping_patience: int = 2,
        early_stopping_min_delta: float = 0.001,
        level_weight_l1: float = 0.5,
        focal_gamma: float = 2.0,
        grad_clip_max_norm: float = GRAD_CLIP_MAX_NORM,
        warmup_ratio: float = 0.1,
        condition_l2_on_l1: bool = False,
    ):
        self.condition_l2_on_l1 = condition_l2_on_l1
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha
        self.save_dir = save_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.project_embedding = project_embedding
        self.pooling = pooling
        self.use_unified_predictions = use_unified_predictions
        self.check_hierarchy_consistency = check_hierarchy_consistency
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.level_weight_l1 = level_weight_l1
        self.focal_gamma = focal_gamma
        self.grad_clip_max_norm = grad_clip_max_norm
        self.warmup_ratio = warmup_ratio

        # DDP info
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.effective_batch_size = (
            batch_size * world_size * gradient_accumulation_steps
        )

        self.results: dict = {}
        self._load_shards(shard_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_shards(self, shard_dir: str) -> None:
        """Load pre-computed data shards and metadata.

        Each rank loads its own training shard. Only rank 0 loads
        validation and test data.
        """
        t0 = time.time()
        print(f"[Rank {self.rank}] Loading shard from {shard_dir}...")

        with open(os.path.join(shard_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.l1_encoder = metadata["l1_encoder"]
        self.l2_encoder = metadata["l2_encoder"]
        self.num_classes_l1 = metadata["num_classes_l1"]
        self.num_classes_l2 = metadata["num_classes_l2"]
        self.l2_to_l1_mapping = metadata["l2_to_l1_mapping"]

        self.parent_child_matrix = build_parent_child_matrix(
            self.l2_to_l1_mapping, self.num_classes_l1, self.num_classes_l2
        ).to(self.device)

        shard_path = os.path.join(shard_dir, f"train_shard_{self.rank}.pkl")
        with open(shard_path, "rb") as f:
            shard = pickle.load(f)

        self.X_train = pd.Series(shard["texts"])
        self.y_l1_train = pd.Series(shard["labels_l1"])
        self.y_l2_train = pd.Series(shard["labels_l2"])
        n_train_local = len(self.X_train)
        del shard
        gc.collect()

        load_time = time.time() - t0
        print(f"[Rank {self.rank}]   Train shard: {n_train_local:,} samples")

        if self.rank == 0:
            with open(os.path.join(shard_dir, "val.pkl"), "rb") as f:
                val_data = pickle.load(f)
            with open(os.path.join(shard_dir, "test.pkl"), "rb") as f:
                test_data = pickle.load(f)

            self.X_val = pd.Series(val_data["texts"])
            self.y_l1_val = pd.Series(val_data["labels_l1"])
            self.y_l2_val = pd.Series(val_data["labels_l2"])
            self.X_test = pd.Series(test_data["texts"])
            self.y_l1_test = pd.Series(test_data["labels_l1"])
            self.y_l2_test = pd.Series(test_data["labels_l2"])

            if mlflow.active_run():
                mlflow.log_params({
                    "n_train_total": metadata["n_train_total"],
                    "n_train_per_gpu": n_train_local,
                    "n_val": len(self.X_val),
                    "n_test": len(self.X_test),
                    "num_classes_l1": self.num_classes_l1,
                    "num_classes_l2": self.num_classes_l2,
                    "data_load_time_sec": round(load_time, 2),
                })

            del val_data, test_data
            print(
                f"[Rank 0]   Val: {len(self.X_val):,}, "
                f"Test: {len(self.X_test):,}"
            )
        else:
            self.X_val = self.y_l1_val = self.y_l2_val = None
            self.X_test = self.y_l1_test = self.y_l2_test = None

        gc.collect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model_dir(self, encoder_name: str, variant: str) -> str:
        """Return the checkpoint directory for a given model configuration."""
        return os.path.join(
            self.save_dir,
            f"HYDRA_{variant}_{encoder_name.replace(' ', '_')}",
        )

    def _enforce_hierarchy(
        self, logits_l1: torch.Tensor, logits_l2: torch.Tensor
    ) -> torch.Tensor:
        """Mask L2 logits to enforce parent-child consistency."""
        return enforce_hierarchy(logits_l1, logits_l2, self.parent_child_matrix)

    @staticmethod
    def _get_base_model(model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap DDP/DP to get the underlying model."""
        if hasattr(model, "module"):
            return model.module
        return model

    def _num_workers(self) -> int:
        """Compute DataLoader num_workers from SLURM environment."""
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
        gpus = int(os.environ.get("GPUS_CE_NOEUD", "1"))
        return max(1, cpus // gpus)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        variant: str,
        collect_errors: bool = False,
    ) -> tuple[dict[str, float], list[dict]]:
        """Evaluate model on a dataset and optionally collect error details.

        Uses the shared metrics module for F1 scores and hierarchy metrics.

        Args:
            model: The (unwrapped) HYDRA model.
            dataloader: Evaluation DataLoader.
            variant: HYDRA variant name.
            collect_errors: Whether to collect per-sample error details.

        Returns:
            Tuple of (metrics_dict, error_details_list).
        """
        model.eval()
        all_preds_l1, all_preds_l2 = [], []
        all_labels_l1, all_labels_l2 = [], []
        error_details = []
        max_errors = 5000

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                level_outputs, unified_output = model(input_ids, attention_mask)
                logits_l1, logits_l2 = level_outputs

                if (
                    self.use_unified_predictions
                    and unified_output is not None
                    and variant in ("global", "nested")
                ):
                    num_l1 = model.num_classes_l1
                    logits_l1 = unified_output[:, :num_l1]
                    logits_l2 = unified_output[:, num_l1:]

                if self.check_hierarchy_consistency:
                    logits_l2 = self._enforce_hierarchy(logits_l1, logits_l2)

                probs_l1 = F.softmax(logits_l1, dim=-1)
                probs_l2 = F.softmax(logits_l2, dim=-1)
                conf_l1, preds_l1_batch = torch.max(probs_l1, dim=-1)
                conf_l2, preds_l2_batch = torch.max(probs_l2, dim=-1)

                preds_l1_np = preds_l1_batch.cpu().numpy()
                preds_l2_np = preds_l2_batch.cpu().numpy()
                labels_l1_batch = batch["label_l1"].numpy()
                labels_l2_batch = batch["label_l2"].numpy()

                all_preds_l1.extend(preds_l1_np)
                all_preds_l2.extend(preds_l2_np)
                all_labels_l1.extend(labels_l1_batch)
                all_labels_l2.extend(labels_l2_batch)

                if collect_errors and len(error_details) < max_errors:
                    for i in range(len(preds_l1_np)):
                        if len(error_details) >= max_errors:
                            break
                        correct_l1 = preds_l1_np[i] == labels_l1_batch[i]
                        correct_l2 = preds_l2_np[i] == labels_l2_batch[i]
                        if not correct_l1 or not correct_l2 or conf_l2[i].item() < 0.4:
                            error_details.append({
                                "true_l1": self.l1_encoder.inverse_transform(
                                    [labels_l1_batch[i]]
                                )[0],
                                "pred_l1": self.l1_encoder.inverse_transform(
                                    [preds_l1_np[i]]
                                )[0],
                                "conf_l1": round(conf_l1[i].item(), 4),
                                "true_l2": self.l2_encoder.inverse_transform(
                                    [labels_l2_batch[i]]
                                )[0],
                                "pred_l2": self.l2_encoder.inverse_transform(
                                    [preds_l2_np[i]]
                                )[0],
                                "conf_l2": round(conf_l2[i].item(), 4),
                                "error_type": (
                                    "L1_ERROR" if not correct_l1
                                    else "L2_ERROR" if not correct_l2
                                    else "LOW_CONFIDENCE"
                                ),
                            })

        # Use shared metrics (includes hierarchy metrics when mapping available)
        metrics = compute_all_metrics(
            all_labels_l1, all_preds_l1,
            all_labels_l2, all_preds_l2,
            l2_to_l1_mapping=self.l2_to_l1_mapping,
        )
        # Attach raw predictions for audit visualizations
        metrics["_preds_l1"] = all_preds_l1
        metrics["_preds_l2"] = all_preds_l2
        metrics["_labels_l1"] = all_labels_l1
        metrics["_labels_l2"] = all_labels_l2
        return metrics, error_details

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_epoch_metrics(
        self,
        epoch_history: dict,
        model_key: str,
        model_dir: str,
    ) -> str:
        """Save a training progress plot.

        Args:
            epoch_history: Dictionary with epoch-level metrics.
            model_key: Model identifier for the plot title.
            model_dir: Directory to save the plot.

        Returns:
            Path to the saved plot file.
        """
        epochs = epoch_history["epoch"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Training Progress: {model_key}", fontsize=14)

        axes[0, 0].plot(epochs, epoch_history["train_loss"], "o-", color="#E74C3C", linewidth=2)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")

        axes[0, 1].plot(
            epochs, epoch_history["val_f1_l2_macro"],
            "D-", color="#E74C3C", label="L2 Macro (val)",
        )
        axes[0, 1].set_title("F1 Macro L2")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Epoch")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(model_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return plot_path

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, encoder_name: str, variant: str = "local") -> dict:
        """Train a single HYDRA model with DDP.

        Implements the full training loop with:
        - Gradient accumulation with DDP no_sync
        - Gradient clipping (max_norm configurable, default 1.0)
        - Intra-epoch evaluation on validation set
        - Early stopping synchronized across all ranks
        - MLflow logging (rank 0 only)
        - Error tracking for hard examples

        Args:
            encoder_name: Name of the encoder (e.g., 'CamemBERT').
            variant: HYDRA variant ('local', 'global', 'nested').

        Returns:
            Results dictionary.
        """
        model_key = f"HYDRA_{variant}_{encoder_name}"
        print_rank0("=" * 70, self.rank)
        print_rank0(f"Training {model_key}", self.rank)
        print_rank0("=" * 70, self.rank)

        # -- MLflow setup (rank 0 only) --------------------------------
        if self.rank == 0:
            mlflow.set_experiment("HYDRA_Hierarchical_Classification")
            mlflow.start_run(run_name=model_key)
            gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            mlflow.log_params({
                "encoder": encoder_name,
                "variant": variant,
                "batch_size_per_gpu": self.batch_size,
                "effective_batch_size": self.effective_batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_length": self.max_length,
                "world_size": self.world_size,
                "total_epochs": self.epochs,
                "grad_clip_max_norm": self.grad_clip_max_norm,
                "hidden_dim": 256,
                "pooling": self.pooling,
                "project_embedding": self.project_embedding,
                "level_weight_l1": self.level_weight_l1,
                "focal_gamma": self.focal_gamma,
                "warmup_ratio": self.warmup_ratio,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gpu_type": gpu_type,
            })

        # -- Data loaders -----------------------------------------------
        encoder_path = ENCODERS[encoder_name]
        tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        nw = self._num_workers()

        train_dataset = HierarchicalProductDataset(
            self.X_train, self.y_l1_train, self.y_l2_train,
            tokenizer, self.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=nw, pin_memory=True, drop_last=True,
        )

        val_loader = test_loader = None
        if self.rank == 0:
            val_loader = DataLoader(
                HierarchicalProductDataset(
                    self.X_val, self.y_l1_val, self.y_l2_val,
                    tokenizer, self.max_length,
                ),
                batch_size=self.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True,
            )
            test_loader = DataLoader(
                HierarchicalProductDataset(
                    self.X_test, self.y_l1_test, self.y_l2_test,
                    tokenizer, self.max_length,
                ),
                batch_size=self.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True,
            )

        # -- Model setup ------------------------------------------------
        model = MODEL_CLASSES[variant](
            encoder_name=encoder_path,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            pooling=self.pooling,
            project_embedding=self.project_embedding,
        )
        model = model.to(self.device)

        # Wrap with DDP or DataParallel
        if self.world_size > 1:
            model = DDP(model, device_ids=[0], find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            print_rank0(
                f"   Single process with {torch.cuda.device_count()} GPUs "
                f"-- using DataParallel.",
                self.rank,
            )
            model = torch.nn.DataParallel(model)

        criterion = HYDRALoss(
            architecture=variant,
            alpha=self.loss_alpha,
            num_classes_l1=self.num_classes_l1,
            num_classes_l2=self.num_classes_l2,
            level_weight_l1=self.level_weight_l1,
            focal_gamma=self.focal_gamma,
        )

        # Weight decay différencié : pas de decay sur bias et LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer = AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ], lr=self.lr)

        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        # Warmup linéaire puis LR constant
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        if self.rank == 0:
            mlflow.log_params({
                "steps_per_epoch": len(train_loader),
                "total_training_steps": total_steps,
            })

        # -- Training state ---------------------------------------------
        start_time = time.time()
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        epoch_history = {
            "epoch": [], "train_loss": [],
            "val_f1_l1_macro": [], "val_f1_l2_macro": [],
            "val_hcr": [],
        }
        model_dir = self._get_model_dir(encoder_name, variant)

        if self.rank == 0:
            os.makedirs(model_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # Dual-speed scheduling parameters
        total_steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        eval_every_n_steps = max(1, total_steps_per_epoch // 10)
        lr_check_every_n_steps = max(1, eval_every_n_steps // 4)
        global_step = 0

        # -- Epoch loop --------------------------------------------------
        for epoch in range(self.epochs):
            model.train()
            interval_loss = 0.0
            interval_batches = 0
            train_error_details = []

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=(self.rank != 0),
            )

            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                l1 = batch["label_l1"].to(self.device)
                l2 = batch["label_l2"].to(self.device)
                s_idxs = batch["sample_idx"]

                is_accumulation_step = (
                    (step + 1) % self.gradient_accumulation_steps != 0
                )
                sync_context = (
                    model.no_sync() if is_accumulation_step and hasattr(model, "no_sync")
                    else nullcontext()
                )

                with sync_context:
                    level_outputs, unified_output = model(input_ids, mask)

                    # Conditionner L2 sur le vrai L1 (teacher forcing)
                    if self.condition_l2_on_l1:
                        logits_l1, logits_l2 = level_outputs
                        valid_mask = self.parent_child_matrix[l1]
                        logits_l2 = logits_l2.clone()
                        logits_l2[valid_mask == 0] = float("-inf")
                        level_outputs = [logits_l1, logits_l2]

                    loss, _, _ = criterion(level_outputs, unified_output, l1, l2)

                    # Collect training errors (rank 0 only, limited)
                    if self.rank == 0 and len(train_error_details) < 1000:
                        with torch.no_grad():
                            p_l1 = torch.argmax(level_outputs[0], -1)
                            p_l2 = torch.argmax(level_outputs[1], -1)
                            for i in range(len(input_ids)):
                                if len(train_error_details) >= 1000:
                                    break
                                if p_l1[i] != l1[i] or p_l2[i] != l2[i]:
                                    train_error_details.append({
                                        "sample_idx": s_idxs[i].item(),
                                        "loss": round(loss.item(), 4),
                                        "true_l1": self.l1_encoder.inverse_transform(
                                            [l1[i].item()]
                                        )[0],
                                        "pred_l1": self.l1_encoder.inverse_transform(
                                            [p_l1[i].item()]
                                        )[0],
                                        "true_l2": self.l2_encoder.inverse_transform(
                                            [l2[i].item()]
                                        )[0],
                                        "pred_l2": self.l2_encoder.inverse_transform(
                                            [p_l2[i].item()]
                                        )[0],
                                    })

                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()

                # Optimizer step with GRADIENT CLIPPING
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.grad_clip_max_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log step-level metrics
                    if self.rank == 0 and global_step % 50 == 0:
                        progress = (global_step / total_steps) * 100
                        mlflow.log_metrics({
                            "loss_step": loss.item(),
                            "lr_step": optimizer.param_groups[0]["lr"],
                            "global_step": global_step,
                            "progress_pct": progress,
                            "current_epoch": epoch + 1,
                            "steps_remaining": total_steps - global_step,
                        }, step=global_step)

                interval_loss += loss.item()
                interval_batches += 1

                # -- Dual-speed scheduling and intra-epoch evaluation ----
                is_lr_step = (
                    global_step > 0
                    and global_step % lr_check_every_n_steps == 0
                    and not is_accumulation_step
                )
                is_eval_step = (
                    global_step > 0
                    and global_step % eval_every_n_steps == 0
                    and not is_accumulation_step
                )
                is_epoch_end = (step + 1) == len(train_loader)

                if is_lr_step or is_eval_step or is_epoch_end:
                    # Synchronized average training loss across ranks
                    avg_loss = interval_loss / max(interval_batches, 1)
                    loss_tensor = torch.tensor([avg_loss], device=self.device)

                    if dist.is_initialized():
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        avg_loss = loss_tensor.item() / self.world_size
                    else:
                        avg_loss = loss_tensor.item()

                    if self.rank == 0:
                        mlflow.log_metric(
                            "avg_train_loss_interval", avg_loss, step=global_step,
                        )

                    # -- Evaluation checkpoint ---------------------------
                    if is_eval_step or is_epoch_end:
                        if self.rank == 0:
                            if train_error_details:
                                mlflow.log_table(
                                    data=pd.DataFrame(train_error_details),
                                    artifact_file=f"hard_train_step_{global_step}.json",
                                )

                            print(
                                f"\n   [Eval] Rank 0 evaluation at step "
                                f"{global_step} (epoch {epoch + 1})..."
                            )
                            eval_start = time.time()
                            base_model = self._get_base_model(model)
                            val_metrics, error_details = self._evaluate_model(
                                base_model, val_loader, variant, collect_errors=True,
                            )
                            eval_duration = time.time() - eval_start

                            # Test set only at epoch end
                            if is_epoch_end:
                                test_metrics, _ = self._evaluate_model(
                                    base_model, test_loader, variant,
                                    collect_errors=False,
                                )
                                mlflow.log_metrics({
                                    "test_f1_l1_macro": test_metrics["f1_l1_macro"],
                                    "test_f1_l2_macro": test_metrics["f1_l2_macro"],
                                }, step=global_step)
                                # Log hierarchy metrics if available
                                if "hcr" in test_metrics:
                                    mlflow.log_metrics({
                                        "test_hcr": test_metrics["hcr"],
                                        "test_constrained_f1_macro": test_metrics.get(
                                            "constrained_f1_macro", 0.0
                                        ),
                                    }, step=global_step)

                            mlflow.log_metrics({
                                "val_f1_l1_macro": val_metrics["f1_l1_macro"],
                                "val_f1_l2_macro": val_metrics["f1_l2_macro"],
                                "eval_duration_sec": eval_duration,
                            }, step=global_step)

                            # Log hierarchy metrics for validation
                            if "hcr" in val_metrics:
                                mlflow.log_metrics({
                                    "val_hcr": val_metrics["hcr"],
                                    "val_constrained_f1_macro": val_metrics.get(
                                        "constrained_f1_macro", 0.0
                                    ),
                                }, step=global_step)

                            if error_details:
                                mlflow.log_table(
                                    data=pd.DataFrame(error_details),
                                    artifact_file=f"hard_val_step_{global_step}.json",
                                )

                            val_f1_avg = (
                                val_metrics["f1_l1_macro"]
                                + val_metrics["f1_l2_macro"]
                            ) / 2
                            print(
                                f"   [Eval] Done. Val F1 avg: {val_f1_avg:.4f}"
                            )

                        # Update epoch history for training curves
                        if is_epoch_end and self.rank == 0:
                            epoch_history["epoch"].append(epoch + 1)
                            epoch_history["train_loss"].append(avg_loss)
                            epoch_history["val_f1_l1_macro"].append(val_metrics["f1_l1_macro"])
                            epoch_history["val_f1_l2_macro"].append(val_metrics["f1_l2_macro"])
                            epoch_history["val_hcr"].append(val_metrics.get("hcr", 0.0))

                        # Synchronized early stopping
                        early_stop_tensor = torch.tensor([0], device=self.device)
                        if self.rank == 0:
                            if val_f1_avg > best_val_f1 + self.early_stopping_min_delta:
                                best_val_f1 = val_f1_avg
                                epochs_without_improvement = 0
                                mlflow.pytorch.log_model(
                                    base_model, "best_model_checkpoint",
                                )
                                torch.save(
                                    {
                                        "model_state_dict": base_model.state_dict(),
                                        "global_step": global_step,
                                        "best_val_f1": best_val_f1,
                                        "config": {
                                            "num_classes_l1": self.num_classes_l1,
                                            "num_classes_l2": self.num_classes_l2,
                                            "variant": variant,
                                            "encoder_name": encoder_name,
                                            "pooling": self.pooling,
                                            "project_embedding": self.project_embedding,
                                        },
                                    },
                                    os.path.join(model_dir, "hydra_model.pt"),
                                )
                            else:
                                epochs_without_improvement += 1

                            if epochs_without_improvement >= self.early_stopping_patience:
                                early_stop_tensor[0] = 1

                        if dist.is_initialized():
                            dist.broadcast(early_stop_tensor, src=0)
                            dist.barrier()

                        if early_stop_tensor.item() == 1:
                            print_rank0(
                                f"[Early stopping] Triggered at step {global_step}",
                                self.rank,
                            )
                            if self.rank == 0:
                                mlflow.end_run()
                            cleanup_ddp()
                            return self.results

                    # Reset interval tracking
                    interval_loss = 0.0
                    interval_batches = 0
                    train_error_details = []
                    model.train()

        # -- End of training: audit artifacts ----------------------------
        if self.rank == 0:
            total_duration = time.time() - start_time
            mlflow.log_metric("total_training_duration_sec", total_duration)

            # Generate audit visualizations on validation set
            try:
                base_model = self._get_base_model(model)
                val_audit_metrics, _ = self._evaluate_model(
                    base_model, val_loader, variant, collect_errors=False,
                )
                all_preds_l1, all_labels_l1 = val_audit_metrics.get("_preds_l1", []), val_audit_metrics.get("_labels_l1", [])
                all_preds_l2, all_labels_l2 = val_audit_metrics.get("_preds_l2", []), val_audit_metrics.get("_labels_l2", [])

                if all_labels_l1:
                    # Confusion matrices
                    fig_cm_l1 = plot_confusion_matrix(
                        all_labels_l1, all_preds_l1, self.l1_encoder,
                        title="Matrice de Confusion — L1 (157 classes)",
                    )
                    mlflow.log_figure(fig_cm_l1, "confusion_matrix_l1.png")
                    plt.close(fig_cm_l1)

                    fig_cm_l2 = plot_confusion_matrix(
                        all_labels_l2, all_preds_l2, self.l2_encoder,
                        title="Matrice de Confusion — L2 (top 50)",
                        top_n=50,
                    )
                    mlflow.log_figure(fig_cm_l2, "confusion_matrix_l2_top50.png")
                    plt.close(fig_cm_l2)

                    # Per-class F1
                    f1_l1 = compute_per_class_f1(all_labels_l1, all_preds_l1, self.l1_encoder)
                    fig_f1_l1 = plot_per_class_f1(f1_l1, "F1 par classe — L1", top_n=30)
                    mlflow.log_figure(fig_f1_l1, "f1_per_class_l1.png")
                    plt.close(fig_f1_l1)
                    mlflow.log_dict(f1_l1, "f1_per_class_l1.json")

                    f1_l2 = compute_per_class_f1(all_labels_l2, all_preds_l2, self.l2_encoder)
                    fig_f1_l2 = plot_per_class_f1(f1_l2, "F1 par classe — L2 (top/bottom 30)", top_n=30)
                    mlflow.log_figure(fig_f1_l2, "f1_per_class_l2.png")
                    plt.close(fig_f1_l2)
                    mlflow.log_dict(
                        {k: v for i, (k, v) in enumerate(f1_l2.items()) if i < 30 or i >= len(f1_l2) - 30},
                        "f1_per_class_l2_extremes.json",
                    )

                # Training curves
                if epoch_history["epoch"]:
                    fig_curves = plot_training_curves(epoch_history)
                    mlflow.log_figure(fig_curves, "training_curves.png")
                    plt.close(fig_curves)

            except Exception as e:
                print(f"   [Audit] Warning: could not generate audit artifacts: {e}")

            # Explainability report (on validation set)
            try:
                from shared.explainability import generate_explainability_report
                val_texts = self.X_val.tolist() if hasattr(self.X_val, "tolist") else list(self.X_val)
                explain_html = generate_explainability_report(
                    model=base_model,
                    tokenizer=tokenizer,
                    texts=val_texts,
                    labels_l1=list(self.y_l1_val),
                    labels_l2=list(self.y_l2_val),
                    preds_l1=all_preds_l1,
                    preds_l2=all_preds_l2,
                    parent_child_matrix=self.parent_child_matrix if self.condition_l2_on_l1 else None,
                    l1_encoder=self.l1_encoder,
                    l2_encoder=self.l2_encoder,
                    device=self.device,
                )
                mlflow.log_text(explain_html, "explainability_report.html")
            except Exception as e:
                print(f"   [Explain] Warning: could not generate explainability report: {e}")

            mlflow.end_run()

        if dist.is_initialized():
            dist.barrier()
        return self.results

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        encoders: Optional[list[str]] = None,
        variants: Optional[list[str]] = None,
    ) -> dict:
        """Run training across multiple encoder/variant combinations.

        Args:
            encoders: List of encoder names. Defaults to all available.
            variants: List of variant names. Defaults to all available.

        Returns:
            Aggregated results dictionary.
        """
        if encoders is None:
            encoders = list(ENCODERS.keys())
        if variants is None:
            variants = HYDRA_VARIANTS

        print_rank0(f"\n{'=' * 70}", self.rank)
        print_rank0(
            f"HYDRA DDP Benchmark ({self.world_size} processes)", self.rank
        )
        print_rank0(f"   Encoders: {encoders}", self.rank)
        print_rank0(f"   Variants: {variants}", self.rank)
        print_rank0(f"   Grad clip: max_norm={self.grad_clip_max_norm}", self.rank)
        print_rank0(f"{'=' * 70}", self.rank)

        for encoder in encoders:
            for variant in variants:
                try:
                    self.train_model(encoder, variant)
                except Exception as e:
                    if self.rank == 0:
                        mlflow.end_run(status="FAILED")
                    print(
                        f"[Rank {self.rank}] ERROR training "
                        f"{encoder}/{variant}: {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    continue

        return self.results
