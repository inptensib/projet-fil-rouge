"""Training loop for HYDRA models (single GPU version).

This is the clean, single-GPU training loop. For distributed training
with DDP, see 03_production/src/trainer_ddp.py.

Corrections vs original implementation:
- Added gradient clipping (norm=1.0) as in the official HYDRA repo.
- Added option for BCE loss to match the official paper.
- Uses hierarchy-aware metrics (Constrained F1, HCR).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data_loader import HierarchicalProductDataset, build_parent_child_matrix
from shared.metrics import compute_all_metrics
from shared.utils import set_seed, get_device

from .models import create_hydra_model
from .losses import HYDRALoss


# Supported encoder models
ENCODERS = {
    "CamemBERT": "camembert-base",
    "XLM-RoBERTa": "xlm-roberta-base",
    "mBERT": "bert-base-multilingual-cased",
}


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
        use_focal: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 2,
        early_stopping_min_delta: float = 0.001,
        project_embedding: bool = True,
        pooling: str = "cls",
        check_hierarchy: bool = False,
        save_dir: str = "models",
        seed: int = 42,
    ):
        set_seed(seed)
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
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.85, patience=1,
            threshold=0.01, min_lr=1e-6,
        )

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
        preds_l1 = torch.argmax(logits_l1, dim=-1)
        valid_l2_mask = self.parent_child_matrix[preds_l1]
        masked = logits_l2.clone()
        masked[valid_l2_mask == 0] = float("-inf")
        return masked

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on a dataloader.

        Returns:
            Dictionary of metrics including F1 scores and hierarchy metrics.
        """
        self.model.eval()
        all_preds_l1, all_preds_l2 = [], []
        all_labels_l1, all_labels_l2 = [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

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

        return compute_all_metrics(
            all_labels_l1, all_preds_l1,
            all_labels_l2, all_preds_l2,
            self.l2_to_l1_mapping,
        )

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dictionary with training history and best metrics.
        """
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

        # Training state
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        history = {"epoch": [], "train_loss": [], "val_f1_l2_macro": [], "val_metrics": []}
        model_dir = self.save_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                l1 = batch["label_l1"].to(self.device)
                l2 = batch["label_l2"].to(self.device)

                level_outputs, unified_output = self.model(input_ids, mask)
                loss, _, _ = self.criterion(level_outputs, unified_output, l1, l2)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping (as in official HYDRA implementation)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                pbar.set_postfix({"loss": f"{epoch_loss / num_batches:.4f}"})

            # Epoch-level metrics
            avg_train_loss = epoch_loss / num_batches
            self.scheduler.step(avg_train_loss)

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_f1_avg = (val_metrics["f1_l1_macro"] + val_metrics["f1_l2_macro"]) / 2

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_f1_l2_macro"].append(val_metrics["f1_l2_macro"])
            history["val_metrics"].append(val_metrics)

            print(
                f"  Loss: {avg_train_loss:.4f} | "
                f"Val F1 L1: {val_metrics['f1_l1_macro']:.4f} | "
                f"Val F1 L2: {val_metrics['f1_l2_macro']:.4f} | "
                f"HCR: {val_metrics.get('hcr', 0):.4f}"
            )

            # Early stopping & checkpointing
            if val_f1_avg > best_val_f1 + self.early_stopping_min_delta:
                best_val_f1 = val_f1_avg
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": {
                            "variant": self.variant,
                            "encoder_name": self.encoder_name,
                            "num_classes_l1": self.num_classes_l1,
                            "num_classes_l2": self.num_classes_l2,
                        },
                        "best_val_f1": best_val_f1,
                        "epoch": epoch + 1,
                    },
                    model_dir / "best_model.pt",
                )
                print(f"  -> New best model saved (F1 avg: {best_val_f1:.4f})")
            else:
                epochs_without_improvement += 1

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

        return {
            "history": history,
            "test_metrics": test_metrics,
            "best_val_f1": best_val_f1,
            "total_time": total_time,
        }
