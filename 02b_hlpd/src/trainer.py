"""Training orchestrator for HLPD.

Key differences from FlatHydraTrainer:
- bfloat16 autocast (no GradScaler)
- ZLPRLoss with binary y_true built on-the-fly
- Intra-epoch checkpointing (every N optimiser steps)
- Gradient accumulation
- DataParallel on multi-GPU (mirrors FlatHydraTrainer pattern)

Public API:
    HLPDConfig  — typed hyperparameter dataclass.
    HLPDTrainer — training loop, evaluate, save, resume.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data_loader import HierarchicalProductDataset  # noqa: E402
from shared.metrics import compute_all_metrics              # noqa: E402

from .losses import ZLPRLoss   # noqa: E402
from .models import HLPD       # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hlpd_3090.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HLPDConfig:
    """Typed hyperparameters for HLPD.

    Canonical values live in ``02b_hlpd/configs/hlpd_3090.yaml``.
    """

    # Encoder
    encoder: str = "camembert-base"
    max_length: int = 128
    freeze_encoder: bool = False

    # Decoder architecture
    num_decoder_layers: int = 2
    nhead: int = 4
    ffn_dim: int = 2048
    decoder_dropout: float = 0.1
    gradient_checkpointing: bool = True

    # Training
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    # Intra-epoch checkpointing
    checkpoint_every_n_steps: int = 5000

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Artefacts
    save_dir: str = "models/hlpd"
    experiment_name: str = "HLPD_CamemBERT"

    # Reproducibility
    seed: int = 42

    # Hardware  ("bfloat16" | "float32")
    precision: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (0.0 <= self.decoder_dropout < 1.0):
            raise ValueError(f"decoder_dropout must be in [0, 1), got {self.decoder_dropout}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError(f"precision must be 'bfloat16' or 'float32', got {self.precision}")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "HLPDConfig":
        """Load from YAML file, then apply keyword overrides."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with p.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        known = {field for field in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        filtered.update(overrides)
        return cls(**filtered)


def _get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class HLPDTrainer:
    """Training orchestrator for HLPD.

    Args:
        data:        Dict from ``shared.data_loader.prepare_hierarchical_data_from_splits()``.
        config:      Hyperparameters.
        device:      Target device. Auto-detected if None.
        resume_from: Path to ``checkpoint_latest.pt`` to resume training.
    """

    def __init__(
        self,
        data: dict,
        config: Optional[HLPDConfig] = None,
        device: Optional[torch.device] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.data = data
        self.config = config or HLPDConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.resume_from = resume_from

        torch.manual_seed(self.config.seed)

        self.num_l1: int = data["num_classes_l1"]
        self.num_l2: int = data["num_classes_l2"]
        self.num_labels: int = self.num_l1 + self.num_l2
        self.l1_encoder = data["l1_encoder"]
        self.l2_encoder = data["l2_encoder"]
        self.l2_to_l1_mapping: dict = data["l2_to_l1_mapping"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder)

        self._base_model = HLPD(
            encoder_name=self.config.encoder,
            num_l1=self.num_l1,
            num_l2=self.num_l2,
            l2_to_l1_mapping=self.l2_to_l1_mapping,
            num_decoder_layers=self.config.num_decoder_layers,
            nhead=self.config.nhead,
            ffn_dim=self.config.ffn_dim,
            decoder_dropout=self.config.decoder_dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
            freeze_encoder=self.config.freeze_encoder,
        ).to(self.device)

        self.model = self._base_model
        if torch.cuda.device_count() > 1:
            print(f"[HLPDTrainer] DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self._base_model)

        self.criterion = ZLPRLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # bfloat16: no GradScaler needed (numerically stable)
        self.use_bf16 = (
            self.config.precision == "bfloat16"
            and self.device.type == "cuda"
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _make_loader(self, texts, labels_l1, labels_l2, shuffle: bool) -> DataLoader:
        ds = HierarchicalProductDataset(
            texts, labels_l1, labels_l2, self.tokenizer, self.config.max_length
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda" and self.config.pin_memory,
            drop_last=shuffle,
        )

    # ------------------------------------------------------------------
    # y_true construction
    # ------------------------------------------------------------------

    def _build_y_true(
        self,
        l1: torch.Tensor,
        l2: torch.Tensor,
    ) -> torch.Tensor:
        """Build binary multi-label target (B, L).

        Sets y_true[b, l1[b]] = 1  (L1 label)
        Sets y_true[b, num_l1 + l2[b]] = 1  (L2 label)
        All other positions remain 0.
        """
        B = l1.shape[0]
        y_true = torch.zeros(B, self.num_labels, device=self.device)
        y_true.scatter_(1, l1.unsqueeze(1), 1.0)
        l2_full = (self.num_l1 + l2).unsqueeze(1)
        y_true.scatter_(1, l2_full, 1.0)
        return y_true

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on a dataloader, returns metrics dict.

        Predictions: argmax on L1 part and L2 part of logits separately.
        """
        self.model.eval()
        all_preds_l1, all_preds_l2 = [], []
        all_labels_l1, all_labels_l2 = [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                logits = self.model(input_ids, mask)  # (B, L)

            logits_l1 = logits[:, :self.num_l1]
            logits_l2 = logits[:, self.num_l1:]

            preds_l1 = logits_l1.argmax(dim=-1)
            preds_l2 = logits_l2.argmax(dim=-1)

            all_preds_l1.extend(preds_l1.cpu().tolist())
            all_preds_l2.extend(preds_l2.cpu().tolist())
            all_labels_l1.extend(batch["label_l1"].tolist())
            all_labels_l2.extend(batch["label_l2"].tolist())

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop with intra-epoch checkpointing.

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

        total_optim_steps = (
            len(train_loader) // self.config.gradient_accumulation_steps
        ) * self.config.epochs
        warmup_steps = int(total_optim_steps * self.config.warmup_ratio)
        self.scheduler = _get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, max(total_optim_steps, 1)
        )

        model_dir = Path(self.config.save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        history: dict = {
            "epoch": [], "train_loss": [],
            "val_f1_l1_macro": [], "val_f1_l2_macro": [],
            "val_hcr": [], "val_constrained_f1_macro": [],
        }
        best_val_f1 = 0.0
        epochs_no_improve = 0
        start_epoch = 0
        global_step = 0
        start_step_in_epoch = 0

        # MLflow (optional — graceful if not running)
        mlflow_active = False
        try:
            import mlflow
            db_path = Path(__file__).resolve().parents[2] / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "encoder": self.config.encoder,
                "max_length": self.config.max_length,
                "num_decoder_layers": self.config.num_decoder_layers,
                "nhead": self.config.nhead,
                "ffn_dim": self.config.ffn_dim,
                "decoder_dropout": self.config.decoder_dropout,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "warmup_ratio": self.config.warmup_ratio,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "precision": self.config.precision,
                "num_l1": self.num_l1,
                "num_l2": self.num_l2,
            })
            mlflow_active = True
            print(f"[MLflow] Tracking active → {db_path}")
        except Exception as e:
            print(f"[MLflow] Disabled ({e})")

        # Resume from checkpoint
        if self.resume_from and Path(self.resume_from).exists():
            print(f"[Resume] Loading {self.resume_from}")
            ckpt = torch.load(
                self.resume_from, map_location=self.device, weights_only=False
            )
            self._base_model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt.get("global_step", 0)
            best_val_f1 = ckpt["best_val_f1"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            history = ckpt.get("history", history)
            start_step_in_epoch = ckpt.get("step_in_epoch", 0)
            print(
                f"[Resume] Epoch {start_epoch}, "
                f"step_in_epoch={start_step_in_epoch}, "
                f"global_step={global_step}, best F1 L2={best_val_f1:.4f}"
            )

        print(f"\n{'=' * 60}")
        print(f"HLPD — {self.config.encoder}")
        print(f"L1: {self.num_l1} | L2: {self.num_l2} | L: {self.num_labels}")
        print(f"Precision: {self.config.precision} | bf16: {self.use_bf16}")
        print(f"{'=' * 60}")

        start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            self.optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for step, batch in enumerate(pbar):
                # Fast-forward when resuming mid-epoch
                if step < start_step_in_epoch:
                    continue

                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                l1 = batch["label_l1"].to(self.device)
                l2 = batch["label_l2"].to(self.device)

                y_true = self._build_y_true(l1, l2)  # (B, L) float32

                with autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    logits = self.model(input_ids, mask)  # (B, L)
                    loss = self.criterion(logits, y_true)
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Intra-epoch checkpoint
                    if global_step % self.config.checkpoint_every_n_steps == 0:
                        self._save_checkpoint(
                            model_dir, epoch, step + 1, global_step,
                            best_val_f1, epochs_no_improve, history
                        )
                        print(f"\n  [Checkpoint] global_step={global_step} saved")

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                n_batches += 1
                pbar.set_postfix({"loss": f"{epoch_loss / n_batches:.4f}"})

            # Reset fast-forward offset for subsequent epochs
            start_step_in_epoch = 0

            # Flush remaining accumulated gradients at end of epoch
            if n_batches % self.config.gradient_accumulation_steps != 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
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

            # VRAM report (once, after epoch 1)
            if epoch == start_epoch and self.device.type == "cuda":
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.max_memory_allocated(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  [VRAM] GPU {i}: {alloc:.1f} / {total:.1f} GB peak")

            # Persist metrics locally
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(history, f, indent=2)

            # MLflow logging
            if mlflow_active:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "train_loss": avg_loss,
                        "val_f1_l1_macro": val_metrics["f1_l1_macro"],
                        "val_f1_l2_macro": val_f1_l2,
                        "val_hcr": val_metrics.get("hcr", 0.0),
                        "val_constrained_f1_macro": val_metrics.get(
                            "constrained_f1_macro", 0.0
                        ),
                    }, step=epoch + 1)
                except Exception as e:
                    print(f"  [MLflow] Metrics failed ({e})")

            # Best model + early stopping
            if val_f1_l2 > best_val_f1 + self.config.early_stopping_min_delta:
                best_val_f1 = val_f1_l2
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self._base_model.state_dict(),
                        "config": asdict(self.config),
                        "num_l1": self.num_l1,
                        "num_l2": self.num_l2,
                        "best_val_f1_l2": best_val_f1,
                        "epoch": epoch + 1,
                    },
                    model_dir / "best_model.pt",
                )
                print(f"  -> Best model saved (F1 L2: {best_val_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    print(
                        f"  [Early stopping] No improvement for "
                        f"{self.config.early_stopping_patience} epochs."
                    )
                    break

            # End-of-epoch resume checkpoint
            self._save_checkpoint(
                model_dir, epoch + 1, 0, global_step,
                best_val_f1, epochs_no_improve, history
            )

        if mlflow_active:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

        (model_dir / "training_complete.txt").write_text(
            f"best_val_f1_l2={best_val_f1:.4f}\n"
            f"epochs_trained={history['epoch'][-1] if history['epoch'] else 0}\n"
        )
        print(f"\n[Done] Best F1 L2: {best_val_f1:.4f} — artefacts in {model_dir}/")
        return history

    def _save_checkpoint(
        self,
        model_dir: Path,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        best_val_f1: float,
        epochs_no_improve: int,
        history: dict,
    ) -> None:
        torch.save(
            {
                "model_state_dict": self._base_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "best_val_f1": best_val_f1,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            },
            model_dir / "checkpoint_latest.pt",
        )

    def save(self, save_dir: Optional[str] = None) -> None:
        """Persist artefacts (tokenizer, encoders, mapping, config)."""
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
        config_dict["num_l1"] = self.num_l1
        config_dict["num_l2"] = self.num_l2
        with open(dest / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"[HLPD] Artefacts saved to {dest}/")
