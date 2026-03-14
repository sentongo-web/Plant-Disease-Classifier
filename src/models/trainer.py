"""
src/models/trainer.py — The Training Engine
============================================

This is the module that actually "teaches" the model.  It runs the training
loop — forward pass, loss computation, backward pass, weight update — for as
many epochs as needed, tracks every metric, saves checkpoints, and integrates
with MLflow for experiment tracking.

The training loop explained step by step
-----------------------------------------
A single training step looks like this:

  1. FORWARD PASS   — feed a batch of images through the model and get logits.
  2. LOSS           — compare logits to the true labels using Cross-Entropy Loss.
  3. BACKWARD PASS  — compute the gradient of the loss with respect to every
                      trainable weight (backpropagation through the computation
                      graph that PyTorch built during the forward pass).
  4. GRADIENT CLIP  — optionally cap gradient norms to prevent exploding grads.
  5. UPDATE         — the optimiser uses the gradients to nudge each weight in
                      the direction that reduces the loss.
  6. ZERO GRADIENTS — PyTorch accumulates gradients by default; we must clear
                      them before the next batch.

This is repeated for every batch in the epoch, then we evaluate on the
validation set (forward pass only — no backward, no update).

Mixed Precision Training (AMP)
--------------------------------
By default, PyTorch uses 32-bit floating-point numbers (float32) for all
computations.  NVIDIA GPUs have special hardware for 16-bit (float16) that
runs roughly 2x faster and uses half the memory.

PyTorch's `torch.cuda.amp.autocast` automatically converts appropriate
operations to float16 during the forward pass, then a `GradScaler` scales
the loss up before the backward pass (to prevent underflow in float16
gradients) and scales it back down before the optimiser step.

The whole thing is opt-in and transparent — you just wrap your forward+backward
in a `with autocast():` block.

Two-Stage Fine-Tuning
----------------------
Stage 1 (epochs 1 → freeze_backbone_epochs):
  - Backbone is frozen (no gradient updates).
  - Only the new classification head trains.
  - Learning rate can be high because we are not risking the pre-trained weights.
  - This "warms up" the head so it produces reasonable gradients before we
    touch the backbone.

Stage 2 (epochs freeze_backbone_epochs → end):
  - Backbone is unfrozen.
  - Both backbone and head are trained together at a lower learning rate.
  - The lower LR prevents catastrophic forgetting of ImageNet features.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_curves

logger = get_logger("trainer")


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Without early stopping, the model will eventually start memorising the
    training set (overfitting).  This class monitors validation loss and
    triggers a stop signal if it hasn't improved by at least `min_delta`
    for `patience` consecutive epochs.

    It also restores the best model weights seen so far when stopping,
    so you always end up with the best model, not the last one.

    Parameters
    ----------
    patience : int
        How many epochs of no improvement before stopping.
    min_delta : float
        Minimum improvement to count as "better".
    verbose : bool
        Print a message every time the best model is updated.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, verbose: bool = True):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = float("inf")
        self.should_stop = False
        self.best_weights: Optional[Dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.
        model : nn.Module
            The model being trained (to save best weights).

        Returns
        -------
        bool
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save a deep copy of the current model weights.
            # state_dict() is a Python dict mapping layer names to tensors.
            self.best_weights = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.6f}.  Saving best weights.")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement for {self.counter}/{self.patience} epochs. "
                    f"(Best: {self.best_loss:.6f}, Current: {val_loss:.6f})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered.")

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load the best-seen weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Best model weights restored.")


class Trainer:
    """
    Orchestrates the full training and validation loop.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    config : dict
    device : torch.device
    mlflow_run : optional MLflow active run (for metric logging)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        mlflow_run=None,
    ):
        if config is None:
            config = load_config()
        self.config = config
        self.cfg_trn = config["training"]

        self.model       = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.mlflow_run   = mlflow_run

        # Select the best available compute device.
        # CUDA → NVIDIA GPU, mps → Apple Silicon, cpu → fallback.
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # ── Loss function ────────────────────────────────────────────────────
        # CrossEntropyLoss = LogSoftmax + NLLLoss combined.
        # It expects raw logits (not softmax outputs) and integer class labels.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # label_smoothing=0.1 softens the hard targets (0/1) to (0.05/0.95).
        # This prevents the model from becoming overconfident and improves
        # generalisation — a technique from "Rethinking the Inception Architecture".

        # ── Optimiser ────────────────────────────────────────────────────────
        # AdamW decouples weight decay from the adaptive learning rate,
        # which fixes a theoretical flaw in the original Adam paper.
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg_trn["learning_rate"],
            weight_decay=self.cfg_trn["weight_decay"],
        )

        # ── Learning-Rate Scheduler ──────────────────────────────────────────
        # CosineAnnealingLR smoothly decays the LR following a cosine curve
        # from the initial value to eta_min over T_max epochs.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg_trn["epochs"],
            eta_min=self.cfg_trn["eta_min"],
        )

        # ── Mixed Precision ──────────────────────────────────────────────────
        self.use_amp = self.cfg_trn.get("use_amp", True) and torch.cuda.is_available()
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── Early Stopping ───────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=self.cfg_trn["early_stopping_patience"],
            min_delta=self.cfg_trn["early_stopping_delta"],
        )

        # ── History ─────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr":         [],
        }

        # ── Checkpoint directory ─────────────────────────────────────────────
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = Path(config["paths"]["models"]) / "best_model.pth"

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns
        -------
        dict
            Training history: {"train_loss": [...], "val_loss": [...], ...}
        """
        epochs            = self.cfg_trn["epochs"]
        freeze_epochs     = self.cfg_trn["freeze_backbone_epochs"]
        save_every        = self.cfg_trn.get("save_every_n_epochs", 5)

        # Stage 1: freeze backbone for the first N epochs
        self.model.freeze_backbone()
        # Re-create optimiser with only the unfrozen (head) parameters
        self._reset_optimizer()

        logger.info(f"Starting training for up to {epochs} epochs.")
        logger.info(f"Stage 1: backbone frozen for first {freeze_epochs} epochs.")

        total_start = time.time()

        for epoch in range(1, epochs + 1):

            # ── Stage transition ──────────────────────────────────────────────
            if epoch == freeze_epochs + 1:
                logger.info(f"\nEpoch {epoch}: Entering Stage 2 — unfreezing backbone.")
                self.model.unfreeze_backbone()
                # Lower LR for fine-tuning the backbone
                self._reset_optimizer(lr_factor=0.1)
                # Reset the scheduler for the remaining epochs
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs - freeze_epochs,
                    eta_min=self.cfg_trn["eta_min"],
                )

            # ── Train one epoch ───────────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(epoch, epochs)

            # ── Validate ──────────────────────────────────────────────────────
            val_loss, val_acc = self._eval_epoch(self.val_loader, desc="Validation")

            # ── Scheduler step ────────────────────────────────────────────────
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Record history ────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # ── Log to MLflow ─────────────────────────────────────────────────
            self._mlflow_log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)

            logger.info(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, val_loss, val_acc)

            # ── Early stopping ────────────────────────────────────────────────
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Training stopped early at epoch {epoch}.")
                break

        # Restore best weights and save the final model
        self.early_stopping.restore_best_weights(self.model)
        self._save_best_model()

        # Plot and save training curves
        fig_path = str(Path(self.config["paths"]["figures"]) / "training_curves.png")
        plot_training_curves(
            self.history["train_loss"], self.history["val_loss"],
            self.history["train_acc"],  self.history["val_acc"],
            save_path=fig_path,
        )
        logger.info(f"Training curves saved to: {fig_path}")

        elapsed = time.time() - total_start
        logger.info(
            f"Training complete in {elapsed/60:.1f} min.  "
            f"Best val loss: {self.early_stopping.best_loss:.6f}"
        )

        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """Run one training epoch.  Returns (mean_loss, accuracy_%)."""
        self.model.train()  # activates dropout, batchnorm in training mode

        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"  Train [{epoch:03d}/{total_epochs}]",
            leave=False,
            unit="batch",
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ── Forward pass (with optional AMP) ─────────────────────────────
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            # ── Backward pass ────────────────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)  # more efficient than zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping — prevents exploding gradients in deep networks.
            # If the norm of all gradients exceeds max_norm, they are scaled down.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Statistics ───────────────────────────────────────────────────
            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total_samples += images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total_samples:.1f}%",
            )

        mean_loss = total_loss / total_samples
        accuracy  = 100.0 * correct / total_samples
        return mean_loss, accuracy

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, desc: str = "Eval"
    ) -> Tuple[float, float]:
        """
        Run a full evaluation pass.  No gradients are computed.

        @torch.no_grad() is a decorator that disables gradient computation for
        the entire function body.  This halves memory usage during evaluation
        because PyTorch doesn't need to store intermediate values for backprop.
        """
        self.model.eval()  # deactivates dropout, batchnorm uses running stats

        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        pbar = tqdm(loader, desc=f"  {desc}", leave=False, unit="batch")
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += images.size(0)

        mean_loss = total_loss / total_samples
        accuracy  = 100.0 * correct / total_samples
        return mean_loss, accuracy

    def _reset_optimizer(self, lr_factor: float = 1.0) -> None:
        """Re-create the AdamW optimiser with currently trainable parameters."""
        base_lr = self.cfg_trn["learning_rate"] * lr_factor
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            weight_decay=self.cfg_trn["weight_decay"],
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        logger.info(f"Optimiser reset.  New LR: {base_lr:.2e}")

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> None:
        """Save a periodic checkpoint to disk."""
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "history":     self.history,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")

    def _save_best_model(self) -> None:
        """Save the best model weights and metadata to models/best_model.pth."""
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":   self.model.state_dict(),
                "num_classes":   self.model.num_classes,
                "backbone_name": self.model.backbone_name,
                "history":       self.history,
                "best_val_loss": self.early_stopping.best_loss,
                "config":        self.config,
            },
            self.best_model_path,
        )
        logger.info(f"Best model saved: {self.best_model_path}")

    def _mlflow_log_metrics(
        self, epoch, train_loss, val_loss, train_acc, val_acc, lr
    ) -> None:
        """Log per-epoch metrics to an active MLflow run if one exists."""
        if self.mlflow_run is None:
            return
        try:
            import mlflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                    "lr":         lr,
                },
                step=epoch,
            )
        except Exception:
            pass  # Never crash training because of a logging failure
