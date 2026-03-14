"""
src/utils/visualization.py — Plotting & Visual Diagnostics
===========================================================

Visualisation is not decoration — it is how you understand what your model
has actually learned.  This module contains reusable functions for:

  - Displaying sample images from each class (sanity-check the data).
  - Plotting training/validation loss and accuracy curves.
  - Drawing confusion matrices to see which classes get confused.
  - Generating Grad-CAM heatmaps to explain WHY the model made a prediction.

All functions save their plots to disk AND return the Matplotlib figure so
they can also be rendered directly in a notebook or a Streamlit app.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image

# Use a clean style that looks good in both notebooks and saved PNGs.
plt.style.use("seaborn-v0_8-whitegrid")


def plot_sample_images(
    dataset,
    class_names: List[str],
    n_per_class: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Display a grid of sample images, one row per class.

    This is the very first thing you should do with any new dataset.
    Seeing the actual images lets you spot label errors, weird crops,
    duplicates, or class imbalance problems before spending hours training.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A PyTorch dataset with `.classes` attribute (e.g., ImageFolder).
    class_names : list of str
        Human-readable class labels.
    n_per_class : int
        How many sample images to show per class.
    save_path : str, optional
        If given, the figure is saved here as a PNG.
    figsize : tuple, optional
        (width, height) in inches.  Auto-computed if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_classes = len(class_names)
    cols = n_per_class
    rows = n_classes

    if figsize is None:
        figsize = (cols * 3, rows * 2.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Sample Images by Class", fontsize=16, fontweight="bold", y=1.01)

    # Build an index: class_id → list of sample indices in the dataset
    class_to_indices: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    for row, (class_id, class_name) in enumerate(zip(range(n_classes), class_names)):
        indices = class_to_indices[class_id][:n_per_class]
        for col, sample_idx in enumerate(indices):
            ax = axes[row, col] if n_classes > 1 else axes[col]
            img_path, _ = dataset.samples[sample_idx]
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    class_name, rotation=0, labelpad=80,
                    fontsize=8, va="center", ha="right"
                )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot loss and accuracy curves from a completed or in-progress training run.

    These two curves tell a story:
      - If training loss keeps falling but validation loss flattens or rises,
        the model is overfitting (memorising instead of generalising).
      - If both curves are still falling at epoch N, you could train longer.
      - If both are flat from epoch 1, the learning rate is probably too low.

    Parameters
    ----------
    train_losses, val_losses : list of float
        Per-epoch average losses.
    train_accs, val_accs : list of float
        Per-epoch accuracies (0–100 scale).
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    # ── Loss plot ────────────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-o", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()

    # Annotate the minimum validation loss point
    min_idx = int(np.argmin(val_losses))
    ax1.annotate(
        f"Best: {val_losses[min_idx]:.4f}",
        xy=(min_idx + 1, val_losses[min_idx]),
        xytext=(min_idx + 1 + 1, val_losses[min_idx] + 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )

    # ── Accuracy plot ────────────────────────────────────────────────────────
    ax2.plot(epochs, train_accs, "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, val_accs,   "r-o", markersize=4, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_ylim(0, 105)
    ax2.legend()

    max_idx = int(np.argmax(val_accs))
    ax2.annotate(
        f"Best: {val_accs[max_idx]:.2f}%",
        xy=(max_idx + 1, val_accs[max_idx]),
        xytext=(max_idx + 1 + 1, val_accs[max_idx] - 5),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 18),
    normalize: bool = True,
) -> plt.Figure:
    """
    Render a colour-coded confusion matrix heatmap.

    A confusion matrix shows you, for every true class, how often the model
    predicted each class.  Diagonal cells = correct predictions.  Off-diagonal
    cells = mistakes.  Dark off-diagonal cells tell you exactly which pairs of
    diseases are getting confused — often visually similar ones.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Raw confusion matrix from sklearn.metrics.confusion_matrix.
    class_names : list of str
        Labels for rows (true) and columns (predicted).
    save_path : str, optional
        Where to save the PNG.
    figsize : tuple
        Figure dimensions in inches.
    normalize : bool
        If True, values are shown as percentages (each row sums to 1).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".0%", 1.0
        title = "Confusion Matrix (Normalised)"
    else:
        cm_plot = cm
        fmt, vmax = "d", cm.max()
        title = "Confusion Matrix (Counts)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing how many images exist per class.

    Class imbalance (some classes having far more images than others) biases
    the model toward the majority class.  This plot helps you spot that problem
    early so you can apply oversampling or class weights as a fix.

    Parameters
    ----------
    class_counts : dict
        {class_name: image_count} mapping.
    save_path : str, optional
        Where to save the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(range(len(names)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution in Training Set", fontsize=14, fontweight="bold")

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(count),
            ha="center", va="bottom", fontsize=7, rotation=90,
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Grad-CAM answers the question: "Which regions of the input image were most
    important for this prediction?"  It does this by computing the gradient of
    the predicted class score with respect to the last convolutional feature
    map.  Regions with high positive gradients lit up the most — those are
    where the model was "looking".

    This is crucial for trust-building: if the model predicts "bacterial blight"
    but the heatmap highlights the soil and not the leaf, something is wrong.

    How it works (simplified):
    ---------------------------
    1. Do a forward pass and record the feature maps from the target layer.
    2. Compute the gradient of the target class score w.r.t. those feature maps.
    3. Average the gradients spatially → one weight per feature-map channel.
    4. Take a weighted sum of the feature maps → a single spatial heatmap.
    5. Apply ReLU (only positive activations matter) and normalise.
    6. Resize and overlay onto the original image.

    Reference: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391

    Usage
    -----
    >>> cam = GradCAM(model, target_layer=model.backbone.conv_head)
    >>> heatmap = cam(image_tensor, class_idx=5)
    >>> fig = cam.overlay(original_pil_image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks that intercept the forward and backward passes
        # at the chosen layer.
        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single input image tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (1, C, H, W)
            Pre-processed image tensor (batch size 1).
        class_idx : int, optional
            Class to visualise.  Defaults to the predicted (argmax) class.

        Returns
        -------
        np.ndarray, shape (H, W)
            Normalised heatmap values in [0, 1].
        """
        self.model.eval()
        x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Zero-out existing gradients then back-propagate through only the
        # target class score.
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions → (channels,)
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps → (1, H', W')
        cam = (weights * self._features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """
        Overlay the heatmap onto the original PIL image.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            The original (un-normalised) image.
        heatmap : np.ndarray
            Output of __call__().
        alpha : float
            Blend weight.  0 = only original image, 1 = only heatmap.

        Returns
        -------
        matplotlib.figure.Figure
        """
        img_array = np.array(pil_image.convert("RGB"))
        heatmap_resized = np.uint8(255 * heatmap)

        # Apply a jet colour map to the grayscale heatmap
        colormap = plt.cm.jet(heatmap_resized)[:, :, :3]  # drop alpha channel
        colormap = (colormap * 255).astype(np.uint8)

        blended = (alpha * colormap + (1 - alpha) * img_array).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_array);          axes[0].set_title("Original Image");   axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
        axes[2].imshow(blended);             axes[2].set_title("Overlay");          axes[2].axis("off")
        plt.tight_layout()
        return fig

    def remove_hooks(self):
        """Clean up forward/backward hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
