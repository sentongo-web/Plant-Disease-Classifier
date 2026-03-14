"""
src/evaluation/metrics.py — Model Evaluation & Metrics
=======================================================

Training a model is only half the story.  The other half is understanding
HOW WELL and HOW RELIABLY it performs — and being honest about its weaknesses.

This module evaluates a trained model on a held-out test set and produces
a suite of diagnostic metrics and visualisations.

Key Metrics Explained
---------------------
Accuracy:
  The fraction of all predictions that were correct.  Easy to understand but
  misleading on imbalanced datasets — a model that always predicts the most
  common class can score 90 % accuracy while being completely useless.

Precision (per class):
  Of all the images the model labelled as "Tomato Early Blight", what fraction
  actually were?  High precision = few false alarms.

Recall (per class):
  Of all actual "Tomato Early Blight" images, what fraction did the model
  catch?  High recall = few missed cases.  In disease detection, you generally
  want high recall — it is worse to miss a sick plant than to over-warn.

F1-Score (per class):
  The harmonic mean of precision and recall.  Balances both concerns into a
  single number.  Macro-F1 is the unweighted average of per-class F1 scores —
  the fairest summary when you care equally about all classes.

Confusion Matrix:
  A 38×38 grid where cell (i, j) = how many images of true class i were
  predicted as class j.  The diagonal is correct; everything else is an error.
  This tells you which diseases are visually similar and get confused.

Top-K Accuracy:
  Instead of requiring the model to get the exact right answer, we check if
  the correct label is anywhere in the model's top-K predictions.  Top-5
  accuracy is usually ~5–10 % higher than top-1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix

logger = get_logger("metrics")


def evaluate_model(
    model: nn.Module,
    loader,
    class_names: List[str],
    device: Optional[torch.device] = None,
    config: Optional[Dict] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run the model on a data loader and compute comprehensive evaluation metrics.

    This is the function you call after training is complete, on the held-out
    test set.  It should be run ONCE, right at the end, to get an unbiased
    estimate of real-world performance.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loader : DataLoader
        Test (or validation) data loader.
    class_names : list of str
        Human-readable class labels.
    device : torch.device, optional
        Where to run inference.
    config : dict, optional
    save_dir : str, optional
        If given, all reports and plots are saved here.

    Returns
    -------
    dict
        {
          "accuracy":   float,
          "macro_f1":   float,
          "macro_precision": float,
          "macro_recall":    float,
          "top5_accuracy":   float,
          "per_class":       dict,
          "confusion_matrix": np.ndarray,
        }
    """
    if config is None:
        config = load_config()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds   = []
    all_labels  = []
    all_probs   = []

    logger.info("Running evaluation on test set ...")

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Testing", unit="batch"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)   # shape: (N, num_classes)

    # ── Core metrics ─────────────────────────────────────────────────────────
    accuracy   = float(np.mean(all_preds == all_labels)) * 100
    macro_f1   = float(f1_score(all_labels, all_preds, average="macro")) * 100
    macro_prec = float(precision_score(all_labels, all_preds, average="macro", zero_division=0)) * 100
    macro_rec  = float(recall_score(all_labels, all_preds, average="macro", zero_division=0)) * 100

    # Top-5 accuracy: correct if true label is in the top-5 softmax scores
    top5_acc = float(
        top_k_accuracy_score(all_labels, all_probs, k=5, labels=list(range(len(class_names))))
    ) * 100

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    # ── Per-class report ──────────────────────────────────────────────────────
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "accuracy":         accuracy,
        "macro_f1":         macro_f1,
        "macro_precision":  macro_prec,
        "macro_recall":     macro_rec,
        "top5_accuracy":    top5_acc,
        "per_class":        report_dict,
        "confusion_matrix": cm,
        "predictions":      all_preds,
        "true_labels":      all_labels,
        "probabilities":    all_probs,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 55)
    logger.info(f"  Top-1 Accuracy : {accuracy:.2f}%")
    logger.info(f"  Top-5 Accuracy : {top5_acc:.2f}%")
    logger.info(f"  Macro F1-Score : {macro_f1:.2f}%")
    logger.info(f"  Macro Precision: {macro_prec:.2f}%")
    logger.info(f"  Macro Recall   : {macro_rec:.2f}%")
    logger.info("=" * 55)

    # ── Save artefacts ────────────────────────────────────────────────────────
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary (easy to read in CI, dashboards, etc.)
        summary = {
            "accuracy":        accuracy,
            "top5_accuracy":   top5_acc,
            "macro_f1":        macro_f1,
            "macro_precision": macro_prec,
            "macro_recall":    macro_rec,
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Classification report as CSV
        df = classification_report_df(all_labels, all_preds, class_names)
        df.to_csv(save_dir / "classification_report.csv", index=True)

        # Confusion matrix plot
        plot_confusion_matrix(
            cm, class_names,
            save_path=str(save_dir / "confusion_matrix.png"),
        )
        logger.info(f"Evaluation artefacts saved to: {save_dir}")

    return results


def classification_report_df(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Return a per-class precision / recall / F1 table as a Pandas DataFrame.

    Parameters
    ----------
    true_labels : np.ndarray
    predictions : np.ndarray
    class_names : list of str

    Returns
    -------
    pd.DataFrame
        Columns: precision, recall, f1-score, support
        Index:   class names
    """
    report = classification_report(
        true_labels, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    # Keep only the per-class rows (drop macro avg, weighted avg, accuracy)
    df = df.loc[class_names]
    df["support"] = df["support"].astype(int)
    return df.round(4)


def compute_topk_accuracy(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute top-K accuracy from probability arrays.

    Parameters
    ----------
    probabilities : np.ndarray, shape (N, num_classes)
    true_labels   : np.ndarray, shape (N,)
    k             : int

    Returns
    -------
    float
        Top-K accuracy as a percentage.
    """
    # For each sample, get the indices of the top-K predicted classes
    topk_preds = np.argsort(probabilities, axis=1)[:, -k:]  # (N, k) — largest k
    correct = np.any(topk_preds == true_labels[:, np.newaxis], axis=1)
    return float(correct.mean()) * 100
