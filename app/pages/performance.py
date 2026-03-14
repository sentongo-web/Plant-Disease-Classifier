"""
app/pages/performance.py — Model Performance Dashboard
=======================================================

This page renders a full model evaluation report inside the Streamlit app.
It reads the metrics.json and classification_report.csv files that were
saved by evaluate_model() at the end of training.

Why expose model performance in the app?
----------------------------------------
Transparency and trust.  Any production ML system should come with a clear
performance report so users know its limitations.  A model that is 99%
accurate overall might be only 85% accurate on a rare class — knowing this
changes how you interpret its outputs.

This page shows:
  - Summary metrics (accuracy, F1, top-5 accuracy).
  - Per-class precision, recall, F1 sorted by performance.
  - An interactive confusion matrix heatmap.
  - Training history curves (loss and accuracy over epochs).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def render_performance_page():
    st.markdown("## 📊 Model Performance Report")
    st.markdown(
        "These metrics measure how well the trained model generalises to "
        "**new images it has never seen before** (the held-out test set)."
    )

    config = st.session_state.get("config") or load_config()
    reports_dir = Path(config["paths"]["reports"])
    figures_dir = Path(config["paths"]["figures"])

    # ── Summary metrics ───────────────────────────────────────────────────────
    metrics_file = reports_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        st.markdown("### Overall Metrics")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col, (key, label, icon) in zip(
            [mc1, mc2, mc3, mc4, mc5],
            [
                ("accuracy",        "Top-1 Accuracy",  "🎯"),
                ("top5_accuracy",   "Top-5 Accuracy",  "🏆"),
                ("macro_f1",        "Macro F1",         "⚖️"),
                ("macro_precision", "Macro Precision",  "🔍"),
                ("macro_recall",    "Macro Recall",     "📡"),
            ],
        ):
            value = metrics.get(key, 0)
            col.metric(f"{icon} {label}", f"{value:.2f}%")

        st.markdown("""
**What do these numbers mean?**

- **Top-1 Accuracy**: For what fraction of test images was the model's single best guess correct?
- **Top-5 Accuracy**: For what fraction of test images was the correct label in the model's top-5 guesses?
- **Macro F1**: The average F1-score across all 38 classes, treating each class equally regardless of size.
- **Macro Precision**: Of all the times the model predicted a specific class, how often was it right?
- **Macro Recall**: Of all the actual instances of each class, what fraction did the model correctly identify?
        """)

    else:
        st.info(
            "📋 No evaluation report found.  "
            "Run `python train.py` to train the model and generate reports."
        )
        return

    st.divider()

    # ── Per-class table ───────────────────────────────────────────────────────
    report_csv = reports_dir / "classification_report.csv"
    if report_csv.exists():
        df = pd.read_csv(report_csv, index_col=0)

        st.markdown("### Per-Class Metrics")
        st.markdown(
            "Sorted by F1-Score.  Classes at the top are those the model handles best.  "
            "Red rows indicate classes where performance is weakest — "
            "these are often visually similar diseases."
        )

        tab1, tab2 = st.tabs(["📈 Bar Chart", "📋 Table"])

        with tab1:
            df_sorted = df.sort_values("f1-score", ascending=True)
            fig = px.bar(
                df_sorted,
                x="f1-score",
                y=df_sorted.index,
                orientation="h",
                color="f1-score",
                color_continuous_scale="RdYlGn",
                range_color=[0.7, 1.0],
                title="Per-Class F1-Score",
                labels={"f1-score": "F1-Score", "y": ""},
                height=900,
            )
            fig.update_layout(
                yaxis=dict(tickfont=dict(size=10)),
                coloraxis_showscale=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            def highlight_low(val):
                if isinstance(val, float) and val < 0.85:
                    return "background-color: #ffcccc"
                return ""
            styled = (
                df.style
                .applymap(highlight_low, subset=["precision", "recall", "f1-score"])
                .format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"})
            )
            st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_img = figures_dir / "confusion_matrix.png"
    if cm_img.exists():
        st.markdown("### Confusion Matrix")
        st.markdown(
            "Each row is a true class; each column is the predicted class.  "
            "The diagonal (top-left to bottom-right) shows correct predictions.  "
            "Off-diagonal cells show which classes get confused with each other."
        )
        st.image(str(cm_img), use_column_width=True)

    st.divider()

    # ── Training curves ───────────────────────────────────────────────────────
    curves_img = figures_dir / "training_curves.png"
    if curves_img.exists():
        st.markdown("### Training History")
        st.markdown(
            "These curves show how the model's loss and accuracy evolved over training.  "
            "The gap between train and val curves is a measure of overfitting: "
            "a small gap means the model generalises well."
        )
        st.image(str(curves_img), use_column_width=True)

    st.divider()

    # ── Class distribution ────────────────────────────────────────────────────
    dist_img = figures_dir / "class_distribution.png"
    if dist_img.exists():
        st.markdown("### Training Data Distribution")
        st.markdown(
            "Shows how many images exist per class.  "
            "A relatively balanced dataset (similar counts across classes) "
            "allows the model to learn all classes equally well."
        )
        st.image(str(dist_img), use_column_width=True)
