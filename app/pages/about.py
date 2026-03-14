"""
app/pages/about.py — About Page
================================
"""

import streamlit as st


def render_about_page():
    st.markdown("## 👨‍💻 About This Project")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
### Paul Sentongo

I'm a data scientist and machine learning engineer with a passion for building
AI systems that solve real problems — not just toy demos.

This project is a full end-to-end demonstration of how I approach ML work:
starting from raw data, through modelling and experiment tracking, to a polished
production-ready application.

**What this project demonstrates:**

- Downloading and working with large-scale image datasets (87k images, 3 GB).
- Professional data pipeline design with transforms, augmentation, and DataLoaders.
- Transfer learning with EfficientNetV2 from the `timm` library.
- Two-stage fine-tuning strategy for optimal performance.
- Experiment tracking with MLflow — every run logged, every metric captured.
- Model explainability with Grad-CAM heatmaps.
- Building a production-quality web app with Streamlit.
- Writing code that others can read, learn from, and build on.

**On writing code that teaches**

I believe code comments and documentation are not optional extras — they are
core to good engineering.  A system that only the original author can understand
is a liability.  I write comments that explain *why*, not just *what*, because
six months from now, the person reading this code might be you, or me.
        """)

    with col2:
        st.markdown("""
### Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.3 |
| Backbone | EfficientNetV2-S (timm) |
| Experiment Tracking | MLflow |
| Data Pipeline | torchvision |
| Web App | Streamlit |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Data Science | NumPy, Pandas, Scikit-learn |
| Dataset | Kaggle / kagglehub |

### Dataset Credit

**New Plant Diseases Dataset**
by Samir Bhattarai (vipoooool)
Published on Kaggle

87,000+ leaf images
38 classes
14 plant species
        """)

    st.divider()

    st.markdown("### Project Structure")
    st.code("""
Plant-Disease-Classifier/
├── plants/                  ← Virtual environment
├── configs/
│   └── config.yaml          ← All hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── download.py      ← Kaggle dataset download
│   │   └── dataset.py       ← Transforms, DataLoaders, augmentation
│   ├── models/
│   │   ├── architecture.py  ← EfficientNetV2 + custom head
│   │   └── trainer.py       ← Training loop, early stopping, AMP
│   ├── evaluation/
│   │   └── metrics.py       ← Accuracy, F1, confusion matrix
│   └── utils/
│       ├── config.py        ← YAML config loader
│       ├── logger.py        ← Centralised logging
│       └── visualization.py ← Plots, Grad-CAM
├── app/
│   ├── streamlit_app.py     ← Main Streamlit entry point
│   ├── pages/
│   │   ├── diagnose.py      ← Prediction / diagnosis page
│   │   ├── performance.py   ← Model evaluation dashboard
│   │   ├── how_it_works.py  ← Educational explainer
│   │   └── about.py         ← This page
│   └── utils/
│       └── inference.py     ← Inference + disease info database
├── models/
│   └── best_model.pth       ← Saved after training
├── reports/
│   ├── metrics.json         ← Evaluation summary
│   ├── classification_report.csv
│   └── figures/             ← Training curves, confusion matrix
├── notebooks/               ← EDA and exploration notebooks
├── train.py                 ← Main training script
├── predict.py               ← CLI prediction script
└── requirements.txt
    """, language="")

    st.divider()

    st.markdown("### 📬 Get in Touch")
    st.markdown(
        "If you have questions about the project, found a bug, or just want to "
        "discuss machine learning — feel free to open an issue on the GitHub repository."
    )

    st.markdown(
        """<div class="footer">
        Built by <strong>Paul Sentongo</strong> &nbsp;·&nbsp; 2024 &nbsp;·&nbsp;
        PyTorch · EfficientNetV2 · MLflow · Streamlit
        </div>""",
        unsafe_allow_html=True,
    )
