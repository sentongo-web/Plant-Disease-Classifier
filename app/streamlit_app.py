"""
app/streamlit_app.py — Main Streamlit Application
==================================================

This is the entry point for the web application.  Run it with:

  streamlit run app/streamlit_app.py

Streamlit is a Python library that turns regular Python scripts into
interactive web applications.  Every time a user interacts with a widget
(a button, a file uploader, a slider), Streamlit re-runs the script from
top to bottom and updates the display.

This file sets up:
  - The global page config (title, icon, layout).
  - Navigation between pages.
  - Model loading with caching so it only happens once.
  - The home page / landing page UI.

The other pages live in app/pages/ and are rendered based on which
navigation option the user selects in the sidebar.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
from PIL import Image

# ── Page configuration — must be the FIRST Streamlit call ────────────────────
st.set_page_config(
    page_title="PlantMD — Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Sentoz/Plant-Disease-Classifier",
        "About": "Built by Paul Sentongo · EfficientNetV2 · PyTorch · MLflow",
    },
)

from src.utils.config import load_config


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background and font */
.main { background-color: #f8fdf8; }
.stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 60%, #40916c 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }

/* Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-left: 4px solid #40916c;
    margin: 8px 0;
}

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
    border-radius: 16px;
    padding: 24px;
    border: 2px solid #52b788;
    margin: 12px 0;
}

/* Alert boxes */
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 14px;
    border-radius: 8px;
}

.success-box {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 14px;
    border-radius: 8px;
}

.danger-box {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 14px;
    border-radius: 8px;
}

/* Title gradient */
.hero-title {
    background: linear-gradient(135deg, #1b4332, #40916c, #74c69d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
}

.subtitle {
    color: #495057;
    font-size: 1.2rem;
    font-weight: 400;
    margin-top: 8px;
}

/* Feature cards on homepage */
.feature-card {
    background: white;
    border-radius: 14px;
    padding: 28px 22px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    border-top: 4px solid #52b788;
    height: 100%;
    transition: transform 0.2s;
}

/* Confidence bar colours */
.conf-high   { color: #27ae60; font-weight: bold; }
.conf-medium { color: #f39c12; font-weight: bold; }
.conf-low    { color: #e74c3c; font-weight: bold; }

/* Footer */
.footer {
    text-align: center;
    color: #6c757d;
    font-size: 0.85rem;
    padding: 20px 0;
    border-top: 1px solid #dee2e6;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading with caching ────────────────────────────────────────────────
# @st.cache_resource runs this function exactly once, stores the model object
# in memory, and returns the same object on every subsequent Streamlit rerun —
# even across multiple browser sessions on HF Spaces.
#
# Model resolution order:
#   1. Local file at models/best_model.pth  (present after local training)
#   2. Download from Hugging Face Hub       (present on HF Spaces deployment)
#   3. Return None with an error message    (demo mode — no model loaded)
#
# On HF Spaces the local file will not exist because we never commit .pth files
# to git.  The Hub download path kicks in automatically.
@st.cache_resource(show_spinner="Loading AI model... (downloading on first run, cached after)")
def load_model_cached(model_path: str, config: dict):
    """
    Load the trained model, falling back to HF Hub download if needed.

    Returns
    -------
    tuple
        (model, device, error_message)
        If loading succeeds, error_message is None.
        If loading fails, model and device are None.
    """
    from src.utils.model_hub import (
        download_model_from_hub,
        model_is_available_locally,
        MODEL_REPO_ID,
        MODEL_FILENAME,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Step 1: check if the model already exists locally (post-training or cached)
    resolved_path = model_path
    if not model_is_available_locally(model_path):
        # Step 2: try downloading from HF Hub
        try:
            local_dir = str(PROJECT_ROOT / "models")
            resolved_path = download_model_from_hub(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=local_dir,
            )
        except Exception as hub_error:
            return None, None, (
                f"Model not found locally and Hub download failed.\n\n"
                f"To fix this:\n"
                f"  1. Train the model:  python train.py\n"
                f"  2. Deploy the model: python deploy.py\n\n"
                f"Hub error: {hub_error}"
            )

    try:
        from predict import load_model
        model = load_model(resolved_path, config, device)
        return model, device, None
    except Exception as e:
        return None, None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(config: dict) -> str:
    """Render the navigation sidebar.  Returns the selected page name."""
    with st.sidebar:
        st.markdown("## 🌿 PlantMD")
        st.markdown("*AI-Powered Plant Disease Diagnostics*")
        st.divider()

        pages = {
            "🏠  Home":             "Home",
            "🔬  Diagnose a Leaf":   "Diagnose",
            "📊  Model Performance": "Performance",
            "📚  How It Works":      "How It Works",
            "👨‍💻  About":             "About",
        }

        selection = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        page = pages[selection]

        st.divider()
        model_path = config["inference"]["model_path"]
        model_exists = Path(PROJECT_ROOT / model_path).exists()

        if model_exists:
            st.success("✅  Model loaded")
        else:
            st.warning(
                "⚠️ No trained model found.\n\n"
                "Run `python train.py` first to train the model."
            )

        st.markdown("---")
        device_name = "GPU 🚀" if torch.cuda.is_available() else "CPU 🖥️"
        st.caption(f"Device: {device_name}")
        st.caption("Model: EfficientNetV2-S")
        st.caption("Classes: 38 plant diseases")

    return page


# ── Home Page ─────────────────────────────────────────────────────────────────
def render_home():
    """Render the landing / home page."""

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<h1 class="hero-title">PlantMD</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="subtitle">AI-Powered Plant Disease Diagnostics<br>'
            'Upload a leaf photo — get an instant diagnosis.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            Every year, **plant diseases destroy up to 40% of global food crops**,
            causing over $220 billion in losses.  Early, accurate detection is the
            difference between saving a harvest and losing an entire season.

            **PlantMD** uses a deep learning model trained on 87,000+ images of
            plant leaves to diagnose 38 different diseases across 14 crop species —
            in under a second.
            """
        )
        st.markdown("")
        if st.button("🔬  Diagnose a Leaf Now", type="primary", use_container_width=False):
            st.session_state["page"] = "Diagnose"
            st.rerun()

    with col2:
        # Stats cards
        for stat in [
            ("87,000+", "Training Images"),
            ("38",      "Disease Classes"),
            ("14",      "Crop Species"),
            ("99%+",    "Validation Accuracy (EfficientNetV2)"),
        ]:
            st.markdown(
                f"""<div class="metric-card">
                    <h2 style="margin:0; color:#1b4332; font-size:2rem">{stat[0]}</h2>
                    <p style="margin:0; color:#6c757d; font-size:0.9rem">{stat[1]}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # Feature highlights
    st.markdown("### What PlantMD Can Do For You")
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🔬", "Instant Diagnosis",     "Upload a photo and get results in under one second, with confidence scores."),
        ("🧠", "Explains Itself",        "Grad-CAM heatmaps show exactly which part of the leaf the AI used to decide."),
        ("💊", "Treatment Advice",       "Every diagnosis comes with specific, actionable treatment recommendations."),
        ("📊", "Transparent Metrics",    "Full model performance report — accuracy, F1, confusion matrix — all visible."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        col.markdown(
            f"""<div class="feature-card">
                <div style="font-size:2.5rem; margin-bottom:12px">{icon}</div>
                <h4 style="color:#1b4332; margin-bottom:8px">{title}</h4>
                <p style="color:#495057; font-size:0.9rem; margin:0">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # Supported plants
    st.markdown("### Supported Crops")
    crops = [
        "🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn", "🍇 Grape",
        "🍊 Orange", "🍑 Peach", "🫑 Bell Pepper", "🥔 Potato",
        "🫐 Raspberry", "🫘 Soybean", "🎃 Squash", "🍓 Strawberry", "🍅 Tomato",
    ]
    crop_cols = st.columns(7)
    for i, crop in enumerate(crops):
        crop_cols[i % 7].markdown(
            f"<div style='text-align:center; padding:8px; background:white; "
            f"border-radius:8px; margin:4px; box-shadow:0 1px 4px rgba(0,0,0,0.08)'>"
            f"{crop}</div>",
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """<div class="footer">
        Built with ❤️ by <strong>Paul Sentongo</strong> &nbsp;|&nbsp;
        PyTorch · EfficientNetV2 · MLflow · Streamlit &nbsp;|&nbsp;
        Dataset: New Plant Diseases Dataset (Kaggle)
        </div>""",
        unsafe_allow_html=True,
    )


# ── Main router ───────────────────────────────────────────────────────────────
def main():
    config = load_config()

    # Load model (cached)
    model_path = str(PROJECT_ROOT / config["inference"]["model_path"])
    model, device, error = load_model_cached(model_path, config)

    # Store in session state so pages can access it
    st.session_state["model"]  = model
    st.session_state["device"] = device
    st.session_state["config"] = config
    st.session_state["model_error"] = error

    page = render_sidebar(config)

    # Route to the appropriate page
    if page == "Home":
        render_home()
    elif page == "Diagnose":
        from app.pages.diagnose import render_diagnose_page
        render_diagnose_page()
    elif page == "Performance":
        from app.pages.performance import render_performance_page
        render_performance_page()
    elif page == "How It Works":
        from app.pages.how_it_works import render_how_it_works_page
        render_how_it_works_page()
    elif page == "About":
        from app.pages.about import render_about_page
        render_about_page()


if __name__ == "__main__":
    main()
