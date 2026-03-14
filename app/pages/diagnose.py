"""
app/pages/diagnose.py — The Diagnosis Page
==========================================

This is the core page of the app — where a user uploads a leaf image
and gets a disease diagnosis.

The user experience flow
------------------------
1. User uploads an image (JPG, PNG, WebP) via the file uploader widget.
2. The image is displayed in a preview column.
3. On clicking "Run Diagnosis", the model runs inference and returns
   top-5 predictions with confidence scores.
4. The top prediction is shown prominently with severity, description,
   and treatment advice.
5. An optional Grad-CAM heatmap shows the model's "attention" area.
6. The top-5 ranked predictions are shown in a chart.
"""

from __future__ import annotations

import sys
from pathlib import Path
import io

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from app.utils.inference import run_inference, generate_gradcam, get_disease_info


def render_diagnose_page():
    st.markdown("## 🔬 Diagnose a Plant Leaf")
    st.markdown(
        "Upload a clear, close-up photo of a single leaf.  "
        "The AI will analyse it and tell you what disease (if any) is present."
    )

    # ── Check model availability ──────────────────────────────────────────────
    model  = st.session_state.get("model")
    device = st.session_state.get("device")
    config = st.session_state.get("config")
    error  = st.session_state.get("model_error")

    if model is None:
        st.error(
            "**Model not loaded.**  "
            "Please train the model first by running `python train.py` "
            "from the project root."
        )
        if error:
            st.code(error)

        st.info(
            "**Demo mode:** You can still explore the app — upload an image "
            "to see how the interface looks, but no real predictions will be made."
        )
        _render_demo_mode()
        return

    # ── Upload widget ─────────────────────────────────────────────────────────
    st.markdown("---")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 1. Upload Your Leaf Image")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of the leaf.  Blurry or far-away photos reduce accuracy.",
        )

        show_gradcam = st.checkbox(
            "🧠 Show Grad-CAM explanation (what the AI looked at)",
            value=False,
            help="Generates a heatmap showing which leaf regions influenced the prediction most.",
        )

        run_btn = st.button("🔬 Run Diagnosis", type="primary", use_container_width=True)

    with right:
        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
            st.markdown("### Preview")
            st.image(pil_image, use_column_width=True, caption="Uploaded leaf image")
        else:
            st.markdown("### Preview")
            st.markdown(
                """<div style="border: 2px dashed #ced4da; border-radius: 12px;
                padding: 60px 20px; text-align: center; color: #6c757d; background: #f8f9fa;">
                <div style="font-size: 3rem">🍃</div>
                <p>Your uploaded image will appear here</p></div>""",
                unsafe_allow_html=True,
            )

    # ── Inference ─────────────────────────────────────────────────────────────
    if run_btn and uploaded is not None:
        with st.spinner("🤖 Analysing leaf..."):
            pil_image = Image.open(uploaded).convert("RGB")
            predictions, image_tensor = run_inference(
                pil_image=pil_image,
                model=model,
                config=config,
                device=device,
                top_k=5,
            )

        _render_results(predictions, pil_image, image_tensor, model, show_gradcam)

    elif run_btn and uploaded is None:
        st.warning("Please upload a leaf image first.")

    # ── Tips ──────────────────────────────────────────────────────────────────
    with st.expander("📸 Tips for best results"):
        st.markdown("""
**For the most accurate diagnosis:**

- Take a **close-up** photo of a single leaf, filling most of the frame.
- Use **natural daylight** — avoid flash which creates hot-spots.
- Make sure the image is **in focus** and not blurry.
- Show the **top surface** of the leaf; include any spots, lesions, or discolouration.
- If possible, photograph **recently developed symptoms** — very old lesions can look similar across diseases.
- Avoid photographing in direct bright sunlight which can wash out colours.
        """)


def _render_results(predictions, pil_image, image_tensor, model, show_gradcam):
    """Render the full results panel after inference."""
    top = predictions[0]
    confidence = top["confidence"]

    st.markdown("---")
    st.markdown("## 🧪 Diagnosis Results")

    # ── Top prediction summary ────────────────────────────────────────────────
    r_left, r_mid, r_right = st.columns([2, 2, 1], gap="large")

    with r_left:
        severity_colours = {
            "None":     ("#27ae60", "#d4edda"),
            "Low":      ("#f39c12", "#fff3cd"),
            "Medium":   ("#e67e22", "#fde8d8"),
            "High":     ("#e74c3c", "#f8d7da"),
            "Critical": ("#c0392b", "#f5c6cb"),
            "Unknown":  ("#6c757d", "#e2e3e5"),
        }
        border_c, bg_c = severity_colours.get(top["severity"], ("#6c757d", "#e2e3e5"))

        st.markdown(
            f"""<div style="background:{bg_c}; border-left: 5px solid {border_c};
            border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <p style="font-size:0.85rem; color:#6c757d; margin:0">DIAGNOSIS</p>
            <h2 style="color:{border_c}; margin:4px 0 8px">{top['emoji']}  {top['class']}</h2>
            <p style="font-size:0.9rem; color:#495057; margin:0">
            Severity: <strong style="color:{border_c}">{top['severity']}</strong>
            &nbsp;|&nbsp; Confidence: <strong>{confidence:.1f}%</strong>
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

        if confidence < 50:
            st.warning(
                f"⚠️ **Low confidence ({confidence:.1f}%).**  "
                "The model is uncertain.  Try a clearer, closer image."
            )
        elif confidence < 75:
            st.info(
                f"ℹ️ Moderate confidence ({confidence:.1f}%).  "
                "The diagnosis is likely correct but not definitive."
            )
        else:
            st.success(f"✅ High confidence ({confidence:.1f}%).  The diagnosis is reliable.")

    with r_mid:
        # Confidence bar chart for top-5 predictions
        names   = [p["class"].split("—")[-1].strip()[:35] for p in predictions]
        confs   = [p["confidence"] for p in predictions]
        colors_bar = [p["color"] for p in predictions]

        fig = go.Figure(go.Bar(
            x=confs[::-1], y=names[::-1],
            orientation="h",
            marker_color=colors_bar[::-1],
            text=[f"{c:.1f}%" for c in confs[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Top-5 Predictions",
            xaxis=dict(range=[0, 105], title="Confidence (%)"),
            height=260, margin=dict(l=10, r=40, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r_right:
        st.image(pil_image, caption="Analysed leaf", use_column_width=True)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if show_gradcam:
        with st.spinner("Generating Grad-CAM explanation..."):
            fig_cam = generate_gradcam(model, image_tensor, pil_image, top["class_idx"])
        if fig_cam is not None:
            st.markdown("### 🧠 What the AI Was Looking At")
            st.markdown(
                "The heatmap below highlights the leaf regions that most influenced "
                "the prediction.  Red/warm areas = high importance.  "
                "If the heatmap points to actual lesions/spots on the leaf, "
                "that is a sign the model is reasoning correctly."
            )
            st.pyplot(fig_cam)
        else:
            st.info("Grad-CAM is only available when a GPU/CUDA device is present.")

    # ── Disease info ──────────────────────────────────────────────────────────
    st.markdown("---")
    info_cols = st.columns(2)

    with info_cols[0]:
        st.markdown("### 📖 About This Disease")
        st.markdown(top["description"])

    with info_cols[1]:
        st.markdown("### 💊 Treatment & Management")
        st.markdown(top["treatment"])

    # ── Other possibilities ───────────────────────────────────────────────────
    with st.expander("🔍 Other possibilities considered"):
        for pred in predictions[1:]:
            st.markdown(
                f"**{pred['rank']}. {pred['class']}** — {pred['confidence']:.1f}%  \n"
                f"_{pred['description'][:150]}..._"
            )
            st.markdown("---")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        """<div class="warning-box">
        <strong>⚠️ Medical / Agricultural Disclaimer</strong><br>
        This tool is for educational and screening purposes only.  For critical
        crop management decisions, always consult a qualified agronomist or
        plant pathologist.  AI diagnoses can be wrong, especially on unusual
        lighting, image quality, or disease stages.
        </div>""",
        unsafe_allow_html=True,
    )


def _render_demo_mode():
    """Show a demo UI when no model is loaded."""
    st.markdown("---")
    st.markdown("### Demo: Interface Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.file_uploader("Upload leaf image (demo only)", type=["jpg", "png"])
        st.button("Run Diagnosis (disabled — model not loaded)", disabled=True)
    with c2:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Tomato_je.jpg/320px-Tomato_je.jpg",
            caption="Example: Tomato leaf",
            use_column_width=True,
        )
