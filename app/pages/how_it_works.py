"""
app/pages/how_it_works.py — Educational Explainer Page
=======================================================

This page teaches the user how the AI system works, in plain English.
Good ML apps don't just give outputs — they build trust by being transparent
about the methodology, limitations, and underlying technology.
"""

import streamlit as st


def render_how_it_works_page():
    st.markdown("## 📚 How PlantMD Works")
    st.markdown(
        "An honest, plain-English explanation of the AI pipeline — "
        "no PhD required."
    )

    # ── Step-by-step pipeline ─────────────────────────────────────────────────
    st.markdown("### The Pipeline at a Glance")

    steps = [
        ("1", "📸", "Image Upload",
         "You upload a photo of a leaf.  The app accepts JPG, PNG, or WebP images."),
        ("2", "🔄", "Pre-processing",
         "The image is resized to 380×380 pixels and normalised using the same statistics "
         "(mean and standard deviation) that were used during training.  "
         "This step is crucial — without normalisation, the model receives values on a "
         "completely different scale than what it learned on."),
        ("3", "🧠", "Feature Extraction",
         "The normalised image passes through EfficientNetV2-S — a convolutional neural "
         "network with 22 million parameters that was pre-trained on 1.2 million ImageNet "
         "images.  It extracts a 1,280-dimensional feature vector that captures textures, "
         "shapes, colours, and lesion patterns."),
        ("4", "🎯", "Classification",
         "The feature vector passes through a custom classification head: "
         "a 2-layer neural network that maps 1,280 features to 38 output scores (logits).  "
         "We apply softmax to convert scores to probabilities."),
        ("5", "📊", "Results",
         "The top-5 predictions are displayed with confidence percentages.  "
         "Treatment advice is attached to each diagnosis."),
        ("6", "🔥", "Grad-CAM Explanation (optional)",
         "Grad-CAM computes which pixels in the image contributed most to the prediction "
         "and overlays them as a heatmap — so you can see exactly where the AI was looking."),
    ]

    for step_num, icon, title, desc in steps:
        with st.container():
            cols = st.columns([0.5, 9.5])
            with cols[0]:
                st.markdown(
                    f"<div style='background:#1b4332; color:white; border-radius:50%; "
                    f"width:36px; height:36px; display:flex; align-items:center; "
                    f"justify-content:center; font-weight:bold; font-size:1rem; "
                    f"margin-top:4px'>{step_num}</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(f"#### {icon} {title}")
                st.markdown(desc)
        st.markdown("")

    st.divider()

    # ── Transfer learning explainer ───────────────────────────────────────────
    st.markdown("### 🔄 What Is Transfer Learning?")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
**The problem with training from scratch**

Suppose you want to teach a child to recognise a cat.  You wouldn't start
from zero — you'd build on the fact that they already know what eyes, fur,
and four-legged shapes look like.

The same principle applies to neural networks.

Training EfficientNetV2 from random weights on 87,000 plant images would
take **weeks on a high-end GPU** and might still not reach good accuracy,
because 87,000 images is relatively small for a deep neural network.

**The transfer learning solution**

Google trained EfficientNetV2 on **1.28 million** ImageNet images and the
model learned to recognise edges, textures, curves, object parts — universal
visual concepts that apply to almost any recognition task.

We borrow those pre-trained weights and add a small classification head for
our 38 plant disease classes.  Training then takes hours instead of weeks,
and final accuracy is significantly higher.
        """)

    with col2:
        st.markdown("""
**Two-stage fine-tuning**

We train in two stages to avoid destroying the pre-trained features:

**Stage 1 — Head-only training (5 epochs)**
The backbone (EfficientNetV2) is frozen.  Only the new 38-class head
is trained.  The high learning rate is safe because we are not touching
the pre-trained weights.

**Stage 2 — Full fine-tuning (remaining epochs)**
The entire network is unfrozen and trained end-to-end with a lower
learning rate (10x smaller).  The backbone slowly adjusts its features
to be more leaf-specific while preserving the general vision knowledge.

This approach consistently outperforms training the full network from
scratch, especially on medium-sized datasets like this one (~87k images).
        """)

    st.divider()

    # ── Grad-CAM explainer ────────────────────────────────────────────────────
    st.markdown("### 🔥 What Is Grad-CAM?")
    st.markdown("""
**Grad-CAM** stands for **Gradient-weighted Class Activation Mapping**.

Neural networks were long criticised as "black boxes" — they give you an
answer but not a reason.  Grad-CAM is a technique that peeks inside the
box by asking: "which spatial regions of the input image caused the
network to activate strongly for the predicted class?"

**The technical idea (simplified)**

1. Run the image through the model and get a prediction.
2. Compute the gradient of the predicted class score with respect to the
   feature maps in the last convolutional layer.
3. Average the gradients across the spatial dimensions to get one importance
   weight per feature-map channel.
4. Take a weighted combination of the feature maps → a single heatmap.
5. Apply ReLU (keep only positive contributions) and resize to input size.
6. Overlay as a colour map on the original image.

**Why it matters for plant disease detection**

If the model says "Tomato Early Blight" and the Grad-CAM heatmap lights
up the brown bull's-eye lesions on the leaf — you can trust the diagnosis.
If the heatmap lights up the background soil instead, something is wrong
and the prediction should be treated with caution.
    """)

    st.divider()

    # ── Data augmentation ─────────────────────────────────────────────────────
    st.markdown("### 🎲 Data Augmentation")
    st.markdown("""
During training, every image is randomly transformed before being shown
to the model.  These transforms preserve the disease label (a flipped leaf
is still a diseased leaf) but change the pixel patterns, effectively
multiplying the dataset.

| Transform | What it does | Why |
|-----------|-------------|-----|
| RandomResizedCrop | Crop a random portion, then resize | Teaches scale invariance |
| HorizontalFlip | Mirror left-right | Leaves look the same either way |
| VerticalFlip | Mirror top-bottom | Same reasoning |
| RandomAffine | Small rotation and translation | Real photos aren't perfectly aligned |
| ColorJitter | Vary brightness, contrast, saturation, hue | Different cameras, lighting conditions |
| RandomErasing | Black out a random rectangle | Simulates partial occlusion (other leaves, shadows) |

Without augmentation, the model would overfit — memorise the training images
rather than learning to generalise.
    """)

    st.divider()

    # ── Metrics explainer ─────────────────────────────────────────────────────
    st.markdown("### 📐 Understanding the Evaluation Metrics")
    st.markdown("""
| Metric | Plain-English Meaning |
|--------|-----------------------|
| **Accuracy** | What fraction of all test images were classified correctly? |
| **Precision** | When the model says "rust", what fraction actually is rust? |
| **Recall** | Of all actual rust images, what fraction did the model catch? |
| **F1-Score** | The harmonic mean of precision and recall — balances both concerns |
| **Top-5 Accuracy** | Was the correct label anywhere in the model's top-5 guesses? |
| **Confusion Matrix** | A grid showing every type of error the model makes |

**Which metric matters most for plant disease detection?**

Recall is arguably the most important metric here.  A false negative
(missing a disease) is far more costly than a false positive (unnecessary
treatment), because:
- Missed disease spreads and destroys the crop.
- Over-treatment wastes money but doesn't lose the harvest.

However, extremely low precision (too many false alarms) leads farmers to
distrust the system and stop using it.  F1-score balances these trade-offs.
    """)
