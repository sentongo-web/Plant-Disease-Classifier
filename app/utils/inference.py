"""
app/utils/inference.py — Inference Utilities for the Streamlit App
===================================================================

This module is the inference backend that the Streamlit pages call.
It handles:
  - Loading the model once and caching it (so it doesn't reload on every user
    interaction — Streamlit re-runs the whole script on each widget change).
  - Pre-processing uploaded images.
  - Running predictions.
  - Generating Grad-CAM explanations.

Streamlit's @st.cache_resource decorator
-----------------------------------------
Streamlit reruns your entire Python script from top to bottom every time the
user interacts with a widget (types text, clicks a button, uploads a file).

Without caching, the 22-million-parameter model would be loaded from disk
and moved to memory on EVERY interaction — taking several seconds each time.

@st.cache_resource tells Streamlit: "run this function once, store the result
in a global cache, and for all future calls with the same arguments, just
return the cached object."  This makes the app feel instantaneous after the
first load.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_transforms, get_class_names
from src.models.architecture import PlantDiseaseModel
from src.utils.config import load_config
from src.utils.visualization import GradCAM


# ── Disease information database ─────────────────────────────────────────────
# Real-world plant disease management info attached to each class.
# This transforms a "% confidence" result into actionable advice — exactly
# what a farmer would need.
DISEASE_INFO: Dict[str, Dict] = {
    "Apple — Apple scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis.  It appears as olive-green to brown spots on leaves and fruit.",
        "treatment": "Apply fungicides (captan, myclobutanil) at bud break.  Remove infected leaves.  Improve air circulation by pruning.",
        "severity": "Medium",
        "color": "#f39c12",
    },
    "Apple — Black rot": {
        "description": "Black rot is caused by Botryosphaeria obtusa.  It causes circular lesions with purple edges on leaves and fruit mummification.",
        "treatment": "Remove mummified fruit and cankers.  Apply copper-based fungicides.  Avoid wounding the bark.",
        "severity": "High",
        "color": "#e74c3c",
    },
    "Apple — Cedar apple rust": {
        "description": "Caused by Gymnosporangium juniperi-virginianae, requiring both apple and cedar/juniper trees to complete its life cycle.",
        "treatment": "Apply fungicides (myclobutanil, trifloxystrobin) before and after infection periods.  Remove nearby cedar galls if possible.",
        "severity": "Medium",
        "color": "#f39c12",
    },
    "Apple — healthy": {
        "description": "The apple leaf shows no signs of disease.  The plant appears healthy.",
        "treatment": "Continue regular monitoring.  Maintain good soil nutrition and irrigation.",
        "severity": "None",
        "color": "#27ae60",
    },
    "Tomato — Early blight": {
        "description": "Early blight is caused by Alternaria solani.  Dark brown bull's-eye lesions appear on older lower leaves first.",
        "treatment": "Remove infected lower leaves.  Apply chlorothalonil or mancozeb.  Water at the base, avoid wetting foliage.",
        "severity": "Medium",
        "color": "#f39c12",
    },
    "Tomato — Late blight": {
        "description": "Late blight (Phytophthora infestans) is the same pathogen behind the Irish Potato Famine.  Water-soaked lesions with white mould on undersides.",
        "treatment": "Remove and destroy infected plants immediately.  Apply copper fungicides preventively.  Ensure good drainage.",
        "severity": "Critical",
        "color": "#c0392b",
    },
    "Tomato — healthy": {
        "description": "The tomato plant is healthy with no visible disease symptoms.",
        "treatment": "Maintain regular watering and fertilisation.  Scout regularly for early signs of pests.",
        "severity": "None",
        "color": "#27ae60",
    },
}

# Default info for classes not in the database above
DEFAULT_INFO = {
    "description": "Detailed disease information is being compiled for this class.",
    "treatment": "Consult a local agricultural extension officer for specific treatment recommendations.",
    "severity": "Unknown",
    "color": "#95a5a6",
}

SEVERITY_EMOJI = {
    "None": "✅",
    "Low": "🟡",
    "Medium": "🟠",
    "High": "🔴",
    "Critical": "🚨",
    "Unknown": "❓",
}


def get_disease_info(class_name: str) -> Dict:
    """
    Return treatment and description info for a given class name.

    Parameters
    ----------
    class_name : str
        Display-friendly class name (from get_class_names(raw=False)).

    Returns
    -------
    dict
    """
    return DISEASE_INFO.get(class_name, DEFAULT_INFO)


@torch.no_grad()
def run_inference(
    pil_image: Image.Image,
    model: PlantDiseaseModel,
    config: Dict,
    device: torch.device,
    top_k: int = 5,
) -> List[Dict]:
    """
    Run the model on a PIL image and return top-K predictions.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Image uploaded by the user.
    model : PlantDiseaseModel
    config : dict
    device : torch.device
    top_k : int

    Returns
    -------
    list of dict
        [{"class": "...", "confidence": 94.3, "class_idx": 29}, ...]
    """
    cfg_img   = config["image"]
    transform = get_transforms(
        image_size=cfg_img["size"],
        mean=cfg_img["mean"],
        std=cfg_img["std"],
        augment=False,
    )

    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(top_k)
    class_names = get_class_names(raw=False)

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        name = class_names[idx]
        info = get_disease_info(name)
        results.append({
            "class":       name,
            "class_idx":   idx,
            "confidence":  round(prob * 100, 2),
            "description": info["description"],
            "treatment":   info["treatment"],
            "severity":    info["severity"],
            "color":       info["color"],
            "emoji":       SEVERITY_EMOJI.get(info["severity"], "❓"),
        })

    return results, tensor


def generate_gradcam(
    model: PlantDiseaseModel,
    image_tensor: torch.Tensor,
    pil_image: Image.Image,
    class_idx: int,
) -> Optional[object]:
    """
    Generate a Grad-CAM explanation figure.

    Parameters
    ----------
    model : PlantDiseaseModel
    image_tensor : torch.Tensor, shape (1, C, H, W)
    pil_image : PIL.Image.Image
    class_idx : int

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    try:
        cam = GradCAM(model, target_layer=model.get_target_layer())
        heatmap = cam(image_tensor, class_idx=class_idx)
        fig = cam.overlay(pil_image, heatmap)
        cam.remove_hooks()
        return fig
    except Exception:
        return None
