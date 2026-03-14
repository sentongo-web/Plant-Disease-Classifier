"""
predict.py — Single-Image Inference Entry Point
================================================

Use this script to run the trained model on a single image from the command
line.  It loads the saved model, preprocesses the image, runs inference,
and prints the top-K predictions with confidence scores.

Usage
-----
  python predict.py --image path/to/leaf.jpg
  python predict.py --image path/to/leaf.jpg --top_k 3
  python predict.py --image path/to/leaf.jpg --model models/best_model.pth
  python predict.py --image path/to/leaf.jpg --gradcam  # show Grad-CAM heatmap

What inference means
--------------------
Inference (or "prediction" in ML, not to be confused with statistical
inference) is the process of running a trained model on new, unseen data
to get a prediction.  Unlike training, there are no gradients, no weight
updates — the model is frozen and we simply ask it: "what do you think this is?"

The inference pipeline
----------------------
  1. Load the saved model weights from disk.
  2. Resize and normalise the image using the same transform as training.
  3. Add a batch dimension (the model always expects a batch, even of size 1).
  4. Run the forward pass with torch.no_grad().
  5. Apply softmax to convert raw scores to probabilities.
  6. Return the top-K class names and their confidence percentages.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_transforms, get_class_names, CLASS_NAMES_DISPLAY
from src.models.architecture import PlantDiseaseModel
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import GradCAM

logger = get_logger("predict")


def load_model(
    model_path: str,
    config: dict,
    device: torch.device,
) -> PlantDiseaseModel:
    """
    Load a trained PlantDiseaseModel from a .pth checkpoint file.

    Parameters
    ----------
    model_path : str
        Path to the saved checkpoint (e.g., models/best_model.pth).
    config : dict
        Project configuration.
    device : torch.device
        Where to load the model.

    Returns
    -------
    PlantDiseaseModel
        Model with loaded weights, in eval mode.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Recreate the model with the same architecture used during training
    model = PlantDiseaseModel(
        num_classes=checkpoint.get("num_classes", config["dataset"]["num_classes"]),
        backbone_name=checkpoint.get("backbone_name", config["model"]["backbone"]),
        pretrained=False,  # We are loading our own weights, not ImageNet
        dropout_rate=config["model"]["dropout_rate"],
        hidden_dim=config["model"]["hidden_dim"],
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from: {model_path}")
    return model


def predict_image(
    image_path: str,
    model: PlantDiseaseModel,
    config: dict,
    device: torch.device,
    top_k: int = 5,
) -> list:
    """
    Predict the plant disease class for a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPG, PNG, etc.).
    model : PlantDiseaseModel
    config : dict
    device : torch.device
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    list of dict
        [{"class": "Tomato — Early blight", "confidence": 94.3}, ...]
    """
    cfg_img = config["image"]
    transform = get_transforms(
        image_size=cfg_img["size"],
        mean=cfg_img["mean"],
        std=cfg_img["std"],
        augment=False,
    )

    # Load and preprocess the image
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image at '{image_path}': {e}")

    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    # .unsqueeze(0) adds a batch dimension: (3, H, W) → (1, 3, H, W)

    with torch.no_grad():
        logits = model(image_tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)  # → (38,)

    # Get top-K predictions
    top_probs, top_indices = probs.topk(top_k)
    class_names = get_class_names(raw=False)

    predictions = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        predictions.append({
            "rank":       len(predictions) + 1,
            "class":      class_names[idx],
            "class_idx":  idx,
            "confidence": round(prob * 100, 2),
        })

    return predictions, pil_image, image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict plant disease from a leaf image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",   required=True, type=str, help="Path to the leaf image")
    parser.add_argument("--model",   type=str, default=None,  help="Path to trained model .pth file")
    parser.add_argument("--top_k",   type=int, default=5,     help="Number of top predictions to show")
    parser.add_argument("--gradcam", action="store_true",      help="Generate and save a Grad-CAM heatmap")
    args = parser.parse_args()

    config = load_config()

    if args.model:
        model_path = args.model
    else:
        model_path = config["inference"]["model_path"]

    if not Path(model_path).exists():
        logger.error(
            f"Model not found at: {model_path}\n"
            "Train the model first by running: python train.py"
        )
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    model = load_model(model_path, config, device)
    predictions, pil_image, image_tensor = predict_image(
        image_path=args.image,
        model=model,
        config=config,
        device=device,
        top_k=args.top_k,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Image: {args.image}")
    print("=" * 55)
    print(f"  {'RANK':<6} {'CLASS':<40} {'CONFIDENCE':>10}")
    print("-" * 55)
    for pred in predictions:
        marker = " ← TOP" if pred["rank"] == 1 else ""
        print(f"  {pred['rank']:<6} {pred['class']:<40} {pred['confidence']:>8.2f}%{marker}")
    print("=" * 55)

    top_pred = predictions[0]
    confidence = top_pred["confidence"]
    threshold  = config["inference"]["confidence_threshold"] * 100

    if confidence < threshold:
        print(f"\n  ⚠  WARNING: Low confidence ({confidence:.1f}% < {threshold:.0f}%).")
        print("     The model is uncertain.  Try a clearer image of the leaf.")
    else:
        print(f"\n  Diagnosis: {top_pred['class']}")
        print(f"  Confidence: {confidence:.1f}%")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if args.gradcam:
        try:
            cam = GradCAM(model, target_layer=model.get_target_layer())
            heatmap = cam(image_tensor, class_idx=top_pred["class_idx"])
            fig = cam.overlay(pil_image, heatmap)
            output_path = Path(args.image).stem + "_gradcam.png"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            print(f"\n  Grad-CAM heatmap saved to: {output_path}")
            cam.remove_hooks()
        except Exception as e:
            print(f"\n  Grad-CAM failed: {e}")


if __name__ == "__main__":
    main()
