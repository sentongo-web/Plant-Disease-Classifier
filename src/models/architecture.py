"""
src/models/architecture.py — Neural Network Architecture
=========================================================

This is the heart of the project — the model that learns to tell a healthy
tomato leaf from one ravaged by early blight.

The approach: Transfer Learning with EfficientNetV2
----------------------------------------------------
Training a convolutional neural network from scratch to recognise 38 plant
disease classes would require millions of images and days of GPU compute.
Instead, we use TRANSFER LEARNING — we start with a model that was already
trained on 1.28 million ImageNet photographs and already knows how to detect
edges, textures, shapes, and colour gradients.

We then replace the final classification layer (which originally predicted
1,000 ImageNet categories) with our own layer that predicts 38 plant disease
categories.  Finally we fine-tune the whole network on our leaf images.

Why EfficientNetV2-S?
---------------------
EfficientNet is a family of CNNs that was designed by Google Brain in 2019
to maximise accuracy per computation.  The key insight was to scale width,
depth, and resolution together (compound scaling) rather than separately.

EfficientNetV2 (2021) improves on the original by replacing early
depthwise convolutions with Fused-MBConv blocks, which train faster.
The "S" (small) variant hits ~84 % ImageNet top-1 accuracy with only
~22 million parameters — a great accuracy/size trade-off for a web app.

Model architecture summary
--------------------------
  Input image: (batch, 3, 380, 380)
                      ↓
  EfficientNetV2-S backbone  [pre-trained on ImageNet, ~22 M params]
  → feature vector of size 1280
                      ↓
  Global Average Pooling     [averages spatial dimensions → (batch, 1280)]
                      ↓
  Dropout(p=0.3)             [regularisation]
                      ↓
  Linear(1280 → 512)         [hidden layer — learns disease-specific features]
                      ↓
  BatchNorm + ReLU            [stabilise activations]
                      ↓
  Dropout(p=0.2)             [more regularisation]
                      ↓
  Linear(512 → 38)           [output logits, one per disease class]
                      ↓
  Output: raw logits (batch, 38)

Note: during inference we apply softmax to convert logits to probabilities.
During training we use CrossEntropyLoss, which includes softmax internally,
so we never apply softmax to training outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm
from torchinfo import summary

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("architecture")


class PlantDiseaseModel(nn.Module):
    """
    Custom classification head on top of an EfficientNetV2-S backbone.

    The class inherits from nn.Module — PyTorch's base class for all neural
    networks.  Every nn.Module must implement:
      - __init__: define all layers.
      - forward: describe how data flows through those layers.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  38 for this dataset.
    backbone_name : str
        timm model identifier, e.g. "tf_efficientnetv2_s".
    pretrained : bool
        Load ImageNet pre-trained weights.
    dropout_rate : float
        Dropout probability for the first dropout layer.
    hidden_dim : int
        Size of the intermediate fully-connected layer.
    """

    def __init__(
        self,
        num_classes: int = 38,
        backbone_name: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.num_classes  = num_classes
        self.backbone_name = backbone_name

        # ── Backbone ─────────────────────────────────────────────────────────
        # timm.create_model loads the architecture + pre-trained weights.
        # num_classes=0 means "remove the original classification head and
        # give me the raw feature vector instead".
        # global_pool="" disables timm's built-in pooling so we can use our own.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # strip the head
            global_pool="avg",      # keep global average pooling
        )
        backbone_out_features = self.backbone.num_features

        # Gradient checkpointing trades ~25 % training speed for ~40 % less
        # GPU memory.  Essential when fine-tuning the full backbone on a 15 GB
        # T4 at any reasonable batch size.  timm exposes this via one call.
        self.backbone.set_grad_checkpointing(enable=True)

        logger.info(
            f"Backbone: {backbone_name} | "
            f"Pretrained: {pretrained} | "
            f"Feature dim: {backbone_out_features} | "
            f"Grad checkpointing: ON"
        )

        # ── Custom Classification Head ────────────────────────────────────────
        # This head is randomly initialised — it knows nothing yet.
        # Stage 1 of training (frozen backbone) trains ONLY this head.
        # Stage 2 (unfrozen) fine-tunes the whole network together.
        self.classifier = nn.Sequential(
            # First dropout — applied to the backbone feature vector.
            # During training, 30 % of features are randomly zeroed each
            # forward pass, which forces the head not to over-rely on any
            # single feature.
            nn.Dropout(p=dropout_rate),

            # The hidden linear layer.  1280 → 512.
            nn.Linear(backbone_out_features, hidden_dim),

            # BatchNorm normalises the activations to have zero mean and
            # unit variance, which makes training much more stable.
            nn.BatchNorm1d(hidden_dim),

            # ReLU activation — clips all negative values to zero.
            # This introduces non-linearity, allowing the network to learn
            # complex decision boundaries.
            nn.ReLU(inplace=True),

            # Second dropout — lighter regularisation before the final layer.
            nn.Dropout(p=0.2),

            # Final projection to class logits.
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialise the linear weights with Kaiming (He) normal init.
        # This keeps gradient magnitudes stable early in training.
        self._initialise_head()

    def _initialise_head(self) -> None:
        """Apply Kaiming normal weight initialisation to the classifier head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: images in, class logits out.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 3, H, W)
            Batch of normalised images.

        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Raw (un-normalised) class scores (logits).
            Apply softmax to convert to probabilities.
        """
        # Step 1: pass images through the backbone feature extractor.
        features = self.backbone(x)        # → (batch, 1280)

        # Step 2: pass features through our custom classification head.
        logits = self.classifier(features) # → (batch, 38)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax probabilities for a batch of images.

        Use this for inference, NOT for computing training loss.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 3, H, W)

        Returns
        -------
        torch.Tensor, shape (batch, num_classes), values in [0, 1], sum to 1
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.

        When a parameter is frozen (requires_grad=False) the optimiser skips
        it — its weights do not change.  We do this in Stage 1 of training to
        protect the pre-trained ImageNet features while the new head learns.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only the classification head will be trained.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters for end-to-end fine-tuning.

        Called at the start of Stage 2.  The learning rate should be
        much lower at this point to avoid destroying the pre-trained features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full end-to-end fine-tuning enabled.")

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.

        Knowing your parameter count is important: too few and the model
        cannot learn; too many and it will overfit and be slow to deploy.

        Returns
        -------
        dict
            {"trainable": N, "frozen": M, "total": N+M}
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen":    total - trainable,
            "total":     total,
        }

    def get_target_layer(self) -> nn.Module:
        """
        Return the last convolutional block — used for Grad-CAM.

        Grad-CAM needs a specific layer to hook into.  The last conv block
        produces the richest spatial feature maps before global pooling
        collapses them.

        Returns
        -------
        nn.Module
        """
        # For EfficientNetV2 in timm, the last feature block is at:
        return self.backbone.blocks[-1]


def build_model(config: Optional[Dict] = None) -> PlantDiseaseModel:
    """
    Convenience factory function: read config and return a ready model.

    Parameters
    ----------
    config : dict, optional
        Project config.  Loaded automatically if None.

    Returns
    -------
    PlantDiseaseModel
    """
    if config is None:
        config = load_config()

    cfg_model = config["model"]

    model = PlantDiseaseModel(
        num_classes=config["dataset"]["num_classes"],
        backbone_name=cfg_model["backbone"],
        pretrained=cfg_model["pretrained"],
        dropout_rate=cfg_model["dropout_rate"],
        hidden_dim=cfg_model["hidden_dim"],
    )

    params = model.count_parameters()
    logger.info(
        f"Model built — "
        f"Total params: {params['total']:,} | "
        f"Trainable: {params['trainable']:,}"
    )

    return model


def print_model_summary(model: PlantDiseaseModel, image_size: int = 380) -> None:
    """
    Print a Keras-style layer-by-layer summary of the model.

    Parameters
    ----------
    model : PlantDiseaseModel
    image_size : int
        The spatial dimension of the input image.
    """
    summary(
        model,
        input_size=(1, 3, image_size, image_size),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3,
        verbose=1,
    )
