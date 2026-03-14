"""
src/data/dataset.py — Data Loading, Augmentation & Preprocessing
=================================================================

This module is the bridge between raw image files on disk and the batched
tensors that the neural network consumes during training.

The three key concepts here are:

  1. Transforms — a pipeline of image operations applied before training.
  2. ImageFolder — PyTorch's built-in class that reads a folder-of-folders.
  3. DataLoader — wraps a dataset and yields shuffled mini-batches.

Why transforms matter
---------------------
A raw leaf photograph is 256×256 pixels with pixel values from 0–255.
The neural network expects:
  - A specific fixed size (380×380 for EfficientNetV2-S).
  - Float values in the range [0, 1] or normalised to ImageNet statistics.
  - Tensors in the shape (channels, height, width) not (height, width, channels).

Augmentation transforms (random flips, rotations, colour jitter) are applied
ONLY to the training set.  The validation set uses a minimal "deterministic"
transform because we want to measure real-world performance, not a lucky flip.

What is ImageFolder?
--------------------
PyTorch's torchvision.datasets.ImageFolder expects your images organised like:

    root/
      class_a/
        img1.jpg
        img2.jpg
      class_b/
        img3.jpg

It automatically assigns integer labels (0, 1, 2, ...) based on alphabetical
folder order and provides a `.class_to_idx` dictionary so you can map back
from integers to human-readable names.

This is exactly the layout the Kaggle dataset uses — one sub-folder per class.

What is a DataLoader?
---------------------
A DataLoader is a worker that:
  - Iterates over the dataset in shuffled random order (for training).
  - Groups individual samples into mini-batches of size `batch_size`.
  - Loads images on background CPU threads (num_workers) while the GPU is busy
    with the previous batch.
  - Applies a `collate_fn` to stack individual (image, label) pairs into
    (batch_images, batch_labels) tensors.

Without DataLoader, you would have to implement all of this yourself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import ImageFolder

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("dataset")


# ── The 38 plant-disease classes in alphabetical order ───────────────────────
# These are the exact folder names in the dataset.  We strip underscores for
# display purposes in the app.
CLASS_NAMES_RAW = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Human-readable names for the UI (spaces, no underscores)
CLASS_NAMES_DISPLAY = [name.replace("___", " — ").replace("_", " ") for name in CLASS_NAMES_RAW]


def get_class_names(raw: bool = False) -> List[str]:
    """
    Return the list of 38 class names.

    Parameters
    ----------
    raw : bool
        If True, returns the raw folder names (with underscores).
        If False, returns the display-friendly version.

    Returns
    -------
    list of str
    """
    return CLASS_NAMES_RAW if raw else CLASS_NAMES_DISPLAY


def get_transforms(
    image_size: int,
    mean: List[float],
    std: List[float],
    augment: bool,
    aug_config: Optional[Dict] = None,
) -> T.Compose:
    """
    Build a torchvision transforms pipeline.

    We use two different pipelines:
      - augment=True  → training pipeline with random transforms
      - augment=False → deterministic pipeline for val/test/inference

    Why normalise with ImageNet statistics?
    ----------------------------------------
    The EfficientNet backbone was pre-trained on ImageNet.  When its weights
    were learned, the input pixels were normalised using ImageNet's mean and
    std.  If we feed in un-normalised pixels, the pre-trained weights expect
    a different scale of inputs and the transfer learning breaks down.

    Parameters
    ----------
    image_size : int
        Target spatial size (both height and width).
    mean, std : list of float
        Normalisation mean and standard deviation per channel.
    augment : bool
        Whether to include random data augmentation transforms.
    aug_config : dict, optional
        Augmentation hyper-parameters from config.yaml.

    Returns
    -------
    torchvision.transforms.Compose
    """
    aug_config = aug_config or {}

    if augment:
        # ── Training pipeline ────────────────────────────────────────────────
        # The order matters:
        # 1. RandomResizedCrop — crop a random portion then resize.
        #    This makes the model invariant to object position and scale.
        # 2. Flips — leaves look the same upside-down or mirrored; the disease
        #    label doesn't change.
        # 3. ColorJitter — simulates different lighting, camera settings.
        # 4. RandomAffine — slight rotation/translation.
        # 5. ToTensor → convert PIL image to float tensor in [0, 1].
        # 6. Normalize → shift to ImageNet statistics.
        # 7. RandomErasing → randomly zero out a patch (simulates occlusion).
        transforms_list = [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=aug_config.get("horizontal_flip_prob", 0.5)),
            T.RandomVerticalFlip(p=aug_config.get("vertical_flip_prob", 0.3)),
            T.RandomAffine(
                degrees=aug_config.get("rotation_degrees", 30),
                translate=(0.1, 0.1),
                shear=10,
            ),
            T.ColorJitter(
                brightness=aug_config.get("color_jitter", {}).get("brightness", 0.2),
                contrast=aug_config.get("color_jitter",  {}).get("contrast",   0.2),
                saturation=aug_config.get("color_jitter",{}).get("saturation", 0.2),
                hue=aug_config.get("color_jitter",        {}).get("hue",       0.1),
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=aug_config.get("random_erasing_prob", 0.1)),
        ]
    else:
        # ── Validation / test / inference pipeline ───────────────────────────
        # No randomness.  We resize to slightly larger than needed, then take
        # the centre crop.  This is the standard "eval" convention.
        transforms_list = [
            T.Resize(int(image_size * 1.143)),   # e.g. 380 → 434
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    return T.Compose(transforms_list)


class PlantDiseaseDataModule:
    """
    Encapsulates all data loading logic in one reusable object.

    A DataModule is a design pattern popularised by PyTorch Lightning.
    It keeps all train/val/test loading logic together so you never copy-paste
    DataLoader boilerplate across different scripts.

    Usage
    -----
    >>> dm = PlantDiseaseDataModule(config)
    >>> dm.setup()
    >>> for images, labels in dm.train_loader:
    ...     # train step
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.train_loader: Optional[DataLoader] = None
        self.val_loader:   Optional[DataLoader] = None
        self.test_loader:  Optional[DataLoader] = None
        self.class_names:  Optional[List[str]] = None
        self.num_classes:  int = config["dataset"]["num_classes"]

    def setup(self, train_dir: str, valid_dir: str) -> None:
        """
        Initialise datasets and DataLoaders from image directories.

        Parameters
        ----------
        train_dir : str
            Path to the folder of training class sub-folders.
        valid_dir : str
            Path to the folder of validation class sub-folders.
        """
        cfg_img = self.config["image"]
        cfg_aug = self.config["augmentation"]
        cfg_trn = self.config["training"]

        img_size = cfg_img["size"]
        mean = cfg_img["mean"]
        std  = cfg_img["std"]

        # ── Build transforms ─────────────────────────────────────────────────
        train_transform = get_transforms(img_size, mean, std, augment=True,  aug_config=cfg_aug)
        eval_transform  = get_transforms(img_size, mean, std, augment=False)

        # ── Load datasets ────────────────────────────────────────────────────
        # ImageFolder reads the directory structure and assigns labels
        # automatically based on alphabetical order of sub-folder names.
        logger.info(f"Loading training images from: {train_dir}")
        train_dataset = ImageFolder(root=train_dir, transform=train_transform)

        logger.info(f"Loading validation images from: {valid_dir}")
        full_val_dataset = ImageFolder(root=valid_dir, transform=eval_transform)

        # ── Split validation into val + test ─────────────────────────────────
        # The dataset comes with a train/val split but no held-out test set.
        # We cut the validation set in half: one half for monitoring training,
        # the other for a final unbiased evaluation after training is done.
        val_size  = int(len(full_val_dataset) * (1 - self.config["dataset"]["test_split"]))
        test_size = len(full_val_dataset) - val_size
        val_dataset, test_dataset = random_split(
            full_val_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(42),  # reproducible split
        )

        self.class_names = train_dataset.classes  # raw folder names
        self.num_classes = len(self.class_names)

        # ── DataLoaders ──────────────────────────────────────────────────────
        # pin_memory=True speeds up CPU→GPU transfers by using page-locked RAM.
        # persistent_workers=True keeps worker processes alive between epochs.
        num_workers = cfg_trn.get("num_workers", 4)
        # On Windows, multiprocessing with DataLoader can crash inside a Jupyter
        # notebook; set num_workers=0 in that case.
        pin = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg_trn["batch_size"],
            shuffle=True,                 # shuffle every epoch for SGD diversity
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            drop_last=True,               # discard the last incomplete batch
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg_trn["batch_size"] * 2,  # can use larger batch for eval
            shuffle=False,                # keep order for reproducible metrics
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=cfg_trn["batch_size"] * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

        logger.info(
            f"Datasets ready — "
            f"Train: {len(train_dataset):,} | "
            f"Val: {len(val_dataset):,} | "
            f"Test: {len(test_dataset):,} | "
            f"Classes: {self.num_classes}"
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the training set.

        When the dataset is imbalanced (some classes have more images than
        others), the model tends to optimise for the majority class.  Passing
        class weights to the loss function compensates by penalising mistakes
        on rare classes more heavily.

        Returns
        -------
        torch.Tensor, shape (num_classes,)
            Normalised inverse-frequency weights.
        """
        if self.train_loader is None:
            raise RuntimeError("Call setup() before get_class_weights().")

        dataset = self.train_loader.dataset
        # Count images per class
        counts = torch.zeros(self.num_classes)
        for _, label in dataset.samples:
            counts[label] += 1

        # Inverse frequency: rare classes get higher weight
        weights = 1.0 / (counts + 1e-8)
        # Normalise so the average weight is 1
        weights = weights / weights.mean()
        return weights
