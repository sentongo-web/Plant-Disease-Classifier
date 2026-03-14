"""
src/data/download.py — Kaggle Dataset Download
===============================================

This module handles downloading the "New Plant Diseases Dataset" from Kaggle
using the `kagglehub` library.  It also takes care of a few housekeeping
details that save you from head-scratching errors later:

  - It checks if the data is already present and skips the download.
  - It saves the download path so every other module knows where to look.
  - It prints a clear summary of what was downloaded.

What is kagglehub?
------------------
`kagglehub` is Kaggle's official lightweight Python client.  Behind the scenes
it authenticates using your Kaggle API key (stored in ~/.kaggle/kaggle.json or
as environment variables KAGGLE_USERNAME and KAGGLE_KEY), downloads the dataset
to a local cache directory, and returns the path to those files.

You only need to install it once and put your API credentials in place.  After
the first download, running this script again is instant because it detects the
cached files.

About the Dataset
-----------------
"New Plant Diseases Dataset" by vipoooool on Kaggle contains:
  - ~87,000 RGB images of plant leaves, 256×256 pixels.
  - 38 classes:  26 diseases + 12 healthy variants across 14 plant species.
  - Pre-split into train (~70,295 images) and validation (~17,572 images).
  - Already balanced — each class has a similar number of images.

The plants covered are:
  Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper,
  Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

Each folder name follows the pattern "PlantName___DiseaseName", e.g.:
  Tomato___Early_blight
  Apple___healthy
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("download")


def download_dataset(config: Optional[Dict] = None) -> str:
    """
    Download the New Plant Diseases Dataset from Kaggle.

    The function uses `kagglehub` to pull the dataset, then saves a small
    JSON metadata file alongside the data so other scripts can locate it.

    Parameters
    ----------
    config : dict, optional
        Project configuration dict (loaded from config.yaml).
        If None, the config is loaded automatically.

    Returns
    -------
    str
        Absolute path to the folder that contains the downloaded images.

    Raises
    ------
    ImportError
        If `kagglehub` is not installed.
    RuntimeError
        If the Kaggle credentials cannot be found.
    """
    if config is None:
        config = load_config()

    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have the data — avoid a 3 GB re-download
    meta_file = raw_dir / "dataset_info.json"
    if meta_file.exists():
        with open(meta_file) as f:
            info = json.load(f)
        cached_path = info.get("dataset_path", "")
        if cached_path and Path(cached_path).exists():
            logger.info(f"Dataset already downloaded at: {cached_path}")
            logger.info("Skipping download.  Delete data/raw/dataset_info.json to force re-download.")
            return cached_path

    # ── Authenticate and download ────────────────────────────────────────────
    logger.info("Starting Kaggle dataset download ...")
    logger.info("Make sure your Kaggle API credentials are set:")
    logger.info("  Option A: Place kaggle.json in ~/.kaggle/")
    logger.info("  Option B: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")

    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed.  Run: pip install kagglehub"
        )

    handle = config["dataset"]["kaggle_handle"]
    logger.info(f"Downloading: {handle}")

    dataset_path = kagglehub.dataset_download(handle)
    logger.info(f"Download complete.  Files at: {dataset_path}")

    # ── Save metadata ────────────────────────────────────────────────────────
    info = {
        "kaggle_handle": handle,
        "dataset_path": dataset_path,
        "train_dir": str(Path(dataset_path) / config["dataset"]["train_dir"]),
        "valid_dir": str(Path(dataset_path) / config["dataset"]["valid_dir"]),
    }
    with open(meta_file, "w") as f:
        json.dump(info, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────
    _print_dataset_summary(Path(dataset_path) / config["dataset"]["train_dir"])

    return dataset_path


def get_dataset_paths(config: Optional[Dict] = None) -> Dict[str, str]:
    """
    Return the absolute paths to the train and validation image directories.

    Call download_dataset() first if the data hasn't been downloaded yet.

    Parameters
    ----------
    config : dict, optional

    Returns
    -------
    dict
        {"train": "/path/to/train", "valid": "/path/to/valid"}
    """
    if config is None:
        config = load_config()

    meta_file = Path(config["paths"]["data_raw"]) / "dataset_info.json"

    if not meta_file.exists():
        raise FileNotFoundError(
            "Dataset not found.  Run: python -m src.data.download  (or call download_dataset())"
        )

    with open(meta_file) as f:
        info = json.load(f)

    return {
        "train": info["train_dir"],
        "valid": info["valid_dir"],
    }


def _print_dataset_summary(train_dir: Path) -> None:
    """
    Walk the training directory and print per-class image counts.

    Parameters
    ----------
    train_dir : Path
        Path to the training folder whose sub-folders are class names.
    """
    if not train_dir.exists():
        logger.warning(f"Train directory not found at: {train_dir}")
        return

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    total = 0

    logger.info("=" * 55)
    logger.info(f"{'CLASS':<45} {'IMAGES':>8}")
    logger.info("=" * 55)

    for class_dir in class_dirs:
        n = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) +
                list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG")))
        logger.info(f"{class_dir.name:<45} {n:>8,}")
        total += n

    logger.info("=" * 55)
    logger.info(f"{'TOTAL':<45} {total:>8,}")
    logger.info(f"Number of classes: {len(class_dirs)}")


if __name__ == "__main__":
    # Allow running directly:  python -m src.data.download
    path = download_dataset()
    print(f"\nDataset ready at: {path}")
