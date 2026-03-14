"""
train.py — Main Training Entry Point
=====================================

Run this script to kick off a complete end-to-end training run:

  python train.py
  python train.py --config configs/config.yaml
  python train.py --backbone tf_efficientnetv2_m --epochs 30 --lr 2e-4

What happens when you run this script
--------------------------------------
1.  Parse command-line arguments (so you can override any config value).
2.  Load the YAML config file.
3.  Start an MLflow experiment run to record everything.
4.  Log all hyper-parameters to MLflow.
5.  Download (or locate cached) dataset using kagglehub.
6.  Build the PlantDiseaseDataModule (transforms, DataLoaders).
7.  Build the EfficientNetV2 model.
8.  Create the Trainer and call .fit().
9.  Evaluate the best model on the held-out test set.
10. Log final metrics and artefacts to MLflow.
11. Generate and save all plots to reports/figures/.

Usage
-----
  # From the project root, with the plants venv active:
  python train.py

  # Override specific settings without editing the YAML:
  python train.py --epochs 20 --batch_size 64 --lr 0.0005

  # Launch MLflow UI to inspect all runs:
  mlflow ui --backend-store-uri mlflow_runs
"""

import argparse
import os
import sys
from pathlib import Path

# ── Make sure the project root is in the Python path ─────────────────────────
# This allows `from src.xxx import yyy` to work whether you run from the
# project root or from inside a sub-directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import torch

from src.data.dataset import PlantDiseaseDataModule, get_class_names
from src.data.download import download_dataset, get_dataset_paths
from src.evaluation.metrics import evaluate_model
from src.models.architecture import build_model, print_model_summary
from src.models.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import plot_class_distribution

logger = get_logger(
    "train",
    log_file=str(PROJECT_ROOT / "logs" / "train.log"),
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    All arguments are optional — they override values from config.yaml.
    This lets you run quick experiments without editing files:
        python train.py --epochs 5 --lr 1e-3
    """
    parser = argparse.ArgumentParser(
        description="Train the Plant Disease Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",    type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--epochs",    type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size",type=int, default=None, help="Batch size")
    parser.add_argument("--lr",        type=float,default=None, help="Learning rate")
    parser.add_argument("--backbone",  type=str, default=None, help="timm model name")
    parser.add_argument("--no_pretrain", action="store_true",  help="Train from scratch (no ImageNet weights)")
    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download check")
    parser.add_argument("--run_name",  type=str, default=None, help="MLflow run name")
    return parser.parse_args()


def apply_arg_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply any command-line overrides to the loaded config dictionary."""
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.no_pretrain:
        config["model"]["pretrained"] = False
    return config


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    config = apply_arg_overrides(config, args)

    logger.info("=" * 60)
    logger.info("  PLANT DISEASE CLASSIFIER — TRAINING PIPELINE")
    logger.info("  By Paul Sentongo")
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS backend.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found.  Training on CPU (this will be slow).")
        logger.info("Consider using Google Colab (free GPU) for this project.")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_uri = str(PROJECT_ROOT / config["paths"]["mlflow_uri"])
    mlflow.set_tracking_uri(f"file://{mlflow_uri}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_name = args.run_name or (
        config["mlflow"]["run_name_prefix"]
        + config["model"]["backbone"].replace("/", "_")
    )

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log all hyper-parameters flat (MLflow stores them as key-value pairs)
        mlflow.log_params({
            "backbone":         config["model"]["backbone"],
            "pretrained":       config["model"]["pretrained"],
            "dropout_rate":     config["model"]["dropout_rate"],
            "hidden_dim":       config["model"]["hidden_dim"],
            "image_size":       config["image"]["size"],
            "batch_size":       config["training"]["batch_size"],
            "epochs":           config["training"]["epochs"],
            "learning_rate":    config["training"]["learning_rate"],
            "weight_decay":     config["training"]["weight_decay"],
            "scheduler":        config["training"]["scheduler"],
            "freeze_epochs":    config["training"]["freeze_backbone_epochs"],
            "early_stop_pat":   config["training"]["early_stopping_patience"],
            "use_amp":          config["training"]["use_amp"],
            "num_classes":      config["dataset"]["num_classes"],
        })

        # ── Data ─────────────────────────────────────────────────────────────
        if not args.skip_download:
            logger.info("Step 1: Checking / downloading dataset ...")
            dataset_root = download_dataset(config)

        paths = get_dataset_paths(config)
        train_dir = paths["train"]
        valid_dir = paths["valid"]

        logger.info("Step 2: Building DataLoaders ...")
        dm = PlantDiseaseDataModule(config)
        dm.setup(train_dir=train_dir, valid_dir=valid_dir)

        # Class distribution plot
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(root=train_dir)
        class_counts = {
            name: sum(1 for _, lbl in train_ds.samples if lbl == idx)
            for idx, name in enumerate(train_ds.classes)
        }
        plot_class_distribution(
            class_counts,
            save_path=str(PROJECT_ROOT / config["paths"]["figures"] / "class_distribution.png"),
        )

        # ── Model ─────────────────────────────────────────────────────────────
        logger.info("Step 3: Building model ...")
        model = build_model(config)
        print_model_summary(model, image_size=config["image"]["size"])

        params = model.count_parameters()
        mlflow.log_params({
            "total_params":     params["total"],
            "trainable_params": params["trainable"],
        })

        # ── Training ──────────────────────────────────────────────────────────
        logger.info("Step 4: Starting training ...")
        trainer = Trainer(
            model=model,
            train_loader=dm.train_loader,
            val_loader=dm.val_loader,
            config=config,
            device=device,
            mlflow_run=run,
        )
        history = trainer.fit()

        # ── Evaluation ────────────────────────────────────────────────────────
        logger.info("Step 5: Evaluating best model on test set ...")
        class_names = get_class_names(raw=False)
        results = evaluate_model(
            model=model,
            loader=dm.test_loader,
            class_names=class_names,
            device=device,
            config=config,
            save_dir=str(PROJECT_ROOT / config["paths"]["reports"]),
        )

        # Log final metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy":    results["accuracy"],
            "test_top5_acc":    results["top5_accuracy"],
            "test_macro_f1":    results["macro_f1"],
            "test_precision":   results["macro_precision"],
            "test_recall":      results["macro_recall"],
        })

        # Log the best model as an MLflow artefact
        best_model_path = str(PROJECT_ROOT / "models" / "best_model.pth")
        if Path(best_model_path).exists():
            mlflow.log_artifact(best_model_path, artifact_path="model")

        logger.info("=" * 60)
        logger.info("  TRAINING COMPLETE")
        logger.info(f"  Test Accuracy : {results['accuracy']:.2f}%")
        logger.info(f"  Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        logger.info(f"  Macro F1      : {results['macro_f1']:.2f}%")
        logger.info(f"  MLflow run    : {run.info.run_id}")
        logger.info("=" * 60)
        logger.info("To view the MLflow dashboard, run:")
        logger.info(f"  mlflow ui --backend-store-uri file://{mlflow_uri}")
        logger.info("To run the Streamlit app, run:")
        logger.info("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
