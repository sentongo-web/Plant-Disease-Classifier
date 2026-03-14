"""
deploy.py — Deploy PlantMD to Hugging Face
==========================================

This script does two things in sequence:

  1. Uploads the trained model weights (best_model.pth) to a Hugging Face
     model repository so the Streamlit app on HF Spaces can download them
     at startup.

  2. Pushes the application code (this entire repository) to a Hugging Face
     Space repository so HF Spaces can serve the Streamlit app.

Why two repositories?
---------------------
HF Hub distinguishes between:

  - "Model" repositories  → store model weights, configs, tokenisers.
  - "Space" repositories  → store application code (Streamlit, Gradio, etc.).

The code lives in the Space.  The weights live in the Model repo.
At runtime, the Streamlit app downloads the weights from the Model repo into
the Space's local filesystem (cached after first download).

This is the standard pattern used by thousands of HF deployments.

Prerequisites
-------------
  1. Create a free account at https://huggingface.co
  2. Generate a write-access token at https://huggingface.co/settings/tokens
  3. Set the token:  export HF_TOKEN=hf_xxxxxxxxxx
     Or add it to your .env file as  HF_TOKEN=hf_xxxxxxxxxx
  4. Train the model:  python train.py
  5. Run this script:  python deploy.py

After running this script
-------------------------
  1. Your model weights are at:
       https://huggingface.co/YOUR_USERNAME/plantmd-disease-classifier
  2. Your app is live at:
       https://huggingface.co/spaces/YOUR_USERNAME/plantmd

  HF Spaces will automatically build and serve the app when the code is pushed.
  The first build takes 3–5 minutes.  After that, the app is live 24/7 for free.

Setting the HF_TOKEN secret in the Space
-----------------------------------------
If your model repository is private, you must also add the HF_TOKEN as a
secret in the Space settings so the app can download the model at runtime:

  HF Space → Settings → Repository Secrets → Add secret: HF_TOKEN
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.model_hub import (
    upload_model_to_hub,
    HF_USERNAME,
    MODEL_REPO_ID,
    MODEL_FILENAME,
)

logger = get_logger("deploy", log_file="logs/deploy.log")

# ── Configuration — update these to your own HF username ─────────────────────
SPACE_REPO_ID = f"{HF_USERNAME}/plantmd"   # the HF Space repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy PlantMD to Hugging Face Hub + Spaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-only", action="store_true",
        help="Only upload the model weights, skip the Space code push."
    )
    parser.add_argument(
        "--space-only", action="store_true",
        help="Only push the Space code, skip the model upload."
    )
    parser.add_argument(
        "--model-path", type=str,
        default=str(PROJECT_ROOT / "models" / "best_model.pth"),
        help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--commit-message", type=str,
        default="Deploy PlantMD — plant disease classifier",
        help="Commit message for the Space push."
    )
    return parser.parse_args()


def check_prerequisites(model_path: str) -> None:
    """Verify everything needed for deployment is in place."""
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN environment variable is not set.\n"
            "Get a write token at: https://huggingface.co/settings/tokens\n"
            "Then run:  export HF_TOKEN=hf_xxxxxxxxxx"
        )
        sys.exit(1)

    if not Path(model_path).exists():
        logger.error(
            f"Model file not found at: {model_path}\n"
            "Train the model first:  python train.py"
        )
        sys.exit(1)

    logger.info("Prerequisites check passed.")


def push_code_to_space(commit_message: str, token: str) -> None:
    """
    Push the repository code to a Hugging Face Space using the HF Hub API.

    Under the hood, HF Spaces are git repositories hosted on HF.
    We use huggingface_hub's Repository class to handle the git operations
    cleanly, including Large File Storage (LFS) for any binary files.

    Parameters
    ----------
    commit_message : str
    token : str
        HF write-access token.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    logger.info(f"Creating / verifying Space: {SPACE_REPO_ID}")
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
        token=token,
        private=False,
    )

    # Files to upload — everything except data, venv, cache, etc.
    # We use upload_folder which respects .gitignore patterns.
    logger.info(f"Uploading code to Space: {SPACE_REPO_ID} ...")

    ignore_patterns = [
        "plants/*",
        "*.venv/*",
        "data/raw/*",
        "data/processed/*",
        "models/best_model.pth",
        "models/checkpoints/*",
        "logs/*",
        "mlflow_runs/*",
        "reports/figures/*",
        "__pycache__/*",
        "*.pyc",
        "*.egg-info/*",
        ".git/*",
        ".env",
    ]

    api.upload_folder(
        repo_id=SPACE_REPO_ID,
        folder_path=str(PROJECT_ROOT),
        repo_type="space",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )

    space_url = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
    logger.info(f"Code pushed to Space:  {space_url}")
    logger.info("HF Spaces will now build and deploy the app automatically.")
    logger.info("This typically takes 3–5 minutes on first deployment.")


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    logger.info("=" * 60)
    logger.info("  PLANTMD — HUGGING FACE DEPLOYMENT")
    logger.info("=" * 60)

    # ── Step 1: Validate ──────────────────────────────────────────────────────
    check_prerequisites(args.model_path)

    # ── Step 2: Upload model weights ──────────────────────────────────────────
    if not args.space_only:
        logger.info(f"\nStep 1: Uploading model to Hub → {MODEL_REPO_ID}")
        model_url = upload_model_to_hub(
            local_model_path=args.model_path,
            repo_id=MODEL_REPO_ID,
            token=token,
            commit_message=args.commit_message,
        )
        logger.info(f"Model repository: https://huggingface.co/{MODEL_REPO_ID}")

    # ── Step 3: Push app code to Space ────────────────────────────────────────
    if not args.model_only:
        logger.info(f"\nStep 2: Pushing app code to Space → {SPACE_REPO_ID}")
        push_code_to_space(commit_message=args.commit_message, token=token)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  DEPLOYMENT COMPLETE")
    if not args.space_only:
        logger.info(f"  Model:  https://huggingface.co/{MODEL_REPO_ID}")
    if not args.model_only:
        logger.info(f"  App:    https://huggingface.co/spaces/{SPACE_REPO_ID}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Open your Space URL above — the build takes 3–5 min.")
    logger.info("  2. If the model repo is private, add HF_TOKEN as a Space secret:")
    logger.info(f"     https://huggingface.co/spaces/{SPACE_REPO_ID}/settings")
    logger.info("  3. Share the Space URL with the world!")


if __name__ == "__main__":
    main()
