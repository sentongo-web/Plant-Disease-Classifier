"""
src/utils/model_hub.py — Hugging Face Hub Model Management
===========================================================

This module handles uploading a trained model to the Hugging Face Hub and
downloading it back at application startup.

Why not just commit the .pth file to git?
------------------------------------------
A trained EfficientNetV2-S checkpoint is around 80–200 MB.  Git was designed
for source code, not binary blobs.  Committing large binaries:
  - Makes the repository slow to clone for everyone.
  - Bloats git history permanently (even if you delete the file later).
  - Hits GitHub's 100 MB hard file limit and HF's default git limit.

The professional solution is to separate code from weights:
  - Code lives in the git repository.
  - Weights live in a model registry (here, the Hugging Face Hub).
  - At deployment time, the app downloads the weights on first startup.

How it works
------------
1. After training locally, you run `python deploy.py` which calls
   upload_model_to_hub() to push best_model.pth to a HF model repository.

2. When the Streamlit app starts on HF Spaces (or locally), load_model_cached()
   calls download_model_from_hub() which:
     a. Checks if the file is already in the local cache (~/.cache/huggingface/).
     b. If not, downloads it from the Hub.
     c. Returns the local path to the downloaded file.

   The download only happens once per deployment — after that it is cached.

3. The model is then loaded from the cached path using PyTorch's torch.load().

Authentication
--------------
Uploading to the Hub requires a Hugging Face write-access token.
Set it as an environment variable:   HF_TOKEN=hf_xxxxxxxxxxxx

Downloading public repositories requires no token.
Set the Space secret HF_TOKEN only if your model repo is private.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("model_hub")

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these two constants to match your Hugging Face username and the
# name you want to give the model repository on the Hub.
HF_USERNAME  = "Sentoz"          # your HF username
MODEL_REPO_ID = f"{HF_USERNAME}/plantmd-disease-classifier"
MODEL_FILENAME = "best_model.pth"


def upload_model_to_hub(
    local_model_path: str,
    repo_id: str = MODEL_REPO_ID,
    token: Optional[str] = None,
    commit_message: str = "Upload trained PlantMD model",
) -> str:
    """
    Upload a local model checkpoint to a Hugging Face model repository.

    If the repository does not exist, this function creates it automatically.
    If it already exists, the file is updated with a new commit.

    Parameters
    ----------
    local_model_path : str
        Path to the local .pth file (e.g. "models/best_model.pth").
    repo_id : str
        The Hub repository in the format "username/repo-name".
    token : str, optional
        HF write token.  Falls back to the HF_TOKEN environment variable.
    commit_message : str
        The git commit message shown in the Hub repository history.

    Returns
    -------
    str
        The URL of the uploaded file on the Hub.

    Example
    -------
    >>> url = upload_model_to_hub("models/best_model.pth")
    >>> print(url)
    https://huggingface.co/Sentoz/plantmd-disease-classifier/...
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Hugging Face token not found.\n"
            "Set it with:  export HF_TOKEN=hf_xxxxxxxxxx\n"
            "or add HF_TOKEN=hf_xxxxxxxxxx to your .env file.\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )

    local_model_path = Path(local_model_path)
    if not local_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {local_model_path}\n"
            "Train the model first with: python train.py"
        )

    api = HfApi(token=token)

    # Create the repository if it doesn't exist yet.
    # exist_ok=True means no error if it already exists.
    logger.info(f"Creating / verifying repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=token,
        private=False,    # set to True if you want a private model repo
    )

    logger.info(f"Uploading {local_model_path} → {repo_id}/{MODEL_FILENAME} ...")
    url = api.upload_file(
        path_or_fileobj=str(local_model_path),
        path_in_repo=MODEL_FILENAME,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    logger.info(f"Upload complete.  File URL: {url}")
    return url


def download_model_from_hub(
    repo_id: str = MODEL_REPO_ID,
    filename: str = MODEL_FILENAME,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a model checkpoint from a Hugging Face repository.

    The file is stored in the Hugging Face local cache directory
    (~/.cache/huggingface/hub/) the first time it is downloaded.
    Subsequent calls return the cached path instantly without re-downloading.

    Parameters
    ----------
    repo_id : str
        The Hub repository identifier (e.g. "Sentoz/plantmd-disease-classifier").
    filename : str
        Name of the file to download from the repository.
    local_dir : str, optional
        If provided, the file is also copied to this local directory.
        Useful when you want the model at a predictable path (models/).
    token : str, optional
        HF token for private repositories.  Falls back to HF_TOKEN env var.

    Returns
    -------
    str
        Absolute path to the downloaded (or cached) model file.

    Raises
    ------
    RuntimeError
        If the download fails (e.g. repository not found, network error).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    token = token or os.getenv("HF_TOKEN") or None  # None = public access

    logger.info(f"Fetching model from Hub: {repo_id}/{filename}")
    logger.info("(Using local cache if already downloaded — this is instant on subsequent runs)")

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=local_dir,    # also copy to local_dir if specified
        )
        logger.info(f"Model ready at: {cached_path}")
        return cached_path

    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from {repo_id}/{filename}.\n"
            f"Error: {e}\n\n"
            "Possible causes:\n"
            "  1. The model has not been uploaded yet.  Run: python deploy.py\n"
            "  2. The repository name is wrong.  Check HF_USERNAME in model_hub.py\n"
            "  3. Network issue on HF Spaces.  Try restarting the Space.\n"
        )


def model_is_available_locally(local_path: str) -> bool:
    """
    Check if a local model file exists and is a valid (non-empty) file.

    Parameters
    ----------
    local_path : str

    Returns
    -------
    bool
    """
    p = Path(local_path)
    return p.exists() and p.stat().st_size > 1_000_000  # at least 1 MB
