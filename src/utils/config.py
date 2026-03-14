"""
src/utils/config.py — Configuration Loader
===========================================

This module has one job: read configs/config.yaml and hand back a plain
Python dictionary that every other module can import and use.

Why centralise config loading here?
------------------------------------
If every module opened the YAML file independently, changing the config file
path would require editing a dozen imports.  By funnelling everything through
this one function, there is exactly one place to update.

It also ensures the path resolution is always relative to the project root,
no matter which directory you run the script from — a common gotcha.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """
    Walk up from this file's location until we find the project root.

    The project root is identified by the presence of configs/config.yaml.
    This approach works whether you run scripts from the root directory,
    a sub-directory, or an IDE that sets a different working directory.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # If we cannot find it, fall back to the current working directory.
    return Path.cwd()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load the YAML configuration file and return it as a nested dictionary.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to a config file.  If None, the function automatically
        finds configs/config.yaml in the project root.

    Returns
    -------
    dict
        Nested dictionary mirroring the YAML structure, e.g.:
        config["training"]["learning_rate"] → 0.0001

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found at the resolved path.
    """
    if config_path is None:
        root = get_project_root()
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            "Make sure you are running from the project root or pass an "
            "explicit config_path argument."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert all path strings in the `paths` section into absolute Path
    # objects so callers never need to worry about relative-path arithmetic.
    root = get_project_root()
    if "paths" in config:
        for key, value in config["paths"].items():
            config["paths"][key] = str(root / value)

    return config
