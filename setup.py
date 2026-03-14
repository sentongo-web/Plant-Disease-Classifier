"""
setup.py — Package Installation Configuration

Running `pip install -e .` installs the project as an editable package.
"Editable" means Python uses the source files directly from this directory —
any code change is immediately reflected without re-installing.

This is the standard way to make `from src.xxx import yyy` work from any
directory on your system, not just from the project root.

Usage
-----
  pip install -e .          # install in editable mode
  pip install -e ".[dev]"   # also install development tools
"""

from setuptools import find_packages, setup

setup(
    name="plant-disease-classifier",
    version="1.0.0",
    author="Paul Sentongo",
    description="End-to-end ML pipeline for plant disease classification using EfficientNetV2",
    python_requires=">=3.10",
    packages=find_packages(exclude=["notebooks", "tests", "plants"]),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "timm>=1.0.3",
        "Pillow>=10.3.0",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.4.2",
        "mlflow>=2.13.0",
        "streamlit>=1.35.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.4",
        "rich>=13.7.1",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "plotly>=5.22.0",
        "kagglehub>=0.2.9",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "black>=24.4.2",
            "flake8>=7.0.0",
            "jupyterlab>=4.2.1",
            "ipykernel>=6.29.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "plantmd-train=train:main",
            "plantmd-predict=predict:main",
        ],
    },
)
