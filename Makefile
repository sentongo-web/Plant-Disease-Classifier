# Makefile — Common Command Shortcuts
#
# A Makefile is not just for C/C++ projects.  In Python ML projects it acts
# as a command menu: instead of remembering long commands, you type `make train`
# or `make app` and the right thing happens.
#
# Usage:  make <target>
#
# On Windows, install `make` via: winget install GnuWin32.Make
# Or use Git Bash / WSL which include make.

.PHONY: help env install download eda train evaluate app clean lint

# ── Default target: print help ────────────────────────────────────────────────
help:
	@echo ""
	@echo "PlantMD — Available Commands"
	@echo "============================="
	@echo "  make env        Create the plants virtual environment"
	@echo "  make install    Install Python dependencies"
	@echo "  make download   Download the Kaggle dataset"
	@echo "  make eda        Open the EDA notebook in JupyterLab"
	@echo "  make train      Train the model (full pipeline)"
	@echo "  make evaluate   Evaluate the best saved model on the test set"
	@echo "  make app        Launch the Streamlit web application"
	@echo "  make mlflow     Launch the MLflow experiment tracking UI"
	@echo "  make lint       Check code style with flake8"
	@echo "  make format     Auto-format code with black"
	@echo "  make clean      Remove generated files (logs, reports, checkpoints)"
	@echo ""

# ── Environment setup ─────────────────────────────────────────────────────────
env:
	python -m venv plants
	@echo "Virtual environment 'plants' created."
	@echo "Activate it with:  plants\\Scripts\\activate  (Windows)"
	@echo "                or source plants/bin/activate  (Linux/Mac)"

install:
	pip install -r requirements.txt
	pip install -e .
	@echo "Dependencies installed."

# ── Data ──────────────────────────────────────────────────────────────────────
download:
	python -m src.data.download

eda:
	jupyter lab notebooks/01_exploratory_data_analysis.ipynb

# ── Training & Evaluation ─────────────────────────────────────────────────────
train:
	python train.py

train-quick:
	python train.py --epochs 5 --batch_size 32

evaluate:
	python -c "from src.evaluation.metrics import evaluate_model; print('Run train.py first to generate a model, then this will evaluate it.')"

# ── Applications ──────────────────────────────────────────────────────────────
app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --backend-store-uri file://mlflow_runs

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	flake8 src/ app/ train.py predict.py --max-line-length=100

format:
	black src/ app/ train.py predict.py --line-length=100

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf logs/*.log
	rm -rf reports/figures/*.png
	rm -rf reports/*.json reports/*.csv
	rm -rf models/checkpoints/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "Cleaned generated files."
