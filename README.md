---
title: PlantMD Plant Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---

## PlantMD — AI-Powered Plant Disease Classifier

### Built by Paul Sentongo

Every single year, plant diseases destroy somewhere between 20 and 40 percent of the world's food supply. That translates to hundreds of billions of dollars in losses, empty markets, and food insecurity for millions of families, especially in regions where a single crop is the difference between eating and not eating. The tragedy is that many of these losses are preventable. Catch the disease early, treat it correctly, and you save the harvest.

The problem has always been access. Getting a qualified plant pathologist to look at your crop requires time, money, and geography cooperating in your favour. Most smallholder farmers in sub-Saharan Africa, South Asia, and Southeast Asia do not have that luxury. They rely on experience and neighbours, and they are often wrong.

This project is my attempt to address that gap. PlantMD is a deep learning system that can look at a photograph of a leaf and tell you, in under a second, which of 38 plant diseases is present across 14 different crops. It also explains what it found, how confident it is, and what you should do about it.

---

## What This Project Covers

This is not a tutorial that stops at training a model and printing an accuracy number. This is a full, professional-grade machine learning pipeline from raw data to a deployed web application, with every decision explained and every piece of code documented for someone who wants to learn from it.

Here is everything that is built here:

**Data Engineering.** Downloading 3 gigabytes of image data from Kaggle using the official API, organising it into train, validation, and test splits, and building a data loading pipeline with transforms and augmentation that feeds batches to the model efficiently.

**Model Architecture.** Transfer learning with EfficientNetV2-S from the timm library. A custom classification head that maps the backbone's 1280-dimensional feature vector to 38 disease classes. Two-stage fine-tuning: freeze the backbone first, train only the head, then unfreeze everything for end-to-end refinement.

**Training Infrastructure.** A full training engine with mixed-precision training (float16 on GPU), gradient clipping, learning-rate scheduling with cosine annealing, early stopping to prevent overfitting, periodic checkpointing, and logging to MLflow for experiment tracking.

**Evaluation.** Top-1 and Top-5 accuracy, macro-averaged precision, recall, and F1 score, per-class breakdowns, and a full confusion matrix visualisation so you can see exactly which diseases get confused with each other.

**Explainability.** Grad-CAM heatmaps that highlight which regions of the leaf the model used to make its prediction. If the model says "bacterial blight" and the heatmap lights up the lesions on the leaf, that is evidence the model is reasoning correctly. If it lights up the background, something is wrong.

**Web Application.** A multi-page Streamlit app that lets you upload a leaf photo and get an instant diagnosis, with confidence scores, treatment recommendations, and optional Grad-CAM explanation. A full model performance dashboard is also built into the app.

---

## Getting Started

### What You Will Need

Before you begin, make sure you have:

Python 3.10 or higher installed on your machine. You can check by opening a terminal and typing `python --version`.

A Kaggle account. The dataset is hosted on Kaggle and requires a free account to download. If you do not have one, go to kaggle.com and sign up.

Your Kaggle API credentials. Once you have an account, go to your profile settings on Kaggle, scroll to the API section, and click "Create New API Token". This downloads a file called `kaggle.json`. Place that file in the `~/.kaggle/` folder on your system (create the folder if it does not exist).

Optionally, an NVIDIA GPU. Training on CPU will work but will be slow. A modern GPU will complete a full training run in a few hours. If you do not have one locally, Google Colab provides free GPU access.

### Setting Up the Environment

The first thing we do is create an isolated Python environment called `plants`. This is a best practice in Python development — every project gets its own environment with its own set of packages, so projects do not interfere with each other.

Open a terminal in the project root directory and run:

```bash
python -m venv plants
```

This creates a new virtual environment in a folder called `plants`. Now activate it:

On Windows:

```bash
plants\Scripts\activate
```

On macOS / Linux:

```bash
source plants/bin/activate
```

You will see `(plants)` appear in your terminal prompt. This tells you the environment is active. Now install the dependencies:

```bash
pip install -r requirements.txt
```

This will take a few minutes the first time. It downloads and installs PyTorch, torchvision, timm, MLflow, Streamlit, and all the other libraries the project needs. Every package is explained in the requirements.txt file itself.

### Downloading the Dataset

With the environment active and your Kaggle credentials in place, run:

```bash
python -m src.data.download
```

This will download the New Plant Diseases Dataset from Kaggle (about 3 GB) and save a metadata file at `data/raw/dataset_info.json` so the rest of the code knows where to find the images. If you run this command again later, it will detect the cached download and skip the re-download.

The dataset contains approximately 87,000 images of plant leaves across 38 classes. It was collected and published on Kaggle by Samir Bhattarai. The images are 256 by 256 pixels, already split into training and validation folders, and cover the following diseases:

Apple scab, Apple black rot, Cedar apple rust, Blueberry healthy, Cherry powdery mildew, Corn grey leaf spot, Corn common rust, Corn northern leaf blight, Grape black rot, Grape black measles, Grape leaf blight, Orange Huanglongbing (citrus greening), Peach bacterial spot, Pepper bacterial spot, Potato early blight, Potato late blight, Raspberry healthy, Soybean healthy, Squash powdery mildew, Strawberry leaf scorch, Tomato bacterial spot, Tomato early blight, Tomato late blight, Tomato leaf mold, Tomato septoria leaf spot, Tomato spider mites, Tomato target spot, Tomato yellow leaf curl virus, Tomato mosaic virus, and healthy variants for Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato.

### Exploring the Data

Before training, open the exploratory data analysis notebook to understand what we are working with:

```bash
jupyter lab notebooks/01_exploratory_data_analysis.ipynb
```

This notebook shows you the class distribution, sample images from each class, image size statistics, and channel statistics. It also identifies which classes are likely to be confused with each other.

### Training the Model

Once the data is downloaded, you can start training:

```bash
python train.py
```

That's it. The script handles everything: building the DataLoaders, constructing the model, running the two-stage fine-tuning, evaluating on the test set, and saving all artefacts.

You can override any setting from the configuration file without editing it:

```bash
python train.py --epochs 20 --batch_size 64 --lr 0.0002
python train.py --backbone tf_efficientnetv2_m      # try a larger backbone
python train.py --run_name my_experiment             # custom MLflow run name
```

Progress is shown with a live progress bar. After training completes, you will see a summary like:

```text
Training complete in 47.3 min.
Test Accuracy : 98.73%
Top-5 Accuracy: 99.91%
Macro F1      : 98.68%
MLflow run    : a1b2c3d4e5f6
```

The best model is saved to `models/best_model.pth`. Evaluation reports (metrics.json, classification_report.csv, confusion matrix plot, training curves) are saved to `reports/`.

### Viewing Experiment Logs in MLflow

MLflow records every training run with its hyperparameters, per-epoch metrics, and the saved model. To open the interactive dashboard:

```bash
mlflow ui --backend-store-uri file://mlflow_runs
```

Open your browser to `http://localhost:5000`. You will see a table of all training runs, can compare metrics side by side, and view the training curves for each run.

### Making Predictions from the Command Line

To predict the disease in a single leaf image:

```bash
python predict.py --image path/to/your/leaf.jpg
```

Output looks like:

```text
=======================================================
  Image: leaf.jpg
=======================================================
  RANK   CLASS                                    CONFIDENCE
-------------------------------------------------------
  1      Tomato — Early blight                     97.43% ← TOP
  2      Tomato — Target Spot                       1.82%
  3      Tomato — Septoria leaf spot                0.41%
  4      Tomato — Bacterial spot                    0.19%
  5      Tomato — healthy                           0.07%
=======================================================

  Diagnosis: Tomato — Early blight
  Confidence: 97.4%
```

Add `--gradcam` to also generate a Grad-CAM heatmap saved as an image file.

### Running the Web App

```bash
streamlit run app/streamlit_app.py
```

Your browser will open to the PlantMD application. You can upload a leaf image, get a diagnosis with treatment advice, see the top-5 predictions visualised as a bar chart, and explore the model performance dashboard with interactive charts.

---

## The Architecture

The choice of architecture matters enormously. Here is why each decision was made.

### Why EfficientNetV2?

When choosing a neural network architecture for image classification, you are making a trade-off between accuracy and computational cost. A ResNet-50 is straightforward and well-understood. A Vision Transformer (ViT) is state-of-the-art but needs enormous datasets to work well. EfficientNet sits in a sweet spot that makes it ideal for this use case.

EfficientNet was developed at Google Brain in 2019. The key insight was compound scaling: instead of making a network wider OR deeper OR processing higher-resolution images independently (as was the previous approach), EfficientNet scales all three dimensions together using a fixed coefficient. The result is a family of models that achieve higher accuracy with fewer parameters than previous architectures.

EfficientNetV2, released in 2021, improves training speed by replacing the depthwise separable convolutions in the early layers with Fused-MBConv blocks, which are more efficient on modern hardware accelerators. The V2-Small variant (which we use) achieves around 84 percent top-1 accuracy on ImageNet with approximately 22 million parameters.

For our plant disease task, 22 million parameters is enough capacity to learn 38 distinct visual patterns while being small enough to load quickly in a web app.

### Why Transfer Learning?

Training from scratch on 87,000 images would produce a mediocre model. Deep neural networks need enormous amounts of data to learn good visual representations — the original ImageNet competition had 1.28 million images and still took weeks of GPU training. When you start from random weights on a dataset of this size, you are likely to overfit.

Transfer learning sidesteps this by reusing representations that were already learned on a much larger dataset. The EfficientNetV2 backbone already knows how to detect edges, textures, shapes, and object parts. Those skills transfer directly to detecting leaf lesions, discolourations, and structural changes.

We use the model's pre-trained weights as a starting point and then fine-tune them on our specific task. The result is significantly higher accuracy in significantly less time.

### Two-Stage Fine-Tuning

Naive fine-tuning — unfreezing all the pre-trained weights immediately and training with a high learning rate — often destroys the features that make transfer learning valuable. Large gradient updates in the first few epochs can push the backbone weights far from their initial values before the new classification head has learned anything meaningful.

The two-stage approach solves this:

In the first stage (five epochs by default), the backbone is frozen. Only the new classification head is trained. This is fast and safe — the pre-trained features cannot be damaged. By the end of stage 1, the head has learned which features are important for plant disease classification.

In the second stage, the backbone is unfrozen and the entire network is trained end-to-end, but at a learning rate ten times smaller than stage 1. The gentle gradient updates allow the backbone to specialise its features for leaves without catastrophic forgetting of the general visual knowledge it started with.

### The Classification Head

After the backbone extracts a 1280-dimensional feature vector, our custom head maps it to 38 class probabilities through this sequence:

Dropout (30%) removes 30% of the features randomly during training, preventing the head from over-relying on any single feature. This is a  form of regularisation — it forces the model to learn redundant representations.

A linear layer maps 1280 dimensions to 512 dimensions. This compression forces the model to distill the most relevant information for disease classification.

Batch normalisation stabilises the activations by normalising them to have approximately zero mean and unit variance. This makes training significantly more stable and allows higher learning rates.

ReLU activation introduces non-linearity. Without it, stacking linear layers would be mathematically equivalent to a single linear layer, no matter how deep.

A second dropout layer (20%) applies lighter regularisation before the final prediction.

The output linear layer maps 512 dimensions to 38, one score per disease class. These raw scores (logits) are converted to probabilities by softmax during inference.

---

## The Training Process

### Mixed-Precision Training

Modern NVIDIA GPUs have specialised hardware for 16-bit floating-point (float16) arithmetic that runs approximately twice as fast as 32-bit (float32) and uses half the memory. PyTorch's Automatic Mixed Precision (AMP) feature automatically converts appropriate operations to float16 while keeping numerically sensitive operations in float32.

A GradScaler is used alongside AMP to prevent gradient underflow: before the backward pass, the loss is scaled up (multiplied by a large number) to bring gradients into the float16 representable range. After the backward pass, the scaler divides the gradients back down before passing them to the optimiser.

The result is faster training and the ability to use larger batch sizes, with no meaningful loss in accuracy.

### The AdamW Optimiser

Adam is an adaptive learning rate optimiser that maintains per-parameter learning rates based on estimates of gradient first and second moments. It converges faster than SGD in most settings and is less sensitive to the initial learning rate.

AdamW fixes a theoretical flaw in Adam's weight decay implementation. Standard Adam applies weight decay by adding it directly to the gradient before computing the adaptive update, which means weight decay is scaled by the per-parameter learning rate. AdamW decouples weight decay from the gradient update, applying it directly to the weights. This is the theoretically correct behaviour and consistently outperforms Adam in practice on fine-tuning tasks.

### Cosine Annealing Learning Rate Schedule

After each epoch, the learning rate follows a cosine curve from its initial value down to a very small minimum. This is motivated by the observation that flat minima (regions of the loss landscape where many nearby weight configurations give similar loss) tend to generalise better than sharp minima. A cosine schedule allows the model to escape sharp local minima early in training (when the LR is high) and then settle into a flat minimum at the end (when the LR is low).

### Label Smoothing

The cross-entropy loss function normally trains the model to produce a probability of exactly 1.0 for the correct class and 0.0 for all others. This makes the model overconfident. Label smoothing modifies the targets: instead of 1.0 and 0.0, the correct class gets 0.9 and the wrong classes share the remaining 0.1. This prevents overconfidence and is a well-established regularisation technique from the "Rethinking the Inception Architecture" paper.

---

## Understanding the Results

### What the Metrics Mean

Top-1 Accuracy tells you the fraction of test images where the model's single best prediction was correct. This is the headline number.

Top-5 Accuracy tells you the fraction of test images where the correct class appeared somewhere in the model's top-5 predictions. This is always higher than top-1 accuracy and is useful for gauging how often the model "knows it doesn't know" — if the right answer is in the top 5 even when it is not first, the model has still narrowed it down significantly.

Macro F1-Score is the average F1-score across all 38 classes, treating each class equally regardless of how many images it has. This is the most honest single summary metric for a multi-class problem because it does not allow the majority class to dominate.

Precision answers the question: when the model says "Tomato Late Blight", how often is it right? High precision means few false alarms.

Recall answers: of all the actual Late Blight images, how many did the model catch? High recall means few missed cases. For disease detection, recall matters most: a missed disease spreads; an unnecessary treatment just costs money.

### The Confusion Matrix

The confusion matrix is a 38-by-38 grid where each cell (i, j) shows how many images of true class i were predicted as class j. The diagonal shows correct predictions. Off-diagonal cells are errors.

In practice, most confusion happens within the same plant species. Tomato Early Blight and Tomato Target Spot are sometimes confused because both produce circular lesions. Potato Early Blight and Tomato Early Blight (both caused by Alternaria species) can look similar because they come from the same pathogen family.

These confusions make biological sense, which is actually reassuring — it means the model is making mistakes that a human expert might also make, rather than making completely random errors.

---

## Project Structure

```text
Plant-Disease-Classifier/
├── plants/                          Virtual environment (not committed to git)
├── configs/
│   └── config.yaml                  The single source of truth for all hyperparameters
├── src/
│   ├── data/
│   │   ├── download.py              Downloads the dataset from Kaggle
│   │   └── dataset.py               Transforms, augmentation, DataLoaders
│   ├── models/
│   │   ├── architecture.py          EfficientNetV2 + custom 38-class head
│   │   └── trainer.py               Training loop, early stopping, AMP
│   ├── evaluation/
│   │   └── metrics.py               Accuracy, F1, confusion matrix, Top-K
│   └── utils/
│       ├── config.py                Loads config.yaml
│       ├── logger.py                Centralised logging to console + file
│       └── visualization.py         Plots, training curves, Grad-CAM
├── app/
│   ├── streamlit_app.py             Main Streamlit entry point
│   ├── pages/
│   │   ├── diagnose.py              Upload and diagnose a leaf
│   │   ├── performance.py           Model evaluation dashboard
│   │   ├── how_it_works.py          Educational explainer
│   │   └── about.py                 Project and author info
│   └── utils/
│       └── inference.py             Inference helpers + disease info database
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── models/
│   ├── best_model.pth               Saved after training
│   └── checkpoints/                 Periodic training checkpoints
├── data/
│   ├── raw/                         Kaggle download metadata
│   ├── processed/                   Reserved for future preprocessing
│   └── splits/                      Reserved for custom splits
├── logs/                            Training log files
├── mlflow_runs/                     MLflow experiment database
├── reports/
│   ├── metrics.json                 Evaluation summary
│   ├── classification_report.csv    Per-class metrics table
│   └── figures/                     All generated plots
├── train.py                         Main training entry point
├── predict.py                       CLI prediction script
├── requirements.txt                 Python dependencies
├── setup.py                         Package installation config
├── Makefile                         Common command shortcuts
└── .env.example                     Template for secrets
```

---

## Reproducibility

Reproducibility is a core value in scientific and engineering work. Anyone who clones this repository should be able to get the same results I got. Here is how this project ensures that:

The random seed is fixed (42) wherever randomness is introduced: the train/val/test split, weight initialisation, and stochastic training operations.

All hyperparameters live in a single config file. There are no magic numbers buried in source code.

The requirements.txt file pins exact package versions so the same software environment can be recreated.

The downloaded dataset is checksummed by kagglehub so we are always working with the same data.

---

## Running on Google Colab

If you do not have a GPU locally, Google Colab provides free NVIDIA T4 GPU access. Here is how to use this project on Colab:

1. Upload your `kaggle.json` credentials file to the Colab session.
2. Clone this repository into Colab.
3. Install the requirements with `pip install -r requirements.txt`.
4. Run the notebook or the training script directly.

Training time on a Colab T4 GPU is typically 3 to 6 hours for a full 50-epoch run, or about 30 minutes for a 5-epoch quick test.

---

## Acknowledgements

The New Plant Diseases Dataset was created and published by Samir Bhattarai on Kaggle. Without this carefully assembled, clean dataset, this project would not exist.

The EfficientNetV2 architecture was designed by Mingxing Tan and Quoc V. Le at Google Brain. The timm library, maintained by Ross Wightman, makes it trivially easy to use hundreds of state-of-the-art image models in PyTorch.

The Grad-CAM technique was introduced by Ramprasaath R. Selvaraju and colleagues in their 2017 paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization".

---

## Author

**Paul Sentongo** is a data scientist and machine learning engineer. This project was built as a demonstration of end-to-end MLOps skills: data engineering, deep learning, experiment tracking, model explainability, and application deployment.

The code is written to be read, not just run. Every module, every function, and every non-obvious line is documented with the reasoning behind the decision. This is how I believe software should be written: as a communication to the next person who reads it, whether that person is a colleague, a student, or myself six months from now.

---

*PlantMD — Making expert plant diagnostics accessible to everyone.*
