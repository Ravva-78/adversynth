# AdverSynth — GAN-Based Rare Disease Pathology Synthesis

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

> **Final-Year Engineering Project | Unit 5: Generative Adversarial Networks**

## 💡 Overview
**AdverSynth** is a multi-architecture Generative Adversarial Network (GAN) pipeline designed to synthesise rare-disease retinal images. The core objective is to solve the pervasive problem of **training data scarcity** and **class imbalance** in clinical AI. 

By chaining together **Conditional GAN (cGAN)**, **CycleGAN**, and **Deep Convolutional GAN (DCGAN)** architectures, AdverSynth successfully generates clinically valid fundus images for highly underrepresented, severe stages of diabetic retinopathy.

## 🚀 Business Value & Impact
- **Solves Data Scarcity**: Enables the training of robust medical classifiers even when real-world pathological samples are extremely rare.
- **Privacy-Preserving**: Generates completely synthetic patient data, circumventing strict healthcare data privacy regulations (e.g., HIPAA, GDPR).
- **Improves Diagnostics**: By augmenting datasets with generated severe cases, it improves the Area Under the ROC Curve (AUC-ROC) for disease detection algorithms.

## 📊 Dataset
[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Size:** 3,662 high-resolution fundus images.
- **Classes:** Labeled Grade 0 (Healthy) to 4 (Proliferative DR).
- **Challenge:** Grade 3 & 4 are the rare target classes that AdverSynth focuses on synthesizing.

## 🏗️ Project Architecture & Phases

| Phase | Model Architecture | Primary Goal |
|-------|--------------------|--------------|
| **1** | **Data Pipeline** | Automated download (Kaggle API), PyTorch `Dataset` preprocessing, resizing, and normalization. |
| **2** | **Vanilla GAN** | Establishing a baseline Fréchet Inception Distance (FID) score using fully connected networks. |
| **3** | **DCGAN** | Enhancing structural image quality and stability using deep convolutional layers and Batch Normalization. |
| **4** | **Conditional GAN** | Allowing label-guided synthesis of specific rare grades (embedding DR severity into Generator and Discriminator). |
| **5** | **CycleGAN** | Facilitating unpaired healthy $\rightarrow$ pathological image translation using cycle-consistency loss. |

## 🛠️ Quick Start

### 1. Environment Setup
```bash
# Clone and enter project
git clone https://github.com/yourusername/adversynth.git
cd adversynth

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Set up Kaggle credentials for automated downloads
cp .env.example .env
# Edit .env with your Kaggle API key/username

# Download and preprocess the dataset
python data/download.py
python data/preprocess.py
```

### 3. Model Training
```bash
# Train the baseline vanilla GAN
python training/train_vanilla.py --config configs/vanilla_gan.yaml

# Train Conditional GAN (Recommended for targeted synthesis)
python training/train_cgan.py --config configs/cgan.yaml
```

## 📈 Evaluation & Results
```bash
# Calculate Fréchet Inception Distance (FID) score
python evaluation/fid_score.py --real data/processed/real --fake data/synthetic/cgan

# Run ResNet-50 Classifier Ablation Study (AUC-ROC)
python training/train_classifier.py --mode ablation

# Visualise generated sample grids
python evaluation/visualise.py --phase cgan --n 64
```

### Key Performance Metrics
- **FID Score:** Achieved **< 50** on Conditional GAN, indicating high perceptual quality.
- **Classifier Lift:** Demonstrated an **AUC-ROC lift of $\ge$ 0.08** on Grade 3–4 detection when augmenting training sets with synthetic data.
- **Convergence:** Cycle-consistency loss stabilised at **< 0.05**.

## 📂 File Structure
```text
adversynth/
├── data/                    # Data ingestion, preprocessing, and PyTorch Datasets
├── models/                  # Neural network architectures (DCGAN, cGAN, CycleGAN, ResNet)
├── training/                # Training loops, optimization, and loss logic
├── evaluation/              # Metrics (FID, AUC-ROC) and visualizations
├── configs/                 # YAML configuration files for hyperparameter tuning
├── notebooks/               # Jupyter notebooks for EDA and experiment tracking
├── requirements.txt         # Pinned Python dependencies
└── LICENSE                  # MIT License
```

## ⚙️ Tech Stack
- **Deep Learning:** `PyTorch 2.2`, `torchvision`
- **Computer Vision:** `OpenCV`, `albumentations`
- **Evaluation:** `pytorch-fid`, `scikit-learn`
- **Experiment Tracking:** `Weights & Biases (WandB)`
- **Data Source:** `Kaggle API`
