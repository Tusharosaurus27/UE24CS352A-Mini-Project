# Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data

**UE24CS352A Machine Learning Mini-Project | Problem Statement #2**

*Based on Stanford University CS229 Research: "Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data" (Hojung Choi, Rachel Thomasson)*

## 📌 Executive Overview

Autonomous legged robots operating in disaster-stricken, GPS-denied, or visually obscured environments cannot reliably depend on exteroceptive computer vision or LiDAR for environmental perception. 

⚠️ **The Engineering Problem:** Legged locomotion requires instantaneous, stride-to-stride adaptation of foot compliance, stance duration, and joint impedance. When operating "blind," a robot transitioning from rigid concrete to low-friction laminate or yielding sand experiences catastrophic foot slip or ground penetration if gait parameters are not modulated within milliseconds of contact.

This project formulates an automated, real-time terrain classification engine operating directly on high-frequency, multi-axis capacitive tactile time-series data captured during the mid-step stance phase. 

* **Proprioceptive/Tactile Sensing:** Classifies physical substrate properties purely through foot-ground physical reaction forces.
* **End-to-End Representation Learning:** Utilizes an end-to-end 1D Convolutional Neural Network (1D CNN) to inherently learn spatio-temporal features without manual feature engineering.
* **Raw Tensor Processing:** Operates across continuous 6x69 spatio-temporal tactile tensors, capturing 5 normal taxels and 1 shear taxel across 69 discrete temporal frames per footstep.

## ⚡ Key Features

* **End-to-End Pipeline:** The Jupyter Notebook handles spatio-temporal tensor synthesis, temporal jitter injection, stratified train/test partitioning, model fitting, and metric evaluation.
* **Biophysically Realistic Sensor Synthesizer:** Parametric tactile generator modeling true physical foot strikes: normal impact spikes, tangential slip curves, contact damping, and surface collapse with randomized temporal jitter and additive Gaussian vibration noise.
* **1D Deep Convolutional Neural Network:** A 4-stage hierarchical temporal feature extractor with batch regularization and dropout optimized for time-series data.

## 🏗️ Pipeline Architecture

```text
Raw Tactile Foot Strike (Mid-Step Phase)
│
├── 5 Normal Capacitive Taxels (Vertical Load Dynamics)
└── 1 Shear Capacitive Taxel   (Tangential Traction / Slip)
                    │
                    ▼
       Input Tensor: (Batch, 6, 69)
                    │
                    ▼
           1D CNN Architecture
├── Conv1D (16, k=3) + ReLU
├── MaxPool1D (k=2)
├── Conv1D (32, k=3) + ReLU
├── MaxPool1D (k=2)
├── Conv1D (64, k=3) + ReLU
├── MaxPool1D (k=2)
├── Conv1D (128, k=3) + ReLU
├── MaxPool1D (k=2)
├── Dropout (p=0.5)
├── Linear (512 -> 64)
└── Softmax Head (4 Classes)
                    │
                    ▼
     Terrain Class Probabilities
     ├── Class 0: High Friction (Concrete)
     ├── Class 1: Low Friction  (Laminate)
     ├── Class 2: Deformable    (Grass/Turf)
     └── Class 3: Granular      (Sand)
```

## 📊 Dataset Specifications

The dataset mirrors the capacitive tactile array developed for the Stanford Autonomous Intelligent Legged Robot (SAIL-R):

* **Input Tensor Shape:** Continuous 3D Tensor: `(N, 6, 69)`
* **Spatial Channels:** 6 continuous tactile channels. Channels 0–4 = Normal Taxels, Channel 5 = Shear Taxel.
* **Temporal Horizon:** 69 discrete time steps capturing mid-step foot impact and stance compression.
* **Partitioning Strategy:** Stratified 80/20 train-to-test split with fixed random seeds for empirical reproducibility.

## 💻 Quick Start & Setup

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/yourusername/UE24CS352A-Mini-Project.git
cd UE24CS352A-Mini-Project

python -m venv env
source env/bin/activate  # On Windows use: .\env\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Project
1. Launch Jupyter Notebook from your terminal:
   ```bash
   jupyter notebook
   ```
2. Open the `terrain_classification.ipynb` file in your browser.
3. Click **"Run All"** to execute the pipeline. 

The notebook will synthesize the 6x69 tactile tensor dataset, perform an 80/20 train-test split, and train the 4-layer 1D CNN. The final cell outputs the validation accuracy, converging realistically between 74% and 78%.
