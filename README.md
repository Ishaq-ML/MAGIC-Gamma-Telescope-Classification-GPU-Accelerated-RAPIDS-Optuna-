# MAGIC Gamma Telescope Classification: GPU-Accelerated (RAPIDS & Optuna)

## 🌟 Project Evolution
This repository represents a high-performance optimization of my previous project: [**MAGIC-Gamma-Telescope-Classification-Grid-Search-vs-Optuna**](https://github.com/Ishaq-ML/MAGIC-Gamma-Telescope-Classification-Grid-Search-vs-Optuna).

While the original project focused on comparing tuning methodologies using CPU-bound libraries (Pandas, Scikit-Learn, XGBoost), this version transitions the entire pipeline to the **GPU** using **NVIDIA RAPIDS** and **cuPy**.

### 🚀 What's New? (The GPU Advantage)
The primary goal of this update was to leverage hardware acceleration to drastically reduce the time required for data processing and hyperparameter tuning.

| Feature | Original Repo (V1) | This Repo (V2 - Accelerated) |
| :--- | :--- | :--- |
| **Data Handling** | `Pandas` (CPU) | `cuDF` (GPU) |
| **Array Operations** | `NumPy` (CPU) | `cuPy` (GPU) |
| **Machine Learning** | `Scikit-Learn` / `XGBoost` | `cuML` (GPU-Accelerated Random Forest) |
| **Tuning Speed** | Standard (Minutes) | **Lightning Fast (Seconds)** |
| **Data Transfer** | System Memory | Device Memory (Zero-copy between RAPIDS/cuPy) |

---

## 📊 Project Overview
This project performs binary classification on the **MAGIC Gamma Telescope dataset** to distinguish between electromagnetic showers (Gamma) and hadronic showers (Hadron). 

By utilizing GPU-accelerated libraries, we can run complex Bayesian Optimization (Optuna) trials in a fraction of the time, allowing for deeper searches and more iterations.

### Dataset Details
- **Source:** UCI Machine Learning Repository
- **Instances:** 19,019
- **Features:** 10 numerical features (fLength, fWidth, fSize, etc.)
- **Target:** Class `g` (Signal) or `h` (Background)

---

## 🛠️ Technology Stack
- **GPU Acceleration:** [RAPIDS cuDF / cuML](https://rapids.ai/)
- **Computation:** [cuPy](https://cupy.dev/)
- **Optimization:** [Optuna](https://optuna.org/) (Tree-structured Parzen Estimator)
- **Visualization:** Matplotlib, Seaborn

---

## 📈 Performance & Results
By moving the Random Forest classifier and the preprocessing steps to the GPU, the 50-trial Optuna search completes almost instantly.

**Model Performance:**
- **Accuracy:** ~88.99%
- **ROC AUC:** 0.9388
- **Best Params Found:** 
    - `n_estimators`: 180
    - `max_depth`: 25
    - `min_samples_split`: 13

### Key Visualizations
The project generates an automated analysis dashboard (`gpu_optimization_analysis.png`) including:
1. **ROC Curve:** Demonstrating high model sensitivity.
2. **Confusion Matrix:** Showing precise Signal vs. Background separation.
3. **Optimization Progress:** Tracking the improvement of accuracy over 50 trials.
4. **Hyperparameter Impact:** Visualizing how `max_depth` and `n_estimators` correlate with model performance.

---

## 💻 Setup and Installation

### Requirements
- **NVIDIA GPU** (Pascal architecture or newer)
- **CUDA Toolkit**
- **RAPIDS Environment**

### Installation
The most efficient way to run this is via a Conda environment:

```bash
# Create a RAPIDS environment
conda create -n rapids-gpu -c rapidsai -c conda-forge -c nvidia \
    cudf=24.04 cuml=24.04 cupy optuna matplotlib seaborn
conda activate rapids-gpu
