# Automated Evaluation of Postharvest Mango Ripening Indices using Hyperspectral Imaging

This repository contains the official code, dataset links, and implementation pipeline for the paper: **"Automated Evaluation of Postharvest Mango Ripening Indices using Hyperspectral Imaging"**.

---

## 📊 Dataset & Setup

### 1. Download the Dataset
Due to the massive file size (**1.8 GB**), the complete postharvest quality dataset (comprising hyperspectral image cubes spanning 395–1,720 nm along with reference truth values for firmness, TSS, carotenoids, and starch) cannot be stored directly on GitHub. 

Please download the dataset from the following secure storage:
* **Google Drive Link:** [Download A_Dataset Here](https://drive.google.com/file/d/10a3Q76GzvPpcDn04thSTBqc_oehmduaW/view?usp=drive_link)

### 2. Directory Structure
After downloading, unzip the data and place it inside your local repository according to the structure below:
```text
Mango-Ripening-Hyperspectral/
├── A_Dataset/                 # Place the downloaded data here
│   ├── raw_hsi_cubes/
│   └── reference_labels.csv
├── src/                       # Source code pipeline
│   ├── preprocessing.py       # Calibration, S-G filter, SNV, Z-score
│   ├── feature_selection.py   # PCA, RF-FS, GA-FS, SLP-FS
│   └── models.py              # Ridge, Lasso, PLSR, MLP, XGBoost, etc.
├── main.py                    # Pipeline execution script
├── .gitignore                 # Configured to ignore local huge datasets
└── README.md# Mango-Ripening-Hyperspectral
Code and model weights for the paper "Automated Evaluation of Postharvest Mango Ripening Indices using Hyperspectral Imaging"
