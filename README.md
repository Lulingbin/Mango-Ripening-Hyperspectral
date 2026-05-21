
---

```markdown
# Mango Ripening Hyperspectral Analysis Pipeline

This repository contains the complete open-source code and workflow for the non-destructive quality inspection and ripening prediction of mangoes using hyperspectral imaging (400–1700 nm). 

## 🔗 Project Links & Data Sources
* **Code Repository:** [https://github.com/Lulingbin/Mango-Ripening-Hyperspectral](https://github.com/Lulingbin/Mango-Ripening-Hyperspectral)
* **Dataset (A_Dataset):** [Google Drive Link](https://drive.google.com/file/d/10a3Q76GzvPpcDn04thSTBqc_oehmduaW/view?usp=drive_link) *(Please download the dataset and place it into the `A_Dataset` directory before running the scripts).*

---

## 📂 Repository Directory Structure

```text
├── A_Dataset                             # Raw and processed datasets (to be downloaded from Google Drive)
├── B_Qutlier                             # Phase 1: Anomaly detection and outlier elimination
│   ├── outlier_isolationforest.py        # Isolation Forest implementation for spectral outlier detection
│   └── outlier_isolationforest           # Visual outputs generated from outlier detection
│       ├── clean.csv / removed.csv       # Filtered normal spectra and detected outliers
│       ├── removed_ids.csv               # Polygon IDs of removed samples
│       ├── plot.png                      # Spectral curves highlighted with outliers
│       ├── isoforest_pca.png             # 2D PCA visual projection of Isolation Forest boundary
│       ├── avg_path_hist.png             # Distribution histogram of average path lengths
│       └── avg_path_step.png             # Step plot of sorted average path lengths
├── C_Data_pre_processing                 # Phase 2: Spectral signal pre-processing
│   └── preprocess_pipeline_save_each_process.py # Sequential pipeline for SG, SNV, and Z-score transformations
├── D_feature_selection                   # Phase 3: Wavelength selection algorithms
│   ├── feature_selection_pipeline.py     # Core algorithms: PCA4FS, RF4FS, SLP4FS (Gedeon), and GA4FS
│   └── build_new_allsingle_and_union_dataset.py # Generates top-K sub-datasets and union-feature sets
├── E_models                              # Model Architecture Definitions
│   ├── regressors.py                     # Scikit-learn wrappers for Ridge, Lasso, PLSR, RF, XGB, SLP, and MLP
│   └── test_all_regressors.py            # Unit tests for verification of all model shapes and dimensions
├── F_training                            # Phase 4: Model training and validation
│   ├── FengZhuang_SLP_AND_MLP_include_all_spec.py # K-Fold GridSearchCV training with early stopping for NN
│   └── predict.py                        # Model inference script
├── G_analysis                            # Phase 5: Paper figure and table replication scripts
│   ├── Fig4                              # 
│   │   ├── _13_hardness_3d_V2.py
│   │   └── _13_hardness_FX17_3d.py
│   ├── Fig6_AND_Fig7                     #
│   │   ├── Fig6.py / Fig7.py
│   │   └── rmse表格_noyolo .xlsx / 表格_noyolo.xlsx
│   ├── predict                           # Model prediction scatter plot generation
│   │   ├── scatter.py 
│   │   └── scatter
│   ├── TableS1_AND_FigS1                 # 
│   │   └── FigS1.py / TableS1.py / 目标.xlsx
│   └── TableS2_S5_AND_FigS2_Fig_S4       #
│       ├── TableS2_S3_AND_FigS2.py
│       ├── Table_S4_AND_FifS3.py
│       ├── Table_S5_AND_Fig_S4.py
│       └── four objectives.xlsx / Results_of_fittings.xlsx
└── H_Predict                             # Evaluation Tools
    └── all_predict.py                    # Automated scanning script to reproduce cross-validation scatter plots

```

---

## ⚙️ Core Pipeline Descriptions

### 1. Anomaly Detection (`B_Qutlier/`)

* **`outlier_isolationforest.py`**: Applies the `IsolationForest` algorithm (200 estimators, 5% contamination rate) to isolate anomalous spectral signatures. Performs dimensionality reduction via 2D-PCA for visual analysis (`isoforest_pca.png`) and plots average path length distributions (`avg_path_hist.png`, `avg_path_step.png`) to mathematically justify the anomaly thresholds.

### 2. Spectral Pre-processing (`C_Data_pre_processing/`)

* **`preprocess_pipeline_save_each_process.py`**: Executes structured transformations to isolate physics-induced scattering and electronic noise:
1. **Savitzky-Golay (SG) Smoothing**: Windows length=5, polynomial order=3 (`train_SG.csv`).
2. **Standard Normal Variate (SNV)**: Center-scale individual spectral matrix to reduce scattering impacts (`train_SNV.csv`).
3. **Z-score Normalization**: Globally scales features using training partitions statistics (`train.csv`).
Dataset partitioning maintains quality attribute distribution via stratified splitting based on 5 quantiles of target metrics (e.g., Hardness).



### 3. Wavelength Selection (`D_feature_selection/`)

* **`feature_selection_pipeline.py`**: Implements four distinct dimensional reduction strategies to capture critical spectral bands (Top-K=10):
* `PCA4FS`: Unsupervised loading weight analysis across main principal components.
* `RF4FS`: Supervised feature importance estimation using Ensembles.
* `SLP4FS`: Implementation of **Gedeon contribution analysis (1997)** by mapping weight connections of an optimized Multi-Layer Perceptron.
* `GA4FS`: Genetic Algorithm wrapper maximizing cross-validated negative RMSE.


* **`build_new_allsingle_and_union_dataset.py`**: Automatically builds individual downscaled datasets for each selection algorithm, structures an optimized union ensemble dataset (`_union.csv`), and exports physical wave indices (`union_features.csv`).

### 4. Regression Engine & Training (`E_models/` & `F_training/`)

* **`regressors.py`**: Establishes cross-compatible Estimator frameworks based on PyTorch and Scikit-Learn for seven core algorithmic families: `Ridge`, `Lasso`, `PLSRegression`, `RandomForestRegressor`, `XGBRegressor`, `SLPRegressorTorch` (Single Layer Perceptron), and `MLPRegressorTorch` (Multi-Layer Perceptron).
* **`FengZhuang_SLP_AND_MLP_include_all_spec.py`**: Comprehensive execution script. Runs a full 5-fold cross-validated grid search (`GridSearchCV`) across model parameter grids. For PyTorch neural networks (`SLP` / `MLP`), it incorporates an `EarlyStopping` routine tied directly to secondary validation checks. Results are fully summarized inside `tuning_summary_all.csv` and evaluation heatmaps.

### 5. Automated Validation & Reproduction (`H_Predict/`)

* **`all_predict.py`**: Scans multiple operational project paths globally to pick up raw `.pkl` model serialization checkpoints. Reads the original dataset contexts, reconstructs underlying neural layer frames, maps cross-validation true vs. predicted regression variables, and outputs fully standardized publication-ready joint performance scatter plots (`output_scatters/`).

---

## 📊 Academic Paper Results Replication (`G_analysis/`)

Scripts located within the `G_analysis/` directory correspond directly to the figures and tables presented in the accompanying manuscript:

---

## 🛠️ Requirements & Installation

Ensure you have Python 3.8+ and the following packages installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch scipy joblib openpyxl sklearn-genetic-opt

```

```

```
