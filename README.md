# CMEPDA Project 2024: Alzheimer's Disease SVM Classification & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-SPM2025-orange?logo=mathworks)](https://it.mathworks.com/products/matlab.html)
![GitHub repo size](https://img.shields.io/github/repo-size/Davideciaba/CMEPDA_project_2024)
![CircleCI](https://circleci.com/gh/Davideciaba/CMEPDA_project_2024/tree/main.svg?style=shield)

The aim of this repository is to test the interpretative capabilities of two Explainable AI (XAI) approaches for Support Vector Machines (SVM) in the field of medical imaging, specifically for the classification of Alzheimer's Disease (AD) versus Healthy Controls (CTRL) using ADNI MRI datasets+. The two XAI approaches (Gaonkar and Haufe) are quantitatively compared against Voxel-Based Morphometry (VBM) analysis and raw SVM weights using NDCG ranking metrics aggregated over Regions of Interest (ROI). The computational framework is developed across MATLAB—ideal for medical image preprocessing via its toolboxes—and Python, which provides the flexibility needed to build the ML models and XAI frameworks.

# Table of contents
+ [Prerequisites](#prerequisites)
+ [Data & Preprocessing](#data--preprocessing)
+ [Validation Framework](#validation-framework)
+ [Model Building and Training](#model-building-and-training)
  + [VBM (The Ground Truth)](#vbm-the-ground-truth)
  + [Linear SVM & XAI](#linear-svm--xai)
+ [Results & Evaluation](#results--evaluation)
  + [Thresholding and ROI Aggregation](#thresholding-and-roi-aggregation)
  + [Heat Map & Feature Importance](#heat-map--feature-importance)
+ [Usage](#usage)
+ [References](#references)

# Prerequisites
The project features a decoupled architecture, meaning the Python and MATLAB pipelines operate independently. If you only wish to execute the Python sections to evaluate the Linear Support Vector Machine (SVM) with the VBM Analysis, you can directly use the pre-computed MATLAB results already provided. Alternatively, the MATLAB sections can be executed natively within your MATLAB Desktop environment (or within your IDE with MATLAB extension).

## Python
* **Environment**: The project requires Python 3.11 or higher.
* **Dependencies**: Install the required Machine Learning packages by running `pip install -r requirements.txt` in your virtual environment.
* **SPM**: Even though the execution is decoupled, the Python environment requires the SPM path configuration. You must duplicate the provided `.env.example` file in the project's root directory, rename it to `.env`, and assign the absolute path of your local SPM folder to the `SPM_DIR` variable (e.g., `SPM_DIR=C:/path/to/spm`).

## MATLAB
* **Environment**: The VBM Analysis and the other MATLAB classes and functions have been tested on MATLAB 2025b.
* **Toolboxes**: The MATLAB pipeline requires the **Image Preprocessing Toolbox** and the **Statistics and Machine Learning Toolbox**. The codebase includes an automated Fail-Fast validator that will block execution if these licenses are missing.
* **SPM**: The VBM Analysis requires SPM. Similarly to Python, you must duplicate the `config.example.json` file in the project's root directory, rename it to `config.json,` and set the `SPM_DIR` key to your local absolute SPM path (e.g., `{"SPM_DIR": "C:\\path\\to\\spm"}`).

# Data & Preprocessing
The dataset utilized in this study stems from the Retico article, comprising 144 subjects affected by AD and 189 healthy controls. 
Before model ingestion, the dataset underwent an extensive preprocessing pipeline governed by the DARTEL algorithm. The standardization steps included:
* Tissue segmentation.
* Generation of a study-specific template.
* Warping of the template and the segmented maps into the standard MNI space.
* Modulation of the volumes altered by spatial warping to preserve gray matter quantities.

<div align="center">

| **Raw MRI (MNI)** | **Preprocessed GM Mask** |
|--------------------|------------------|
| <img src="Readme_images/raw_mri.png" alt="Raw" width="200"> | <img src="Readme_images/gm_mask.png" alt="Preprocessed" width="200"> |

</div>

# Validation Framework
We trained a classical linear SVM model, aligning with the methodology proposed by Retico. To ensure robustness, the dataset was adapted using a Double Cross-Validation strategy:
* **Outer 5-fold CV** for rigorous evaluation.
* **Inner 5-fold CV** for hyperparameter tuning.

For both inner and outer folds, classic predictive metrics are evaluated: Accuracy, Balanced Accuracy, AUROC, F1-Score, Sensitivity, and Specificity. 
To strictly avoid **Data Leakage**, the scaler is calibrated exclusively on the training data (Inner and Outer Fold respectively) and subsequently applied to scale both the training and validation sets.

# Model Building and Training

## VBM (The Ground Truth)
The Ground Truth is defined via VBM analysis, a method that allows determining regional differences in tissue volume (specifically, the gray matter of patients, which is the most affected by Alzheimer's) and evaluating the statistical contribution of each voxel to atrophy. 
* **Methodology:** A General Linear Model based on a Two-Sample T-test is computed over the entire dataset, incorporating age, sex, and Total Intracranial Volume (TIV) as regressors.
* **Output:** The output generates a Global VBM Mask. The threshold is determined via False Discovery Rate correction, combined with a specific TPM threshold.

## Linear SVM & XAI
The core of the project focuses on interpretability methods, as the raw weights of the linear SVM (defined as the "backward model") assign high values to voxels acting as suppression variables, thereby masking crucial regions. To overcome these limitations, the training and XAI extraction pipelines have been completely decoupled.

### Model Building and Tuning
* **Data Ingestion:** The input 3D NIfTI volumes are loaded using `nibabel` and masked via boolean vectorization, extracting valid voxels directly into a flattened 1D array to optimize computational performance.
* **Pipeline Setup:** A `Linear SVM` is implemented using Scikit-Learn's `SVC` combined with a `StandardScaler` within a unified pipeline to prevent data leakage.
* **Hyperparameter Tuning:** Inside each fold, hyperparameter optimization is performed using `GridSearchCV`. The scoring metric used to select the optimal model is `balanced_accuracy`.
* **Serialization:** After executing the Nested CV, the best-fitted pipelines for each fold are saved to disk (`.joblib`) for independent XAI extraction.

The hyperparameters explored during the grid search are shown in the following table:

<div align="center">

| Hyperparameter | Values |
| -------------- | ------ |
| `kernel`       | `linear` |
| `class_weight` | `balanced`|
| `C`            | `1e-4, 1e-3` |

</div>

### Spatial Interpretability (XAI)
A standalone XAI orchestrator loads the pre-trained pipelines from disk and flanks the model with advanced mathematical transformations:
* **Raw Weights:** Extracted directly from the SVM coefficients (`svc.coef_`).
* **Haufe Transformation:** This transforms the "backward model" into a "forward model" by calculating the covariance between the scaled training variables and the decision function scores. This extracts Activation Patterns that isolate the actual neurophysiological distribution and gray matter variation.
* **Gaonkar Transformation:** An analytic margin-aware transformation is applied to the weight vector, generating p-value maps that allow building a multivariate statistical inference complementary to VBM. It dynamically utilizes the optimal `C` parameter and the number of support vectors extracted from the trained model.

Finally, the 1D arrays are reconstructed into 3D NIfTI volumes using the original spatial affine matrix and rendered as 3D activation maps over a reference background image (e.g., CTRL-117).

# Results & Evaluation
The generated maps represent the general decision rules learned over the entire training set, making them inherently global. 

## Thresholding and ROI Aggregation
Before comparison, specific thresholds were applied to isolate relevant voxels based on the mathematical nature of the maps:
* **Continuous Maps (Haufe and Raw Weights):** We selected the top 1% (99th percentile) and top 5% (95th percentile) of voxels with the highest intensity to isolate clusters with the greatest feature importance.
* **Statistical Maps (Gaonkar):** Based on a strict statistical approach, we utilized an alpha = 0.05 for the Bonferroni method and q = 0.1 for the Benjamini-Yekutieli method (FDR).

To properly compare the various generated maps with the VBM (which visualizes local atrophy), we performed an aggregation within Regions of Interest (ROI) using the SPM Neuromorphometric atlas. 

## Heat Map & Feature Importance
The feature importance—used for the construction of global heatmaps and the correlation matrix based on the NDCG metric—is calculated by considering the **absolute value** of the attributions, as our global models generate both positive and negative attributes. 
To preserve the original positive/negative signature for each region, these absolute maps are accompanied by **diverging bar plots**.

*Visual Comparison of XAI outputs (SVM vs VBM):*
<div align="center">
<img src="Readme_images/heat_map.png" width="800"> 
</div>

# Usage
Due to the dimension of the model's weights we were not able to upload them on github. To avoid the problem and make the user able to test the code using our tuning results we upload them at the following links: [Weights](https://drive.google.com/drive/folders/1vJJEUFFgmrWvrM_9QjdptaV1J5LFpE6y?usp=sharing). The user has to move the folders to the root directory of the project before running the code.

Clone this repository and run the main orchestrator using default parameters (In case you are running this code for the first time remember to install the requirements.):
```bash
cd CMEPDA_project_2024
pip install -r requirements.txt 
python main.py
```
Refer to help for the different section of the main orchestrator:
```bash
python main.py --help
usage: main.py [-h] [-log] [-out OUTPUT_DIR] [-in INPUT_DIR] [-csv CSV_NAME] [-set] [-svm] [-xai] [-up] [-bg] [-cv C_VALUES [C_VALUES ...]] [-xc] [-of OUTER_FOLDS] [-inf INNER_FOLDS]

Launch Machine Learning and XAI Pipelines.

options:
  -h, --help            show this help message and exit
  -log, --enable-file-logging
                        Enable writing .log files to disk for all executed modules (Default: False).
  -out OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Target root directory for all outputs (Default: Python_Results).
  -in INPUT_DIR, --input-dir INPUT_DIR
                        Source directory containing NIfTI and CSV files (Default: AD_CTRL in project root).
  -csv CSV_NAME, --csv-name CSV_NAME
                        Name of the clinical covariate CSV file (Default: covariateADCTRLsexAgeTIV.csv).
  -set, --run-setup     Execute common setup (generates CV folds & cohort registry).
  -svm, --run-svm       Execute the Linear SVM classification pipeline.
  -xai, --run-xai       Execute the XAI pipeline for the trained Linear SVM.
  -up, --use-pretrained
                        If passed, skips training and loads pre-trained models. Default: False (Forces full retraining).
  -bg, --bypass-grid    If passed, skips GridSearch and uses historical Optimal_C from CSV. Default: False (Executes full GridSearch).
  -cv C_VALUES [C_VALUES ...], --c-values C_VALUES [C_VALUES ...]
                        List of values for the SVM 'C' hyperparameter. Example: -cv 0.0001 0.001 0.01 (Default: [1e-4]).
  -xc, --run-compare    Execute the Comparative XAI benchmarking against MATLAB VBM Ground Truth.
  -of OUTER_FOLDS, --outer-folds OUTER_FOLDS
                        Number of outer cross-validation folds. Default: 5
  -inf INNER_FOLDS, --inner-folds INNER_FOLDS
                        Number of inner cross-validation folds. Default: 5
```

For the preprocessing part you need **Matlab**, version 25.2, not included in the project's requirements. 
# References

The original dataset comes from the Alzheimer Dataset used in the following paper:
- **Retico, A., Bosco, P., Cerello, P., Fiorina, E., Chincarini, A., & Fantacci, M. E.** (2014). Predictive Models Based on Support Vector Machines: Whole-Brain versus Regional Analysis of Structural MRI in the Alzheimer's Disease. Journal of Neuroimaging, 25(4), 552–563. [DOI](https://doi.org/10.1111/jon.12163).

The Machine Learning Model and the Explainable-AI approaches comes from the following articles:

- **Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.** (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96–110. [DOI](http://dx.doi.org/10.1016/j.neuroimage.2013.10.067)

- **Gaonkar, B., & Davatzikos, C.** (2013). Analytic estimation of statistical significance maps for support vector machine based multi-variate image analysis and classification. NeuroImage, 78, 270-283. [DOI](10.1016/j.media.2015.06.008.)

- **Bloch, L., & Friedrich, C. M.** (2024). Systematic comparison of 3D Deep learning and classical machine learning explanations for Alzheimer's Disease detection. Scientific Reports, 14. [DOI](https://doi.org/10.1016/j.compbiomed.2024.108029)

- **Ridgway, G., Omar, R., Ourselin, S., Hill, D., Warren, J., & Fox, N.** (2009). Issues with threshold masking in voxel-based morphometry of atrophied brains. NeuroImage, 44(1), 99–111. [DOI](10.1016/j.neuroimage.2008.08.045)