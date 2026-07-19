# CMEPDA Project 2024: Alzheimer's Disease SVM Classification & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-SPM2025-orange?logo=mathworks)](https://it.mathworks.com/products/matlab.html)
![GitHub repo size](https://img.shields.io/github/repo-size/Davideciaba/CMEPDA_project_2024)
![CircleCI](https://circleci.com/gh/Davideciaba/CMEPDA_project_2024/tree/main.svg?style=shield)

The aim of this repository is to test the interpretative capabilities of two Explainable AI (XAI) approaches for Support Vector Machines (SVM) in the field of medical imaging, specifically for the classification of Alzheimer's Disease (AD) versus Healthy Controls (CTRL) using ADNI MRI datasets. The two XAI approaches ([Gaonkar et al. (2015)](https://doi.org/10.1016/j.media.2015.06.008) and [Haufe et al. (2014)](http://dx.doi.org/10.1016/j.neuroimage.2013.10.067)) are quantitatively compared against Voxel-Based Morphometry (VBM) analysis and raw SVM weights using NDCG ranking metrics aggregated over Regions of Interest (ROI). The computational framework is developed across MATLAB—ideal for medical image preprocessing via its toolboxes—and Python, which provides the flexibility needed to build the ML models and XAI frameworks.

# Table of contents
+ [Prerequisites](#prerequisites)
+ [Data & Preprocessing](#data--preprocessing)
+ [Validation Framework](#validation-framework)
+ [Models and XAI approches explored](#model-and-xai-approches-explored)
  + [VBM (The Ground Truth)](#vbm-the-ground-truth)
  + [Linear SVM & XAI](#linear-svm--xai)
+ [Results & Evaluation](#results--evaluation)
  + [Thresholding and ROI Aggregation](#thresholding-and-roi-aggregation)
  + [Heat Map & Feature Importance](#heat-map--feature-importance)
+ [Usage](#usage)
+ [References](#references)

# Prerequisites
The project features a decoupled architecture, meaning the Python and MATLAB pipelines operate independently. If you only wish to execute the Python sections to evaluate the Linear Support Vector Machine (SVM) with the VBM Analysis, you can directly use the pre-computed MATLAB results already provided. Alternatively, the MATLAB sections can be executed natively within your MATLAB Desktop environment (or within your IDE with MATLAB extension). Both the environments requires [SPM](https://doi.org/10.21105/joss.08103). Please refer to SPM documentation: [https://www.fil.ion.ucl.ac.uk/spm/docs/](https://www.fil.ion.ucl.ac.uk/spm/docs/) to download the latest release (SPM25 tested).

## Python
* **Environment**: The project requires Python 3.11 or higher.
* **Dependencies**: Install the required packages by running `pip install -r requirements.txt` in the project directory.
* **SPM**: Even though the execution is decoupled, the Python environment requires the SPM path configuration. You must duplicate the provided `.env.example` file in the project's root directory, rename it to `.env`, and assign the absolute path of your local SPM folder to the `SPM_DIR` variable (e.g., `SPM_DIR=C:/path/to/spm`).

## MATLAB
* **Environment**: The VBM Analysis and the other MATLAB classes and functions have been tested on MATLAB 2025b.
* **Toolboxes**: The MATLAB pipeline requires the **Image Preprocessing Toolbox** and the **Statistics and Machine Learning Toolbox**. The codebase includes an automated Fail-Fast validator that will block execution if these licenses are missing.
* **SPM**: The VBM Analysis requires SPM. Similarly to Python, you can duplicate the `config.example.json` file in the project's root directory, rename it to `config.json,` and set the `SPM_DIR` key to your local absolute SPM path (e.g., `{"SPM_DIR": "C:\\path\\to\\spm"}`).

# Data & Preprocessing
The dataset utilized in this study comes from the [Retico et al. (2014)](https://doi.org/10.1111/jon.12163), comprising 144 subjects affected by AD and 189 healthy controls. 
Before model ingestion, the dataset underwent an extensive preprocessing pipeline governed by the DARTEL algorithm. The standardization steps included:
* Tissue segmentation.
* Generation of a study-specific template.
* Warping of the template and the segmented maps into the standard MNI space.
* Smoothing with isotropic Gaussian kernel (scale = 3 mm).

# Validation Framework
We trained a Linear SVM model, aligning with the methodology proposed by [Retico et al. (2014)](https://doi.org/10.1111/jon.12163). To ensure robustness, the dataset was adapted using a Double Cross-Validation strategy:
* **Outer 5-fold CV** for rigorous evaluation.
* **Inner 5-fold CV** for hyperparameter tuning.

For both inner and outer folds, classic predictive metrics are evaluated: Accuracy, Balanced Accuracy, AUROC, F1-Score, Sensitivity, and Specificity. 
To strictly avoid **Data Leakage**, the scaler is calibrated exclusively on the training data (Inner and Outer Fold respectively) and subsequently applied to scale both the training and validation sets.

# Models and XAI approches explored

## VBM (The Ground Truth)
The Ground Truth is defined via VBM analysis, a method that allows determining regional differences in tissue volume (specifically, the gray matter of patients, which is the most affected by Alzheimer's) and evaluating the statistical contribution of each voxel to atrophy.
* **Preprocess:** The input 3D NIfTI volumes are loaded and masked using a boolean mask based on SPM Tissue Probability Map (TPM).
* **Methodology:** A General Linear Model based on a Two-Sample T-test is computed over the entire dataset, incorporating age, sex, and Total Intracranial Volume (TIV) as covariates.
* **Output:** The output generates a Global VBM Mask. The threshold is determined via Family-Wise Error correction at alpha = 0.05.

<div align="center">

| **VBM TPM Mask (FWE 0.05)** |
|--------------------
| <img src="MATLAB_Results\VBM_Pipeline_Results\Plots\VBM_TPM_Mask_FWE_005.png" alt="Raw" width="600"> |

</div>

## Linear SVM & XAI
The core of the project focuses on binary classification and interpretability methods.

### Model Building and Tuning
* **Data Ingestion:** The input 3D NIfTI volumes are loaded using `nibabel` and masked via TPM mask, extracting valid voxels directly into a flattened 1D array to optimize computational performance.
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
A standalone XAI orchestrator loads the trained models and flanks them with the following XAI strategies:
* **Raw Weights:** Extracted directly from the SVM coefficients (`svc.coef_`).


* **Haufe transformation:** This transforms the "backward model" into a "forward model" by calculating the covariance between the scaled training variables and the decision function scores. This extracts Activation Patterns that isolate the actual neurophysiological distribution and gray matter variation.


* **Gaonkar Transformation:** An analytic margin-aware transformation is applied to the weight vector, generating p-value maps that allow building a multivariate statistical inference complementary to VBM.

Finally, the 1D arrays are reconstructed into 3D NIfTI volumes using the original spatial affine matrix and rendered as 3D activation maps over a reference background image (e.g., CTRL-117).

# Results & Evaluation
The generated maps represent the general decision rules learned over the entire training set, making them inherently global.

## Classification results
In the following we report the ROC Curve plot of our models:

<div align="center">

| **SVM AD vs CTRL ROC Curve** |
|--------------------
| <img src="Python_Results\SVM_Classification_Results\Plots\SVM_ROC.png" alt="Raw" width="600"> |

</div>


## Thresholding and ROI Aggregation
Before comparison, specific thresholds were applied to isolate relevant voxels based on the nature of the maps:
* **Continuous Maps ([Haufe](http://dx.doi.org/10.1016/j.neuroimage.2013.10.067) and Raw Weights):** We selected the top 1% (over 99th percentile) and top 5% (over 95th percentile) of raw weights and Haufe activation patterns by absolute value to isolate clusters with the greatest feature importance.
<div align="center">

| **SVM Raw Weights (Top 1% - Fold 3)** |
|--------------------
| <img src="Python_Results\SVM_XAI_Results\Plots\SVM_RawWeights_Fold_3_Top1.png" alt="Raw" width="600"> |

</div>

<div align="center">

| **Haufe Transformation (Top 1% - Fold 3)** |
|--------------------
| <img src="Python_Results\SVM_XAI_Results\Plots\SVM_Haufe_Fold_3_Top1.png" alt="Raw" width="600"> |

</div>

* **Statistical Maps ([Gaonkar](https://doi.org/10.1016/j.media.2015.06.008)):** Based on a strict statistical approach, we utilized an alpha = 0.05 for the Bonferroni method and q = 0.1 for the Benjamini-Yekutieli method (FDR).

<div align="center">

| **Gaonkar Transformation (Bonferroni 0.05 - Fold 3)** |
|--------------------
| <img src="Python_Results\SVM_XAI_Results\Plots\SVM_Gaonkar_Fold_3_bonf005.png" alt="Raw" width="600"> |

</div>

To properly compare the various generated maps with the VBM, we performed an aggregation within Regions of Interest (ROI) using the SPM Neuromorphometric atlas. 

## Heat Map & Feature Importance
The feature importance — used for the construction of global heatmaps and the correlation matrix based on the NDCG metric—is calculated by considering mean of the **absolute value** of the attributions, as our global models generate both positive and negative attributes. Due to graphical restraints in the NDCG correlation matrix, the numerical values have been added in a .csv file in the `Python_Results\XAI_Comparison_Results\Plots` folder, with a range of [0.68, 1] of the normalized score.

*Visual Comparison of XAI outputs (SVM vs VBM):*
<div align="center">
<img src="Python_Results\XAI_Comparison_Results\Plots\Heatmap_AllFolds.png" width="800"> 
</div>

*nDCG Correlation Matrix (SVM vs VBM):*
<div align="center">
<img src="Python_Results\XAI_Comparison_Results\Plots\nDCG_Correlation_Matrix_Extended.png" width="800"> 
</div>

# Usage
Due to the dimension of the model's weights we were not able to upload them on github. To avoid the problem and make the user able to test the code using our tuned results we uploaded them at the following link: [Weights](https://drive.google.com/drive/folders/1vJJEUFFgmrWvrM_9QjdptaV1J5LFpE6y?usp=sharing). The user has to move the folder to the `\Python_Results\SVM_Classification_Results` folder and replace the `Results` folder before running the code.

In case you are running this code for the first time remember to install the requirements:

```bash
pip install -r requirements.txt 
```

In order to run the Python pipeline as intended, you need to pass the functions with this order:
```bash
cd CMEPDA_project_2024/CMEPDA_project_2024/Python
python main.py -set -svm -xai -cv 0.0001 0.001 -xc    
```

To execute the MATLAB pipeline, you can run the following commands in the MATLAB Command Window:
```bash
% Execute the standard VBM Pipeline
cd CMEPDA_project_2024/CMEPDA_project_2024/MATLAB
main('runVBM', true)
```
```bash
% Execute the Mask Comparison
cd CMEPDA_project_2024/CMEPDA_project_2024/MATLAB
main('runMaskComp', true)
```

Refer to help for the different sections of the Matlab and Python orchestrators:
```bash
>> help main
  MAIN Entry point for the MATLAB pipeline.
  Orchestrates the RunVBMPipeline and RunMaskComparison scripts.

  Options (Name-Value arguments):
    'runVBM'              (logical) Execute the VBM Pipeline. Default: false
    'runMaskComp'         (logical) Execute the Mask Comparison script. Default: false
    'enableFileLogging'   (logical) Enable writing .log files to disk. Default: false
    'outputDir'           (char)    Target root directory for outputs. Default: 'MATLAB_Results'
    'inputDir'            (char)    Source directory containing NIfTI/CSV. Default: 'AD_CTRL'
    'csvName'             (char)    Name of the clinical covariate CSV file. Default: 'covariateADCTRLsexAgeTIV.csv'

```
```bash
python main.py --help
usage: main.py [-h] [-log] [-out OUTPUT_DIR] [-in INPUT_DIR] [-csv CSV_NAME] [-set] [-svm] [-xai] [-up] [-bg] [-cv C_VALUES [C_VALUES ...]] [-xc] [-of OUTER_FOLDS] [-inf INNER_FOLDS]

This script performs Alzheimer's Disease vs Healthy Control binary classification using Support Vector Machine and tests the interpretative capabilites of Haufe transformation and Gaonkar statistics against VBM analysis and SVM's raw weights making a qualitative and quantitative analysis.

options:
  -h, --help            show this help message and exit
  -log, --enable-file-logging
                        Enable writing .log files to disk for all executed modules (Default: False).
  -out OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Target root directory for all outputs (Default: Python_Results).
  -in INPUT_DIR, --input-dir INPUT_DIR
                        Source directory containing NIfTI and CSV files (Default: AD_CTRL in project root).
  -csv CSV_NAME, --csv-name CSV_NAME
                        Name of the clinical covariate CSV file (Default: covariateADCTRLsexAgeTIV.csv in AD_CTRL directory).
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

# References

The original dataset comes from the Alzheimer Dataset used in the following paper:
- **Retico, A., Bosco, P., Cerello, P., Fiorina, E., Chincarini, A., & Fantacci, M. E.** (2014). Predictive Models Based on Support Vector Machines: Whole-Brain versus Regional Analysis of Structural MRI in the Alzheimer's Disease. Journal of Neuroimaging, 25(4), 552–563. [DOI](https://doi.org/10.1111/jon.12163).

The Machine Learning Model and the Explainable-AI approaches comes from the following articles:

- **Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.** (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96–110. [DOI](http://dx.doi.org/10.1016/j.neuroimage.2013.10.067)

- **Gaonkar, B., Russell, T. S., Davatzikos, C.** (2015). Interpreting support vector machine models for multivariate group wise analysis in neuroimaging. Medical Image Analysis, 24, 1, 190-204. [DOI](https://doi.org/10.1016/j.media.2015.06.008)

- **Bloch, L., & Friedrich, C. M.** (2024). Systematic comparison of 3D Deep learning and classical machine learning explanations for Alzheimer's Disease detection. Computers in Biology and Medicine, 170. [DOI](https://doi.org/10.1016/j.compbiomed.2024.108029)

Stastistical Parametric Mapping (SPM):

- **Tierney et al.,** (2025). SPM 25: open source neuroimaging analysis software. Journal of Open Source Software, 10(110), 8103, [DOI](https://doi.org/10.21105/joss.08103)