# CMEPDA Project 2024: Alzheimer's Disease 3D Classification & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-SPM2025-orange?logo=mathworks)](https://it.mathworks.com/products/matlab.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-3D_DL-red?logo=pytorch)](https://pytorch.org/)

The aim of this repository is to build and train Machine Learning and Deep Learning models for an image-based medical classification. Specifically, we compare a Linear Support Vector Machine (SVM) and a 3D EfficientNet Convolutional Neural Network (CNN) for the classification of Alzheimer's Disease (AD) versus Healthy Controls (CTRL). Starting from 3D structural MRI, the models infer the clinical diagnosis with a strong emphasis on spatial interpretability and Explainable AI (XAI)[cite: 31]. This project is developed using a strictly synchronized hybrid architecture comprising both Python (OOP) and MATLAB scripts[cite: 31].

# Table of contents
+ [Data & Preprocessing](#data--preprocessing)
+ [Validation Framework](#validation-framework)
+ [Model Building and Training](#model-building-and-training)
  + [VBM (The Ground Truth)](#vbm-the-ground-truth)
  + [Linear SVM & XAI](#linear-svm--xai)
  + [3D Deep Learning & XAI](#3d-deep-learning--xai)
+ [Results & Evaluation](#results--evaluation)
+ [Usage](#usage)

# Data & Preprocessing
The input baseline consists of normalized, modulated, and smoothed Gray Matter (GM) 3D maps of AD and CTRL subjects, registered in the MNI space[cite: 31]. 
All derived maps (VBM, SVM weights-map, SVM Haufe, SVM Gaonkar, DL IG) MUST strictly adhere to the same MNI grid[cite: 31]. To ensure voxel-perfect spatial alignment for downstream XAI evaluations, all maps are masked using an identical boolean GM Mask[cite: 31].

<div align="center">

| **Raw MRI (MNI)** | **Preprocessed GM Mask** |
|--------------------|------------------|
| <img src="Readme_images/raw_mri.png" alt="Raw" width="200"> | <img src="Readme_images/gm_mask.png" alt="Preprocessed" width="200"> |

</div>

# Validation Framework
The architecture is based on Object-Oriented Programming (OOP) in Python and Deep Learning modules are strictly Hardware-Agnostic (`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`)[cite: 31]. 
The validation scheme relies on a Double Cross-Validation (Double CV) design[cite: 31]:
* An Outer K-fold CV (default K=5) for evaluation[cite: 31].
* An Inner K-fold CV (default K=5) for Hyperparameter Tuning[cite: 31].

To prevent data leakage, the Outer K-fold split is generated centrally by the Python Orchestrator, utilizing a stratification strategy based on the AD/CTRL labels[cite: 31]. 
Data scaling is applied dynamically within each fold: `StandardScaler` (or `RobustScaler`) is used for the SVM, while `NormalizeIntensity` (or `ScaleIntensity`) is used for the CNN after calculating the custom mean and standard deviation of the current fold[cite: 31].

# Model Building and Training

## VBM (The Ground Truth)
The Voxel-Based Morphometry (VBM) analysis is utilized purely as an independent epidemiological gold standard[cite: 31].
* **Environment:** Implemented in MATLAB with the SPM2025 toolkit, executed asynchronously via `matlab.engine` (Python orchestrated)[cite: 31]. All MATLAB scripts instantiate a custom Logger for Inter-Process Communication (IPC)[cite: 31].
* **Methodology:** A General Linear Model (GLM) based on a Two-Sample T-test (AD vs CTRL) is computed over the entire dataset[cite: 31]. Age, Sex, and TIV are included as nuisance regressors[cite: 31].
* **Output:** A Global VBM Mask thresholded using False Discovery Rate (FDR) correction[cite: 31].

## Linear SVM & XAI
A Linear Support Vector Machine is trained on the flattened 3D GM maps using Scikit-Learn pipelines[cite: 31]. To bypass the biologically uninterpretable nature of standard SVM backward weights ($W$), we implemented advanced XAI algorithms[cite: 31]:
* **Haufe Transformation:** We compute biological Activation Patterns $A$ via covariance: $A = Cov(X_{train}, \hat{s}) = \frac{1}{N_{train}-1} \sum_{n=1}^{N_{train}} X_{centr., n}^{T} \hat{s}_{centr., n}$[cite: 31].
* **Gaonkar Transformation:** We employ Gaonkar's analytic representation to evaluate the decision patterns of the SVM without performing permutation tests[cite: 31]. This is designed for High-Dimension Low-Sample-Size (HDLSS) regimes (verifying $m/d < 0.2$ and `np.linalg.cond(K) < 10^{4}`)[cite: 31]. Exact p-values are computed and FWE-corrected[cite: 31].

## 3D Deep Learning & XAI
A `3D EfficientNet` architecture is implemented in PyTorch/TensorFlow[cite: 31].
* **Training:** Includes a dynamic Early Stopping method in the Inner CV (based on `best_val_loss` and a `patience_counter`)[cite: 31]. The final model is calculated by applying Polyak-Ruppert Averaging on the parameters of the last N=5 epochs[cite: 31].
* **Interpretability:** We use the Integrated Gradient (IG) algorithm with a zero-matrix baseline to interpret the 3D EfficientNet and extract subject-wise spatial attributions[cite: 31].

# Results & Evaluation
The predictive performances of both models are evaluated on the respective Outer Fold Test Sets[cite: 31]. To quantitatively measure this, we employ: Accuracy, AUROC, Balanced Accuracy (BACC), F1-Score, Sensitivity, and Specificity ($Mean \pm Std$)[cite: 31].

## Heat Map
As part of the analysis, we include the possibility to "visualize" what the model has learnt using a heat map, which highlights the regions of input images which are relevant in the decision making process. 
To quantitatively evaluate whether the Machine Learning representations (SVM and DL) successfully captured the true biological variance (VBM ground truth), we employ the Normalized Discounted Cumulative Gain (NDCG) to evaluate the continuous ranking of voxel importance[cite: 31].

*Visual Comparison of XAI outputs (SVM vs CNN vs VBM):*
<div align="center">
<img src="Readme_images/heat_map.png" width="800"> 
</div>

# Usage
We warmly invite the user to run the code on a GPU, because of its computational cost[cite: 30]. 

Clone this repository and run the main orchestrator using default parameters[cite: 30]:
```bash
cd CMEPDA_project_2024
python main.py
```

# References

The original dataset comes from the Alzheimer Dataset used in the following paper:
- **Retico, A., Bosco, P., Cerello, P., Fiorina, E., Chincarini, A., & Fantacci, M. E.** (2014). Predictive Models Based on Support Vector Machines: Whole-Brain versus Regional Analysis of Structural MRI in the Alzheimer's Disease. Journal of Neuroimaging, 25(4), 552–563. [DOI](https://doi.org/10.1111/jon.12163).

The Machine Learning Models and the Explainable-AI approaches comes from the following articles:

- **Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.** (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96–110. [DOI](http://dx.doi.org/10.1016/j.neuroimage.2013.10.067)

- **Gaonkar, B., & Davatzikos, C.** (2013). Analytic estimation of statistical significance maps for support vector machine based multi-variate image analysis and classification. NeuroImage, 78, 270-283. [DOI](10.1016/j.media.2015.06.008.)

- **Bloch, L., & Friedrich, C. M.** (2024). Systematic comparison of 3D Deep learning and classical machine learning explanations for Alzheimer's Disease detection. Scientific Reports, 14. [DOI](https://doi.org/10.1016/j.compbiomed.2024.108029)

- **Di Wang, et al.** (2023). DL network heatmaps capture AD patterns reported in a large meta-analysis of neuroimaging studies. NeuroImage. [DOI](https://doi.org/10.1016/j.neuroimage.2023.119929.)