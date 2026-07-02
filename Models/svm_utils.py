"""
SVM Utilities Module.

This module encapsulates all functional primitives required for the Linear SVM pipeline.
It utilizes a static utility class (`SVMUtils`) to ensure a strict Object-Oriented design
without the overhead of instantiating stateful objects for pure mathematical or I/O operations.
Designed to be fully compatible with automated documentation generators (e.g., Sphinx).
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix
)

class SVMUtils:
    """
    Static utility class for Support Vector Machine (SVM) operations.
    
    This class provides static methods for NIfTI data extraction, hyperparameter 
    optimization via Grid Search, and safe statistical metric evaluation. It is designed 
    to be strictly stateless.
    """

    @staticmethod
    def load_real_data(csv_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads 3D NIfTI neuroimaging files, applies a spatial Gray Matter mask, 
        and flattens the resulting volumetric data into 1D feature vectors suitable for Linear SVMs.
        
        Purpose:
            Standard SVMs cannot process 3D/4D tensors natively. This function performs 
            'Boolean Indexing' using a brain mask to discard background noise (skull, air) 
            and flattens only the informative gray matter voxels, drastically reducing RAM usage.
            
        Parameters:
            csv_path (str): Absolute or relative path to the CSV dataset registry. 
                            Must contain columns: 'subject_id', 'file_path', 'label'.
            mask_path (str): Path to the 3D binary mask NIfTI file (e.g., MNI template).
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - subjects: 1D array of subject identifier strings.
                - X_data: 2D array (N_samples, N_features) containing flattened masked voxels.
                - y_data: 1D array of binary integer labels (0 for CTRL, 1 for AD).
                
        Raises:
            FileNotFoundError: If either the CSV registry or the NIfTI mask cannot be located on disk.
        """
        if not os.path.exists(csv_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing Data files. Verify paths:\nCSV: {csv_path}\nMask: {mask_path}")

        df = pd.read_csv(csv_path)
        
        # WHY: Loading the mask once and keeping it as a boolean array allows us to use 
        # advanced NumPy indexing on every patient volume without looping over three axes.
        mask_bool = nib.load(mask_path).get_fdata() > 0
        
        subjects, X_list, y_list = [], [], []
        
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            
            img_data = nib.load(row['file_path']).get_fdata()
            
            # WHY: img_data[mask_bool] extracts only elements where mask_bool is True.
            # This automatically collapses the 3D volume into a 1D array of valid voxels.
            masked_flat_vector = img_data[mask_bool]
            X_list.append(masked_flat_vector)
            
        return np.array(subjects), np.array(X_list), np.array(y_list)

    @staticmethod
    def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, list], inner_folds: int, random_state: int) -> Tuple[float, SVC]:
        """
        Executes an internal Grid Search Cross-Validation to identify the optimal SVM 
        regularization parameter (C) for a specific training subset.
        
        Purpose:
            To prevent data leakage during hyperparameter tuning. This function represents 
            the 'Inner Loop' of the Nested CV architecture. It isolates the hyperparameter 
            selection from the final out-of-sample evaluation.
            
        Parameters:
            X_train (np.ndarray): 2D feature matrix for the current outer fold.
            y_train (np.ndarray): 1D array of target labels.
            param_grid (Dict[str, list]): Dictionary specifying the parameter space (e.g., {'C': [...]}).
            inner_folds (int): Number of splits for the internal cross-validation.
            random_state (int): Seed for deterministic fold generation.
            
        Returns:
            Tuple[float, SVC]: 
                - The optimal floating-point value for the 'C' parameter.
                - The scikit-learn SVC model fitted on the entire X_train using the optimal 'C'.
        """
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        
        # WHY: 'class_weight=balanced' is critical in medical datasets to heavily penalize 
        # misclassifications of the minority class (e.g., rare diseases).
        # 'probability=True' enables Platt scaling, required to compute the AUROC later.
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True)
        
        grid_search = GridSearchCV(
            estimator=svm_base, 
            param_grid=param_grid, 
            cv=inner_cv, 
            scoring='balanced_accuracy', 
            n_jobs=-1 # Utilize all available CPU cores for concurrent tuning
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_['C'], grid_search.best_estimator_

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Computes standard clinical classification metrics, providing fail-safes for mathematical edge cases.
        
        Purpose:
            To evaluate model performance safely. Small or highly imbalanced folds can cause 
            divisions by zero (e.g., 0 True Positives + 0 False Negatives) or trigger 
            ValueErrors in AUROC calculation if only one class is present in the test set.
            
        Parameters:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred (np.ndarray): Predicted discrete labels (0 or 1).
            y_prob (np.ndarray): Predicted continuous probabilities for the positive class (1).
            
        Returns:
            Dict[str, float]: A dictionary containing 'Accuracy', 'Balanced_Accuracy', 
                              'F1_Score', 'Sensitivity', 'Specificity', and 'AUROC'.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # WHY: Safe division logic. If the denominator is zero, it defaults to 0.0 instead of crashing.
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            # WHY: roc_auc_score raises ValueError if y_true contains only one class.
            # Returning NaN ensures the pipeline continues running and the anomaly is flagged.
            auc_score = float('nan')

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUROC': auc_score
        }