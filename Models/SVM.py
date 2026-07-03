"""
Main Execution and Orchestration Script for the Predictive Linear SVM.

This module houses the entire Support Vector Machine pipeline. It is fully 
self-contained, handling spatial data flattening via Gray Matter masks, inner loop 
hyperparameter selection via Grid Search, and unbiased out-of-sample prediction 
via Nested Cross-Validation.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, List, Tuple, Any
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix
)

from py_logger import CustomLogger

# --- GLOBAL CONFIGURATION CONSTANTS ---
DEFAULT_RANDOM_STATE = 42
DEFAULT_INNER_FOLDS = 5
DEFAULT_OUTER_FOLDS = 5
SVM_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class SVMPredictiveEngine:
    """
    Monolithic Orchestration Engine for Double Cross-Validation using a Linear SVM.
    
    Encapsulates all processing operations, including masked data loaders, 
    GridSearch tuning grids, and clinical evaluation statistics.
    """

    def __init__(self, logger: Any, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        """
        Initializes the SVM Engine with specific structural hyperparameters.

        Args:
            logger (Any): Injected instance of the CustomLogger.
            inner_folds (int): Splits for hyperparameter tuning (Inner Loop).
            outer_folds (int): Splits for generalization evaluation (Outer Loop).
        """
        self.logger = logger
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.param_grid = {'C': SVM_C_GRID}
        self.random_state = DEFAULT_RANDOM_STATE

    @staticmethod
    def load_real_data(csv_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads 3D NIfTI neuroimaging volumes, extracts relevant brain voxels using 
        a spatial binary mask, and flattens the result into 1D feature arrays.

        Mathematical Background:
            Standard SVM formulations operate on data tensors of shape (N_samples, N_features). 
            To convert a 3D structural MRI volume (D, H, W) into this format without 
            including meaningless structural noise (skull, air, CSF), we apply spatial masking.

        Steps:
            1. Read the CSV manifest containing patient details.
            2. Extract a Boolean Mask from the MNI NIfTI template (voxels > 0).
            3. For each subject, open the NIfTI scan and index it with the Boolean Mask.
               This instantly drops non-brain voxels and collapses the space to 1D.

        Args:
            csv_path (str): Path to CSV mapping 'subject_id', 'file_path', and 'label'.
            mask_path (str): Path to the 3D binary brain mask NIfTI file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - subjects: String array of anonymized subject identifiers.
                - X_data: 2D Feature matrix of shape (N_samples, N_masked_voxels).
                - y_data: 1D Integer array of ground-truth binary targets (0: CTRL, 1: AD).
        """
        if not os.path.exists(csv_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing Data files:\nCSV: {csv_path}\nMASK: {mask_path}")

        df = pd.read_csv(csv_path)
        
        # WHY: Pre-loading the spatial mask as a boolean matrix enables NumPy 
        # vectorized vector selection, eliminating slow triple nested loops (X, Y, Z).
        mask_bool = nib.load(mask_path).get_fdata() > 0
        
        subjects, X_list, y_list = [], [], []
        
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            
            # Extract voxel values from the patient's structural brain scan
            img_data = nib.load(row['file_path']).get_fdata()
            
            # WHY: Advanced NumPy Boolean Indexing. Collapses the 3D grid into a 1D vector.
            X_list.append(img_data[mask_bool])
            
        return np.array(subjects), np.array(X_list), np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Computes clinical metrics safely, guarding against zero variance anomalies.

        Why this is necessary:
            In clinical validation folds with small validation samples, it is mathematically 
            possible to encounter zero denominators (e.g., zero True Positives and zero 
            False Negatives). This method intercepts the division before it triggers 
            a runtime crash.

        Args:
            y_true (np.ndarray): Ground truth arrays.
            y_pred (np.ndarray): Discrete model classifications.
            y_prob (np.ndarray): Continuous prediction probabilities for the positive class.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # WHY: Explicit fallback handles zero denominators during severe fold splits.
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            # WHY: Thrown by scikit-learn if a fold contains only 1 class during CV splitting.
            auc_score = float('nan')

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUROC': auc_score
        }

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, SVC]:
        """
        Executes the Inner Loop of the Nested CV using a hyperparameter search grid.

        Purpose:
            Selects the best 'C' boundary penalization coefficient without looking at 
            the outer test fold, preventing optimizational data leakage.
        """
        inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
        
        # WHY: class_weight='balanced' automatically recalibrates class weights inversely 
        # proportional to class frequencies, forcing the SVM hyperplane to respect clinical minority groups.
        # probability=True activates Platt scaling via 5-fold cross-validation inside scikit-learn, 
        # which is required to output calibrated continuous probabilities for AUROC calculation.
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True)
        
        grid_search = GridSearchCV(
            estimator=svm_base, 
            param_grid=self.param_grid, 
            cv=inner_cv, 
            scoring='balanced_accuracy', 
            n_jobs=-1 # Parallel processing utilizing all CPU cores concurrently
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_['C'], grid_search.best_estimator_

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Orchestrates the macro Double Cross-Validation pipeline.

        Steps:
            1. Partition dataset into outer training and out-of-sample validation folds.
            2. Pass training folds into the inner tuning loop to find the best 'C'.
            3. Extract out-of-sample probabilities using the locked optimal estimator.
            4. Compute unbiased out-of-sample metrics.
        """
        if len(np.unique(y)) < 2:
            raise ValueError("Target array 'y' lacks variance. Binary classes required.")

        self.logger.info(f"Starting SVM Nested CV: {self.outer_folds} Outer Folds, {self.inner_folds} Inner Folds.")
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics_list, fold_artifacts = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{self.outer_folds} ---")
            
            # Slice outer partition arrays
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Execute Inner Loop optimization
            best_c, best_model = self._tune_hyperparameters(X_train, y_train)
            self.logger.debug(f"Fold {fold_idx} tuning complete. Optimal C: {best_c}")

            # Out-of-sample inference
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            # Parse results
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            # WHY: Saving raw true values, predictions and continuous probabilities 
            # enables downstream XAI evaluation and statistical plotting without retraining.
            fold_artifacts.append({
                'fold_id': fold_idx,
                'optimal_C': best_c,
                'test_subjects': subjects[test_idx],
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            })

        self.logger.info("Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts


if __name__ == "__main__":
    # Self-test block replicating a standalone execution
    logger = CustomLogger()
    logger.add_console_handler(level="DEBUG")
    
    with logger.context(session_id="SVM_Predictive_Run"):
        logger.info("--- Starting SVM Monolithic Pipeline ---")
        
        USE_DUMMY_DATA = True
        if USE_DUMMY_DATA:
            logger.info("Generating DUMMY DATA matrix for mathematical validation...")
            N_SAMPLES, N_FEATURES = 100, 5000
            np.random.seed(DEFAULT_RANDOM_STATE)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            X_data = np.random.randn(N_SAMPLES, N_FEATURES)
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            
            # Induce deliberate linear variance separation to ensure the search grid converges
            X_data[y_data == 1] += 0.5 
            logger.success("Dummy Data matrix initialized.")
        else:
            subjects, X_data, y_data = SVMPredictiveEngine.load_real_data("data/dataset_info.csv", "data/mask.nii")

        engine = SVMPredictiveEngine(logger=logger)
        df_metrics, artifacts = engine.execute_nested_cv(X_data, y_data, subjects)
        
        print("\n--- FINAL OUT-OF-SAMPLE SVM PERFORMANCE ---")
        print(df_metrics.to_string(index=False))