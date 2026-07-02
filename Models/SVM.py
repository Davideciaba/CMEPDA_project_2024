"""
SVM Predictive Engine and Main Execution Script.

This module houses the `SVMPredictiveEngine` class, which manages the Nested Cross-Validation 
routing. It acts as the primary entry point for executing the classical Machine Learning branch 
of the project, utilizing the static methods defined in `svm_utils.py`.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold

from py_logger import CustomLogger
from CMEPDA_project_2024.Models.svm_utils import SVMUtils

# --- GLOBAL CONFIGURATION CONSTANTS ---
DEFAULT_RANDOM_STATE = 42
DEFAULT_INNER_FOLDS = 5
DEFAULT_OUTER_FOLDS = 5
SVM_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class SVMPredictiveEngine:
    """
    Orchestration Engine for Double Cross-Validation using a Linear SVM.
    
    This class handles the macro-architecture of the training process. It generates 
    the external data splits and iterates through them, delegating the heavy mathematical 
    lifting (tuning, metric calculation) to the `SVMUtils` static class.
    """

    def __init__(self, logger: Any, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        """
        Initializes the Orchestrator with CV parameters.
        
        Parameters:
            logger (Any): The injected custom logger instance for unified tracing.
            inner_folds (int): Number of folds for hyperparameter tuning (Inner Loop).
            outer_folds (int): Number of folds for model generalization evaluation (Outer Loop).
        """
        self.logger = logger
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.param_grid = {'C': SVM_C_GRID}

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes the exhaustive Double CV training and evaluation pipeline.
        
        Purpose:
            Provides a completely unbiased estimate of the model's predictive power. 
            By hiding the test set from both the model training AND the hyperparameter tuning 
            phases, it perfectly simulates the model's deployment in a real clinical scenario.

        Parameters:
            X (np.ndarray): 2D Feature matrix.
            y (np.ndarray): 1D Target array.
            subjects (np.ndarray): 1D Subject identifiers to track patient-level predictions.

        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 
                - df_metrics: Pandas DataFrame containing aggregated classification metrics per fold.
                - artifacts: A list of dictionaries preserving optimal parameters and raw probabilities, 
                             essential for downstream Explainable AI (XAI) extraction.
        
        Raises:
            ValueError: If the dataset does not meet the minimum variance requirements (e.g., single class).
        """
        if len(np.unique(y)) < 2:
            raise ValueError("Target array 'y' lacks variance. Must contain at least two classes.")

        self.logger.info(f"Starting SVM Nested CV: {self.outer_folds} Outer Folds, {self.inner_folds} Inner Folds.")
        
        # WHY: StratifiedKFold guarantees that the ratio of AD to CTRL subjects remains 
        # strictly constant across all generated folds, preventing severe class imbalances.
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        
        fold_metrics_list = []
        fold_artifacts = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{self.outer_folds} ---")
            
            # 1. Data Partitioning for the current Outer Fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subjects = subjects[test_idx]

            # 2. Inner CV: Hyperparameter Tuning (Delegated to Utility)
            # Find the best 'C' using ONLY the X_train dataset.
            best_c, best_model = SVMUtils.tune_hyperparameters(
                X_train, y_train, self.param_grid, self.inner_folds, DEFAULT_RANDOM_STATE
            )
            self.logger.debug(f"Fold {fold_idx} tuning complete. Optimal C: {best_c}")

            # 3. Predictive Inference on unseen Out-of-Sample data
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            # 4. Metric Evaluation (Delegated to Utility)
            metrics = SVMUtils.evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            # 5. Artifact Preservation
            # WHY: Saving these artifacts allows us to completely decouple the predictive 
            # phase from the XAI phase. We can load these probabilities later without retraining.
            fold_artifacts.append({
                'fold_id': fold_idx,
                'optimal_C': best_c,
                'test_subjects': test_subjects,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            })

        self.logger.info("Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts


if __name__ == "__main__":
    # Main execution block
    logger = CustomLogger(enable_file_logging=False, level="DEBUG")
    
    with logger.context(session_id="SVM_Predictive_Run"):
        logger.info("--- Starting SVM Predictive Pipeline ---")
        
        # --- Environment Configuration ---
        USE_DUMMY_DATA = True
        CSV_DATASET_PATH = "data/dataset_info.csv"
        MASK_PATH = "data/gm_mask_MNI.nii.gz"
        
        if USE_DUMMY_DATA:
            logger.info("Generating DUMMY DATA for standalone testing...")
            N_SAMPLES, N_FEATURES = 100, 5000
            np.random.seed(DEFAULT_RANDOM_STATE)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            X_data = np.random.randn(N_SAMPLES, N_FEATURES)
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            
            # WHY: Injecting a constant scalar to the target class forces the random data 
            # to be linearly separable. Without this, the SVM tuning grid search would fail to find 
            # meaningful gradients, resulting in 50% accuracy (coin flip).
            X_data[y_data == 1] += 0.5 
            logger.success("Dummy Data successfully generated.")
        else:
            logger.info(f"Loading REAL DATA from {CSV_DATASET_PATH}...")
            subjects, X_data, y_data = SVMUtils.load_real_data(CSV_DATASET_PATH, MASK_PATH)
            logger.success("Real NIfTI data loaded and flattened successfully.")

        logger.debug(f"Feature Matrix Shape: {X_data.shape}. Class Distribution: {np.bincount(y_data)}")

        # --- Engine Execution ---
        engine = SVMPredictiveEngine(logger=logger)
        df_metrics, artifacts = engine.execute_nested_cv(X_data, y_data, subjects)
        
        logger.success("SVM Pipeline execution completed. Results overview:")
        print("\n--- FINAL SVM METRICS (OUT-OF-SAMPLE) ---")
        print(df_metrics.to_string(index=False))