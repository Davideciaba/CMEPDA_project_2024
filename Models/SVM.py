"""
Predictive Linear SVM Engine Module.

This module houses the fully encapsulated Support Vector Machine pipeline.
It incorporates decoupled atomic methods for training (GridSearch tuning) 
and pure inference, ensuring API uniformity with the Deep Learning ecosystem.

Designed as a pure library module without global execution blocks.
"""
import os
import sys
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

# --- Path Injection ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from py_logger import CustomLogger

# --- GLOBAL CONFIGURATION CONSTANTS ---
DEFAULT_RANDOM_STATE = 42
DEFAULT_INNER_FOLDS = 5
DEFAULT_OUTER_FOLDS = 5
SVM_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class SVMPredictiveEngine:
    """
    Monolithic Orchestration Engine for Double Cross-Validation using a Linear SVM.
    Encapsulates all mathematical processing operations and decoupled inference APIs.
    """

    def __init__(self, logger: Any, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        self.logger = logger
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.param_grid = {'C': SVM_C_GRID}
        self.random_state = DEFAULT_RANDOM_STATE

    @staticmethod
    def load_real_data(csv_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads NIfTI volumes, extracts valid voxels via mask, and flattens to 1D."""
        if not os.path.exists(csv_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing Data files:\nCSV: {csv_path}\nMASK: {mask_path}")

        df = pd.read_csv(csv_path)
        mask_bool = nib.load(mask_path).get_fdata() > 0
        
        subjects, X_list, y_list = [], [], []
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            img_data = nib.load(row['file_path']).get_fdata()
            X_list.append(img_data[mask_bool])
            
        return np.array(subjects), np.array(X_list), np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Computes clinical metrics safely, guarding against mathematical edge-cases."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = float('nan')

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUROC': auc_score
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, SVC]:
        """
        Unified Training API.
        Executes the Inner Loop tuning to find the optimal 'C' regularization coefficient 
        and fits the final estimator on the full training fold.
        """
        inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True)
        
        grid_search = GridSearchCV(
            estimator=svm_base, 
            param_grid=self.param_grid, 
            cv=inner_cv, 
            scoring='balanced_accuracy', 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_['C'], grid_search.best_estimator_

    def predict(self, model: SVC, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure Inference API (Deploy & XAI Ready).
        Accepts a trained model and a feature matrix, returning discrete predictions 
        and continuous probabilities safely.
        """
        # WHY: Wrapping scikit-learn's native methods provides a uniform interface 
        # identical to the Deep Learning engine, allowing polymorphic orchestration.
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return y_pred, y_prob

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Orchestrates the macro Double Cross-Validation pipeline."""
        if len(np.unique(y)) < 2:
            raise ValueError("Target array 'y' lacks variance. Binary classes required.")

        self.logger.info(f"Starting SVM Nested CV: {self.outer_folds} Outer Folds, {self.inner_folds} Inner Folds.")
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics_list, fold_artifacts = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{self.outer_folds} ---")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # --- Training Phase ---
            best_c, best_model = self.train(X_train, y_train)
            
            # --- Inference Phase ---
            y_pred, y_prob = self.predict(best_model, X_test)

            # --- Evaluation Phase ---
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            fold_artifacts.append({
                'fold_id': fold_idx, 'optimal_C': best_c, 'test_subjects': subjects[test_idx],
                'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob
            })

        self.logger.info("Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts