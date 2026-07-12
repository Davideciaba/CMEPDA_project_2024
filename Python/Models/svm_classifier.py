"""
Predictive Linear SVM Engine Module.

This module houses the fully encapsulated Support Vector Machine pipeline.
It incorporates decoupled atomic methods for training (GridSearch tuning) 
and pure inference, ensuring API uniformity with the Deep Learning ecosystem.

Designed as a pure library module without global execution blocks or manual path injections.
"""
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, List, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix,
    roc_curve
)
from Python.utils.py_logger import CustomLogger


class SVMClassifier:
    """
    Monolithic Orchestration Engine for Double Cross-Validation using a Linear SVM.
    Encapsulates all mathematical processing operations and decoupled inference APIs.
    """

    def __init__(self, logger: CustomLogger, param_grid: Dict[str, List[Any]]):
        self.logger = logger
        self.param_grid = param_grid

    @staticmethod
    def load_data(csv_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads NIfTI volumes, extracts valid voxels via mask, and flattens to 1D."""
        
        df = pd.read_csv(csv_path)
        
        # WHY: Pre-loading the mask as a boolean matrix enables vectorized NumPy 
        # indexing, eliminating slow triple nested loops (X, Y, Z).
        mask_bool = nib.load(mask_path).get_fdata() > 0
        
        subjects, X_list, y_list = [], [], []
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            img_data = nib.load(row['file_path']).get_fdata(dtype=np.float32)
            
            # WHY: Boolean Indexing directly extracts the valid values into a 1D vector.
            X_list.append(img_data[mask_bool])
            
        return np.array(subjects), np.array(X_list, dtype=np.float32), np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_decision: np.ndarray) -> Dict[str, float]:
        """Computes clinical metrics safely, guarding against mathematical edge-cases."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_decision)
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, inner_cv_iterator: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float, float, Any]:
        """
        Unified Training API.
        Executes the Inner Loop tuning to find the optimal 'C' regularization coefficient 
        and fits the final estimator on the full training fold.
        Returns:
            - best_c: The optimal hyperparameter
            - mean_cv_bal_acc: Mean balanced accuracy across the Inner Folds for best_c
            - std_cv_bal_acc: Standard deviation of balanced accuracy for best_c
            - best_pipeline: The fitted pipeline object
        """
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear', class_weight='balanced'))
        ])
        
        grid_params = {f"svc__{k}": v for k, v in self.param_grid.items()}
        
        grid_search = GridSearchCV(
            estimator=base_pipeline, 
            param_grid=grid_params, 
            cv=inner_cv_iterator,
            scoring='balanced_accuracy', 
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        
        # Extract hyperparameters and their internal CV statistics
        best_c = grid_search.best_params_['svc__C']
        best_idx = grid_search.best_index_
        mean_cv_bal_acc = grid_search.cv_results_['mean_test_score'][best_idx]
        std_cv_bal_acc = grid_search.cv_results_['std_test_score'][best_idx]
        
        best_pipeline = grid_search.best_estimator_
        
        return best_c, mean_cv_bal_acc, std_cv_bal_acc, best_pipeline

    def predict(self, model: SVC, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure Inference API (Deploy & XAI Ready).
        Accepts a trained model and a feature matrix, returning discrete predictions 
        and continuous probabilities safely.
        """
        y_pred = model.predict(X_test)
        y_decision = model.decision_function(X_test)
        return y_pred, y_decision

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, cv_splits: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Orchestrates the macro Double Cross-Validation pipeline."""

        self.logger.info(f"Starting SVM Nested CV: {len(cv_splits)} Outer Folds, {len(cv_splits[0]['inner_splits_relative'])} Inner Folds.")
        
        fold_metrics_list, fold_artifacts = [], []

        for split in cv_splits:
            fold_idx = split['fold']
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{len(cv_splits)} ---")
            
            # Map Absolute Outer Indices
            train_idx, test_idx = split['outer_train_idx'], split['outer_test_idx']
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Map Relative Inner Indices for GridSearchCV
            inner_iterator = split['inner_splits_relative']
            
            best_c, mean_cv_acc, std_cv_acc, best_model = self.train(X_train, y_train, inner_iterator)
            y_pred, y_decision = self.predict(best_model, X_test)

            metrics = self._evaluate_classification(y_test, y_pred, y_decision)
            metrics['Fold'] = fold_idx
            metrics['Optimal_C'] = best_c
            metrics['Inner_CV_BalAcc_Mean'] = mean_cv_acc
            metrics['Inner_CV_BalAcc_Std'] = std_cv_acc

            fold_metrics_list.append(metrics)

            # Calculate True/False Positive Rates for the Renderer
            fpr, tpr, _ = roc_curve(y_test, y_decision)
            
            fold_artifacts.append({
                'fold_id': fold_idx, 
                'optimal_C': best_c, 
                'test_subjects': subjects[test_idx],
                'y_true': y_test, 
                'y_pred': y_pred, 
                'y_decision': y_decision,
                'roc_fpr': fpr,
                'roc_tpr': tpr,
                'model': best_model,
                'train_idx': train_idx
            })

        self.logger.info("Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts