"""
Module: svm_classifier.py

Linear SVM Classifier.

PURPOSE:
    - Instantiate Linear Support Vector Machine Classifier
    - Perform Double Cross-Validation based on cv_splits in cohort registry
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
    Orchestrator for Double Cross-Validation using a Linear SVM.
    """

    def __init__(self, logger: CustomLogger, param_grid: Dict[str, List[Any]], inner_folds: int = 5):
        """
        Initializes the SVM Engine.
        
        Args:
            logger (CustomLogger): Centralized logging instance.
            param_grid (Dict[str, List[Any]]): The hyperparameter search space (e.g., {'C': [0.001, 0.01]}).
            inner_folds (int): The number of inner folds used for internal hyperparameter tuning.
        """
        self.logger = logger
        self.param_grid = param_grid
        self.inner_folds = inner_folds

    @staticmethod
    def load_data(csv_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads NIfTI volumes, extracts valid voxels via mask, and flattens to 1D.
            
        Args:
            csv_path (str): Absolute path to the normalized cohort CSV.
            mask_path (str): Absolute path to the binarized TPM Mask.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - subjects: Array of subject IDs.
                - X_list: Flattened 2D feature matrix (Samples x Voxels).
                - y_list: Target binary labels array.
        """
        
        df = pd.read_csv(csv_path)
        mask_bool = nib.load(mask_path).get_fdata() > 0
        
        subjects, X_list, y_list = [], [], []
        for _, row in df.iterrows():
            subjects.append(str(row['subject_id']))
            y_list.append(int(row['label']))
            img_data = nib.load(row['file_path']).get_fdata(dtype=np.float32)
            
            X_list.append(img_data[mask_bool])
            
        return np.array(subjects), np.array(X_list, dtype=np.float32), np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_decision: np.ndarray) -> Dict[str, float]:
        """
        Computes clinical metrics.
        
        Args:
            y_true (np.ndarray): True class labels.
            y_pred (np.ndarray): Predicted discrete class labels.
            y_decision (np.ndarray): Continuous decision scores from the SVM.
            
        Returns:
            Dict[str, float]: Dictionary mapping metric names to computed values.
        """
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
        Unified Training method.
        
        PURPOSE:
            Executes the Grid Search to find the optimal 'C' regularization coefficient 
            and fits the final estimator on the full training fold. Uses StandardScaler inside a Pipeline.
            
        Args:
            X_train (np.ndarray): Flattened training feature matrix.
            y_train (np.ndarray): Training labels.
            inner_cv_iterator (List): Pre-computed nested validation topology relative indices.
            
        Returns:
            Tuple[float, float, float, Any]:
                - best_c: The optimal hyperparameter chosen.
                - mean_cv_bal_acc: Mean balanced accuracy across the Inner Folds.
                - std_cv_bal_acc: Standard deviation of balanced accuracy.
                - best_pipeline: The fitted sklearn Pipeline object.
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
        Inference method
        
        PURPOSE:
            Accepts a trained model and a feature matrix, returning discrete 
            predictions and continuous probabilities.
            
        Args:
            model (SVC): The trained sklearn pipeline.
            X_test (np.ndarray): The outer fold feature matrix to evaluate.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Discrete predictions and decision scores.
        """
        y_pred = model.predict(X_test)
        y_decision = model.decision_function(X_test)
        return y_pred, y_decision

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, cv_splits: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Orchestrates the Double Cross-Validation pipeline.
        
        PURPOSE:
            Iterates through the frozen CV topology, tuning on the 
            inner folds and  evaluation on the outer folds.
            
        Args:
            X (np.ndarray): The complete flattened cohort tensor.
            y (np.ndarray): Complete binary label array.
            subjects (np.ndarray): Subject IDs corresponding to rows in X.
            cv_splits (List[Dict]): The frozen evaluation topology computed by CVManager.
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]:
                - fold_metrics_list: DataFrame capturing all clinical metrics per fold.
                - fold_artifacts: List of dictionaries packing models and artifacts for XAI/rendering.
        """

        self.logger.info(f"Starting SVM Nested CV: {len(cv_splits)} Outer Folds, {self.inner_folds} Inner Folds.")
        
        fold_metrics_list, fold_artifacts = [], []

        for split in cv_splits:
            fold_idx = split['fold']
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{len(cv_splits)} ---")
            
            # Map absolute outer indecis
            train_idx, test_idx = split['outer_train_idx'], split['outer_test_idx']
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Map relative inner indices for GridSearchCV
            inner_iterator = split['inner_splits_relative']
            
            best_c, mean_cv_acc, std_cv_acc, best_model = self.train(X_train, y_train, inner_iterator)
            y_pred, y_decision = self.predict(best_model, X_test)

            metrics = self._evaluate_classification(y_test, y_pred, y_decision)
            metrics['Fold'] = fold_idx
            metrics['Optimal_C'] = best_c
            metrics['Inner_CV_BalAcc_Mean'] = mean_cv_acc
            metrics['Inner_CV_BalAcc_Std'] = std_cv_acc

            fold_metrics_list.append(metrics)

            # Calculate true/false positive rates for the renderer
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