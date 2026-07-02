"""
Unit Testing Suite for SVM Predictive Utilities.

This module validates the static methods of the `SVMUtils` class. It includes tests for:
1. Mathematical correctness of clinical evaluation metrics.
2. Safety mechanisms for extreme edge cases (e.g., single-class AUC calculation).
3. The structural integrity of the GridSearchCV tuning output.
4. I/O Data Loading logic, utilizing `unittest.mock` to safely simulate file system interactions
   (CSV and NIfTI files) without requiring actual neuroimaging datasets on disk.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC

# Import the updated static utility class
from CMEPDA_project_2024.Models.svm_utils import SVMUtils


class TestSVMUtils(unittest.TestCase):
    """Test suite covering the mathematical and I/O contracts of the Linear SVM engine."""
    
    def test_evaluate_classification_standard(self):
        """
        Validates the correct computation of standard clinical classification metrics.
        
        Purpose:
            Ensures that the confusion matrix properly translates into Sensitivity, 
            Specificity, and Accuracy.
        """
        # Ground Truth: 2 Controls (0), 2 Alzheimer's Patients (1)
        y_true = np.array([0, 0, 1, 1])
        # Predictions: 1 False Positive (Index 1)
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.9, 0.85])

        metrics = SVMUtils.evaluate_classification(y_true, y_pred, y_prob)

        # Expected Confusion Matrix: True Positives=2, False Negatives=0, False Positives=1, True Negatives=1
        self.assertEqual(metrics['Accuracy'], 0.75, "Accuracy should be 3/4 (0.75)")
        self.assertEqual(metrics['Sensitivity'], 1.0, "Sensitivity (Recall) must be 1.0 (2/2)")
        self.assertEqual(metrics['Specificity'], 0.5, "Specificity must be 0.5 (1/2)")

    def test_evaluate_single_class_auc_safety(self):
        """
        Verifies the fail-safe mechanism during Area Under the ROC Curve (AUROC) computation.
        
        Purpose:
            If a test fold randomly ends up containing only one class, Scikit-Learn throws 
            a ValueError. This test ensures the utility gracefully catches the error and returns NaN.
        """
        y_true = np.array([1, 1, 1, 1]) # Extreme edge case: Only positive samples
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.9, 0.85])
        
        metrics = SVMUtils.evaluate_classification(y_true, y_pred, y_prob)
        
        self.assertTrue(math.isnan(metrics['AUROC']), "AUROC must return NaN if target variance is zero")

    def test_tune_hyperparameters(self):
        """
        Ensures the GridSearchCV inner loop correctly extracts the optimal parameter and the fitted model.
        """
        X_train = np.random.randn(20, 10) # Mock 20 patients, 10 features
        y_train = np.array([0] * 10 + [1] * 10)
        param_grid = {'C': [0.1, 1.0]}
        
        best_c, best_model = SVMUtils.tune_hyperparameters(X_train, y_train, param_grid, inner_folds=2, random_state=42)

        self.assertIn(best_c, [0.1, 1.0], "Extracted parameter 'C' must belong to the provided grid")
        self.assertIsInstance(best_model, SVC, "Output model must be a valid Scikit-Learn SVC instance")

    @patch('svm_utils.os.path.exists')
    @patch('svm_utils.pd.read_csv')
    @patch('svm_utils.nib.load')
    def test_load_real_data_mocked(self, mock_nib_load, mock_read_csv, mock_exists):
        """
        Mocks filesystem interactions to validate the 3D-to-1D vector flattening logic.
        
        Purpose:
            To test `load_real_data` without requiring massive NIfTI files. 
            Mocks intercept `pandas` and `nibabel` calls to inject synthetic volumetric data.
        """
        # 1. Mock file existence checks to always pass
        mock_exists.return_value = True
        
        # 2. Mock the CSV registry with 2 dummy patients
        mock_df = pd.DataFrame({
            'subject_id': ['SUB_01', 'SUB_02'],
            'file_path': ['fake_path_1.nii', 'fake_path_2.nii'],
            'label': [0, 1]
        })
        mock_read_csv.return_value = mock_df
        
        # 3. Mock the Nibabel NIfTI loader
        mock_img = MagicMock()
        # Create a synthetic 3D volume (10x10x10) filled with 1s.
        mock_img.get_fdata.return_value = np.ones((10, 10, 10))
        mock_nib_load.return_value = mock_img
        
        # Execute the function with dummy paths (intercepted by decorators)
        subjects, X_data, y_data = SVMUtils.load_real_data("dummy.csv", "dummy_mask.nii")
        
        # Assertions
        self.assertEqual(len(subjects), 2, "Should extract exactly 2 subjects based on the mocked CSV")
        # Since the mock mask is filled with 1s (True), all 10*10*10 = 1000 voxels should be extracted and flattened
        self.assertEqual(X_data.shape, (2, 1000), "3D volume must be flattened to a 1D vector of 1000 elements")
        self.assertEqual(list(y_data), [0, 1], "Labels must be correctly parsed as integers")


if __name__ == '__main__':
    unittest.main()