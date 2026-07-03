"""
Unit Testing Suite for the Monolithic SVM Predictive Engine.

This module validates the encapsulated methods within the `SVMPredictiveEngine` class.
It leverages `unittest.mock` to bypass actual hardware I/O and isolates mathematical 
evaluation boundaries (like AUROC edge-cases).
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC

# Import the monolithic Engine
from SVM import SVMPredictiveEngine


class TestSVMEngine(unittest.TestCase):
    """Test suite covering the mathematical and I/O contracts of the Linear SVM engine."""
    
    def setUp(self):
        """Initializes the Engine with a Mock Logger for silent, rapid testing."""
        self.mock_logger = MagicMock()
        # Instantiate with minimal folds to speed up testing
        self.engine = SVMPredictiveEngine(logger=self.mock_logger, inner_folds=2, outer_folds=2)

    def test_evaluate_classification_standard(self):
        """Validates the correct computation of standard clinical classification metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.9, 0.85])

        # Testing the protected internal evaluation method
        metrics = self.engine._evaluate_classification(y_true, y_pred, y_prob)

        self.assertEqual(metrics['Accuracy'], 0.75, "Accuracy should be 3/4 (0.75)")
        self.assertEqual(metrics['Sensitivity'], 1.0, "Sensitivity (Recall) must be 1.0 (2/2)")
        self.assertEqual(metrics['Specificity'], 0.5, "Specificity must be 0.5 (1/2)")

    def test_evaluate_single_class_auc_safety(self):
        """Verifies the fail-safe mechanism during Area Under the ROC Curve computation."""
        y_true = np.array([1, 1, 1, 1]) # Extreme edge case: Only positive samples
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.9, 0.85])
        
        metrics = self.engine._evaluate_classification(y_true, y_pred, y_prob)
        self.assertTrue(math.isnan(metrics['AUROC']), "AUROC must return NaN if target variance is zero")

    def test_tune_hyperparameters(self):
        """Ensures the internal Grid Search correctly extracts the optimal parameters."""
        X_train = np.random.randn(20, 10)
        y_train = np.array([0] * 10 + [1] * 10)
        
        # Override parameter grid for a faster micro-test
        self.engine.param_grid = {'C': [0.1, 1.0]}
        
        best_c, best_model = self.engine._tune_hyperparameters(X_train, y_train)

        self.assertIn(best_c, [0.1, 1.0], "Extracted 'C' must belong to the testing grid")
        self.assertIsInstance(best_model, SVC, "Output model must be a Scikit-Learn SVC")

    @patch('SVM.os.path.exists')
    @patch('SVM.pd.read_csv')
    @patch('SVM.nib.load')
    def test_load_real_data_mocked(self, mock_nib_load, mock_read_csv, mock_exists):
        """Mocks filesystem interactions to validate the 3D-to-1D Boolean vector flattening logic."""
        mock_exists.return_value = True
        
        # Mock CSV registry
        mock_df = pd.DataFrame({
            'subject_id': ['SUB_01', 'SUB_02'],
            'file_path': ['fake_path_1.nii', 'fake_path_2.nii'],
            'label': [0, 1]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock Nibabel to return a 10x10x10 cube of 1s (all true for masking)
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.ones((10, 10, 10))
        mock_nib_load.return_value = mock_img
        
        # Execute static method on the Class
        subjects, X_data, y_data = SVMPredictiveEngine.load_real_data("dummy.csv", "dummy_mask.nii")
        
        self.assertEqual(len(subjects), 2)
        self.assertEqual(X_data.shape, (2, 1000), "3D volume must be flattened to a 1D vector of 1000 elements")


if __name__ == '__main__':
    unittest.main()