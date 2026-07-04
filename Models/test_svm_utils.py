"""
Unit and Integration Testing Suite for the SVM Predictive Engine.
Validates encapsulated methods, decoupled APIs (train/predict), and executes E2E pipelines.
Designed for a flat directory structure (tests and models in the same folder).
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC

from SVM import SVMPredictiveEngine
from py_logger import CustomLogger

class TestSVMEngine(unittest.TestCase):
    
    def setUp(self):
        """Initializes the Engine with a Mock Logger and reduced fold parameters."""
        self.logger = CustomLogger(name="TestSVM")
        self.logger.add_console_handler(level="INFO")
        self.engine = SVMPredictiveEngine(logger=self.logger, inner_folds=2, outer_folds=2)

    def test_evaluate_classification_standard(self):
        """Validates the mathematical calculation of clinical metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.9, 0.85])

        metrics = self.engine._evaluate_classification(y_true, y_pred, y_prob)
        self.assertEqual(metrics['Accuracy'], 0.75)
        self.assertEqual(metrics['Sensitivity'], 1.0)

    def test_evaluate_single_class_auc_safety(self):
        """Verifies AUROC fallback mechanism for zero-variance data folds."""
        y_true = np.array([1, 1, 1, 1]) 
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.9, 0.85])
        
        metrics = self.engine._evaluate_classification(y_true, y_pred, y_prob)
        self.assertTrue(math.isnan(metrics['AUROC']))

    def test_train_method(self):
        """Ensures the decoupled training API correctly performs Grid Search and fitting."""
        X_train = np.random.randn(20, 10)
        y_train = np.array([0] * 10 + [1] * 10)
        
        self.engine.param_grid = {'C': [0.1, 1.0]}
        best_c, best_model = self.engine.train(X_train, y_train)

        self.assertIn(best_c, [0.1, 1.0])
        self.assertIsInstance(best_model, SVC)

    def test_predict_method(self):
        """Validates the pure inference API structure and array returns."""
        mock_model = MagicMock(spec=SVC)
        mock_model.predict.return_value = np.array([0, 1, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
        
        X_test = np.random.randn(3, 10) 
        
        y_pred, y_prob = self.engine.predict(mock_model, X_test)
        
        self.assertEqual(len(y_pred), 3)
        self.assertEqual(y_prob[1], 0.8, "Probability extraction failed to target the positive class.")

    # I target dei patch ora puntano al namespace locale "SVM"
    @patch('SVM.os.path.exists')
    def test_load_real_data_file_missing(self, mock_exists):
        """Validates that the Engine correctly aborts if data files are missing."""
        mock_exists.return_value = False 
        
        with self.assertRaises(FileNotFoundError):
            SVMPredictiveEngine.load_real_data("missing.csv", "missing_mask.nii")

    @patch('SVM.os.path.exists')
    @patch('SVM.pd.read_csv')
    @patch('SVM.nib.load')
    def test_load_real_data_success(self, mock_nib_load, mock_read_csv, mock_exists):
        """Mocks the OS file system to validate structural 3D-to-1D flattening and parsing."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            'subject_id': ['SUB_01', 'SUB_02'], 
            'file_path': ['fake1.nii', 'fake2.nii'], 
            'label': [1, 0]
        })
        
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.ones((10, 10, 10)) 
        mock_nib_load.return_value = mock_img
        
        subjects, X_data, y_data = SVMPredictiveEngine.load_real_data("dummy.csv", "mask.nii")
        
        self.assertEqual(X_data.shape, (2, 1000))
        self.assertEqual(len(subjects), 2)
        self.assertEqual(subjects[0], "SUB_01")
        self.assertEqual(y_data[0], 1)

    def test_execute_nested_cv_integration(self):
        """Integration Test: Ensures the decoupled Double CV pipeline executes end-to-end."""
        with self.logger.context(session_id="SVM_Integration"):
            N_SAMPLES, N_FEATURES = 20, 50
            np.random.seed(42)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            X_data = np.random.randn(N_SAMPLES, N_FEATURES)
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            
            X_data[y_data == 1] += 0.5 
            self.engine.param_grid = {'C': [0.1, 1.0]}
            
            df_metrics, artifacts = self.engine.execute_nested_cv(X_data, y_data, subjects)
            
            self.assertIsInstance(df_metrics, pd.DataFrame)
            self.assertEqual(len(df_metrics), 2)
            self.assertEqual(len(artifacts), 2)

if __name__ == '__main__':
    unittest.main()