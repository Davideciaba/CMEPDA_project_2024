"""
Module: test_svm.py

Unit and Integration Testing Suite for the SVM Predictive Engine.
Validates encapsulated methods, decoupled APIs (train/predict), and executes 
E2E pipelines using randomized datasets. Designed for a flat directory structure.
"""
import pathlib
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

current_dir= pathlib.Path(__file__).resolve().parent
project_dir= current_dir.parent
python_dir = project_dir / "CMEPDA_project_2024" / "Python" 

# Add the project and Python directory to sys.path
sys.path.append(str(project_dir))
sys.path.append(str(python_dir))


from CMEPDA_project_2024.Python.svm_classifier import SVMClassifier
from CMEPDA_project_2024.Python.utils.py_logger import CustomLogger

class TestSVMEngine(unittest.TestCase):
    """
    Test suite for SVMClassifier.
    
    PURPOSE:
        Validates clinical metric calculations, pipeline, GridSearch 
        execution, and edge-case metric guarding (like single-class AUROC).
    """
    
    def setUp(self) -> None:
        """Initializes the Engine with a Mock Logger and reduced fold parameters."""
        self.logger = CustomLogger(name="TestSVM")
        self.logger.add_console_handler(level="INFO")
        default_param_grid = {'C': [0.1, 1.0]}
        self.engine = SVMClassifier(logger=self.logger, param_grid=default_param_grid)

    def test_evaluate_classification_standard(self) -> None:
        """
        Validates the mathematical calculation of clinical metrics.
        """
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_decision = np.array([0.1, 0.8, 0.9, 0.85])

        metrics = self.engine._evaluate_classification(y_true, y_pred, y_decision)
        self.assertEqual(metrics['Accuracy'], 0.75)
        self.assertEqual(metrics['Sensitivity'], 1.0)

    def test_evaluate_single_class_auc_safety(self) -> None:
        """
        Verifies AUROC fallback mechanism for zero-variance data folds.
        
        PURPOSE:
            If a test set only contains 1 class, standard Scikit-Learn AUC throws 
            a ValueError. This tests the protective try/except block.
        """
        y_true = np.array([1, 1, 1, 1]) 
        y_pred = np.array([1, 1, 1, 1])
        y_decision = np.array([0.9, 0.8, 0.9, 0.85])
                 
        metrics = self.engine._evaluate_classification(y_true, y_pred, y_decision)
        self.assertTrue(math.isnan(metrics['AUROC']))

    def test_train_method(self) -> None:
        """
        Ensures the decoupled training API correctly performs Grid Search and fitting.
        
        PURPOSE:
            Validates that the returned model is a full standard Pipeline 
            and not a vulnerable raw algorithm, thus preventing Data Leakage.
        """
        X_train = np.random.randn(20, 10)
        y_train = np.array([0] * 10 + [1] * 10)
             
        self.engine.param_grid = {'C': [0.1, 1.0]}
                 
        # Balanced splits to ensure SVM sees both classes
        train_indices = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        test_indices = np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])
        dummy_inner_cv = [(train_indices, test_indices)]
             
        best_c, _, _, best_model = self.engine.train(X_train, y_train, inner_cv_iterator=dummy_inner_cv)
                 
        self.assertIn(best_c, [0.1, 1.0])
        self.assertIsInstance(best_model, Pipeline)
        self.assertIsInstance(best_model.named_steps['svc'], SVC)

    def test_predict_method(self) -> None:
        """
        Validates the pure inference API structure and array returns.
        """
        mock_model = MagicMock(spec=SVC)
        mock_model.predict.return_value = np.array([0, 1, 1])
        mock_model.decision_function.return_value = np.array([-0.9, 0.8, 0.7])
                 
        X_test = np.random.randn(3, 10) 
                 
        y_pred, y_prob = self.engine.predict(mock_model, X_test)
                 
        self.assertEqual(len(y_pred), 3)
        self.assertEqual(y_prob[1], 0.8, "Probability extraction failed to target the positive class.")
             
    @patch('CMEPDA_project_2024.Python.svm_classifier.pd.read_csv', side_effect=FileNotFoundError)
    def test_load_real_data_file_missing(self, mock_read_csv: MagicMock) -> None:
        """Validates that the Engine correctly aborts if data files are missing."""
        with self.assertRaises(FileNotFoundError):
            SVMClassifier.load_data("missing.csv", "missing_mask.nii", "dummy_base_dir")

    @patch('CMEPDA_project_2024.Python.svm_classifier.pd.read_csv')
    @patch('CMEPDA_project_2024.Python.svm_classifier.nib.load')
    def test_load_real_data_success(self, mock_nib_load, mock_read_csv) -> None:
        """
        Mocks the OS file system to validate structural 3D-to-1D flattening and parsing.
        """
        mock_read_csv.return_value = pd.DataFrame({
            'subject_id': ['SUB_01', 'SUB_02'], 
            'file_path': ['fake1.nii', 'fake2.nii'], 
            'label': [1, 0]
        })
                 
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.ones((10, 10, 10))
        mock_nib_load.return_value = mock_img
                 
        subjects, X_data, y_data = SVMClassifier.load_data("dummy.csv", "mask.nii", "dummy_base_dir")
                 
        self.assertEqual(X_data.shape, (2, 1000))
        self.assertEqual(len(subjects), 2)
        self.assertEqual(subjects[0], "SUB_01")
        self.assertEqual(y_data[0], 1)

    def test_execute_nested_cv_integration(self) -> None:
        """
        Integration Test: Ensures the decoupled Double CV pipeline executes end-to-end.
        
        PURPOSE:
            Generates completely random internal matrices to test the full pipeline 
            without relying on local datasets, guaranteeing robust code architecture.
        """
        with self.logger.context(session_id="SVM_Integration"):
            N_SAMPLES, N_FEATURES = 20, 50
            np.random.seed(42)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            X_data = np.random.randn(N_SAMPLES, N_FEATURES)
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            X_data[y_data == 1] += 0.5
            
            dummy_cv_splits = [{
                'fold': 1,
                'outer_train_idx': np.arange(10),
                'outer_test_idx': np.arange(10, 20),
                'inner_splits_relative': [(np.arange(8), np.arange(8, 10))]
            }]
            
            df_metrics, artifacts = self.engine.execute_nested_cv(X_data, y_data, subjects, cv_splits=dummy_cv_splits)
            
            self.assertIsInstance(df_metrics, pd.DataFrame)
            self.assertEqual(len(df_metrics), 1)
            self.assertEqual(len(artifacts), 1)

if __name__ == '__main__':
    unittest.main()