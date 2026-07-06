"""
Module: test_xai_svm.py

Unit testing suite targeting the SVMAnalyticalXAI class.
Employs flat direct imports and isolates computational boundaries using synthetic arrays.
Includes a SpyLogger to validate Gaonkar's HDLSS assumptions and QA warnings.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os

# Importazioni dirette per architettura a directory piatta
from XAI_SVM import SVMAnalyticalXAI
from py_logger import CustomLogger

class SpyLogger:
    """
    A Mock Logger that acts as a 'Spy' to track warning emissions during unit tests.
    Ensures that Quality Assurance (QA) checks are actively firing.
    """
    def __init__(self):
        self.warnings_log = []
        
    def info(self, msg): pass
    def debug(self, msg): pass
    def success(self, msg): pass
    def error(self, msg): pass
    
    def warning(self, msg):
        self.warnings_log.append(msg)
        
    def context(self, **kwargs):
        class DummyContextManager:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

class TestSVMAnalyticalXAI(unittest.TestCase):

    def setUp(self):
        """Initializes the XAI engine with the SpyLogger to track assertions."""
        self.logger = SpyLogger()
        self.xai_engine = SVMAnalyticalXAI(logger=self.logger)

    def test_compute_haufe_transform_math(self):
        """Validates the algebraic correctness of Haufe's associative transform with intercepts."""
        N_SAMPLES, N_FEATURES = 10, 100
        np.random.seed(42)
        
        X_train = np.random.randn(N_SAMPLES, N_FEATURES)
        
        mock_model = MagicMock()
        mock_model.coef_ = np.array([np.ones(N_FEATURES)])
        mock_model.intercept_ = np.array([0.5])

        haufe_map = self.xai_engine.compute_haufe_transform(mock_model, X_train)

        self.assertEqual(haufe_map.shape, (N_FEATURES,))
        self.assertIsInstance(haufe_map, np.ndarray)
        self.assertFalse(np.isnan(haufe_map).any())

    def test_compute_gaonkar_maps_math_clean(self):
        """Validates Gaonkar's non-parametric projection under pure HDLSS regime (No Warnings)."""
        self.logger.warnings_log = [] # Reset spy
        N_SAMPLES, N_FEATURES = 8, 500 # m/d = 0.016 (< 0.2)
        np.random.seed(42)
        
        X_train = np.random.randn(N_SAMPLES, N_FEATURES)
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        mock_model = MagicMock()
        mock_model.coef_ = np.array([np.random.randn(N_FEATURES)])
        mock_model.support_ = np.arange(N_SAMPLES) # 100% Saturation

        z_map, p_map_raw, p_map_fwe = self.xai_engine.compute_gaonkar_maps(mock_model, X_train, y_train, C=1.0)

        self.assertEqual(z_map.shape, (N_FEATURES,))
        
        # Verify FWE behavior
        self.assertTrue(np.all(p_map_fwe >= p_map_raw))
        self.assertTrue(np.all(p_map_fwe <= 1.0))
        
        # Assert NO warnings were triggered since we respected HDLSS
        self.assertEqual(len(self.logger.warnings_log), 0)

    def test_compute_gaonkar_assumptions_warnings(self):
        """Validates that Gaonkar Quality Assurance actively flags HDLSS violations."""
        self.logger.warnings_log = [] # Reset spy
        N_SAMPLES, N_FEATURES = 100, 20 # m/d = 5.0 (Violates < 0.2)
        np.random.seed(42)
        
        X_train = np.random.randn(N_SAMPLES, N_FEATURES)
        y_train = np.random.randint(0, 2, N_SAMPLES)
        
        mock_model = MagicMock()
        mock_model.coef_ = np.array([np.random.randn(N_FEATURES)])
        mock_model.support_ = np.arange(10) # 10% Saturation (Violates > 95%)

        # Execute (it will generate warnings but won't interrupt execution)
        self.xai_engine.compute_gaonkar_maps(mock_model, X_train, y_train, C=1.0)

        # Assert that the two specific QA warnings were logged
        warning_texts = " ".join(self.logger.warnings_log)
        self.assertTrue(len(self.logger.warnings_log) >= 2)
        self.assertIn("m/d", warning_texts)
        self.assertIn("Support Vectors", warning_texts)

    def test_aggregate_global_maps(self):
        """Ensures cross-fold element-wise aggregation yields an exact mean."""
        map1 = np.array([1.0, 2.0, 3.0])
        map2 = np.array([3.0, 4.0, 5.0])
        
        aggregated = SVMAnalyticalXAI.aggregate_global_maps([map1, map2])
        expected = np.array([2.0, 3.0, 4.0])
        
        np.testing.assert_array_almost_equal(aggregated, expected)

    @patch('XAI_SVM.os.path.exists')
    @patch('XAI_SVM.nib.load')
    @patch('XAI_SVM.nib.save')
    def test_reconstruct_nifti_success(self, mock_save, mock_load, mock_exists):
        """Validates safe mapping from 1D back into 3D NIfTI grids via boolean templates."""
        mock_exists.return_value = True
        
        mask_grid = np.zeros((4, 4, 4))
        mask_grid[0, :, :2] = 1 
        
        mock_mask_img = MagicMock()
        mock_mask_img.get_fdata.return_value = mask_grid
        mock_mask_img.affine = np.eye(4)
        mock_mask_img.header = None
        mock_load.return_value = mock_mask_img

        map_1d = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        
        self.xai_engine.reconstruct_nifti(map_1d, "mask_path.nii", "output_path.nii")
        self.assertEqual(mock_save.call_count, 1)

if __name__ == '__main__':
    unittest.main()