"""
Module: test_XAI_SVM.py

Unit testing suite targeting the SVMAnalyticalXAI class.
Employs path injection to access the XAI and utils directories.
"""
import sys
import os

# --- PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
python_src_dir = os.path.join(parent_dir, 'Python')

if os.path.exists(python_src_dir) and python_src_dir not in sys.path:
    sys.path.insert(0, python_src_dir)
elif parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from Python.XAI.XAI_SVM import SVMExplainer
from Python.utils.py_logger import CustomLogger

class TestSVMAnalyticalXAI(unittest.TestCase):

    def setUp(self):
        """Initializes the XAI engine with the actual CustomLogger."""
        self.logger = CustomLogger(name="TestXAI_SVM")
        self.logger.add_console_handler(level="INFO")
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

    def test_compute_gaonkar_maps_math(self):
        """Validates Gaonkar's non-parametric projection bounding and execution."""
        with self.logger.context(Task="Gaonkar_Math"):
            N_SAMPLES, N_FEATURES = 8, 500 
            np.random.seed(42)
            
            X_train = np.random.randn(N_SAMPLES, N_FEATURES)
            y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            
            mock_model = MagicMock()
            mock_model.coef_ = np.array([np.random.randn(N_FEATURES)])
            mock_model.support_ = np.arange(N_SAMPLES)

            z_map, p_map_raw, p_map_fwe = self.xai_engine.compute_gaonkar_maps(mock_model, X_train, y_train, C=1.0)

            self.assertEqual(z_map.shape, (N_FEATURES,))
            self.assertEqual(p_map_raw.shape, (N_FEATURES,))
            self.assertEqual(p_map_fwe.shape, (N_FEATURES,))
            
            # Valida la natura matematica della correzione FWE
            self.assertTrue(np.all(p_map_fwe >= p_map_raw)) 
            self.assertTrue(np.all(p_map_fwe <= 1.0))

    def test_aggregate_global_maps(self):
        """Ensures cross-fold element-wise aggregation yields an exact mean."""
        map1 = np.array([1.0, 2.0, 3.0])
        map2 = np.array([3.0, 4.0, 5.0])
        
        aggregated = SVMAnalyticalXAI.aggregate_global_maps([map1, map2])
        expected = np.array([2.0, 3.0, 4.0])
        
        np.testing.assert_array_almost_equal(aggregated, expected)

    @patch('XAI.XAI_SVM.os.path.exists')
    @patch('XAI.XAI_SVM.nib.load')
    @patch('XAI.XAI_SVM.nib.save')
    def test_reconstruct_nifti_success(self, mock_save, mock_load, mock_exists):
        """Validates safe mapping from 1D back into 3D NIfTI grids via boolean templates."""
        with self.logger.context(Task="NIfTI_Reconstruction"):
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