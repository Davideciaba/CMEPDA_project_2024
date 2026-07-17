"""
Module: test_XAI_SVM.py

Unit testing suite targeting the SVMExplainer class.
Employs path injection to access the XAI and utils directories.
Validates the dense linear algebra associated with the Haufe Transform and Gaonkar Maps.
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
from unittest.mock import patch
import numpy as np

from Python.utils.xai_svm import SVMExplainer
from Python.utils.py_logger import CustomLogger

class TestSVMAnalyticalXAI(unittest.TestCase):
    """
    Test suite for Python.utils.xai_svm.SVMExplainer.
    
    PURPOSE:
        Validates algebraic matrix manipulations, checking if dimensionality 
        is preserved when decoding backward weights into forward significance maps.
    """

    def setUp(self) -> None:
        """Initializes the XAI engine with the actual CustomLogger."""
        self.logger = CustomLogger(name="TestXAI_SVM")
        self.logger.add_console_handler(level="INFO")
        self.xai_engine = SVMExplainer(logger=self.logger)

    def test_compute_haufe_transform_math(self) -> None:
        """
        Validates the algebraic correctness of Haufe's associative transform.
        
        PURPOSE:
            Guarantees that Cov(X, S) generates an output vector strictly matching 
            the number of original input features (N_FEATURES).
        """
        N_SAMPLES, N_FEATURES = 10, 100
        np.random.seed(42)
        X_train = np.random.randn(N_SAMPLES, N_FEATURES)
        decision_scores = np.random.randn(N_SAMPLES)
        
        haufe_map = self.xai_engine.compute_haufe_patterns(X_train, decision_scores)
        self.assertEqual(haufe_map.shape[0], N_FEATURES)

    def test_compute_gaonkar_maps_math(self) -> None:
        """
        Validates Gaonkar's execution.
        
        PURPOSE:
            Simulates a High-Dimension Low-Sample-Size (HDLSS) scenario (8 vs 500) 
            to assert that analytical Z-scores calculate without raising 
            Singular Matrix or Division by Zero errors.
        """
        with self.logger.context(Task="Gaonkar_Math"):
            N_SAMPLES, N_FEATURES = 8, 500
            np.random.seed(42)
            X_train = np.random.randn(N_SAMPLES, N_FEATURES)
            y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            svm_weights = np.random.randn(N_FEATURES)
    
            SVM_C_PARAM = 1.0
    
            z_map, p_map_raw = self.xai_engine.compute_gaonkar_maps(
                X_train, y_train, svm_weights, 
                C_param=SVM_C_PARAM
            )
            self.assertEqual(z_map.shape[0], N_FEATURES)

    @patch('os.path.exists')
    @patch('nibabel.load')
    @patch('nibabel.save')
    def test_reconstruct_nifti_success(self, mock_save, mock_load, mock_exists) -> None:
        """
        Validates safe mapping from 1D back into 3D NIfTI.
        
        PURPOSE:
            Ensures that a 1D vector exactly matching the active voxel count 
            is inflated correctly into a 3D geometry without raising mismatch errors.
        """
        with self.logger.context(Task="NIfTI_Reconstruction"):
            mock_exists.return_value = True
            
            # Boolean mask as required by the new class
            brain_mask = np.zeros((4, 4, 4), dtype=bool)
            brain_mask[0, :, :2] = True  # 8 active voxels
            
            map_1d = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
            affine = np.eye(4)
    
            self.xai_engine.reconstruct_and_save_3d(map_1d, brain_mask, affine, "output_path.nii")
            mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()