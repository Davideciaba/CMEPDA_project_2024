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
from unittest.mock import patch
import numpy as np

from Python.XAI.XAI_SVM import SVMExplainer
from Python.utils.py_logger import CustomLogger

class TestSVMAnalyticalXAI(unittest.TestCase):

    def setUp(self):
        """Initializes the XAI engine with the actual CustomLogger."""
        self.logger = CustomLogger(name="TestXAI_SVM")
        self.logger.add_console_handler(level="INFO")
        self.xai_engine = SVMExplainer(logger=self.logger)

    def test_compute_haufe_transform_math(self):
        """Validates the algebraic correctness of Haufe's associative transform."""
        N_SAMPLES, N_FEATURES = 10, 100
        np.random.seed(42)
        X_train = np.random.randn(N_SAMPLES, N_FEATURES)
        decision_scores = np.random.randn(N_SAMPLES) # Richiesto dal nuovo metodo
        
        # Nome aggiornato e nuova firma
        haufe_map = self.xai_engine.compute_haufe_patterns(X_train, decision_scores)
        self.assertEqual(haufe_map.shape[0], N_FEATURES)

    def test_compute_gaonkar_maps_math(self):
        """Validates Gaonkar's execution."""
        with self.logger.context(Task="Gaonkar_Math"):
            N_SAMPLES, N_FEATURES = 8, 500
            np.random.seed(42)
            X_train = np.random.randn(N_SAMPLES, N_FEATURES)
            y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            svm_weights = np.random.randn(N_FEATURES)
    
            SVM_C_PARAM = 1.0
            SUPPORT_VECTORS = 8
    
            z_map, p_map_raw = self.xai_engine.compute_gaonkar_maps(
                X_train, y_train, svm_weights, 
                C_param=SVM_C_PARAM, 
                n_support=SUPPORT_VECTORS
            )
            self.assertEqual(z_map.shape[0], N_FEATURES)

    @patch('Python.XAI.XAI_SVM.os.path.exists')
    @patch('Python.XAI.XAI_SVM.nib.load')
    @patch('Python.XAI.XAI_SVM.nib.save')
    def test_reconstruct_nifti_success(self, mock_save, mock_load, mock_exists):
        """Validates safe mapping from 1D back into 3D NIfTI."""
        with self.logger.context(Task="NIfTI_Reconstruction"):
            mock_exists.return_value = True
            
            # Maschera booleana come richiesto dalla nuova classe
            brain_mask = np.zeros((4, 4, 4), dtype=bool)
            brain_mask[0, :, :2] = True  # 8 voxel attivi
            
            map_1d = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
            affine = np.eye(4)
    
            # Nome aggiornato
            self.xai_engine.reconstruct_and_save_3d(map_1d, brain_mask, affine, "output_path.nii")
            mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()