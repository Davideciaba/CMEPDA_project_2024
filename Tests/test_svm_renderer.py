"""
Module: test_svm_renderer.py

Unit testing suite targeting the SVMRenderer visualization class.
"""
import sys
import os

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

from XAI.svm_renderer import SVMRenderer
from utils.py_logger import CustomLogger

class TestSVMRenderer(unittest.TestCase):

    def setUp(self):
        self.logger = CustomLogger(name="TestSVMRenderer")
        self.svm_renderer = SVMRenderer(logger=self.logger)

    def test_voxel_to_mni_conversion(self):
        """Validates the bidirectional Z-axis MNI geometric conversions."""
        affine_mat = np.eye(4)
        affine_mat[2, 2] = 2.0  
        affine_mat[2, 3] = -10.0 
        
        active_mask = np.zeros((10, 10, 10), dtype=bool)
        active_mask[5, 5, 4:6] = True 
        
        z_voxels, z_mms = self.svm_renderer._get_voxel_indices_from_mni(affine_mat, (10,10,10), active_mask, step_mm=2.0)
        
        self.assertTrue(len(z_voxels) > 0)
        self.assertEqual(len(z_voxels), len(z_mms))

    @patch('XAI.svm_renderer.plt.savefig')
    @patch('XAI.svm_renderer.nib.load')
    def test_svm_renderer_execution(self, mock_nib_load, mock_savefig):
        """Validates end-to-end execution of the SVM Sequential plotting pipeline."""
        mock_bg = MagicMock()
        mock_bg.get_fdata.return_value = np.random.rand(10, 10, 10)
        mock_bg.affine = np.eye(4)
        
        mock_map_svm = MagicMock()
        mock_map_svm_data = np.zeros((10, 10, 10))
        mock_map_svm_data[5, 5, 5] = 1.0 
        mock_map_svm.get_fdata.return_value = mock_map_svm_data
        
        mock_nib_load.side_effect = [mock_bg, mock_map_svm]
        
        self.svm_renderer.plot_svm_map(
            "dummy_svm.nii", "dummy_bg.nii", "dummy_svm_out.png", threshold=0.5, step_mm=5.0
        )
        self.assertEqual(mock_savefig.call_count, 1)

if __name__ == '__main__':
    unittest.main()