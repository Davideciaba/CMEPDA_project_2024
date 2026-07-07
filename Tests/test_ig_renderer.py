"""
Module: test_ig_renderer.py

Unit testing suite targeting the IGRenderer visualization class.
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

from XAI.ig_renderer import IGRenderer
from utils.py_logger import CustomLogger

class TestIGRenderer(unittest.TestCase):

    def setUp(self):
        self.logger = CustomLogger(name="TestIGRenderer")
        self.ig_renderer = IGRenderer(logger=self.logger)

    @patch('XAI.ig_renderer.plt.savefig')
    @patch('XAI.ig_renderer.nib.load')
    def test_ig_renderer_execution(self, mock_nib_load, mock_savefig):
        """Validates end-to-end execution of the IG Diverging plotting pipeline."""
        mock_bg = MagicMock()
        mock_bg.get_fdata.return_value = np.random.rand(10, 10, 10)
        mock_bg.affine = np.eye(4)
        
        mock_map_ig = MagicMock()
        mock_map_ig_data = np.zeros((10, 10, 10))
        mock_map_ig_data[5, 5, 5] = -2.0 
        mock_map_ig_data[6, 6, 6] = 1.5  
        mock_map_ig.get_fdata.return_value = mock_map_ig_data
        
        mock_nib_load.side_effect = [mock_bg, mock_map_ig]
        
        self.ig_renderer.plot_ig_map(
            "dummy_ig.nii", "dummy_bg.nii", "dummy_ig_out.png", threshold=1.0, step_mm=5.0
        )
        self.assertEqual(mock_savefig.call_count, 1)

if __name__ == '__main__':
    unittest.main()