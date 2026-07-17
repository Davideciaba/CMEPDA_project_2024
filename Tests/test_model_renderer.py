"""
Module: test_model_renderer.py

Unit testing suite targeting the ModelRenderer class.
Validates the structural robustness of the rendering logic, ensuring that 
dimension mismatches are blocked before attempting complex Matplotlib operations.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
import os

# Path injection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.utils.model_renderer import ModelRenderer
from Python.utils.py_logger import CustomLogger

class TestModelRenderer(unittest.TestCase):
    """
    Test suite for Python.utils.model_renderer.ModelRenderer.
    
    PURPOSE:
        Isolates the visualization engine using mock patching to avoid physical 
        disk I/O, ensuring that exceptions and data-flow behave properly.
    """

    def setUp(self) -> None:
        """
        Initializes the rendering engine and mock data before each test.
        """
        self.logger = CustomLogger(name="TestRenderer")
        self.output_dir = "dummy_plots"
        self.renderer = ModelRenderer(logger=self.logger, output_dir=self.output_dir)
        
        # Base data for plotting tests (XAI plots)
        self.base_df = pd.DataFrame({
            'ROI_Name': ['A', 'B', 'C'], 
            'Mean_ROI_Signal': [0.5, -0.8, 0.2]
        })

    @patch('Python.utils.model_renderer.plt.savefig')
    @patch('Python.utils.model_renderer.nib.load')
    def test_plot_3d_activation_map_dimension_mismatch(self, mock_nib_load, mock_savefig) -> None:
        """
        Simulates the LBYL (Look Before You Leap) logic and verifies dimensional mismatches.
        
        PURPOSE:
            Guarantees that attempting to render a statistical map over an incompatible 
            anatomical background triggers a clear ValueError, preventing deep Numpy crashes.
        """
        # Setup mock NIfTI Reference
        mock_bg = MagicMock()
        mock_bg.get_fdata.return_value = np.zeros((10, 10, 10))
        mock_bg.affine = np.eye(4)
        
        # Setup mock Map mismatching (5x5x5 vs 10x10x10)
        mock_stats = MagicMock()
        mock_stats.get_fdata.return_value = np.zeros((5, 5, 5)) 
        
        mock_mask = MagicMock()
        mock_mask.get_fdata.return_value = np.zeros((10, 10, 10))
        
        # The return follows the order in which nib.load() is called in the code
        mock_nib_load.side_effect = [mock_bg, mock_stats, mock_mask] 
        
        # We expect Numpy to raise a ValueError when applying the mask with wrong dimensions
        with self.assertRaises(ValueError):
            self.renderer.plot_3d_activation_map("bg.nii", "stats.nii", "mask.nii", "Title", "out.png")

    @patch('matplotlib.figure.Figure.savefig')
    def test_plot_top_rois(self, mock_savefig) -> None:
        """Test the generation of horizontal bar plots."""
        self.renderer.plot_top_rois(self.base_df, score_col='Mean_ROI_Signal', title='Test Top', filename="fake.png")
        mock_savefig.assert_called_once()

    @patch('matplotlib.figure.Figure.savefig')
    def test_plot_diverging_bars(self, mock_savefig) -> None:
        """Test the directional diverging bar plot creation."""
        self.renderer.plot_diverging_bars(self.base_df, score_col='Mean_ROI_Signal', title='Test Div', filename="fake_div.png")
        mock_savefig.assert_called_once()

    @patch('matplotlib.figure.Figure.savefig')
    def test_bloch_heatmap(self, mock_savefig) -> None:
        """Verify the heatmap safely accepts the matrix configuration."""
        matrix = pd.DataFrame({'Model1': [0.1, 0.9], 'Model2': [0.2, 0.8]}, index=['R1', 'R2'])
        self.renderer.plot_heatmap(matrix, "heatmap.png")
        mock_savefig.assert_called_once()
        
    @patch('matplotlib.figure.Figure.savefig')
    def test_ndcg_matrix(self, mock_savefig) -> None:
        """Verify the All-to-All nDCG plot rendering."""
        matrix = pd.DataFrame({'Model1': [1.0, 0.85], 'Model2': [0.85, 1.0]}, index=['Model1', 'Model2'])
        self.renderer.plot_ndcg_matrix(matrix, "ndcg.png")
        mock_savefig.assert_called_once()

if __name__ == '__main__':
    unittest.main()