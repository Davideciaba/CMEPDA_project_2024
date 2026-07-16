import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.XAI.plot_ndcg import XAIPlotter
from Python.utils.py_logger import CustomLogger

class TestXAIPlotter(unittest.TestCase):
    def setUp(self):
        self.logger = CustomLogger(name="TestPlotter")
        self.plotter = XAIPlotter(logger=self.logger)
        
        # Dati base per testare il funzionamento dei plot
        self.base_df = pd.DataFrame({
            'ROI_Name': ['A', 'B', 'C'], 
            'Mean_ROI_Signal': [0.5, -0.8, 0.2]
        })

    @patch('Python.XAI.plot_ndcg.plt.savefig')
    @patch('Python.XAI.plot_ndcg.os.makedirs')
    def test_plot_top_rois(self, mock_makedirs, mock_savefig):
        """Test the generation of horizontal bar plots."""
        self.plotter.plot_top_rois(self.base_df, score_col='Mean_ROI_Signal', title='Test Top', out_path="fake.png")
        mock_savefig.assert_called_once()

    @patch('Python.XAI.plot_ndcg.plt.savefig')
    def test_plot_diverging_bars(self, mock_save):
        """Test the directional diverging bar plot creation."""
        self.plotter.plot_diverging_bars(self.base_df, score_col='Mean_ROI_Signal', title='Test Div', out_path="fake_div.png")
        mock_save.assert_called_once()

    @patch('Python.XAI.plot_ndcg.plt.savefig')
    def test_bloch_heatmap(self, mock_save):
        """Verify the heatmap safely accepts the matrix configuration."""
        matrix = pd.DataFrame({'Model1': [0.1, 0.9], 'Model2': [0.2, 0.8]}, index=['R1', 'R2'])
        self.plotter.plot_bloch_style_heatmap(matrix, "heatmap.png")
        mock_save.assert_called_once()
        
    @patch('Python.XAI.plot_ndcg.plt.savefig')
    def test_ndcg_matrix(self, mock_save):
        """Verify the All-to-All nDCG plot rendering."""
        matrix = pd.DataFrame({'Model1': [1.0, 0.85], 'Model2': [0.85, 1.0]}, index=['Model1', 'Model2'])
        self.plotter.plot_ndcg_matrix(matrix, "ndcg.png")
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()