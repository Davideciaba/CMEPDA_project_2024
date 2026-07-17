"""
Module: test_roi_analyzer.py

Unit testing suite targeting the ROIAnalyzer class.
Validates the extraction of features via discrete SPM atlases, checking filtering 
logic and the exact mathematical correctness of the nDCG ranking equations.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
import pathlib
# Dynamically resolve paths using pathlib
current_dir= pathlib.Path(__file__).resolve().parent
parent_dir= current_dir.parent

# Add the parent directory to sys.path to allow imports from there
sys.path.append(str(parent_dir))

from Python.utils.roi_analyzer import ROIAnalyzer
from Python.utils.py_logger import CustomLogger

class TestROIAnalyzer(unittest.TestCase):
    """
    Test suite for Python.utils.roi_analyzer.ROIAnalyzer.
    
    PURPOSE:
        Ensures that White Matter and Cerebellum are successfully excluded 
        from the XAI analysis and that ranking correlations match the theoretical formulas.
    """

    def setUp(self) -> None:
        """Initializes the engine with a Mock Logger."""
        self.logger = CustomLogger(name="TestROI")
        self.analyzer = ROIAnalyzer(logger=self.logger)

    @patch('Python.utils.roi_analyzer.os.path.exists')
    @patch('Python.utils.roi_analyzer.nib.load')
    @patch('Python.utils.roi_analyzer.pd.read_csv')
    def test_extract_regional_importance_with_filtering(self, mock_read_csv, mock_nib_load, mock_exists) -> None:
        """
        Tests the ROI extraction and the strict filtering of White Matter/Ventricles.
        
        PURPOSE:
            Asserts that the string-matching logic successfully parses SPM labels 
            and discards non-Grey-Matter structures from the dataframe.
        """
        mock_exists.return_value = True
        
        # Simulate a CSV with mixed regions (GM, WM, Ventricles)
        mock_df = pd.DataFrame({
            'ROI_ID': [1, 2, 3, 4],
            'ROI_Name': ['Left Hippocampus', 'Right Cerebral White Matter', 'Left Lateral Ventricle', 'Right Amygdala']
        })
        mock_read_csv.return_value = mock_df
        
        # Simulate an atlas (4 regions, each with 2 voxels for simplicity)
        atlas_vol = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])
        mock_atlas = MagicMock()
        mock_atlas.get_fdata.return_value = atlas_vol
        
        # Simulate an XAI map with arbitrary values
        xai_vol = np.array([
            [0.5, 0.5], # Hippocampus (Mean = 0.5)
            [1.0, 1.0], # White Matter (Should be filtered)
            [2.0, 2.0], # Ventricle (Should be filtered)
            [0.8, 0.8]  # Amygdala (Mean = 0.8)
        ])
        mock_xai = MagicMock()
        mock_xai.get_fdata.return_value = xai_vol
        
        mock_nib_load.side_effect = [mock_xai, mock_atlas]
        
        # Execution
        df_result = self.analyzer.extract_regional_importance('fake_xai.nii', 'fake_atlas.nii', 'fake_csv.csv')
        
        # Only 2 regions must remain (Hippocampus and Amygdala)
        self.assertEqual(len(df_result), 2)
        
        # Filtered regions must not exist
        region_names = df_result['ROI_Name'].tolist()
        self.assertNotIn('Right Cerebral White Matter', region_names)
        self.assertNotIn('Left Lateral Ventricle', region_names)
        
        # Amygdala (0.8) must precede Hippocampus (0.5) since results are sorted descending
        self.assertEqual(df_result.iloc[0]['ROI_Name'], 'Right Amygdala')
        self.assertEqual(df_result.iloc[1]['ROI_Name'], 'Left Hippocampus')

    def test_ndcg_calculation(self) -> None:
        """
        Tests the mathematical accuracy of the Normalized Discounted Cumulative Gain.
        
        PURPOSE:
            Replicates a manual mathematical nDCG calculation scenario to verify 
            that the array sorting and logarithmic discounts match exactly 1.0 and 0.2768.
        """
        # Standard test scenario
        # True relevance values (ground truth)
        true_scores = np.array([3, 2, 3, 0, 1, 2])
        # Predicted values (model predictions for those same elements)
        predicted_scores = np.array([2, 1, 3, 0, 0, 1]) 
        
        
        ndcg_k3 = self.analyzer.calculate_ndcg(predicted_scores, true_scores, k=3)
        self.assertAlmostEqual(ndcg_k3, 1.0, places=3)
        
        # Test a case with terrible predictions
        bad_predictions = np.array([0, 0, 0, 3, 2, 1])
        ndcg_bad_k3 = self.analyzer.calculate_ndcg(bad_predictions, true_scores, k=3)
        self.assertTrue(ndcg_bad_k3 < 1.0)
        self.assertAlmostEqual(ndcg_bad_k3, 0.2768, places=3)

if __name__ == '__main__':
    unittest.main()