"""
Module: test_roi_analyzer.py

Unit testing suite targeting the ROIAnalyzer class.
Validates the NumPy vectorized boolean masking and absolute importance algebra 
without hitting real disk memory.
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# --- BULLETPROOF PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
python_src_dir = os.path.join(parent_dir, 'Python')

if os.path.exists(python_src_dir) and python_src_dir not in sys.path:
    sys.path.insert(0, python_src_dir)
elif parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Python.XAI.roi_analyzer import ROIAnalyzer
from Python.utils.py_logger import CustomLogger

class TestROIAnalyzer(unittest.TestCase):

    def setUp(self):
        """Initializes the ROI Analyzer with a muted test logger."""
        self.logger = CustomLogger(name="TestROI")
        self.analyzer = ROIAnalyzer(logger=self.logger)

    @patch('XAI.roi_analyzer.os.path.exists')
    @patch('XAI.roi_analyzer.nib.load')
    @patch('XAI.roi_analyzer.pd.read_csv')
    def test_regional_extraction_math(self, mock_read_csv, mock_nib_load, mock_exists):
        """
        Validates that boolean masking accurately isolates discrete anatomical 
        regions and calculates the absolute mean of diverging XAI signals.
        """
        # Bypassa il controllo di sicurezza fisico sul disco
        mock_exists.return_value = True
        
        # Mock Atlas CSV Dictionary
        mock_df = pd.DataFrame({
            'ROI_ID': [1, 2], 
            'ROI_Name': ['Hippocampus', 'Amygdala']
        })
        mock_read_csv.return_value = mock_df
        
        # Mock 3D SPM Atlas (3x3x3 tensor)
        atlas_vol = np.zeros((3, 3, 3))
        atlas_vol[0, :, :] = 1 # Z=0 is Hippocampus
        atlas_vol[1, :, :] = 2 # Z=1 is Amygdala
        
        mock_atlas = MagicMock()
        mock_atlas.get_fdata.return_value = atlas_vol
        
        # Mock 3D XAI Map (e.g., Integrated Gradients with diverging logic)
        xai_vol = np.zeros((3, 3, 3))
        xai_vol[0, 0, 0] = 0.5  # Positive attribution in Hippocampus
        xai_vol[0, 1, 0] = -0.5 # Negative attribution in Hippocampus
        xai_vol[1, 0, 0] = -1.0 # Strong negative attribution in Amygdala
        
        mock_xai = MagicMock()
        mock_xai.get_fdata.return_value = xai_vol
        
        # Binding the sequential loads
        mock_nib_load.side_effect = [mock_xai, mock_atlas]
        
        # Execute Evaluation
        df = self.analyzer.extract_regional_importance('xai_path.nii', 'atlas_path.nii', 'csv_path.csv', threshold=0.1)
        
        # Assertions
        self.assertEqual(len(df), 2)
        
        # Based on absolute sorting, Amygdala (| -1.0 | = 1.0) should be ranked 1st
        self.assertEqual(df.iloc[0]['ROI_Name'], 'Amygdala')
        self.assertEqual(df.iloc[0]['Mean_Abs_Importance'], 1.0)
        
        # Hippocampus should be ranked 2nd. Mean of |0.5| and |-0.5| is 0.5
        self.assertEqual(df.iloc[1]['ROI_Name'], 'Hippocampus')
        self.assertEqual(df.iloc[1]['Mean_Abs_Importance'], 0.5)

    @patch('XAI.roi_analyzer.os.path.exists')
    @patch('XAI.roi_analyzer.nib.load')
    @patch('XAI.roi_analyzer.pd.read_csv')
    def test_dimension_mismatch_exception(self, mock_read_csv, mock_nib_load, mock_exists):
        """Ensures the analyzer fails safely if maps are not in the same spatial grid."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'ROI_ID': [1], 'ROI_Name': ['A']})
        
        mock_xai = MagicMock()
        mock_xai.get_fdata.return_value = np.zeros((10, 10, 10))
        
        mock_atlas = MagicMock()
        mock_atlas.get_fdata.return_value = np.zeros((12, 12, 12)) # Mismatch!
        
        mock_nib_load.side_effect = [mock_xai, mock_atlas]
        
        with self.assertRaises(ValueError):
            self.analyzer.extract_regional_importance('xai.nii', 'atlas.nii', 'labels.csv')

if __name__ == '__main__':
    unittest.main()