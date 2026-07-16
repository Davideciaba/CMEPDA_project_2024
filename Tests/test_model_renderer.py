import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Path injection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.utils.model_renderer import ModelRenderer
from Python.utils.py_logger import CustomLogger

class TestModelRenderer(unittest.TestCase):
    def setUp(self):
        self.logger = CustomLogger(name="TestRenderer")
        self.output_dir = "dummy_plots"
        self.renderer = ModelRenderer(logger=self.logger, output_dir=self.output_dir)

    @patch('Python.utils.model_renderer.plt.savefig')
    @patch('Python.utils.model_renderer.nib.load')
    def test_plot_3d_activation_map_dimension_mismatch(self, mock_nib_load, mock_savefig):
        """Simula il LBYL e verifica che venga bloccato il mismatch dimensionale (come in MATLAB)."""
        # Setup mock NIfTI Reference
        mock_bg = MagicMock()
        mock_bg.get_fdata.return_value = np.zeros((10, 10, 10))
        mock_bg.affine = np.eye(4)
        
        # Setup mock Map mismatching
        mock_stats = MagicMock()
        mock_stats.get_fdata.return_value = np.zeros((5, 5, 5)) 
        
        mock_mask = MagicMock()
        mock_mask.get_fdata.return_value = np.zeros((10, 10, 10))
        
        # Il ritorno segue l'ordine in cui i nib.load() vengono chiamati nel codice
        mock_nib_load.side_effect = [mock_bg, mock_stats, mock_mask] 
        
        # Ci aspettiamo che Numpy sollevi ValueError cercando di applicare la maschera con dimensioni errate
        with self.assertRaises(ValueError):
            self.renderer.plot_3d_activation_map("bg.nii", "stats.nii", "mask.nii", "Title", "out.png")

if __name__ == '__main__':
    unittest.main()