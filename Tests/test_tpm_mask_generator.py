import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import nibabel as nib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.utils.tpm_mask_generator import TpmMaskGenerator
from Python.utils.py_logger import CustomLogger

class TestTpmMaskGenerator(unittest.TestCase):
    def setUp(self):
        self.logger = CustomLogger(name="TestTPM")
        self.gen = TpmMaskGenerator(logger=self.logger)

    @patch('Python.utils.tpm_mask_generator.nib.load')
    @patch('Python.utils.tpm_mask_generator.pd.read_csv')
    @patch('Python.utils.tpm_mask_generator.nib.save')
    def test_generate_mask_flow(self, mock_save, mock_csv, mock_load):
        """Validates the standard operating flow of the generator."""
        mock_csv.return_value = pd.DataFrame({'file_path': ['fake.nii']})
        
        mock_img = MagicMock()
        mock_img.shape = (10, 10, 10)
        mock_img.affine = np.eye(4)
        mock_img.get_fdata.return_value = np.ones((10, 10, 10))
        mock_img.header = nib.Nifti1Header()
        
        mock_load.side_effect = [mock_img, mock_img] # Returns Reference, then TPM
        
        self.gen.generate_mask("reg.csv", "tpm.nii", "mask.nii")
        
        mock_save.assert_called_once()

    @patch('Python.utils.tpm_mask_generator.nib.load')
    @patch('Python.utils.tpm_mask_generator.pd.read_csv')
    def test_spatial_validation_mismatch(self, mock_csv, mock_load):
        """Asserts that a ValueError is raised if the TPM dimension does not match the cohort."""
        mock_csv.return_value = pd.DataFrame({'file_path': ['fake.nii']})
        
        mock_ref = MagicMock()
        mock_ref.shape = (10, 10, 10)
        mock_ref.affine = np.eye(4)
        
        mock_tpm = MagicMock()
        mock_tpm.shape = (5, 5, 5) # Dimensione errata di proposito
        mock_tpm.affine = np.eye(4)
        mock_tpm.get_fdata.return_value = np.zeros((5, 5, 5))
        
        mock_load.side_effect = [mock_ref, mock_tpm]
        
        with self.assertRaises(ValueError):
            self.gen.generate_mask("reg.csv", "tpm.nii", "mask.nii")

if __name__ == '__main__':
    unittest.main()