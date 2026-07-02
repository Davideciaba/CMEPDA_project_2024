"""
Unit Testing Suite for CNN Predictive Utilities.

This module validates the Deep Learning primitives provided by the `CNNUtils` static class.
It includes tests for PyTorch forward/backward pass execution, tensor dimensional transformations 
(expanding 3D data to 5D required by MONAI), and safely mocking data I/O streams.
Executes forced CPU tensors to ensure rapid, asynchronous CI/CD testing.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the updated static utility class
from CMEPDA_project_2024.Models.efficientnet_utils import CNNUtils


class TestCNNUtils(unittest.TestCase):
    """Test suite covering network instantiations, gradient graph building, and data I/O for EfficientNet."""
    
    def setUp(self):
        """
        Initializes lightweight PyTorch objects and synthetic tensors for instantaneous testing.
        """
        self.device = torch.device("cpu") # Force CPU to avoid CUDA initialization overhead during tests
        
        # Construct a synthetic 5D PyTorch Tensor (Batch, Channels, D, H, W)
        # 4 Patients, 1 Channel, aggressively down-sampled volume (16x16x16)
        self.X = torch.randn(4, 1, 64, 64, 64)
        self.y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        
        self.dataset = TensorDataset(self.X, self.y)
        # drop_last=False is safe here because batch_size (2) perfectly divides the dataset (4)
        self.loader = DataLoader(self.dataset, batch_size=2, drop_last=False) 
        
        # Initialize the actual MONAI EfficientNet architecture
        self.model = CNNUtils.build_model(model_name="efficientnet-b0", in_channels=1, num_classes=2)

    def test_model_output_tensor_shape(self):
        """
        Validates the structural contract of the Neural Network's final classification layer.
        
        Purpose:
            Ensures that passing a 5D volumetric tensor correctly reduces to a 2D 
            classification matrix without broadcasting errors.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X)
            
        # The model MUST return 4 distinct predictions distributed across 2 output classes
        expected_shape = (4, 2)
        self.assertEqual(output.shape, expected_shape, f"Mismatch in final classification layer geometry.")

    def test_train_one_epoch_execution(self):
        """
        Verifies the computational graph generation during the training loop.
        
        Purpose:
            Ensures that the Backpropagation sequence (zero_grad, backward, step) operates 
            flawlessly without raising computational exceptions.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        loss = CNNUtils.train_one_epoch(self.model, self.loader, optimizer, criterion, self.device)
        
        self.assertIsInstance(loss, float, "Epoch loss must be collapsed into a scalar float")
        self.assertTrue(loss > 0, "Cross-Entropy loss must be strictly positive")

    def test_evaluate_model_execution(self):
        """
        Validates the evaluation loop, ensuring the Autograd engine remains inactive.
        """
        criterion = nn.CrossEntropyLoss()
        val_loss = CNNUtils.evaluate_model(self.model, self.loader, criterion, self.device)
        
        self.assertIsInstance(val_loss, float, "Validation loss must be a scalar float")

    @patch('efficientnet_utils.os.path.exists')
    @patch('efficientnet_utils.pd.read_csv')
    @patch('efficientnet_utils.nib.load')
    def test_load_real_data_mocked(self, mock_nib_load, mock_read_csv, mock_exists):
        """
        Mocks the filesystem to validate the expansion of 3D NIfTI volumes into 5D PyTorch streams.
        
        Purpose:
            Unlike SVM, PyTorch requires a 'Channel' dimension. This test verifies that 
            `CNNUtils` successfully injects `axis=0` into the numpy array.
        """
        mock_exists.return_value = True
        
        # Mock CSV containing a single patient
        mock_df = pd.DataFrame({'subject_id': ['SUB_01'], 'file_path': ['fake.nii'], 'label': [1]})
        mock_read_csv.return_value = mock_df
        
        # Mock Nibabel to return a standard 3D brain volume (e.g., 32x32x32)
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.zeros((32, 32, 32))
        mock_nib_load.return_value = mock_img
        
        # Execute logic
        subjects, X_data, y_data = CNNUtils.load_real_data("dummy.csv")
        
        # The expected shape is (1 Patient, 1 Channel, 32 Depth, 32 Height, 32 Width) -> (1, 1, 32, 32, 32)
        self.assertEqual(X_data.shape, (1, 1, 32, 32, 32), "NIfTI volume must be expanded with a Channel dimension")
        self.assertEqual(y_data[0], 1, "Label parsing failed")


if __name__ == '__main__':
    unittest.main()