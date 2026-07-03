"""
Unit Testing Suite for the Monolithic CNN Predictive Engine.

This module validates the Deep Learning primitives encapsulated inside `CNNPredictiveEngine`.
It tests the unified DRY loop (`_train_and_validate`), tensor structural contracts, and 
MONAI lazy loading pointers. Executes on CPU for CI/CD speed.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# MONAI Imports
from monai.data import DataLoader, Dataset

# Import the monolithic Engine
from EfficientNet import CNNPredictiveEngine


class TestCNNEngine(unittest.TestCase):
    """Test suite covering network instantiations, gradient graph building, and Dictionary Data I/O."""
    
    def setUp(self):
        """Initializes the Engine in Dummy Mode to bypass disk I/O and build synthetic tensors."""
        self.mock_logger = MagicMock()
        self.device = torch.device("cpu")
        
        # Initialize the Engine in is_dummy=True mode
        self.engine = CNNPredictiveEngine(logger=self.mock_logger, device=self.device, is_dummy=True, inner_folds=2, outer_folds=2)
        
        # Construct synthetic Dictionary-based batch (4 Patients, 64x64x64)
        self.data_dicts = [
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(0, dtype=torch.long)},
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(1, dtype=torch.long)},
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(0, dtype=torch.long)},
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(1, dtype=torch.long)}
        ]
        
        # Create Loaders and Model using the Engine's internal methods
        self.loader = self.engine._create_dataloader(self.data_dicts, batch_size=2, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        self.model = self.engine._prepare_model_for_parallelism()

    def test_model_output_tensor_shape(self):
        """Validates the structural contract of the Neural Network's final classification layer."""
        self.model.eval()
        
        batch_images = torch.stack([d["image"] for d in self.data_dicts])
        with torch.no_grad():
            output = self.model(batch_images)
            
        expected_shape = (4, 2)
        self.assertEqual(output.shape, expected_shape, "Mismatch in final classification layer geometry.")

    def test_train_and_validate_fast_mode(self):
        """
        Verifies the DRY Unified Optimization Loop in 'Grid Search Mode' (No patience).
        Ensures gradients are processed and loss is computed without checkpoint saving.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Execute 1 epoch. patience=None forces it to skip weight cloning logic
        v_loss, best_state = self.engine._train_and_validate(
            self.model, self.loader, self.loader, optimizer, criterion, max_epochs=1, patience=None
        )
        
        self.assertIsInstance(v_loss, float, "Loss must be collapsed into a scalar float")
        self.assertTrue(v_loss > 0, "Cross-Entropy loss must be strictly positive")
        self.assertIsNone(best_state, "Fast tuning mode should not return a saved state dictionary")

    @patch('EfficientNet.os.path.exists')
    @patch('EfficientNet.pd.read_csv')
    def test_load_data_dicts_mocked(self, mock_read_csv, mock_exists):
        """Mocks the filesystem to validate the generation of MONAI Lazy Loading pointers."""
        mock_exists.return_value = True
        
        # Mock CSV containing a single patient
        mock_df = pd.DataFrame({'subject_id': ['SUB_01'], 'file_path': ['fake.nii'], 'label': [1]})
        mock_read_csv.return_value = mock_df
        
        # Call static method on the Class
        subjects, data_dicts, y_data = CNNPredictiveEngine.load_data_dicts("dummy.csv")
        
        self.assertEqual(len(subjects), 1)
        self.assertEqual(data_dicts[0]["image"], "fake.nii", "Image path must be mapped correctly for Lazy Loading")
        self.assertEqual(data_dicts[0]["label"], 1, "Label parsing failed")


if __name__ == '__main__':
    unittest.main()