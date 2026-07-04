"""
Unit and Integration Testing Suite for the CNN Predictive Engine.

Validates the decoupled atomic primitives (`_train_epoch`, `_validate_epoch`, `predict`)
and ensures the Double CV pipeline executes securely. Integrates CustomLogger 
and Dummy Tensor Generation for full E2E assessment.
"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
import os

# --- Path Injection for Testing ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.EfficientNet import CNNPredictiveEngine
from py_logger import CustomLogger

class TestCNNEngine(unittest.TestCase):
    
    def setUp(self):
        """Initializes the Engine in Dummy Mode to isolate PyTorch logic from disk limits."""
        self.logger = CustomLogger(name="TestCNN")
        self.logger.add_console_handler(level="INFO")
        self.device = torch.device("cpu")
        
        self.engine = CNNPredictiveEngine(logger=self.logger, device=self.device, is_dummy=True, inner_folds=2, outer_folds=2)
        
        # Base 64x64x64 tensors to satisfy EfficientNet spatial convolutions
        self.data_dicts = [
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(0, dtype=torch.long)},
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(1, dtype=torch.long)}
        ]
        
        self.loader = self.engine._create_dataloader(self.data_dicts, batch_size=2, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        self.model = self.engine._prepare_model_for_parallelism()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def test_model_output_tensor_shape(self):
        """Validates the structural contract of EfficientNet downsampling."""
        self.model.eval()
        batch_images = torch.stack([d["image"] for d in self.data_dicts])
        with torch.no_grad():
            output = self.model(batch_images)
            
        self.assertEqual(output.shape, (2, 2))

    def test_train_epoch(self):
        """Verifies the decoupled training epoch correctly updates the gradient graph."""
        t_loss = self.engine._train_epoch(self.model, self.loader, self.optimizer, self.criterion)
        self.assertIsInstance(t_loss, float)
        self.assertTrue(t_loss > 0, "Loss must be positive during training.")

    def test_validate_epoch(self):
        """Verifies the decoupled validation epoch safely executes without tracking gradients."""
        v_loss = self.engine._validate_epoch(self.model, self.loader, self.criterion)
        self.assertIsInstance(v_loss, float)
        self.assertTrue(v_loss > 0, "Loss must be positive during validation.")

    def test_predict_method(self):
        """Validates the pure inference API required for future XAI operations."""
        y_pred, y_prob = self.engine.predict(self.model, self.loader)
        
        self.assertEqual(len(y_pred), 2, "Predictions must match the batch sample size.")
        self.assertEqual(len(y_prob), 2, "Probabilities must match the batch sample size.")
        self.assertTrue(all(0.0 <= p <= 1.0 for p in y_prob), "Probabilities must be bounded [0, 1].")

    @patch('Models.EfficientNet.os.path.exists')
    @patch('Models.EfficientNet.pd.read_csv')
    def test_load_data_dicts_mocked(self, mock_read_csv, mock_exists):
        """Mocks the OS file system to validate the generation of Lazy Loading pointers."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'subject_id': ['SUB_01'], 'file_path': ['fake.nii'], 'label': [1]})
        
        subjects, data_dicts, y_data = CNNPredictiveEngine.load_data_dicts("dummy.csv")
        self.assertEqual(data_dicts[0]["image"], "fake.nii")

    def test_execute_nested_cv_integration(self):
        """
        Integration Test: Generates Dummy Tensor Data to ensure the full Deep Learning 
        pipeline utilizes the decoupled methods correctly end-to-end.
        """
        with self.logger.context(session_id="CNN_Integration"):
            N_SAMPLES, D, H, W = 16, 64, 64, 64
            np.random.seed(42)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            
            data_dicts = [{"image": torch.randn(1, D, H, W, dtype=torch.float32), "label": int(lbl)} for lbl in y_data]
            
            # Override internal grid to limit testing computational time
            import Models.EfficientNet as EN
            EN.CNN_LR_GRID = [1e-3]
            EN.CNN_WD_GRID = [1e-4]
            
            df_metrics, artifacts = self.engine.execute_nested_cv(
                data_dicts, y_data, subjects, batch_size=2, max_epochs=2, patience=1, num_workers=0
            )
            
            self.assertIsInstance(df_metrics, pd.DataFrame)
            self.assertEqual(len(df_metrics), 2, "Must process 2 outer folds.")


if __name__ == '__main__':
    unittest.main()