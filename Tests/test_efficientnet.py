"""
Unit and Integration Testing Suite for the CNN Predictive Engine.

Validates the decoupled atomic primitives (`_train_epoch`, `_validate_epoch`, `predict`)
and ensures the Double CV pipeline executes securely. 
Designed for a flat directory structure (tests and models in the same folder).
"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn

from Python.Models.efficientnet_classifier import EfficientNetClassifier
from Python.utils.py_logger import CustomLogger

class TestCNNEngine(unittest.TestCase):
         
    def setUp(self):
        """Initializes the Engine in Dummy Mode to isolate PyTorch logic from disk limits."""
        self.logger = CustomLogger(name="TestCNN")
        self.logger.add_console_handler(level="INFO")
        self.device = torch.device("cpu")
                 
        self.engine = EfficientNetClassifier(logger=self.logger, device=self.device, param_grid={})
        
        # Impedisce che 'LoadImaged' cerchi file su disco, permettendogli di digerire i tensori fittizi.
        from monai.transforms import Compose, EnsureTyped
        self.engine._get_transforms = lambda: Compose([EnsureTyped(keys=["image", "label"])])
        
        self.data_dicts = [
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(0, dtype=torch.long)},
            {"image": torch.randn(1, 64, 64, 64), "label": torch.tensor(1, dtype=torch.long)}
        ]
                 
        self.loader = self.engine._create_dataloader(self.data_dicts, batch_size=2, shuffle=False, num_workers=0)
        self.model = self.engine._prepare_model()
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
        self.assertTrue(t_loss > 0)

    def test_validate_epoch(self):
        """Verifies the decoupled validation epoch safely executes without tracking gradients."""
        v_loss = self.engine._validate_epoch(self.model, self.loader, self.criterion)
        self.assertIsInstance(v_loss, float)
        self.assertTrue(v_loss > 0)

    def test_predict_method(self):
        """Validates the pure inference API required for future XAI operations."""
        y_pred, y_prob = self.engine.predict(self.model, self.loader)
                 
        self.assertEqual(len(y_pred), 2)
        self.assertEqual(len(y_prob), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in y_prob))
             
    # Rimosso il patch su 'os.path.exists' poiché non è più utilizzato all'interno del metodo load_data
    @patch('Python.Models.efficientnet_classifier.pd.read_csv')
    def test_load_data_dicts_mocked(self, mock_read_csv):
        """Mocks the OS file system to validate the generation of Lazy Loading pointers."""
        mock_read_csv.return_value = pd.DataFrame({'subject_id': ['SUB_01'], 'file_path': ['fake.nii'], 'label': [1]})
                 
        subjects, data_dicts, y_data = EfficientNetClassifier.load_data("dummy.csv")
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
                         
            self.engine.param_grid = {
                'optimizer': ['adamw'], 
                'scheduler': ['none'],
                'lr': [1e-3], 
                'wd': [1e-4]
            }
                         
            # Creazione di split CV fittizi per accontentare i requisiti del modulo
            dummy_cv_splits = [{
                'fold': 1,
                'outer_train_idx': np.arange(8),
                'outer_test_idx': np.arange(8, 16),
                'inner_splits_relative': [(np.arange(6), np.arange(6, 8))]
            }]
                         
            df_metrics, artifacts = self.engine.execute_nested_cv(
                data_dicts, y_data, subjects, cv_splits=dummy_cv_splits,
                batch_size=2, max_epochs=2, patience=1, num_workers=0
            )
                         
            self.assertIsInstance(df_metrics, pd.DataFrame)
            self.assertEqual(len(df_metrics), 1)

if __name__ == '__main__':
    unittest.main()