"""
Module: test_xai_efficientnet.py

Unit testing suite targeting the DLExplainableAI class.
Employs flat direct imports and isolates computational boundaries 
using minimal PyTorch synthetic tensors to validate Integrated Gradients logic.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Importazioni locali dalla cartella unificata Models
from XAI_EfficientNet import DLExplainableAI
from py_logger import CustomLogger


class Simple3DCNN(nn.Module):
    """Minimal CNN to trace and mathematically validate gradient backpropagation."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(4*4*4, 2)
        
        # Override weights deterministically for testing
        nn.init.ones_(self.conv.weight)
        nn.init.ones_(self.fc.weight)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestDLExplainableAI(unittest.TestCase):

    def setUp(self):
        self.logger = CustomLogger(name="TestXAI_DL")
        self.logger.add_console_handler(level="INFO")
        self.device = torch.device("cpu")
        self.xai_engine = DLExplainableAI(logger=self.logger, device=self.device)
        self.model = Simple3DCNN().to(self.device)

    def test_integrated_gradients_math(self):
        """Validates that Riemann approximation converges and gradients flow backward to input."""
        with self.logger.context(Task="IG_Math"):
            input_tensor = torch.ones((1, 1, 4, 4, 4), dtype=torch.float32)
            baseline = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
            
            ig_map = self.xai_engine.compute_integrated_gradients(
                self.model, input_tensor, target_class=1, baseline=baseline, steps=10
            )
            
            # Ensure shape mapping is perfectly aligned
            self.assertEqual(ig_map.shape, (1, 1, 4, 4, 4))
            
            # Since input=1, baseline=0, and weights=1, IG must be strictly positive
            self.assertTrue(torch.all(ig_map > 0))

    def test_benjamini_hochberg_fdr(self):
        """Validates the native NumPy FDR multi-test correction thresholding."""
        # 5 p-values: first 2 are significant, last 3 are noise
        p_values = np.array([0.001, 0.015, 0.04, 0.3, 0.9])
        
        mask = self.xai_engine._benjamini_hochberg_fdr(p_values, alpha=0.05)
        
        # BH FDR expects indices 0 and 1 to pass, but index 2 (0.04) fails the stringency 
        # because 0.04 > (3/5 * 0.05 = 0.03)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])

    def test_compute_statistical_mask(self):
        """
        Integration test verifying Shapiro-Wilk routing and final 3D shape reconstruction.
        Uses small dimensions (2x2x2 = 8 voxels) to execute rapidly.
        """
        with self.logger.context(Task="Statistical_Stitching"):
            N_AD, N_CTRL = 6, 6
            D, H, W = 2, 2, 2
            
            np.random.seed(42)
            # Create separable arrays so T-Test flags them as significant
            ad_maps = np.random.normal(loc=1.0, scale=0.1, size=(N_AD, 1, D, H, W))
            ctrl_maps = np.random.normal(loc=-1.0, scale=0.1, size=(N_CTRL, 1, D, H, W))
            
            all_ig_maps = np.concatenate([ad_maps, ctrl_maps], axis=0)[:, 0, ...]
            all_labels = np.array([1]*N_AD + [0]*N_CTRL)
            
            stat_mask = self.xai_engine.compute_statistical_mask(all_ig_maps, all_labels, alpha=0.05)
            
            self.assertEqual(stat_mask.shape, (D, H, W))
            # Given the extreme synthetic mean separation, all 8 voxels should survive FDR
            self.assertEqual(np.sum(stat_mask), 8)

if __name__ == '__main__':
    unittest.main()