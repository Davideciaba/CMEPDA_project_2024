"""
Module: test_xai_efficientnet.py

Unit testing suite targeting the DLExplainableAI class.
Employs flat direct imports. Integrates natively with the actual CNNPredictiveEngine 
to validate gradient backpropagation directly on the project's MONAI EfficientNet topology.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Importazioni locali dalla cartella unificata Models
from XAI_EfficientNet import DLExplainableAI
from py_logger import CustomLogger


class TestDLExplainableAI(unittest.TestCase):

    def setUp(self):
        """Initializes the engine with strict CustomLogger dependency and true network topology."""
        self.logger = CustomLogger(name="TestXAI_DL")
        self.logger.add_console_handler(level="INFO")
        self.device = torch.device("cpu")
        
        self.xai_engine = DLExplainableAI(logger=self.logger, device=self.device)
        
        # Instantiate the real CNN Engine in Dummy Mode to extract the actual architecture
        self.dl_engine = CNNPredictiveEngine(logger=self.logger, device=self.device, is_dummy=True)
        self.model = self.dl_engine._prepare_model_for_parallelism()

    def test_integrated_gradients_math(self):
        """Validates that Riemann approximation computes properly and matches ig.py shape logic."""
        with self.logger.context(Task="IG_Math_RealNet"):
            # Shape mapping: (Batch, Channel, Depth, Height, Width)
            input_tensor = torch.ones((1, 1, 32, 32, 32), dtype=torch.float32)
            
            # Using steps=2 to keep unit testing rapid on CPU constraints
            ig_map = self.xai_engine.compute_integrated_gradients(
                self.model, input_tensor, target_class=1, baseline_name="z", steps=2
            )
            
            # The returned map has the batch dimension squeezed out (Channel, Depth, Height, Width)
            self.assertEqual(ig_map.shape, (1, 32, 32, 32))
            
            # Ensures Autograd successfully tracked through all EfficientNet blocks without NaN decay
            self.assertFalse(np.isnan(ig_map).any())

    def test_aggregate_global_maps(self):
        """Ensures cross-fold element-wise aggregation yields an exact mean."""
        map1 = np.array([1.0, 2.0, 3.0])
        map2 = np.array([3.0, 4.0, 5.0])
        
        aggregated = DLExplainableAI.aggregate_global_maps([map1, map2])
        expected = np.array([2.0, 3.0, 4.0])
        
        np.testing.assert_array_almost_equal(aggregated, expected)

if __name__ == '__main__':
    unittest.main()