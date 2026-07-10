"""
Module: test_XAI_EfficientNet.py

Unit testing suite targeting the DLExplainableAI class.
Employs a bulletproof path injection to reliably locate project namespaces.
"""
import sys
import os

# --- BULLETPROOF PATH INJECTION ---
# Catturiamo il percorso assoluto in cui si trova QUESTO file (la cartella Tests)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Saliamo di un livello. In base a come hai strutturato il progetto, 
# questo dovrebbe portarci dentro CMEPDA_project_2024 o dentro Python
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Se c'è una sottocartella 'Python', quella è la nostra root dei moduli.
# Altrimenti, la root è la parent_dir stessa.
python_src_dir = os.path.join(parent_dir, 'Python')

if os.path.exists(python_src_dir) and python_src_dir not in sys.path:
    sys.path.insert(0, python_src_dir)
elif parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# =====================================================================
# DEBUG: Decommenta questa riga se l'errore persiste, 
# stamperà esattamente dove Python sta cercando i file.
# print("PYTHON PATH ATTUALI:", sys.path)
# =====================================================================

import unittest
import numpy as np
import torch

# Importiamo dai rispettivi package della struttura del progetto
from Python.XAI.XAI_EfficientNet import DLExplainableAI

from Python.Models.efficientnet_classifier import EfficientNetClassifier as ModelEngine

from Python.utils.py_logger import CustomLogger


class TestDLExplainableAI(unittest.TestCase):

    def setUp(self):
        """Initializes the engine with the actual CustomLogger and true network topology."""
        self.logger = CustomLogger(name="TestXAI_DL")
        self.logger.add_console_handler(level="INFO")
        self.device = torch.device("cpu")
        
        self.xai_engine = DLExplainableAI(logger=self.logger, device=self.device)
        
        # Instantiate the real CNN Engine in Dummy Mode to extract the actual architecture
        # Usiamo l'alias ModelEngine gestito nel blocco di importazione
        self.dl_engine = ModelEngine(logger=self.logger, device=self.device, is_dummy=True)
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