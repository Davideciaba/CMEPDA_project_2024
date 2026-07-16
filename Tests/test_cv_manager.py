import unittest
import numpy as np
import sys
import os
import tempfile
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.utils.cv_manager import CVManager

class TestCVManager(unittest.TestCase):
    def test_generate_splits(self):
        """Verifies the creation of multi-layered nested folds."""
        cv = CVManager(outer_folds=2, inner_folds=2)
        # 8 Campioni, 4 di classe 0 e 4 di classe 1
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        splits = cv.generate_splits(y)
        
        self.assertEqual(len(splits), 2, "Dovrebbero essere generate 2 outer folds.")
        self.assertIn('outer_train_idx', splits[0])
        self.assertIn('inner_splits_relative', splits[0])
        
        # Verifica che nessun dato venga perso
        train_len = len(splits[0]['outer_train_idx'])
        test_len = len(splits[0]['outer_test_idx'])
        self.assertEqual(train_len + test_len, 8, "La somma di test e train deve pareggiare il dataset.")

    def test_serialization(self):
        """Verifies the security and integrity of the JSON CV Artifacts."""
        cv = CVManager(outer_folds=2, inner_folds=2)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        subjects = np.array([f"S_{i}" for i in range(8)])
        
        splits = cv.generate_splits(y)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_folds.json")
            CVManager.save_to_json(splits, subjects, out_file)
            
            loaded_splits = CVManager.load_from_json(out_file)
            
            self.assertEqual(len(loaded_splits), 2)
            self.assertIn('security_test_subjects', loaded_splits[0])
            self.assertEqual(len(loaded_splits[0]['security_test_subjects']), 4)

if __name__ == '__main__':
    unittest.main()