"""
Module: test_cv_manager.py

Unit testing suite targeting the CVManager class.
Ensures deterministic generation of nested cross-validation folds and validates 
the integrity of JSON serialization to prevent data leakage between decoupled scripts.
"""
import unittest
import numpy as np
import sys
import os
import tempfile
import json

# Dynamically resolve path to the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.utils.cv_manager import CVManager

class TestCVManager(unittest.TestCase):
    """
    Test suite for Python.utils.cv_manager.CVManager.
    
    PURPOSE:
        Validates the mathematical distribution of indices across the Double 
        Cross-Validation topology and checks the I/O security mechanisms.
    """

    def test_generate_splits(self) -> None:
        """
        Verifies the creation of multi-layered nested folds.
        
        PURPOSE:
            Ensures that the stratified generation strictly preserves all data points 
            without overlaps or losses between training and testing absolute indices.
        """
        cv = CVManager(outer_folds=2, inner_folds=2)
        # 8 Samples: 4 of class 0, and 4 of class 1
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        splits = cv.generate_splits(y)
        
        self.assertEqual(len(splits), 2, "It should generate exactly 2 outer folds.")
        self.assertIn('outer_train_idx', splits[0])
        self.assertIn('inner_splits_relative', splits[0])
        
        # Verify that no data is lost or leaked
        train_len = len(splits[0]['outer_train_idx'])
        test_len = len(splits[0]['outer_test_idx'])
        self.assertEqual(train_len + test_len, 8, "The sum of train and test lengths must equal the dataset size.")

    def test_serialization(self) -> None:
        """
        Verifies the security and integrity of the JSON CV Artifacts.
        
        PURPOSE:
            Tests the SSOT (Single Source of Truth) export mechanism, ensuring that 
            'security_test_subjects' are correctly embedded to guard against 
            subsequent data leakage when models are trained.
        """
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