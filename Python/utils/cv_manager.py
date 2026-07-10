"""
Cross-Validation Manager Module.

Centralizes the generation of Outer and Inner folds to guarantee absolute 
synchronization between independent predictive engines (e.g., SVM and EfficientNet).
"""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Dict, Any

class CVManager:
    """
    Generates deterministic splits for Nested Cross-Validation.
    Provides both Absolute indices (for Outer folds) and Relative indices 
    (for Inner GridSearchCV and Deep Learning Early Stopping).
    """
    def __init__(self, outer_folds: int = 5, inner_folds: int = 5, random_state: int = 42):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state

    def generate_splits(self, y: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes the Stratified K-Fold indices based on the target array 'y'.
        
        Returns:
            A list of dictionaries. Each dictionary contains:
            - 'fold': Integer fold number.
            - 'outer_train_idx': Absolute indices for training.
            - 'outer_test_idx': Absolute indices for testing.
            - 'inner_splits_relative': List of tuples (in_tr, in_val) relative to outer_train_idx.
        """
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        splits_registry = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            y_train = y[train_idx]
            
            # INNER CV: Generated relatively to y_train for Scikit-Learn GridSearchCV compatibility
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
            inner_splits_relative = list(inner_cv.split(np.zeros(len(y_train)), y_train))

            splits_registry.append({
                'fold': fold_idx,
                'outer_train_idx': train_idx,
                'outer_test_idx': test_idx,
                'inner_splits_relative': inner_splits_relative
            })
            
        return splits_registry

    @staticmethod
    def save_to_json(splits_registry: List[Dict[str, Any]], subjects: np.ndarray, filepath: str) -> None:
        """
        Serializes Numpy indices to standard JSON lists and injects Subject ID 
        signatures to prevent Data Leakage in decoupled scripts.
        """
        serializable_splits = []
        for split in splits_registry:
            s_dict = {}
            for k, v in split.items():
                if isinstance(v, np.ndarray):
                    s_dict[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
                    # Handle inner_splits_relative tuples of numpy arrays
                    s_dict[k] = [[tr.tolist(), val.tolist()] for tr, val in v]
                else:
                    s_dict[k] = v
                    
            # Inject SECURITY SIGNATURE: The exact subject IDs expected in the test fold
            s_dict['security_test_subjects'] = subjects[split['outer_test_idx']].tolist()
            serializable_splits.append(s_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_splits, f, indent=4)

    @staticmethod
    def load_from_json(filepath: str) -> List[Dict[str, Any]]:
        """Loads serialized splits. Lists function identically to arrays for Numpy indexing."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)