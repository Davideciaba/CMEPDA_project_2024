"""
Cross-Validation Manager Module.

Centralizes the generation of Outer and Inner folds to guarantee absolute 
synchronization between independent predictive engines (e.g., SVM and EfficientNet).
"""
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
            - 'final_train_idx_relative': 80% split relative to outer_train_idx (for CNNs).
            - 'final_val_idx_relative': 20% split relative to outer_train_idx (for CNNs).
        """
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=False, random_state=self.random_state)
        splits_registry = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            y_train = y[train_idx]
            
            # INNER CV: Generated relatively to y_train for Scikit-Learn GridSearchCV compatibility
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
            inner_splits_relative = list(inner_cv.split(np.zeros(len(y_train)), y_train))
            
            # FINAL STRUCTURAL SPLIT: 80/20 split on the Training Set for Deep Learning Early Stopping
            # Stratified to maintain class balance in the validation set
            tr_rel_idx, val_rel_idx = train_test_split(
                np.arange(len(y_train)), 
                test_size=0.2, 
                stratify=y_train, 
                random_state=self.random_state
            )

            splits_registry.append({
                'fold': fold_idx,
                'outer_train_idx': train_idx,
                'outer_test_idx': test_idx,
                'inner_splits_relative': inner_splits_relative,
                'final_train_idx_relative': tr_rel_idx,
                'final_val_idx_relative': val_rel_idx
            })
            
        return splits_registry