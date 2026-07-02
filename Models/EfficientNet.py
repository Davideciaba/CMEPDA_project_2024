"""
EfficientNet Predictive Engine and Main Execution Script.

This module houses the `CNNPredictiveEngine` class, which governs the Nested Cross-Validation 
and High-Performance hardware distribution (Multi-GPU). It delegates the internal epoch 
loops to the `CNNUtils` static class, functioning as the main entry point for the Deep Learning pipeline.
"""
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, List, Tuple, Any

from py_logger import CustomLogger
from CMEPDA_project_2024.Models.efficientnet_utils import CNNUtils

# --- GLOBAL CONFIGURATION CONSTANTS ---
DEFAULT_RANDOM_STATE = 42
DEFAULT_INNER_FOLDS = 3 # Reduced compared to SVM due to DL computational cost
DEFAULT_OUTER_FOLDS = 5
CNN_LR_GRID = [1e-3, 1e-4]
CNN_WD_GRID = [1e-4, 1e-5]


class CNNPredictiveEngine:
    """
    Orchestration Engine for 3D EfficientNet Nested CV evaluation.
    
    This class manages data splitting, manual grid search for hyperparameter tuning, 
    and incorporates hardware-aware parallelization (DataParallel and Multi-Processing) 
    to maximize VRAM utilization.
    """

    def __init__(self, logger: Any, device: torch.device, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        """
        Initializes the CNN Engine and interrogates the hardware topology.
        
        Parameters:
            logger (Any): Injected logger instance.
            device (torch.device): Detected primary compute unit (CUDA/CPU).
            inner_folds (int): Hyperparameter tuning splits.
            outer_folds (int): Generalization evaluation splits.
        """
        self.logger = logger
        self.device = device
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        
        # Hardware topology detection for Multi-GPU environments
        self.gpu_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        if self.gpu_count > 1:
            self.logger.info(f"Hardware parallelization activated: Detected {self.gpu_count} GPUs.")

    def _prepare_model_for_parallelism(self) -> nn.Module:
        """
        Instantiates the MONAI EfficientNet3D and safely wraps it for Distributed execution.
        
        Purpose:
            If a server has 4 GPUs, initializing standard PyTorch will only utilize GPU:0, 
            causing an immediate memory overflow. Wrapping the model in `nn.DataParallel` 
            instructs PyTorch to split the incoming batch across all detected GPUs automatically.
        """
        model = CNNUtils.build_model(model_name="efficientnet-b0")
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool = False) -> DataLoader:
        """
        Constructs a multi-threaded PyTorch DataLoader optimized for 3D tensors.
        
        Purpose:
            Translates static NumPy arrays into a highly optimized data stream. The loader 
            uses `num_workers` to pre-fetch files using background CPU threads.
            
        Parameters:
            X, y: Feature and target tensors.
            batch_size: Iteration step size.
            shuffle: Whether to randomize selection (True for Training).
            num_workers: Number of active CPU background threads.
            pin_memory: Whether to lock memory pages for ultra-fast GPU transfer.
            drop_last: (CRITICAL) If True, discards the last incomplete batch. 
                       Must be True for Training to prevent BatchNorm 3D failures on batch_size=1.
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            drop_last=drop_last
        )

    def execute_nested_cv(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, 
                          batch_size: int = 4, max_epochs: int = 30, patience: int = 5,
                          num_workers: int = 4) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes the exhaustive Double CV training and evaluation pipeline for Deep Learning.
        
        Purpose:
            Conducts an internal grid search for Learning Rate and Weight Decay optimization, 
            followed by a final training pass with Early Stopping to prevent overfitting 
            on the outer fold dataset.
        """
        pin_memory = True if self.device.type == 'cuda' else False
        self.logger.info(f"Starting Multi-Threaded EfficientNet Nested CV: {self.outer_folds} Outer Folds.")
        
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        fold_metrics_list = []
        fold_artifacts = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{self.outer_folds} ---")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subjects = subjects[test_idx]

            # --- INNER CV: Hyperparameter Tuning ---
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            best_grid_loss = float('inf')
            best_lr, best_wd = CNN_LR_GRID[0], CNN_WD_GRID[0]

            for lr in CNN_LR_GRID:
                for wd in CNN_WD_GRID:
                    combo_losses = []
                    for in_tr_idx, in_val_idx in inner_cv.split(X_train, y_train):
                        # WHY: drop_last=True is enforced on training loaders to ensure BatchNorm stability.
                        in_tr_loader = self._create_dataloader(X_train[in_tr_idx], y_train[in_tr_idx], batch_size, True, num_workers, pin_memory, drop_last=True)
                        in_val_loader = self._create_dataloader(X_train[in_val_idx], y_train[in_val_idx], batch_size, False, num_workers, pin_memory, drop_last=False)

                        model_cv = self._prepare_model_for_parallelism()
                        optimizer_cv = optim.AdamW(model_cv.parameters(), lr=lr, weight_decay=wd)
                        criterion = nn.CrossEntropyLoss()

                        # Fast tuning: Limits epochs to prevent grid search from taking weeks.
                        fold_min_val = float('inf')
                        for _ in range(5):
                            CNNUtils.train_one_epoch(model_cv, in_tr_loader, optimizer_cv, criterion, self.device)
                            v_loss = CNNUtils.evaluate_model(model_cv, in_val_loader, criterion, self.device)
                            fold_min_val = min(fold_min_val, v_loss)
                        combo_losses.append(fold_min_val)
                    
                    avg_loss = float(np.mean(combo_losses))
                    if avg_loss < best_grid_loss:
                        best_grid_loss = avg_loss
                        best_lr, best_wd = lr, wd
            
            self.logger.debug(f"Fold {fold_idx} tuning complete. Optimal LR: {best_lr}, WD: {best_wd}")

            # --- OUTER CV: Final Model Training (Early Stopping) ---
            # Generate a 20% validation split specifically to monitor for overfitting
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=DEFAULT_RANDOM_STATE)
            
            tr_loader = self._create_dataloader(X_tr, y_tr, batch_size, True, num_workers, pin_memory, drop_last=True)
            val_loader = self._create_dataloader(X_val, y_val, batch_size, False, num_workers, pin_memory, drop_last=False)
            te_loader = self._create_dataloader(X_test, y_test, batch_size, False, num_workers, pin_memory, drop_last=False)

            model = self._prepare_model_for_parallelism()
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
            criterion = nn.CrossEntropyLoss()
            
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_state_dict = None
            
            for epoch in range(max_epochs):
                CNNUtils.train_one_epoch(model, tr_loader, optimizer, criterion, self.device)
                val_loss = CNNUtils.evaluate_model(model, val_loader, criterion, self.device)
                
                # Early Stopping Logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # WHY: v.cpu().clone() explicitly copies the weights to CPU RAM. 
                    # Without clone(), PyTorch maintains a pointer to the live GPU tensor, 
                    # and the "best state" would be overwritten during the next epoch.
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        self.logger.debug(f"Early Stopping triggered at epoch {epoch+1}")
                        break
            
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            # --- OUTER CV: Predictive Evaluation (Out-of-Sample) ---
            model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for inputs, _ in te_loader:
                    logits = model(inputs.to(self.device))
                    # Extract the softmax probability belonging to the positive class (Index 1)
                    all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    
            y_pred, y_prob = np.array(all_preds), np.array(all_probs)
            
            metrics = CNNUtils.evaluate_dl_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            fold_artifacts.append({
                'fold_id': fold_idx,
                'optimal_lr': best_lr,
                'optimal_wd': best_wd,
                'test_subjects': test_subjects,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            })

        self.logger.info("EfficientNet Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts


if __name__ == "__main__":
    # Windows OS requires explicit multiprocessing protection
    if sys.platform == "win32":
        multiprocessing.freeze_support()
        
    logger = CustomLogger(enable_file_logging=False, level="DEBUG")
    
    with logger.context(session_id="CNN_Predictive_Run"):
        logger.info("--- Starting EfficientNet 3D Predictive Pipeline ---")
        
        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if comp_device.type == 'cuda':
            logger.success(f"GPU Engine Engaged: {torch.cuda.get_device_name(0)}")
        else:
            logger.warn("WARNING: No CUDA device found. Executing on CPU.")
            
        # Dynamically allocate I/O threads based on Operating System
        OS_WORKERS = 0 if sys.platform == "win32" else 2

        USE_DUMMY_DATA = True
        CSV_DATASET_PATH = "data/dataset_info.csv"

        if USE_DUMMY_DATA:
            logger.info("Generating DUMMY DATA (3D Volumes) for standalone testing...")
            N_SAMPLES, N_CHANNELS, D, H, W = 40, 1, 32, 32, 32
            np.random.seed(DEFAULT_RANDOM_STATE)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            X_data = np.random.randn(N_SAMPLES, N_CHANNELS, D, H, W)
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            logger.success("Dummy 3D Data successfully generated.")
        else:
            logger.info(f"Loading REAL DATA from {CSV_DATASET_PATH}...")
            subjects, X_data, y_data = CNNUtils.load_real_data(CSV_DATASET_PATH)
            logger.success("Real NIfTI data loaded and expanded to 5D PyTorch format.")

        logger.debug(f"Input Tensor Shape: {X_data.shape}. Class Distribution: {np.bincount(y_data)}")

        engine = CNNPredictiveEngine(logger=logger, device=comp_device)
        df_metrics, artifacts = engine.execute_nested_cv(
            X_data, y_data, subjects, 
            batch_size=4, max_epochs=20, patience=5, num_workers=OS_WORKERS
        )
        
        logger.success("EfficientNet Pipeline execution completed. Results overview:")
        print("\n--- FINAL CNN METRICS (OUT-OF-SAMPLE) ---")
        print(df_metrics.to_string(index=False))