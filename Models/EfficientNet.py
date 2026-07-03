"""
Main Execution and Orchestration Script for the 3D EfficientNet CNN.

This module provides a fully self-contained deep learning pipeline built on MONAI 
and PyTorch. It implements Dictionary-based Lazy Loading, medical pre-processing transforms, 
Multi-GPU DataParallel scaling, and an optimized unified execution loop (_train_and_validate).
"""
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix
)

# MONAI Framework Native Components
from monai.networks.nets import EfficientNetBN
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped

from py_logger import CustomLogger

# --- GLOBAL CONFIGURATION CONSTANTS ---
DEFAULT_RANDOM_STATE = 42
DEFAULT_INNER_FOLDS = 3 
DEFAULT_OUTER_FOLDS = 5
CNN_LR_GRID = [1e-3, 1e-4]
CNN_WD_GRID = [1e-4, 1e-5]


class CNNPredictiveEngine:
    """
    Monolithic Orchestration Engine for 3D EfficientNet-B0 architectures.
    Houses hardware topology maps, MONAI transform queues, training gradient 
    graphs, and Early Stopping evaluation checkpoints.
    """

    def __init__(self, logger: Any, device: torch.device, is_dummy: bool = False, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        self.logger = logger
        self.device = device
        self.is_dummy = is_dummy
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.random_state = DEFAULT_RANDOM_STATE
        
        self.gpu_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        if self.gpu_count > 1:
            self.logger.info(f"HPC Parallelization: Distributing batches across {self.gpu_count} GPUs via DataParallel.")

    @staticmethod
    def load_data_dicts(csv_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """Generates lightweight Dictionary pointers mapping strings to disk assets (Lazy Loading)."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing registry CSV file: {csv_path}")

        df = pd.read_csv(csv_path)
        subjects, data_dicts, y_list = [], [], []
        
        for _, row in df.iterrows():
            lbl = int(row['label'])
            subjects.append(str(row['subject_id']))
            y_list.append(lbl)
            data_dicts.append({"image": row['file_path'], "label": lbl})
            
        return np.array(subjects), data_dicts, np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Internal helper to compute safe clinical metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = float('nan')

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUROC': auc_score
        }

    def _get_transforms(self) -> Compose:
        """Defines the MONAI data-stream transformation queue."""
        keys = ["image"]
        transforms = []
        if not self.is_dummy:
            transforms.extend([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), ScaleIntensityd(keys=keys)])
        transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
        return Compose(transforms)

    def _prepare_model_for_parallelism(self) -> nn.Module:
        """Instantiates EfficientNet-B0 3D, scaling it to Multi-GPU containers if available."""
        model = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2)
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _create_dataloader(self, subset_dicts: List[Dict[str, Any]], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool = False) -> DataLoader:
        """Assembles a multi-threaded asynchronous MONAI stream iterator."""
        dataset = Dataset(data=subset_dicts, transform=self._get_transforms())
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )

    def _train_and_validate(self, model: nn.Module, tr_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, max_epochs: int, patience: int = None) -> Tuple[float, Dict]:
        """
        Unified Optimization Loop (DRY Pattern Core).
        If 'patience' is None, it acts as a rapid grid-search explorer. 
        If 'patience' is passed, it activates full Early Stopping monitoring.
        """
        best_val_loss, epochs_no_improve, best_state_dict = float('inf'), 0, None
        
        for epoch in range(max_epochs):
            # Gradient Extraction Phase
            model.train()
            for batch_data in tr_loader:
                inputs = batch_data["image"].to(self.device, non_blocking=True)
                labels = batch_data["label"].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                
            # Validation Audit Phase (Autograd Frozen)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data in val_loader:
                    inputs = batch_data["image"].to(self.device, non_blocking=True)
                    labels = batch_data["label"].to(self.device, non_blocking=True)
                    val_loss += criterion(model(inputs), labels).item()
            val_loss /= max(1, len(val_loader))

            # Early Stopping Checkpoint Logic
            if val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = val_loss, 0
                if patience is not None:
                    # Deep copy of the weights to system RAM
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if patience and epochs_no_improve >= patience:
                    break
                    
        return best_val_loss, best_state_dict

    def execute_nested_cv(self, data_dicts: List[Dict[str, Any]], y: np.ndarray, subjects: np.ndarray, 
                          batch_size: int = 4, max_epochs: int = 30, patience: int = 5, num_workers: int = 4) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Orchestrates the macro Deep Learning Double Cross-Validation architecture."""
        pin_memory = True if self.device.type == 'cuda' else False
        self.logger.info(f"Starting MONAI-Native EfficientNet Nested CV: {self.outer_folds} Outer Folds.")
        
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        fold_metrics_list, fold_artifacts = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            self.logger.debug(f"--- Processing Outer Fold {fold_idx}/{self.outer_folds} ---")
            
            train_dicts = [data_dicts[i] for i in train_idx]
            test_dicts = [data_dicts[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- INNER CV: Hyperparameter Grid Search ---
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
            best_grid_loss, best_lr, best_wd = float('inf'), CNN_LR_GRID[0], CNN_WD_GRID[0]

            for lr in CNN_LR_GRID:
                for wd in CNN_WD_GRID:
                    combo_losses = []
                    for in_tr_idx, in_val_idx in inner_cv.split(np.zeros(len(y_train)), y_train):
                        in_tr_loader = self._create_dataloader([train_dicts[i] for i in in_tr_idx], batch_size, True, num_workers, pin_memory, drop_last=True)
                        in_val_loader = self._create_dataloader([train_dicts[i] for i in in_val_idx], batch_size, False, num_workers, pin_memory, drop_last=False)

                        model_cv = self._prepare_model_for_parallelism()
                        optimizer_cv = optim.AdamW(model_cv.parameters(), lr=lr, weight_decay=wd)
                        
                        # Call unified method in tuning acceleration mode
                        v_loss, _ = self._train_and_validate(model_cv, in_tr_loader, in_val_loader, optimizer_cv, nn.CrossEntropyLoss(), max_epochs=3, patience=None)
                        combo_losses.append(v_loss)
                    
                    avg_loss = float(np.mean(combo_losses))
                    if avg_loss < best_grid_loss:
                        best_grid_loss, best_lr, best_wd = avg_loss, lr, wd

            self.logger.debug(f"Fold {fold_idx} tuning complete. Selected LR: {best_lr}, WD: {best_wd}")

            # --- OUTER CV: Final Structural Training with Early Stopping ---
            X_tr_idx, X_val_idx, y_tr, _ = train_test_split(np.arange(len(train_dicts)), y_train, test_size=0.2, stratify=y_train, random_state=self.random_state)
            
            tr_loader = self._create_dataloader([train_dicts[i] for i in X_tr_idx], batch_size, True, num_workers, pin_memory, drop_last=True)
            val_loader = self._create_dataloader([train_dicts[i] for i in X_val_idx], batch_size, False, num_workers, pin_memory, drop_last=False)
            te_loader = self._create_dataloader(test_dicts, batch_size, False, num_workers, pin_memory, drop_last=False)

            model = self._prepare_model_for_parallelism()
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
            
            # Execute training with patience limits active to capture regularized weights
            _, best_state = self._train_and_validate(model, tr_loader, val_loader, optimizer, nn.CrossEntropyLoss(), max_epochs=max_epochs, patience=patience)
            if best_state: 
                model.load_state_dict(best_state)

            # --- OUTER CV: Out-of-Sample Predictive Evaluation ---
            model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for batch_data in te_loader:
                    logits = model(batch_data["image"].to(self.device, non_blocking=True))
                    all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    
            y_pred, y_prob = np.array(all_preds), np.array(all_probs)
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            fold_artifacts.append({
                'fold_id': fold_idx, 'optimal_lr': best_lr, 'optimal_wd': best_wd,
                'test_subjects': subjects[test_idx], 'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob
            })

        self.logger.info("EfficientNet Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts


if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
        
    logger = CustomLogger()
    logger.add_console_handler(level="DEBUG")
    
    with logger.context(session_id="CNN_Predictive_Run"):
        logger.info("--- Starting EfficientNet Monolithic Pipeline ---")
        
        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OS_WORKERS = 0 if sys.platform == "win32" else 2

        USE_DUMMY_DATA = True
        CSV_DATASET_PATH = "data/dataset_info.csv"

        if USE_DUMMY_DATA:
            logger.info("Generating spatial volume dictionary mappings (64x64x64)...")
            N_SAMPLES, D, H, W = 40, 64, 64, 64
            np.random.seed(DEFAULT_RANDOM_STATE)
            
            subjects = np.array([f"SUBJ_{i:03d}" for i in range(N_SAMPLES)])
            y_data = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))
            np.random.shuffle(y_data)
            
            data_dicts = [{"image": np.random.randn(1, D, H, W).astype(np.float32), "label": int(lbl)} for lbl in y_data]
            logger.success("Dummy dictionary structures generated safely.")
        else:
            subjects, data_dicts, y_data = CNNPredictiveEngine.load_data_dicts(CSV_DATASET_PATH)

        engine = CNNPredictiveEngine(logger=logger, device=comp_device, is_dummy=USE_DUMMY_DATA)
        df_metrics, artifacts = engine.execute_nested_cv(
            data_dicts, y_data, subjects, batch_size=4, max_epochs=20, patience=5, num_workers=OS_WORKERS
        )
        
        print("\n--- FINAL OUT-OF-SAMPLE CNN PERFORMANCE ---")
        print(df_metrics.to_string(index=False))