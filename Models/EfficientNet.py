"""
Predictive 3D EfficientNet CNN Engine Module.

This module provides a fully self-contained deep learning pipeline built on MONAI 
and PyTorch. It implements Dictionary-based Lazy Loading, medical pre-processing transforms, 
Multi-GPU DataParallel scaling, and decoupled atomic methods for training, validation, and inference.

Designed as a pure library module without global execution blocks.
"""
import os
import sys
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

# MONAI Native Components
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
    Houses hardware topology maps, MONAI transform queues, and decoupled logic 
    for training, validation, and raw inference (XAI-ready).
    """

    def __init__(self, logger: Any, device: torch.device, is_dummy: bool = False, inner_folds: int = DEFAULT_INNER_FOLDS, outer_folds: int = DEFAULT_OUTER_FOLDS):
        """Initializes the DL Engine and inventories the hardware state."""
        self.logger = logger
        self.device = device
        self.is_dummy = is_dummy
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.random_state = DEFAULT_RANDOM_STATE
        
        self.gpu_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        if self.gpu_count > 1:
            self.logger.info(f"HPC Parallelization: Using {self.gpu_count} GPUs.")

    @staticmethod
    def load_data_dicts(csv_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Generates lightweight Dictionary pointers mapping strings to disk assets.

        Purpose:
            Prevents OS memory saturation via Lazy Loading. Generates file coordinate mappings 
            instead of loading massive 3D arrays into RAM. Data is read incrementally during epochs.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing registry CSV file: {csv_path}")

        df = pd.read_csv(csv_path)
        subjects, data_dicts, y_list = [], [], []
        
        for _, row in df.iterrows():
            lbl = int(row['label'])
            subjects.append(str(row['subject_id']))
            y_list.append(lbl)
            # Standard Dictionary format requested by MONAI Datasets
            data_dicts.append({"image": row['file_path'], "label": lbl})
            
        return np.array(subjects), data_dicts, np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Internal helper to compute safe clinical metrics (Accuracy, AUROC)."""
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
        """
        Defines the MONAI data-stream preprocessing queue.
        Purpose: Ensures channel dimensions (1, D, H, W) and standardizes MR intensities.
        """
        keys = ["image"]
        transforms = []
        if not self.is_dummy:
            transforms.extend([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), ScaleIntensityd(keys=keys)])
        transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
        return Compose(transforms)

    def _prepare_model_for_parallelism(self) -> nn.Module:
        """Instantiates EfficientNet-B0 3D, scaling to Multi-GPU via nn.DataParallel."""
        model = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2)
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _create_dataloader(self, subset_dicts: List[Dict[str, Any]], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool = False) -> DataLoader:
        """Assembles a multi-threaded asynchronous MONAI stream iterator."""
        dataset = Dataset(data=subset_dicts, transform=self._get_transforms())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """
        Executes a single Training Epoch (Forward and Backward pass).
        Updates network weights and returns the average batch loss.
        """
        model.train()
        epoch_loss = 0.0
        
        for batch_data in loader:
            # WHY: non_blocking=True allows async memory transfers to the GPU, preventing CPU starvation.
            inputs = batch_data["image"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / max(1, len(loader))

    def _validate_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        """
        Executes a single Validation Epoch.
        Freezes the Autograd engine to save VRAM and evaluates on unseen folds.
        """
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in loader:
                inputs = batch_data["image"].to(self.device, non_blocking=True)
                labels = batch_data["label"].to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        return val_loss / max(1, len(loader))

    def predict(self, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure Inference API.
        Accepts a trained model and a DataLoader, returning discrete predictions 
        and continuous probabilities. Highly decoupled for subsequent XAI extraction.
        """
        model.eval()
        all_preds, all_probs = [], []
        
        with torch.no_grad():
            for batch_data in loader:
                logits = model(batch_data["image"].to(self.device, non_blocking=True))
                # WHY: Extracts probability of the positive class (AD) via Softmax index 1
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                # Evaluates the class with the highest probability
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                
        return np.array(all_preds), np.array(all_probs)

    def execute_nested_cv(self, data_dicts: List[Dict[str, Any]], y: np.ndarray, subjects: np.ndarray, 
                          batch_size: int = 4, max_epochs: int = 30, patience: int = 5, num_workers: int = 4) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Orchestrates the macro Deep Learning Double Cross-Validation architecture."""
        pin_memory = True if self.device.type == 'cuda' else False
        self.logger.info(f"Starting MONAI-Native EfficientNet Nested CV: {self.outer_folds} Folds.")
        
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        fold_metrics_list, fold_artifacts = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            train_dicts, test_dicts = [data_dicts[i] for i in train_idx], [data_dicts[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- INNER CV: Grid Search ---
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
                        criterion_cv = nn.CrossEntropyLoss()
                        
                        # Fast tuning mode: run 3 epochs of training, evaluate only at the end
                        for _ in range(3):
                            self._train_epoch(model_cv, in_tr_loader, optimizer_cv, criterion_cv)
                        
                        v_loss = self._validate_epoch(model_cv, in_val_loader, criterion_cv)
                        combo_losses.append(v_loss)
                    
                    if np.mean(combo_losses) < best_grid_loss:
                        best_grid_loss, best_lr, best_wd = float(np.mean(combo_losses)), lr, wd

            # --- OUTER CV: Final Structural Training ---
            X_tr_idx, X_val_idx, y_tr, _ = train_test_split(np.arange(len(train_dicts)), y_train, test_size=0.2, stratify=y_train, random_state=self.random_state)
            
            tr_loader = self._create_dataloader([train_dicts[i] for i in X_tr_idx], batch_size, True, num_workers, pin_memory, drop_last=True)
            val_loader = self._create_dataloader([train_dicts[i] for i in X_val_idx], batch_size, False, num_workers, pin_memory, drop_last=False)
            te_loader = self._create_dataloader(test_dicts, batch_size, False, num_workers, pin_memory, drop_last=False)

            model = self._prepare_model_for_parallelism()
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
            criterion = nn.CrossEntropyLoss()
            
            best_val_loss, epochs_no_improve, best_state = float('inf'), 0, None
            
            for epoch in range(max_epochs):
                self._train_epoch(model, tr_loader, optimizer, criterion)
                val_loss = self._validate_epoch(model, val_loader, criterion)
                
                # Early Stopping Checkpoint Logic
                if val_loss < best_val_loss:
                    best_val_loss, epochs_no_improve = val_loss, 0
                    # Deep copy of weights to avoid GPU state drifts
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if patience and epochs_no_improve >= patience:
                        self.logger.debug(f"Early Stopping triggered at epoch {epoch+1}")
                        break
            
            if best_state: 
                model.load_state_dict(best_state)

            # --- OUTER CV: Out-of-Sample Predictive Evaluation ---
            # Utilizing the decoupled inference method
            y_pred, y_prob = self.predict(model, te_loader)
            
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            fold_artifacts.append({
                'fold_id': fold_idx, 'optimal_lr': best_lr, 'optimal_wd': best_wd,
                'test_subjects': subjects[test_idx], 'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob
            })

        return pd.DataFrame(fold_metrics_list), fold_artifacts