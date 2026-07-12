"""
Predictive 3D EfficientNet CNN Engine Module.

This module provides a fully self-contained deep learning pipeline built on MONAI 
and PyTorch. It implements Dictionary-based Lazy Loading, medical pre-processing transforms, 
Multi-GPU DataParallel scaling, and decoupled atomic methods for training, validation, and inference.

Designed as a pure library module without global execution blocks.
"""
import collections
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
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
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    EnsureTyped,
    ResizeWithPadOrCropd
)

from Python.utils.py_logger import CustomLogger

# --- GLOBAL CONFIGURATION CONSTANTS ---
CNN_LR_GRID = [1e-3, 1e-4]
CNN_WD_GRID = [1e-4, 1e-5]


class EfficientNetClassifier:
    """
    Monolithic Orchestration Engine for 3D EfficientNet-B0 architectures.
    Houses hardware topology maps, MONAI transform queues, and decoupled logic 
    for training, validation, and raw inference (XAI-ready).
    """

    def __init__(self, logger: CustomLogger, device: torch.device, param_grid: Dict[str, List[Any]]):
        """Initializes the DL Engine and inventories the hardware state."""
        self.logger = logger
        self.device = device
        self.param_grid = param_grid
        
        self.gpu_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        if self.gpu_count > 1:
            self.logger.info(f"HPC Parallelization: Using {self.gpu_count} GPUs.")

    @staticmethod
    def load_data(csv_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Generates lightweight Dictionary pointers mapping strings to disk assets.
        """
        df = pd.read_csv(csv_path)

        subjects, data_dicts, y_list = [], [], []
        
        for _, row in df.iterrows():
            lbl = int(row['label'])
            subjects.append(str(row['subject_id']))
            y_list.append(lbl)
            data_dicts.append({"image": row['file_path'], "label": lbl})
            
        return np.array(subjects), data_dicts, np.array(y_list)

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_decision: np.ndarray) -> Dict[str, float]:
        """Internal helper to compute safe clinical metrics (Accuracy, AUROC)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc_score = roc_auc_score(y_true, y_decision)
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
        """Defines the MONAI data-stream preprocessing queue."""
        keys = ["image"]
        transforms = []
        transforms.extend([
            LoadImaged(keys=keys), 
            EnsureChannelFirstd(keys=keys),
            # Spatial Formatting: Adapts 121x145x121 inputs to strict 128x128x128 dimensions.
            # Mode "constant" ensures background voxel additions equal 0 by default.
            # Method "symmetric" balances both padding and cropping mathematically.
            ResizeWithPadOrCropd(
                keys=keys,
                spatial_size=(128, 128, 128),
                method="symmetric",
                mode="constant"
            ),
            ScaleIntensityd(keys=keys)
        ])
        transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
        return Compose(transforms)

    def _prepare_model(self) -> nn.Module:
        """Instantiates EfficientNet-B0 3D, scaling to Multi-GPU via nn.DataParallel."""
        model = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2)
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _create_dataloader(self, subset_dicts: List[Dict[str, Any]], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        """Assembles a multi-threaded asynchronous MONAI stream iterator."""
        pin_memory = True if self.device.type == 'cuda' else False
        dataset = Dataset(data=subset_dicts, transform=self._get_transforms())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Executes a single Training Epoch (Forward and Backward pass)."""
        model.train()
        epoch_loss = 0.0
        
        for batch_data in loader:
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
        """Executes a single Validation Epoch."""
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
    
    def _average_weights(self, state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Performs Polyak/Stochastic Weight Averaging across a list of model states."""
        avg_dict = {k: v.clone() for k, v in state_dicts[0].items()}
        for key in avg_dict.keys():
            for i in range(1, len(state_dicts)):
                avg_dict[key] += state_dicts[i][key]
            avg_dict[key] = torch.div(avg_dict[key], len(state_dicts))
        return avg_dict

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
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                
        return np.array(all_preds), np.array(all_probs)

    def execute_nested_cv(self, data_dicts: List[Dict[str, Any]], y: np.ndarray, subjects: np.ndarray, 
                          cv_splits: List[Dict[str, Any]], batch_size: int = 4, max_epochs: int = 30, 
                          use_early_stopping: bool = True, use_swa: bool = True, 
                          patience: int = 5, min_delta: float = 1e-4, swa_n: int = 5, 
                          num_workers: int = 4) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Orchestrates Deep Learning CV with Nested Grid Search.
        Provides Modular toggles for Inner-Fold Early Stopping and SWA.
        """
        self.logger.info(f"Starting CNN Nested CV (Grid Search: {len(CNN_LR_GRID)} LR x {len(CNN_WD_GRID)} WD).")
        self.logger.info(f"Features - Early Stopping: {use_early_stopping} | SWA: {use_swa}")
        
        fold_metrics_list, fold_artifacts = [], []

        for split in cv_splits:
            fold_idx = split['fold']
            self.logger.info(f"--- Processing Outer Fold {fold_idx}/{len(cv_splits)} ---")
            
            train_idx, test_idx = split['outer_train_idx'], split['outer_test_idx']
            train_dicts = [data_dicts[i] for i in train_idx]
            test_dicts = [data_dicts[i] for i in test_idx]
            y_test = y[test_idx]
            inner_iterator = split['inner_splits_relative']
            
            # --- PHASE 1: NESTED GRID SEARCH ON INNER CV ---
            best_grid_val_loss = float('inf')
            best_lr, best_wd = self.param_grid['lr'][0], self.param_grid['wd'][0]
            best_outer_target_epochs = max_epochs

            for lr in self.param_grid['lr']:
                for wd in self.param_grid['wd']:
                    self.logger.debug(f"Grid Search Combo -> LR: {lr}, WD: {wd}")
                    combo_val_losses = []
                    combo_best_epochs = []
                    combo_early_stopped = []

                    for in_tr_idx, in_val_idx in inner_iterator:
                        in_tr_loader = self._create_dataloader([train_dicts[i] for i in in_tr_idx], batch_size, True, num_workers)
                        in_val_loader = self._create_dataloader([train_dicts[i] for i in in_val_idx], batch_size, False, num_workers)

                        model_cv = self._prepare_model()
                        optimizer_cv = optim.AdamW(model_cv.parameters(), lr=lr, weight_decay=wd)
                        criterion_cv = nn.CrossEntropyLoss()
                        
                        in_best_val_loss = float('inf')
                        patience_counter = 0
                        best_epoch = max_epochs # Default if ES not used
                        stopped = False

                        # Sliding window buffer for Stochastic Weight Averaging
                        swa_buffer = collections.deque(maxlen=swa_n)

                        for epoch in range(max_epochs):
                            self._train_epoch(model_cv, in_tr_loader, optimizer_cv, criterion_cv)
                            epoch_val_loss = self._validate_epoch(model_cv, in_val_loader, criterion_cv)
                            
                            if use_swa:
                                swa_buffer.append(copy.deepcopy(model_cv.state_dict()))

                            if use_early_stopping:
                                if epoch_val_loss < in_best_val_loss - min_delta:
                                    in_best_val_loss = epoch_val_loss
                                    best_epoch = epoch
                                    patience_counter = 0
                                else:
                                    patience_counter += 1

                                if patience_counter >= patience:
                                    stopped = True
                                    break
                            else:
                                # If no early stopping, we just track the loss for the grid search comparison
                                in_best_val_loss = epoch_val_loss

                        # End of Inner Fold Training
                        if use_swa and len(swa_buffer) > 0:
                            avg_inner_state = self._average_weights(list(swa_buffer))
                            model_cv.load_state_dict(avg_inner_state)
                            # Optional: Re-evaluate val_loss with SWA weights to make Grid Search decision
                            in_best_val_loss = self._validate_epoch(model_cv, in_val_loader, criterion_cv)

                        combo_val_losses.append(in_best_val_loss)
                        combo_best_epochs.append(best_epoch)
                        combo_early_stopped.append(stopped)

                    # Evaluate Grid Combination
                    avg_combo_val_loss = float(np.mean(combo_val_losses))
                    
                    if avg_combo_val_loss < best_grid_val_loss:
                        best_grid_val_loss = avg_combo_val_loss
                        best_lr, best_wd = lr, wd
                        
                        # Determine Outer Target Epochs for this winning combo
                        if use_early_stopping:
                            majority_stopped = sum(combo_early_stopped) > (len(inner_iterator) / 2)
                            if majority_stopped:
                                # Conservative approach
                                best_outer_target_epochs = max(combo_best_epochs) + patience
                            else:
                                best_outer_target_epochs = max_epochs
                        else:
                            best_outer_target_epochs = max_epochs

            self.logger.info(f"Optimal Grid Combo -> LR: {best_lr}, WD: {best_wd} | Target Epochs: {best_outer_target_epochs}")

            # --- PHASE 2: FINAL OUTER CV TRAINING (FULL TRAIN SET) ---
            full_tr_loader = self._create_dataloader(train_dicts, batch_size, True, num_workers)
            te_loader = self._create_dataloader(test_dicts, batch_size, False, num_workers)

            model = self._prepare_model()
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
            criterion = nn.CrossEntropyLoss()
            
            outer_swa_buffer = collections.deque(maxlen=swa_n)
            
            for epoch in range(best_outer_target_epochs):
                self._train_epoch(model, full_tr_loader, optimizer, criterion)
                
                if use_swa:
                    outer_swa_buffer.append(copy.deepcopy(model.state_dict()))

            if use_swa and len(outer_swa_buffer) > 0:
                final_swa_state = self._average_weights(list(outer_swa_buffer))
                model.load_state_dict(final_swa_state)

            # --- PHASE 3: PREDICTION & EVALUATION ---
            y_pred, y_prob = self.predict(model, te_loader)
            
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            fold_metrics_list.append(metrics)
            
            fold_artifacts.append({
                'fold_id': fold_idx, 'optimal_lr': best_lr, 'optimal_wd': best_wd,
                'target_epochs': best_outer_target_epochs, 'test_subjects': subjects[test_idx], 
                'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob
            })

        self.logger.info("EfficientNet Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts