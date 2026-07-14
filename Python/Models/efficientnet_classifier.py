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
    confusion_matrix,
    roc_curve
)

# MONAI Native Components
from monai.networks.nets import EfficientNetBN
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,  
    EnsureTyped,
    SpatialPadd
)

from Python.utils.py_logger import CustomLogger



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

        # Ensure default grid keys exist to avoid KeyError if not provided by user
        if 'optimizer' not in self.param_grid: self.param_grid['optimizer'] = ['adamw']
        if 'scheduler' not in self.param_grid: self.param_grid['scheduler'] = ['none']
        if 'lr' not in self.param_grid: self.param_grid['lr'] = [1e-3]
        if 'wd' not in self.param_grid: self.param_grid['wd'] = [1e-2]
        
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
            EnsureChannelFirstd(keys=keys), # Ensure the correct dimension (C, H, W)
            # Spatial Padding: Adapts 121x145x121 inputs to 160x160x160 dimensions.
            # Mode "constant" ensures background voxel additions equal 0 by default.
            # Method "symmetric" balances padding mathematically.
            SpatialPadd(
                keys=keys,
                spatial_size=(160, 160, 160),
                method="symmetric",
                mode="constant"
            )
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

    def _configure_optimizer(self, model: nn.Module, opt_name: str, lr: float, wd: float) -> optim.Optimizer:
        """Dynamically configures the optimizer based on string name."""
        opt_name = opt_name.lower()
        if opt_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    def _configure_scheduler(self, optimizer: optim.Optimizer, sched_name: str):
        """Dynamically configures the learning rate scheduler."""
        sched_name = sched_name.lower()
        if sched_name == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10)
        elif sched_name == 'exp':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return None

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
                          cv_splits: List[Dict[str, Any]], batch_size: int = 4, max_epochs: int = 50, 
                          use_early_stopping: bool = True, use_swa: bool = True, 
                          patience: int = 10, min_delta: float = 1e-4, swa_n: int = 5, 
                          num_workers: int = 4) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Orchestrates Deep Learning CV with Nested Grid Search.
        Provides Modular toggles for Inner-Fold Early Stopping and SWA.
        """
        total_combos = len(self.param_grid['lr']) * len(self.param_grid['wd']) * len(self.param_grid['optimizer']) * len(self.param_grid['scheduler'])
        self.logger.info(f"Starting EfficientNet Nested CV (Grid Search: {total_combos} combinations).")
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
            best_params = {'lr': None, 'wd': None, 'opt': None, 'sched': None}
            best_outer_target_epochs = max_epochs

            # Telemetry for plotting inner folds
            optimal_inner_loss_history = {} 
            best_inner_bal_acc_mean = 0.0

            for opt_name in self.param_grid['optimizer']:
                for sched_name in self.param_grid['scheduler']:
                    for lr in self.param_grid['lr']:
                        for wd in self.param_grid['wd']:
                            combo_str = f"Opt:{opt_name}, Sched:{sched_name}, LR:{lr}, WD:{wd}"
                            self.logger.debug(f"Grid Search Combo -> {combo_str}")
                            
                            combo_val_losses = []
                            combo_best_epochs = []
                            combo_early_stopped = []
                            combo_bal_accs = []
                            combo_loss_history = {} # Store history for this combo

                            for inner_fold_idx, (in_tr_idx, in_val_idx) in enumerate(inner_iterator):
                                in_tr_loader = self._create_dataloader([train_dicts[i] for i in in_tr_idx], batch_size, True, num_workers)
                                in_val_loader = self._create_dataloader([train_dicts[i] for i in in_val_idx], batch_size, False, num_workers)

                                model_cv = self._prepare_model()
                                optimizer_cv = self._configure_optimizer(model_cv, opt_name, lr, wd)
                                scheduler_cv = self._configure_scheduler(optimizer_cv, sched_name)
                                criterion_cv = nn.CrossEntropyLoss()
                                
                                in_best_val_loss = float('inf')
                                patience_counter = 0
                                best_epoch = max_epochs # Default if ES not used
                                stopped = False
                                history = {'train_loss': [], 'val_loss': []}

                                # Sliding window buffer for Stochastic Weight Averaging
                                swa_buffer = collections.deque(maxlen=swa_n)

                                for epoch in range(max_epochs):
                                    tr_loss = self._train_epoch(model_cv, in_tr_loader, optimizer_cv, criterion_cv)
                                    val_loss = self._validate_epoch(model_cv, in_val_loader, criterion_cv)
                                    
                                    if scheduler_cv:
                                        scheduler_cv.step()
                                        
                                    history['train_loss'].append(tr_loss)
                                    history['val_loss'].append(val_loss)

                                    if use_swa:
                                        swa_buffer.append(copy.deepcopy(model_cv.state_dict()))

                                    if use_early_stopping:
                                        if val_loss < in_best_val_loss - min_delta:
                                            in_best_val_loss = val_loss
                                            best_epoch = epoch
                                            patience_counter = 0
                                        else:
                                            patience_counter += 1

                                        if patience_counter >= patience:
                                            stopped = True
                                            break
                                    else:
                                        # If no early stopping, we just track the loss for the grid search comparison
                                        in_best_val_loss = val_loss

                                # End of Inner Fold Training
                                if use_swa and len(swa_buffer) > 0:
                                    avg_inner_state = self._average_weights(list(swa_buffer))
                                    model_cv.load_state_dict(avg_inner_state)
                                    # Re-evaluate val_loss with SWA weights to make Grid Search decision
                                    in_best_val_loss = self._validate_epoch(model_cv, in_val_loader, criterion_cv)

                                # Calculate Inner Balanced Accuracy for this fold
                                y_val_true = np.array([train_dicts[i]['label'] for i in in_val_idx])
                                y_val_pred, _ = self.predict(model_cv, in_val_loader)
                                combo_bal_accs.append(balanced_accuracy_score(y_val_true, y_val_pred))

                                combo_val_losses.append(in_best_val_loss)
                                combo_best_epochs.append(best_epoch)
                                combo_early_stopped.append(stopped)
                                combo_loss_history[f"inner_fold_{inner_fold_idx}"] = history

                            # Evaluate Grid Combination
                            avg_combo_val_loss = float(np.mean(combo_val_losses))
                            
                            if avg_combo_val_loss < best_grid_val_loss:
                                best_grid_val_loss = avg_combo_val_loss
                                best_params = {'lr': lr, 'wd': wd, 'opt': opt_name, 'sched': sched_name}
                                optimal_inner_loss_history = combo_loss_history
                                best_inner_bal_acc_mean = float(np.mean(combo_bal_accs))
                                
                                # Determine Outer Target Epochs for this winning combo
                                if use_early_stopping:
                                    majority_stopped = sum(combo_early_stopped) > (len(inner_iterator) / 2)
                                    if majority_stopped:
                                        # Use the max of the best epochs (plus a tiny 10% buffer)
                                        base_epochs = max(combo_best_epochs)
                                        best_outer_target_epochs = int(round(base_epochs + 1 + base_epochs/10))
                                        # Ensure we don't exceed max_epochs
                                        best_outer_target_epochs = min(best_outer_target_epochs, max_epochs)
                                    else:
                                        best_outer_target_epochs = max_epochs
                                else:
                                    best_outer_target_epochs = max_epochs

            self.logger.info(f"Optimal Grid Combo -> {best_params} | Target Epochs: {best_outer_target_epochs}")

            # --- PHASE 2: FINAL OUTER CV TRAINING (FULL TRAIN SET) ---
            full_tr_loader = self._create_dataloader(train_dicts, batch_size, True, num_workers)
            te_loader = self._create_dataloader(test_dicts, batch_size, False, num_workers)

            model = self._prepare_model()
            optimizer = self._configure_optimizer(model, best_params['opt'], best_params['lr'], best_params['wd'])
            scheduler = self._configure_scheduler(optimizer, best_params['sched'])
            criterion = nn.CrossEntropyLoss()
            
            outer_swa_buffer = collections.deque(maxlen=swa_n)
            outer_loss_history = {'train_loss': [], 'test_loss': []} # Track test loss purely for telemetry/plotting
            
            for epoch in range(best_outer_target_epochs):
                tr_loss = self._train_epoch(model, full_tr_loader, optimizer, criterion)
                
                # Evaluate on test set just for the learning curves (NOT for early stopping)
                te_loss = self._validate_epoch(model, te_loader, criterion)
                
                if scheduler:
                    scheduler.step()
                    
                outer_loss_history['train_loss'].append(tr_loss)
                outer_loss_history['test_loss'].append(te_loss)
                
                if use_swa:
                    outer_swa_buffer.append(copy.deepcopy(model.state_dict()))

            if use_swa and len(outer_swa_buffer) > 0:
                final_swa_state = self._average_weights(list(outer_swa_buffer))
                model.load_state_dict(final_swa_state)

            # --- PHASE 3: PREDICTION & EVALUATION ---
            y_pred, y_prob = self.predict(model, te_loader)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            metrics['Fold'] = fold_idx
            metrics['Inner_CV_BalAcc_Mean'] = best_inner_bal_acc_mean # Added for pipeline aggregation
            fold_metrics_list.append(metrics)
            
            # Save comprehensive artifacts for ModelRenderer
            fold_artifacts.append({
                'fold_id': fold_idx, 
                'optimal_params': best_params,
                'target_epochs': best_outer_target_epochs, 
                'test_subjects': subjects[test_idx], 
                'y_true': y_test, 
                'y_pred': y_pred, 
                'y_prob': y_prob,
                'roc_fpr': fpr, # Added for ROC plotting
                'roc_tpr': tpr, # Added for ROC plotting
                'inner_loss_history': optimal_inner_loss_history, # Added for plotting
                'outer_loss_history': outer_loss_history # Added for plotting
            })

        self.logger.info("EfficientNet Nested CV Evaluation Completed.")
        return pd.DataFrame(fold_metrics_list), fold_artifacts