import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os
import json
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score

# Local imports
from logging_utils import CustomLogger
from efficientnet_utils import (
    get_or_create_splits, 
    compute_integrated_gradients, 
    voxel_wise_permutation_test, 
    build_efficientnet_3d, 
    load_real_data_efficientnet
)

# Global default initialization (console only), will be overwritten in the main block
logger = CustomLogger(enable_file_logging=False, level="DEBUG")

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, device, batch_size=2, max_epochs=10, patience=3, n_splits_int=5):
    """Training and evaluation with Nested CV, AdamW, Scheduler, Early Stopping and XAI."""
    
    # 1. Baseline IG Calculation: Mean image of CTRL subjects in the training fold
    idx_ctrl_train = (y_train == 0)
    if isinstance(X_train, torch.Tensor):
        ctrl_mean_baseline = X_train[idx_ctrl_train].mean(dim=0, keepdim=True).to(device)
    else:
        ctrl_mean_baseline = torch.tensor(X_train[idx_ctrl_train].mean(axis=0, keepdims=True), dtype=torch.float32).to(device)
    
    # --- 2. INTERNAL GRID SEARCH (Nested CV) ---
    param_grid = {'lr': [1e-3, 1e-4], 'weight_decay': [1e-4, 1e-5]}
    best_grid_loss = float('inf')
    best_lr = 1e-3
    best_wd = 1e-4
    
    logger.info(f"Starting Internal Grid Search ({n_splits_int}-fold)...")
    skf_inner = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=42)

    for lr in param_grid['lr']:
        for wd in param_grid['weight_decay']:
            combo_val_losses = []
            
            for in_tr_idx, in_val_idx in skf_inner.split(X_train, y_train):
                X_in_tr, y_in_tr = X_train[in_tr_idx], y_train[in_tr_idx]
                X_in_val, y_in_val = X_train[in_val_idx], y_train[in_val_idx]
                
                if not isinstance(X_in_tr, torch.Tensor): X_in_tr = torch.tensor(X_in_tr, dtype=torch.float32)
                if not isinstance(y_in_tr, torch.Tensor): y_in_tr = torch.tensor(y_in_tr, dtype=torch.long)
                if not isinstance(X_in_val, torch.Tensor): X_in_val = torch.tensor(X_in_val, dtype=torch.float32)
                if not isinstance(y_in_val, torch.Tensor): y_in_val = torch.tensor(y_in_val, dtype=torch.long)

                in_tr_loader = DataLoader(TensorDataset(X_in_tr, y_in_tr), batch_size=batch_size, shuffle=True, drop_last=True)
                in_val_loader = DataLoader(TensorDataset(X_in_val, y_in_val), batch_size=batch_size, shuffle=False)

                model_cv = build_efficientnet_3d().to(device)
                optimizer_cv = optim.AdamW(model_cv.parameters(), lr=lr, weight_decay=wd) 
                criterion = nn.CrossEntropyLoss()
                scheduler_cv = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cv, mode='min', factor=0.5, patience=3)

                fold_best_val = float('inf')
                for epoch in range(max_epochs):
                    model_cv.train()
                    for inputs, labels in in_tr_loader:
                        optimizer_cv.zero_grad()
                        loss = criterion(model_cv(inputs.to(device)), labels.to(device))
                        loss.backward()
                        optimizer_cv.step()

                    model_cv.eval()
                    with torch.no_grad():
                        val_loss = sum(criterion(model_cv(i.to(device)), l.to(device)).item() for i, l in in_val_loader) / max(1, len(in_val_loader))
                    
                    scheduler_cv.step(val_loss)
                    if val_loss < fold_best_val: fold_best_val = val_loss

                combo_val_losses.append(fold_best_val)
                
            avg_val_loss = np.mean(combo_val_losses)
            logger.debug(f"Grid Params [lr={lr}, wd={wd}] -> Avg Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_grid_loss:
                best_grid_loss = avg_val_loss
                best_lr = lr
                best_wd = wd
                
    logger.info(f"Best hyperparameters found: lr={best_lr}, weight_decay={best_wd}")
    fold_hyperparams = {"learning_rate": best_lr, "weight_decay": best_wd, "batch_size": batch_size}

    # --- 3. FINAL TRAINING ON OUTER FOLD (with Early Stopping) ---
    logger.debug("Starting final model training with optimal parameters...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    if not isinstance(X_tr, torch.Tensor): X_tr = torch.tensor(X_tr, dtype=torch.float32)
    if not isinstance(y_tr, torch.Tensor): y_tr = torch.tensor(y_tr, dtype=torch.long)
    if not isinstance(X_val, torch.Tensor): X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor): y_val = torch.tensor(y_val, dtype=torch.long)
    if not isinstance(X_test, torch.Tensor): X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor): y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    model = build_efficientnet_3d().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch_stopped = max_epochs
    
    for epoch in range(max_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(i.to(device)), l.to(device)).item() for i, l in val_loader) / max(1, len(val_loader))
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logger.debug(f"Early stopping at epoch {epoch+1}. Loss did not improve for {patience} epochs.")
            best_epoch_stopped = epoch + 1 - patience
            break
            
    fold_hyperparams["early_stopping_epoch"] = best_epoch_stopped
            
    # -- 4. TEST & PREDICTIONS --
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_targets.extend(labels.numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    
    try:
        auc_val = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_val = float('nan')
        logger.warn("ROC AUC could not be calculated: only one class present in the test set.")

    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'auc': auc_val,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(all_targets, all_preds, zero_division=0)
    }
    
    # -- 5. XAI: Integrated Gradients --
    ig_maps_ad, ig_maps_ctrl, all_ig_maps, all_ig_labels = [], [], [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        baseline_batch = ctrl_mean_baseline.expand(inputs.size(0), -1, -1, -1, -1)
        attributions = compute_integrated_gradients(inputs, model, target_class=1, steps=20, baseline=baseline_batch)
        
        for i in range(inputs.size(0)):
            attr_np = attributions[i].cpu().numpy()
            attr_norm = attr_np / (np.sum(np.abs(attr_np)) + 1e-8) 
            all_ig_maps.append(attr_norm)
            all_ig_labels.append(labels[i].item())
            
            if labels[i] == 1: ig_maps_ad.append(attr_norm)
            else: ig_maps_ctrl.append(attr_norm)
                
    ig_mean_AD = np.mean(ig_maps_ad, axis=0) if ig_maps_ad else None
    ig_mean_CTRL = np.mean(ig_maps_ctrl, axis=0) if ig_maps_ctrl else None
    ig_contrast = ig_mean_AD - ig_mean_CTRL if (ig_mean_AD is not None and ig_mean_CTRL is not None) else None
    
    # -- 6. IG Maps Significance --
    p_map, sig_mask = voxel_wise_permutation_test(np.stack(all_ig_maps), all_ig_labels, n_perm=100, alpha=0.05)
    
    ig_contrast_masked = np.copy(ig_contrast) if ig_contrast is not None else None
    if ig_contrast_masked is not None: 
        ig_contrast_masked[~sig_mask] = 0.0

    xai_maps = {
        'ig_mean_AD': ig_mean_AD,
        'ig_mean_CTRL': ig_mean_CTRL,
        'ig_contrast_AD_minus_CTRL': ig_contrast,
        'ig_pmap_perm_fdr': p_map,
        'ig_contrast_masked': ig_contrast_masked
    }
    
    return metrics, xai_maps, fold_hyperparams, all_probs, all_preds

if __name__ == "__main__":
    run_id = f"run_{int(time.time())}_seed42_EfficientNet"
    base_out_dir = f"results/runs/{run_id}"
    os.makedirs(base_out_dir, exist_ok=True)
    summary_dir = os.path.join(base_out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    logger = CustomLogger(log_file_path=f"{base_out_dir}/pipeline_efficientnet.log", enable_file_logging=True, level="DEBUG")
    
    # Configurazione e logging del Device (CUDA/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        logger.info(f"✅ GPU Acceleration enabled: {torch.cuda.get_device_name(0)} (CUDA available)")
    else:
        logger.warn("⚠️ CUDA not available! Training is running on CPU and might be extremely slow.")
        
    USE_DUMMY_DATA = True
    
    if USE_DUMMY_DATA:
        torch.manual_seed(42)
        n_subj = 40
        subjects = np.array([f"sub_{i:03d}" for i in range(n_subj)])
        X_data = torch.randn(n_subj, 1, 32, 32, 32) 
        # Balanced dummy data generation to avoid ROC AUC errors
        y_np = np.array([0]*(n_subj//2) + [1]*(n_subj//2))
        np.random.shuffle(y_np)
        y_data = torch.tensor(y_np, dtype=torch.long)
    else:
        subjects, X_data, y_data = load_real_data_efficientnet("data/dataset_info.csv")
    
    splits = get_or_create_splits(subjects, y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data, n_splits_ext=5, n_repeats=2, logger=logger)
    
    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'sensitivity': [], 'specificity': [], 'f1_score': []}
    all_hyperparams = []
    all_ig_contrasts = []
    all_ig_masked = []
    
    for outer_id, train_idx, test_idx in splits:
        logger.info(f"--- Starting Outer Fold {outer_id} ---")
        fold_dir = os.path.join(base_out_dir, "folds", f"outer_{outer_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        test_subjects = subjects[test_idx]
        test_y_true = y_data[test_idx].numpy() if isinstance(y_data, torch.Tensor) else y_data[test_idx]
        
        metrics, xai_maps, fold_hyp, probs, preds = train_and_evaluate_fold(
            X_data[train_idx], y_data[train_idx], X_data[test_idx], y_data[test_idx], 
            device, batch_size=2, n_splits_int=5
        )

        if xai_maps['ig_contrast_AD_minus_CTRL'] is not None:
            all_ig_contrasts.append(xai_maps['ig_contrast_AD_minus_CTRL'])
            all_ig_masked.append(xai_maps['ig_contrast_masked'])
        
        logger.info(f"Fold {outer_id} Metrics -> Bal. Acc: {metrics['balanced_accuracy']:.3f} | Sens: {metrics['sensitivity']:.3f} | Spec: {metrics['specificity']:.3f} | F1: {metrics['f1_score']:.3f} | AUC: {metrics['auc']:.3f}")
        
        with open(os.path.join(fold_dir, "hyperparams_selected.json"), 'w') as f:
            json.dump(fold_hyp, f, indent=4)
        with open(os.path.join(fold_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        pd.DataFrame({"subject_id": test_subjects, "y_true": test_y_true, "score": probs, "y_pred": preds}).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        
        fold_hyp['fold_id'] = outer_id
        all_hyperparams.append(fold_hyp)
        for k in fold_metrics.keys(): fold_metrics[k].append(metrics[k])

    df_hyper = pd.DataFrame(all_hyperparams)
    df_hyper.to_csv(os.path.join(summary_dir, "hyperparams_distribution.csv"), index=False)
    
    consensus_data = {
        "median_early_stopping_epoch": df_hyper["early_stopping_epoch"].median(),
        "learning_rate_used": float(df_hyper["learning_rate"].mode()[0]),
        "weight_decay_used": float(df_hyper["weight_decay"].mode()[0]),
        "batch_size_used": int(df_hyper["batch_size"].mode()[0])
    }
    with open(os.path.join(summary_dir, "hyperparams_consensus.json"), 'w') as f:
        json.dump(consensus_data, f, indent=4)

    df_met = pd.DataFrame(fold_metrics)
    df_met['Fold'] = [f"outer_F{i}" for i in range(len(df_met))]
    df_met.to_csv(os.path.join(summary_dir, "metrics_all_folds.csv"), index=False)    
        
    logger.success(f"Hyperparameter Consensus calculated: Median Stopping Epoch = {consensus_data['median_early_stopping_epoch']}")
    logger.success("=== AVERAGE FINAL RESULTS ===")
    logger.success(f"Average Balanced Accuracy: {np.mean(fold_metrics['balanced_accuracy']):.3f} ± {np.std(fold_metrics['balanced_accuracy']):.3f}")
    logger.success(f"Average Sensitivity: {np.mean(fold_metrics['sensitivity']):.3f} ± {np.std(fold_metrics['sensitivity']):.3f}")
    logger.success(f"Average Specificity: {np.mean(fold_metrics['specificity']):.3f} ± {np.std(fold_metrics['specificity']):.3f}")
    logger.success(f"Average F1 Score: {np.mean(fold_metrics['f1_score']):.3f} ± {np.std(fold_metrics['f1_score']):.3f}")
    logger.success(f"Average Accuracy: {np.mean(fold_metrics['accuracy']):.3f} ± {np.std(fold_metrics['accuracy']):.3f}")
    logger.success(f"Average ROC AUC: {np.mean(fold_metrics['auc']):.3f} ± {np.std(fold_metrics['auc']):.3f}")

    maps_dir = "data/maps"
    os.makedirs(maps_dir, exist_ok=True)
    mask_path = "data/gm_mask_MNI.nii.gz"

    if os.path.exists(mask_path) and len(all_ig_contrasts) > 0:
        logger.info("Calculating averages and saving 3D IG NIfTI maps...")
        ref_img = nib.load(mask_path)
        
        mean_ig_contrast = np.mean(all_ig_contrasts, axis=0)
        mean_ig_masked = np.mean(all_ig_masked, axis=0)
        
        ig_3d = np.squeeze(mean_ig_contrast)
        ig_masked_3d = np.squeeze(mean_ig_masked)
        
        nib.save(nib.Nifti1Image(ig_3d, ref_img.affine, ref_img.header), os.path.join(maps_dir, "effnet_ig_mean.nii.gz"))
        nib.save(nib.Nifti1Image(ig_masked_3d, ref_img.affine, ref_img.header), os.path.join(maps_dir, "effnet_ig_masked_mean.nii.gz"))
        
        logger.success(f"IG NIfTI maps successfully saved in {maps_dir}/")
    else:
        logger.warn(f"Mask {mask_path} not found or empty maps. Skipping NIfTI save (normal with Dummy Data without mask).")