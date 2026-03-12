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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from monai.networks.nets import EfficientNetBN
from statsmodels.stats.multitest import fdrcorrection
from loguru import logger

def get_or_create_splits(subjects, y, n_splits_ext=5, n_repeats=5, n_splits_int=5, random_state=42, splits_dir="data/splits"):
    """Genera la Repeated Stratified K-Fold oppure la carica dai JSON se esistono."""
    outer_json_path = os.path.join(splits_dir, "outer_splits.json")
    if os.path.exists(outer_json_path):
        with open(outer_json_path, 'r') as f:
            outer_splits_data = json.load(f)
        splits_to_yield = []
        for split_info in outer_splits_data["splits"]:
            train_idx = np.where(np.isin(subjects, split_info["train_subjects"]))[0]
            test_idx = np.where(np.isin(subjects, split_info["test_subjects"]))[0]
            splits_to_yield.append((split_info["outer_id"], train_idx, test_idx))
        return splits_to_yield

    rskf = RepeatedStratifiedKFold(n_splits=n_splits_ext, n_repeats=n_repeats, random_state=random_state)
    outer_splits_data = {"n_splits_ext": n_splits_ext, "n_repeats": n_repeats, "random_state": random_state, "splits": []}
    splits_to_yield = []
    
    os.makedirs(os.path.join(splits_dir, "inner_splits"), exist_ok=True)
    for split_idx, (train_idx, test_idx) in enumerate(rskf.split(subjects, y)):
        current_repeat, current_fold = split_idx // n_splits_ext, split_idx % n_splits_ext
        outer_id = f"R{current_repeat}_F{current_fold}"
        outer_splits_data["splits"].append({
            "outer_id": outer_id, "repeat": current_repeat, "fold": current_fold,
            "train_subjects": subjects[train_idx].tolist(), "test_subjects": subjects[test_idx].tolist()
        })
        splits_to_yield.append((outer_id, train_idx, test_idx))
        
    with open(outer_json_path, 'w') as f:
        json.dump(outer_splits_data, f, indent=4)
    return splits_to_yield

def compute_integrated_gradients(inputs, model, target_class=1, steps=10, baseline=None):
    """Calcola gli Integrated Gradients (IG) rispetto a una baseline specifica."""
    if baseline is None: 
        baseline = torch.zeros_like(inputs)
        
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_()
        score = model(scaled_input)[:, target_class].sum()
        model.zero_grad()
        score.backward()
        grads.append(scaled_input.grad.detach())
        
    # Approssimazione dell'integrale e moltiplicazione per (input - baseline)
    return (inputs - baseline) * torch.mean(torch.stack(grads), dim=0)

def voxel_wise_permutation_test(ig_maps, labels, n_perm=1000, alpha=0.05):
    """Esegue il permutation test voxel-wise sulle mappe IG (Punto 4.3)."""
    shape_orig = ig_maps.shape
    abs_ig = np.abs(ig_maps.reshape(shape_orig[0], -1))
    labels = np.array(labels)
    if sum(labels == 1) == 0 or sum(labels == 0) == 0: 
        return np.ones(shape_orig[1:]), np.zeros(shape_orig[1:], dtype=bool)

    true_stat = np.mean(abs_ig[labels == 1], axis=0) - np.mean(abs_ig[labels == 0], axis=0)
    count_greater = np.zeros_like(true_stat)
    
    for _ in range(n_perm):
        perm_labels = np.random.permutation(labels)
        p_stat = np.mean(abs_ig[perm_labels == 1], axis=0) - np.mean(abs_ig[perm_labels == 0], axis=0)
        count_greater += (np.abs(p_stat) >= np.abs(true_stat))
        
    p_values_flat = (count_greater + 1) / (n_perm + 1)
    rejected, pval_corrected_flat = fdrcorrection(p_values_flat, alpha=alpha, method='indep')
    return pval_corrected_flat.reshape(shape_orig[1:]), rejected.reshape(shape_orig[1:])

def build_efficientnet_3d(model_name="efficientnet-b0", in_channels=1, num_classes=2):
    return EfficientNetBN(model_name=model_name, spatial_dims=3, in_channels=in_channels, num_classes=num_classes, pretrained=False)

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, device, batch_size=4, max_epochs=10, patience=3, n_splits_int=5):
    """Training e valutazione con Nested CV (Grid Search), IG e test di permutazione."""
    
    # 1. Calcolo Baseline IG: Immagine media dei CTRL del training fold
    idx_ctrl_train = (y_train == 0)
    if isinstance(X_train, torch.Tensor):
        ctrl_mean_baseline = X_train[idx_ctrl_train].mean(dim=0, keepdim=True).to(device)
    else:
        ctrl_mean_baseline = torch.tensor(X_train[idx_ctrl_train].mean(axis=0, keepdims=True), dtype=torch.float32).to(device)
    
    # --- 2. GRID SEARCH INTERNA (Nested CV) ---
    # Griglia degli iperparametri da esplorare (puoi ampliarla!)
    param_grid = {'lr': [1e-3, 1e-4], 'weight_decay': [1e-4, 1e-5]}
    best_grid_loss = float('inf')
    best_lr = 1e-3
    best_wd = 1e-4
    
    logger.info(f"Avvio Grid Search Interna ({n_splits_int}-fold)...")
    skf_inner = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=42)
    
    for lr in param_grid['lr']:
        for wd in param_grid['weight_decay']:
            combo_val_losses = []
            
            for in_tr_idx, in_val_idx in skf_inner.split(X_train, y_train):
                X_in_tr, y_in_tr = X_train[in_tr_idx], y_train[in_tr_idx]
                X_in_val, y_in_val = X_train[in_val_idx], y_train[in_val_idx]
                
                # TensorDataset
                if not isinstance(X_in_tr, torch.Tensor): X_in_tr = torch.tensor(X_in_tr, dtype=torch.float32)
                if not isinstance(y_in_tr, torch.Tensor): y_in_tr = torch.tensor(y_in_tr, dtype=torch.long)
                if not isinstance(X_in_val, torch.Tensor): X_in_val = torch.tensor(X_in_val, dtype=torch.float32)
                if not isinstance(y_in_val, torch.Tensor): y_in_val = torch.tensor(y_in_val, dtype=torch.long)

                in_tr_loader = DataLoader(TensorDataset(X_in_tr, y_in_tr), batch_size=batch_size, shuffle=True, drop_last=True)
                in_val_loader = DataLoader(TensorDataset(X_in_val, y_in_val), batch_size=batch_size, shuffle=False)

                model_cv = build_efficientnet_3d().to(device)
                optimizer_cv = optim.Adam(model_cv.parameters(), lr=lr, weight_decay=wd)
                criterion = nn.CrossEntropyLoss()

                # Addestramento rapido per valutare la combinazione
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
                    if val_loss < fold_best_val: fold_best_val = val_loss

                combo_val_losses.append(fold_best_val)
                
            avg_val_loss = np.mean(combo_val_losses)
            logger.debug(f"Grid Params [lr={lr}, wd={wd}] -> Avg Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_grid_loss:
                best_grid_loss = avg_val_loss
                best_lr = lr
                best_wd = wd
                
    logger.info(f"Migliori iperparametri trovati: lr={best_lr}, weight_decay={best_wd}")
    fold_hyperparams = {"learning_rate": best_lr, "weight_decay": best_wd, "batch_size": batch_size}

    # --- 3. ADDESTRAMENTO FINALE SUL FOLD ESTERNO (con Early Stopping) ---
    logger.debug("Inizio addestramento modello finale con parametri ottimali...")
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
    optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)
    
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            best_epoch_stopped = epoch + 1 - patience
            break
            
    fold_hyperparams["early_stopping_epoch"] = best_epoch_stopped
            
    # -- 4. TEST E PREDICITONS --
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_targets.extend(labels.numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    
    # Prevenzione per dati Dummy: se nel fold di test c'è una sola classe
    try:
        auc_val = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_val = float('nan')
        logger.warning("ROC AUC non calcolabile: solo una classe presente nel test set.")

    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'auc': auc_val,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    # -- 5. XAI: Integrated Gradients --
    ig_maps_ad, ig_maps_ctrl, all_ig_maps, all_ig_labels = [], [], [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        baseline_batch = ctrl_mean_baseline.expand(inputs.size(0), -1, -1, -1, -1)
        attributions = compute_integrated_gradients(inputs, model, target_class=1, steps=10, baseline=baseline_batch)
        
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
    
    # -- 6. Significatività delle Mappe IG --
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

def load_real_data_efficientnet(csv_path):
    df = pd.read_csv(csv_path)
    subjects = df['subject_id'].values
    y_tensor = torch.tensor(df['label'].values, dtype=torch.long)
    volumes = []
    for idx, row in df.iterrows():
        img_tensor = torch.tensor(nib.load(row['nifti_path']).get_fdata(), dtype=torch.float32).unsqueeze(0) 
        volumes.append(img_tensor)
    return subjects, torch.stack(volumes), y_tensor

if __name__ == "__main__":
    run_id = f"run_{int(time.time())}_seed42_EfficientNet"
    base_out_dir = f"results/runs/{run_id}"
    os.makedirs(base_out_dir, exist_ok=True)
    summary_dir = os.path.join(base_out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    logger.add(f"{base_out_dir}/pipeline_efficientnet.log", rotation="10 MB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    USE_DUMMY_DATA = True
    
    if USE_DUMMY_DATA:
        torch.manual_seed(42)
        n_subj = 40
        subjects = np.array([f"sub_{i:03d}" for i in range(n_subj)])
        X_data = torch.randn(n_subj, 1, 32, 32, 32) 
        y_data = torch.randint(0, 2, (n_subj,))
    else:
        subjects, X_data, y_data = load_real_data_efficientnet("data/dataset_info.csv")
    
    splits = get_or_create_splits(subjects, y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data, n_splits_ext=5, n_repeats=2)
    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'sensitivity': [], 'specificity': []}
    all_hyperparams = []

    all_ig_contrasts = []
    all_ig_masked = []
    
    for outer_id, train_idx, test_idx in splits:
        logger.info(f"--- Inizio Fold Esterno {outer_id} ---")
        fold_dir = os.path.join(base_out_dir, "folds", f"outer_{outer_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        test_subjects = subjects[test_idx]
        test_y_true = y_data[test_idx].numpy() if isinstance(y_data, torch.Tensor) else y_data[test_idx]
        
        metrics, xai_maps, fold_hyp, probs, preds = train_and_evaluate_fold(
            X_data[train_idx], y_data[train_idx], X_data[test_idx], y_data[test_idx], 
            device, batch_size=2, n_splits_int=5   # <-- Aggiunto n_splits_int
        )

        if xai_maps['ig_contrast_AD_minus_CTRL'] is not None:
            all_ig_contrasts.append(xai_maps['ig_contrast_AD_minus_CTRL'])
            all_ig_masked.append(xai_maps['ig_contrast_masked'])
        
        # Stampa a schermo le metriche
        logger.info(f"Metriche Fold {outer_id} -> Bal. Acc: {metrics['balanced_accuracy']:.3f} | Sens: {metrics['sensitivity']:.3f} | Spec: {metrics['specificity']:.3f} | AUC: {metrics['auc']:.3f}")
        
        # Salvataggio iperparametri, metriche e predizioni per fold
        with open(os.path.join(fold_dir, "hyperparams_selected.json"), 'w') as f:
            json.dump(fold_hyp, f, indent=4)
        with open(os.path.join(fold_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        pd.DataFrame({"subject_id": test_subjects, "y_true": test_y_true, "score": probs, "y_pred": preds}).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        
        fold_hyp['fold_id'] = outer_id
        all_hyperparams.append(fold_hyp)
        for k in fold_metrics.keys(): fold_metrics[k].append(metrics[k])

    # --- REPORTING IPERPARAMETRI E CONSENSO ---
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

    # NUOVO: Salva il recap delle metriche di tutti i fold in un unico CSV
    df_met = pd.DataFrame(fold_metrics)
    df_met['Fold'] = [f"outer_F{i}" for i in range(len(df_met))]
    df_met.to_csv(os.path.join(summary_dir, "metrics_all_folds.csv"), index=False)    
        
    # STAMPA A SCHERMO I RISULTATI FINALI MEDI
    logger.success(f"Consenso Iperparametri calcolato: Median Stopping Epoch = {consensus_data['median_early_stopping_epoch']}")
    logger.success("=== RISULTATI FINALI MEDI ===")
    logger.success(f"Balanced Accuracy Media: {np.mean(fold_metrics['balanced_accuracy']):.3f} ± {np.std(fold_metrics['balanced_accuracy']):.3f}")
    logger.success(f"Sensitivity Media: {np.mean(fold_metrics['sensitivity']):.3f} ± {np.std(fold_metrics['sensitivity']):.3f}")
    logger.success(f"Specificity Media: {np.mean(fold_metrics['specificity']):.3f} ± {np.std(fold_metrics['specificity']):.3f}")
    logger.success(f"Accuracy Media: {np.mean(fold_metrics['accuracy']):.3f} ± {np.std(fold_metrics['accuracy']):.3f}")
    logger.success(f"AUC ROC Media: {np.mean(fold_metrics['auc']):.3f} ± {np.std(fold_metrics['auc']):.3f}")

    # --- MEDIA E SALVATAGGIO MAPPE NIFTI 3D EFFICIENTNET ---
    maps_dir = "data/maps"
    os.makedirs(maps_dir, exist_ok=True)
    mask_path = "data/gm_mask_MNI.nii.gz"

    if os.path.exists(mask_path) and len(all_ig_contrasts) > 0:
        logger.info("Calcolo medie e salvataggio mappe NIfTI IG 3D...")
        ref_img = nib.load(mask_path)
        
        # 1. Media sui fold
        mean_ig_contrast = np.mean(all_ig_contrasts, axis=0)
        mean_ig_masked = np.mean(all_ig_masked, axis=0)
        
        # 2. Rimuove la dimensione del canale (1, D, H, W) -> (D, H, W) per NIfTI
        ig_3d = np.squeeze(mean_ig_contrast)
        ig_masked_3d = np.squeeze(mean_ig_masked)
        
        # 3. Salvataggio
        nib.save(nib.Nifti1Image(ig_3d, ref_img.affine, ref_img.header), os.path.join(maps_dir, "effnet_ig_mean.nii.gz"))
        nib.save(nib.Nifti1Image(ig_masked_3d, ref_img.affine, ref_img.header), os.path.join(maps_dir, "effnet_ig_masked_mean.nii.gz"))
        
        logger.success(f"Mappe IG NIfTI salvate con successo in {maps_dir}/")
    else:
        logger.warning(f"Maschera {mask_path} non trovata o mappe vuote. Salvataggio NIfTI saltato (normale con Dummy Data senza maschera).")