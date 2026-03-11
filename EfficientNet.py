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

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, device, lr=1e-3, weight_decay=1e-4, batch_size=4, max_epochs=10, patience=3):
    """Training e valutazione con IG, baseline media CTRL e test di permutazione."""
    
    # 1. Calcolo Baseline IG: Immagine media dei CTRL del training fold (Punto 4.2)
    idx_ctrl_train = (y_train == 0)
    if isinstance(X_train, torch.Tensor):
        ctrl_mean_baseline = X_train[idx_ctrl_train].mean(dim=0, keepdim=True).to(device)
    else:
        ctrl_mean_baseline = torch.tensor(X_train[idx_ctrl_train].mean(axis=0, keepdims=True), dtype=torch.float32).to(device)
    
    # 2. Semplificazione Nested CV: Hold-out interno per Early Stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    model = build_efficientnet_3d().to(device)
    criterion = nn.CrossEntropyLoss()
    # Aggiunta weight_decay per la regolarizzazione richiesta (Punto 4.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch_stopped = max_epochs
    
    logger.debug("Inizio addestramento EfficientNet...")
    for epoch in range(max_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = sum(criterion(model(i.to(device)), l.to(device)).item() for i, l in val_loader) / len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            best_epoch_stopped = epoch + 1 - patience
            break
            
    fold_hyperparams = {"learning_rate": lr, "weight_decay": weight_decay, "batch_size": batch_size, "early_stopping_epoch": best_epoch_stopped}
            
    # -- 3. TEST E PREDICITONS --
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_targets.extend(labels.numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    # -- 4. XAI: Integrated Gradients (Punto 4.2) --
    ig_maps_ad, ig_maps_ctrl, all_ig_maps, all_ig_labels = [], [], [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        # Espande la baseline media CTRL per fare match con la dimensione del batch
        baseline_batch = ctrl_mean_baseline.expand(inputs.size(0), -1, -1, -1, -1)
        
        attributions = compute_integrated_gradients(inputs, model, target_class=1, steps=10, baseline=baseline_batch)
        
        for i in range(inputs.size(0)):
            attr_np = attributions[i].cpu().numpy()
            # Normalizzazione per soggetto (somma di |IG| costante)
            attr_norm = attr_np / (np.sum(np.abs(attr_np)) + 1e-8) 
            
            all_ig_maps.append(attr_norm)
            all_ig_labels.append(labels[i].item())
            
            if labels[i] == 1: 
                ig_maps_ad.append(attr_norm)
            else: 
                ig_maps_ctrl.append(attr_norm)
                
    ig_mean_AD = np.mean(ig_maps_ad, axis=0) if ig_maps_ad else None
    ig_mean_CTRL = np.mean(ig_maps_ctrl, axis=0) if ig_maps_ctrl else None
    ig_contrast = ig_mean_AD - ig_mean_CTRL if (ig_mean_AD is not None and ig_mean_CTRL is not None) else None
    
    # -- 5. Significatività delle Mappe IG (Punto 4.3) --
    # Nota: su dati dummy usiamo 100 permutazioni per rapidità, ma su run definitive porta n_perm=1000
    p_map, sig_mask = voxel_wise_permutation_test(np.stack(all_ig_maps), all_ig_labels, n_perm=100, alpha=0.05)
    
    # Mappa IG finale mascherata dai voxel con p < soglia
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
        n_subj = 40
        subjects = np.array([f"sub_{i:03d}" for i in range(n_subj)])
        X_data = torch.randn(n_subj, 1, 32, 32, 32) 
        y_data = torch.randint(0, 2, (n_subj,))
    else:
        subjects, X_data, y_data = load_real_data_efficientnet("data/dataset_info.csv")
    
    splits = get_or_create_splits(subjects, y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data, n_splits_ext=5, n_repeats=2)
    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'sensitivity': [], 'specificity': []}
    all_hyperparams = []
    
    for outer_id, train_idx, test_idx in splits:
        logger.info(f"--- Inizio Fold Esterno {outer_id} ---")
        fold_dir = os.path.join(base_out_dir, "folds", f"outer_{outer_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        test_subjects = subjects[test_idx]
        test_y_true = y_data[test_idx].numpy() if isinstance(y_data, torch.Tensor) else y_data[test_idx]
        
        metrics, xai_maps, fold_hyp, probs, preds = train_and_evaluate_fold(
            X_data[train_idx], y_data[train_idx], X_data[test_idx], y_data[test_idx], device, batch_size=2
        )
        
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