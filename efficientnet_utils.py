import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from scipy.stats import ttest_ind
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import fdrcorrection
from monai.networks.nets import EfficientNetBN

def get_or_create_splits(subjects, y, n_splits_ext=5, n_repeats=5, random_state=42, splits_dir="data/splits", logger=None):
    """Generates ONLY the outer splits (Repeated Stratified K-Fold) or loads them from JSON if they exist."""
    outer_json_path = os.path.join(splits_dir, "outer_splits.json")
    os.makedirs(splits_dir, exist_ok=True)
    
    if os.path.exists(outer_json_path):
        if logger: logger.info(f"Splits file found in {outer_json_path}. Loading...")
        with open(outer_json_path, 'r') as f:
            outer_splits_data = json.load(f)
        splits_to_yield = []
        for split_info in outer_splits_data["splits"]:
            train_idx = np.where(np.isin(subjects, split_info["train_subjects"]))[0]
            test_idx = np.where(np.isin(subjects, split_info["test_subjects"]))[0]
            splits_to_yield.append((split_info["outer_id"], train_idx, test_idx))
        return splits_to_yield

    if logger: logger.info("Splits file not found. Generating and saving JSON...")
    rskf = RepeatedStratifiedKFold(n_splits=n_splits_ext, n_repeats=n_repeats, random_state=random_state)
    outer_splits_data = {"n_splits_ext": n_splits_ext, "n_repeats": n_repeats, "random_state": random_state, "splits": []}
    splits_to_yield = []
    
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

def compute_integrated_gradients(inputs, model, target_class=1, steps=20, baseline=None):
    """
    Calculates the Integrated Gradients (IG) using Riemann sum approximation.
    Excludes step 0 (pure baseline) to maintain mathematical correctness.
    """
    if baseline is None: 
        baseline = torch.zeros_like(inputs)
        
    # Create interpolation steps (from 1 to 'steps')
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)]
    
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_()
        score = model(scaled_input)[:, target_class].sum()
        model.zero_grad()
        score.backward()
        grads.append(scaled_input.grad.detach())
    
    # Average of gradients across all steps
    avg_grads = torch.mean(torch.stack(grads), dim=0)
    
    # Final multiplication: (input - baseline) * average_gradients
    integrated_gradients = (inputs - baseline) * avg_grads
    return integrated_gradients

def voxel_wise_permutation_test(ig_maps, labels, n_perm=1000, alpha=0.05):
    """
    Performs voxel-wise permutation test using T-statistic (Welch's t-test) 
    and applies Benjamini-Hochberg FDR correction.
    """
    shape_orig = ig_maps.shape
    abs_ig = np.abs(ig_maps.reshape(shape_orig[0], -1))
    labels = np.array(labels)
    if sum(labels == 1) == 0 or sum(labels == 0) == 0: 
        return np.ones(shape_orig[1:]), np.zeros(shape_orig[1:], dtype=bool)

    # 1. Real T-statistic calculation (equal_var=False for Welch's T-test)
    true_t_stat, _ = ttest_ind(abs_ig[labels == 1], abs_ig[labels == 0], axis=0, equal_var=False)
    true_t_stat = np.nan_to_num(true_t_stat) # Handle potential division by zero
    
    count_greater = np.zeros_like(true_t_stat)
    
    for _ in range(n_perm):
        perm_labels = np.random.permutation(labels)
        
        # 2. Permuted T-statistic calculation
        p_t_stat, _ = ttest_ind(abs_ig[perm_labels == 1], abs_ig[perm_labels == 0], axis=0, equal_var=False)
        p_t_stat = np.nan_to_num(p_t_stat)
        
        # Count how many times the absolute permuted T-stat exceeds the real one
        count_greater += (np.abs(p_t_stat) >= np.abs(true_t_stat))
        
    p_values_flat = (count_greater + 1) / (n_perm + 1)
    
    # Using 'indep' (Benjamini-Hochberg) for spatial correlation consistency
    rejected, pval_corrected_flat = fdrcorrection(p_values_flat, alpha=alpha, method='indep')
    return pval_corrected_flat.reshape(shape_orig[1:]), rejected.reshape(shape_orig[1:])

def build_efficientnet_3d(model_name="efficientnet-b0", in_channels=1, num_classes=2):
    """Initializes and returns a 3D EfficientNet model from MONAI."""
    return EfficientNetBN(model_name=model_name, spatial_dims=3, in_channels=in_channels, num_classes=num_classes, pretrained=False)

def load_real_data_efficientnet(csv_path):
    """Loads 3D NIfTI images into PyTorch tensors."""
    df = pd.read_csv(csv_path)
    subjects = df['subject_id'].values
    y_tensor = torch.tensor(df['label'].values, dtype=torch.long)
    volumes = []
    for idx, row in df.iterrows():
        img_tensor = torch.tensor(nib.load(row['nifti_path']).get_fdata(), dtype=torch.float32).unsqueeze(0) 
        volumes.append(img_tensor)
    return subjects, torch.stack(volumes), y_tensor