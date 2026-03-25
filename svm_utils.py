import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import fdrcorrection

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
        current_repeat = split_idx // n_splits_ext
        current_fold = split_idx % n_splits_ext
        outer_id = f"R{current_repeat}_F{current_fold}"
        
        outer_splits_data["splits"].append({
            "outer_id": outer_id, "repeat": current_repeat, "fold": current_fold,
            "train_subjects": subjects[train_idx].tolist(), "test_subjects": subjects[test_idx].tolist()
        })
        splits_to_yield.append((outer_id, train_idx, test_idx))
            
    with open(outer_json_path, 'w') as f:
        json.dump(outer_splits_data, f, indent=4)
        
    return splits_to_yield

def calculate_gaonkar_pmap_and_mask(best_svm, haufe_pattern, alpha=0.05):
    """
    Calculates the analytical Gaonkar p-map and masks the Haufe pattern.
    Approximates the analytical variance of the margin under permutation using support vectors.
    """
    w = best_svm.coef_.flatten()
    dual_coef = best_svm.dual_coef_.flatten() # alpha * y
    support_vectors = best_svm.support_vectors_
    
    w_var = np.sum((dual_coef[:, None] * support_vectors)**2, axis=0)
    w_var[w_var == 0] = 1e-10 
    
    z_scores = w / np.sqrt(w_var)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    rejected, pval_corrected = fdrcorrection(p_values, alpha=alpha, method='indep') 
    
    haufe_masked = np.copy(haufe_pattern)
    haufe_masked[~rejected] = 0.0 
    
    return p_values, pval_corrected, haufe_masked

def calculate_top_k_overlap(map1, map2, k_percent=5.0):
    """Calculates the overlap (Jaccard/Dice index) between the top K% voxels of two maps."""
    k = max(1, int(len(map1) * (k_percent / 100.0)))
    top_idx1 = set(np.argsort(np.abs(map1))[-k:])
    top_idx2 = set(np.argsort(np.abs(map2))[-k:])
    overlap = len(top_idx1.intersection(top_idx2)) / k
    return overlap

def compare_xai_maps(w, haufe_pattern, vbm_map, k_percent=5.0):
    """Performs systematic comparisons between w, a, and VBM (Point 3.2)."""
    results = {}
    
    results['corr_w_vbm'] = pearsonr(w, vbm_map)[0]
    results['corr_a_vbm'] = pearsonr(haufe_pattern, vbm_map)[0]
    results['corr_w_a'] = pearsonr(w, haufe_pattern)[0]
    
    results[f'overlap_top{int(k_percent)}_w_vbm'] = calculate_top_k_overlap(w, vbm_map, k_percent)
    results[f'overlap_top{int(k_percent)}_a_vbm'] = calculate_top_k_overlap(haufe_pattern, vbm_map, k_percent)
    results[f'overlap_top{int(k_percent)}_w_a'] = calculate_top_k_overlap(w, haufe_pattern, k_percent)
    
    return results

def load_real_data_svm(csv_path, mask_path):
    """Loads NIfTI files and extracts only the voxels within the gray matter mask."""
    mask_boolean = nib.load(mask_path).get_fdata() > 0
    df = pd.read_csv(csv_path)
    X = np.zeros((len(df), np.sum(mask_boolean)))
    for idx, row in df.iterrows():
        X[idx, :] = nib.load(row['nifti_path']).get_fdata()[mask_boolean]
    return df['subject_id'].values, X, df['label'].values