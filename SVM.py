import numpy as np
import time
import os
import json
import math
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

# Imports locali
from py_logger import CustomLogger
from svm_utils import (
    get_or_create_splits, 
    calculate_gaonkar_pmap_and_mask, 
    compare_xai_maps, 
    load_real_data_svm
)

# Global default initialization (console only), will be overwritten in the main block
logger = CustomLogger(enable_file_logging=False, level="DEBUG")

def pipeline_svm_cmepda(subjects, X, y, base_out_dir, n_splits_ext=5, n_repeats=5, n_splits_int=5):
    """Voxel-wise linear SVM branch with Nested CV and XAI comparisons."""
    splits = get_or_create_splits(subjects, y, n_splits_ext, n_repeats, logger=logger)
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    
    metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'sensitivity': [], 'specificity': []} 
    all_comparisons = []
    best_c_values = []
    all_w, all_haufe, all_haufe_masked, all_vbm = [], [], [], []
    
    for outer_id, train_idx, test_idx in splits:
        logger.info(f"--- Starting Outer Fold {outer_id} ---")
        fold_dir = os.path.join(base_out_dir, "folds", f"outer_{outer_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = subjects[test_idx]
        
        # 1. SVM Tuning
        cv_interno = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=42)
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True)
        grid_search = GridSearchCV(estimator=svm_base, param_grid=param_grid, cv=cv_interno, scoring='balanced_accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train)

        best_c = grid_search.best_params_['C']
        best_svm = grid_search.best_estimator_
        best_c_values.append(best_c)
        logger.info(f"Tuning completed. Best C: {best_c}")
        
        with open(os.path.join(fold_dir, "hyperparams_selected.json"), 'w') as f:
            json.dump({"C": best_c, "log10_C": math.log10(best_c)}, f, indent=4)
        
        # 2. Metrics
        y_pred = best_svm.predict(X_test)
        y_prob = best_svm.predict_proba(X_test)[:, 1] 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        
        fold_met = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
        
        logger.info(f"Fold {outer_id} Metrics -> Bal. Acc: {fold_met['balanced_accuracy']:.3f} | Sens: {fold_met['sensitivity']:.3f} | Spec: {fold_met['specificity']:.3f} | AUC: {fold_met['auc']:.3f}")
        for k, v in fold_met.items(): metrics[k].append(v)
        
        with open(os.path.join(fold_dir, "metrics.json"), 'w') as f:
            json.dump(fold_met, f, indent=4)
        pd.DataFrame({"subject_id": test_subjects, "y_true": y_test, "score": y_prob, "y_pred": y_pred}).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        
        # 3. XAI: Weights and Haufe (Point 3.2)
        w = best_svm.coef_.flatten() 
        s_train = best_svm.decision_function(X_train)
        X_train_centered = X_train - np.mean(X_train, axis=0)
        haufe_pattern = np.dot(X_train_centered.T, s_train - np.mean(s_train)) / (X_train.shape[0] - 1) 

        # 4. VBM Map
        t_stat, _ = ttest_ind(X_train[y_train == 1], X_train[y_train == 0], axis=0, equal_var=False)
        vbm_map = np.nan_to_num(t_stat)
        
        # 5. Systematic XAI Comparison
        comps = compare_xai_maps(w, haufe_pattern, vbm_map, k_percent=5.0)
        comps['fold_id'] = outer_id
        all_comparisons.append(comps)
        with open(os.path.join(fold_dir, "xai_comparisons.json"), 'w') as f:
            json.dump(comps, f, indent=4)
            
        logger.debug(f"XAI Comparison - Correlations | w-VBM: {comps['corr_w_vbm']:.3f} | a-VBM: {comps['corr_a_vbm']:.3f}")
        
        # 6. Gaonkar Significance (Point 3.3)
        p_values, pval_corrected, haufe_masked = calculate_gaonkar_pmap_and_mask(best_svm, haufe_pattern, alpha=0.05)
        
        all_w.append(w)
        all_haufe.append(haufe_pattern)
        all_haufe_masked.append(haufe_masked)
        all_vbm.append(vbm_map)
    
    mean_w = np.mean(all_w, axis=0)
    mean_haufe = np.mean(all_haufe, axis=0)
    mean_haufe_masked = np.mean(all_haufe_masked, axis=0)
    mean_vbm = np.mean(all_vbm, axis=0)

    return metrics, all_comparisons, best_c_values, (mean_w, mean_haufe, mean_haufe_masked, mean_vbm)


if __name__ == "__main__":
    run_id = f"run_{int(time.time())}_seed42_SVM"
    base_out_dir = f"results/runs/{run_id}"
    os.makedirs(base_out_dir, exist_ok=True)
    summary_dir = os.path.join(base_out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    logger = CustomLogger(log_file_path=f"{base_out_dir}/pipeline_svm.log", enable_file_logging=True, level="DEBUG")
    
    USE_DUMMY_DATA = True 
    if USE_DUMMY_DATA:
        n_subj, n_vox = 40, 5000
        subjects = np.array([f"sub_{i:03d}" for i in range(n_subj)])
        X_data = np.random.randn(n_subj, n_vox)
        y_data = np.array([0]*(n_subj//2) + [1]*(n_subj//2))
        np.random.shuffle(y_data) # Mixing labels
    else:
        subjects, X_data, y_data = load_real_data_svm("data/dataset_info.csv", "data/gm_mask_MNI.nii.gz")

    met, xai_comps, C_opt, mean_maps = pipeline_svm_cmepda(subjects, X_data, y_data, base_out_dir, n_splits_ext=5, n_repeats=2)
    mean_w, mean_haufe, mean_haufe_masked, mean_vbm = mean_maps
    
    log10_c_values = [math.log10(c) for c in C_opt]
    consensus_log10_c = np.median(log10_c_values)
    
    pd.DataFrame({"fold_idx": range(1, len(C_opt)+1), "C_selected": C_opt, "log10_C": log10_c_values}).to_csv(os.path.join(summary_dir, "hyperparams_distribution.csv"), index=False)
    
    with open(os.path.join(summary_dir, "hyperparams_consensus.json"), 'w') as f:
        json.dump({"consensus_log10_C_median": consensus_log10_c, "consensus_C_value": 10**consensus_log10_c, "frequencies": pd.Series(C_opt).value_counts().to_dict()}, f, indent=4)
        
    pd.DataFrame(xai_comps).to_csv(os.path.join(summary_dir, "xai_comparisons_summary.csv"), index=False)
        
    df_met = pd.DataFrame(met)
    df_met['Fold'] = [f"outer_F{i}" for i in range(len(df_met))]
    df_met.to_csv(os.path.join(summary_dir, "metrics_all_folds.csv"), index=False)

    logger.success(f"Hyperparameter Consensus calculated: Median log10(C) = {consensus_log10_c:.3f}")
    logger.success("=== AVERAGE FINAL RESULTS ===")
    logger.success(f"Average Balanced Accuracy: {np.mean(met['balanced_accuracy']):.3f} ± {np.std(met['balanced_accuracy']):.3f}")
    logger.success(f"Average Sensitivity: {np.mean(met['sensitivity']):.3f} ± {np.std(met['sensitivity']):.3f}")
    logger.success(f"Average Specificity: {np.mean(met['specificity']):.3f} ± {np.std(met['specificity']):.3f}")
    logger.success(f"Average Accuracy: {np.mean(met['accuracy']):.3f} ± {np.std(met['accuracy']):.3f}")
    logger.success(f"Average ROC AUC: {np.mean(met['auc']):.3f} ± {np.std(met['auc']):.3f}")
    
    maps_dir = "data/maps"
    os.makedirs(maps_dir, exist_ok=True)
    mask_path = "data/gm_mask_MNI.nii.gz"

    if os.path.exists(mask_path):
        logger.info("Reconstructing and saving 3D NIfTI maps...")
        mask_img = nib.load(mask_path)
        mask_bool = mask_img.get_fdata() > 0
        
        def save_3d_map(vector_1d, out_filename):
            vol_3d = np.zeros(mask_img.shape)
            vol_3d[mask_bool] = vector_1d      
            new_img = nib.Nifti1Image(vol_3d, mask_img.affine, mask_img.header)
            nib.save(new_img, os.path.join(maps_dir, out_filename))
            
        save_3d_map(mean_vbm, "vbm_tstat_map.nii.gz")
        save_3d_map(mean_w, "svm_weights_mean.nii.gz")
        save_3d_map(mean_haufe, "svm_haufe_mean.nii.gz")
        save_3d_map(mean_haufe_masked, "svm_haufe_masked_mean.nii.gz")
        
        logger.success(f"3D maps successfully saved in {maps_dir}/")
    else:
        logger.warn(f"Mask {mask_path} not found. Unable to save NIfTI (normal if using Dummy Data).")

    df_comps = pd.DataFrame(xai_comps)
    logger.success("=== AVERAGE XAI COMPARISONS ===")
    logger.success(f"Haufe(a) vs VBM Correlation: {df_comps['corr_a_vbm'].mean():.3f}")
    logger.success(f"Top-5% Haufe(a) vs VBM Overlap: {df_comps['overlap_top5_a_vbm'].mean()*100:.1f}%")