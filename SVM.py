import numpy as np
import time
import os
import json
import math
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_ind, pearsonr
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from statsmodels.stats.multitest import fdrcorrection
from loguru import logger

def get_or_create_splits(subjects, y, n_splits_ext=5, n_repeats=5, n_splits_int=5, random_state=42, splits_dir="data/splits"):
    """Genera la Repeated Stratified K-Fold oppure la carica dai JSON se esistono."""
    outer_json_path = os.path.join(splits_dir, "outer_splits.json")
    inner_splits_dir = os.path.join(splits_dir, "inner_splits")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(inner_splits_dir, exist_ok=True)
    
    if os.path.exists(outer_json_path):
        logger.info(f"File degli split trovato in {outer_json_path}. Caricamento in corso...")
        with open(outer_json_path, 'r') as f:
            outer_splits_data = json.load(f)
            
        splits_to_yield = []
        for split_info in outer_splits_data["splits"]:
            train_idx = np.where(np.isin(subjects, split_info["train_subjects"]))[0]
            test_idx = np.where(np.isin(subjects, split_info["test_subjects"]))[0]
            splits_to_yield.append((split_info["outer_id"], train_idx, test_idx))
        return splits_to_yield

    logger.info("File degli split non trovati. Generazione e salvataggio JSON in corso...")
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
        
        skf_inner = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=random_state)
        inner_splits_data = {"outer_id": outer_id, "splits": []}
        inner_subjects, inner_y = subjects[train_idx], y[train_idx]
        
        for inner_fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(skf_inner.split(inner_subjects, inner_y)):
            inner_splits_data["splits"].append({
                "inner_fold": inner_fold_idx,
                "train_subjects": inner_subjects[inner_tr_idx].tolist(),
                "val_subjects": inner_subjects[inner_val_idx].tolist()
            })
            
        with open(os.path.join(inner_splits_dir, f"outer_{outer_id}.json"), 'w') as f:
            json.dump(inner_splits_data, f, indent=4)
            
    with open(outer_json_path, 'w') as f:
        json.dump(outer_splits_data, f, indent=4)
        
    return splits_to_yield

def calculate_gaonkar_pmap_and_mask(best_svm, haufe_pattern, alpha=0.05):
    """
    Calcola la p-map analitica di Gaonkar e maschera il pattern di Haufe.
    Approssima la varianza analitica del margine sotto permutazione usando i support vectors.
    """
    w = best_svm.coef_.flatten()
    dual_coef = best_svm.dual_coef_.flatten() # alpha * y
    support_vectors = best_svm.support_vectors_
    
    # Varianza analitica di ogni peso w_j sotto ipotesi nulla (Gaonkar approssimato)
    w_var = np.sum((dual_coef[:, None] * support_vectors)**2, axis=0)
    w_var[w_var == 0] = 1e-10 # Evita divisioni per zero
    
    z_scores = w / np.sqrt(w_var)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Correzione FDR
    rejected, pval_corrected = fdrcorrection(p_values, alpha=alpha, method='indep')
    
    haufe_masked = np.copy(haufe_pattern)
    haufe_masked[~rejected] = 0.0 
    
    return p_values, pval_corrected, haufe_masked

def calculate_top_k_overlap(map1, map2, k_percent=5.0):
    """Calcola l'overlap (indice di Jaccard/Dice) tra i top K% voxel di due mappe."""
    k = max(1, int(len(map1) * (k_percent / 100.0)))
    top_idx1 = set(np.argsort(np.abs(map1))[-k:])
    top_idx2 = set(np.argsort(np.abs(map2))[-k:])
    overlap = len(top_idx1.intersection(top_idx2)) / k
    return overlap

def compare_xai_maps(w, haufe_pattern, vbm_map, k_percent=5.0):
    """Esegue i confronti sistematici tra w, a e VBM (Punto 3.2)."""
    results = {}
    
    # Correlazioni di Pearson
    results['corr_w_vbm'] = pearsonr(w, vbm_map)[0]
    results['corr_a_vbm'] = pearsonr(haufe_pattern, vbm_map)[0]
    results['corr_w_a'] = pearsonr(w, haufe_pattern)[0]
    
    # Overlap Top K%
    results[f'overlap_top{int(k_percent)}_w_vbm'] = calculate_top_k_overlap(w, vbm_map, k_percent)
    results[f'overlap_top{int(k_percent)}_a_vbm'] = calculate_top_k_overlap(haufe_pattern, vbm_map, k_percent)
    results[f'overlap_top{int(k_percent)}_w_a'] = calculate_top_k_overlap(w, haufe_pattern, k_percent)
    
    return results

def pipeline_svm_cmepda(subjects, X, y, base_out_dir, n_splits_ext=5, n_repeats=5, n_splits_int=5):
    """Branch SVM lineare voxel-wise con Nested CV e confronti XAI."""
    splits = get_or_create_splits(subjects, y, n_splits_ext, n_repeats, n_splits_int)
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    
    metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'sensitivity': [], 'specificity': []}
    all_comparisons = []
    best_c_values = []
    
    for outer_id, train_idx, test_idx in splits:
        logger.info(f"--- Inizio Fold Esterno {outer_id} ---")
        fold_dir = os.path.join(base_out_dir, "folds", f"outer_{outer_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = subjects[test_idx]
        
        # 1. Tuning SVM
        cv_interno = StratifiedKFold(n_splits=n_splits_int, shuffle=True, random_state=42)
        svm_base = SVC(kernel='linear', class_weight='balanced', probability=True)
        grid_search = GridSearchCV(estimator=svm_base, param_grid=param_grid, cv=cv_interno, scoring='balanced_accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train)
        
        best_c = grid_search.best_params_['C']
        best_svm = grid_search.best_estimator_
        best_c_values.append(best_c)
        logger.info(f"Tuning completato. Best C: {best_c}")
        
        with open(os.path.join(fold_dir, "hyperparams_selected.json"), 'w') as f:
            json.dump({"C": best_c, "log10_C": math.log10(best_c)}, f, indent=4)
        
        # 2. Metriche
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
        
        logger.info(f"Metriche Fold {outer_id} -> Bal. Acc: {fold_met['balanced_accuracy']:.3f} | Sens: {fold_met['sensitivity']:.3f} | Spec: {fold_met['specificity']:.3f} | AUC: {fold_met['auc']:.3f}")
        for k, v in fold_met.items(): metrics[k].append(v)
        
        with open(os.path.join(fold_dir, "metrics.json"), 'w') as f:
            json.dump(fold_met, f, indent=4)
        pd.DataFrame({"subject_id": test_subjects, "y_true": y_test, "score": y_prob, "y_pred": y_pred}).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        
        # 3. XAI: Pesi e Haufe (Punto 3.2)
        w = best_svm.coef_.flatten() 
        s_train = best_svm.decision_function(X_train)
        X_train_centered = X_train - np.mean(X_train, axis=0)
        haufe_pattern = np.dot(X_train_centered.T, s_train - np.mean(s_train)) / (X_train.shape[0] - 1)
        
        # 4. Mappa VBM (T-test AD vs CTRL sul training fold)
        t_stat, _ = ttest_ind(X_train[y_train == 1], X_train[y_train == 0], axis=0, equal_var=False)
        vbm_map = np.nan_to_num(t_stat)
        
        # 5. Confronto Sistematico XAI
        comps = compare_xai_maps(w, haufe_pattern, vbm_map, k_percent=5.0)
        comps['fold_id'] = outer_id
        all_comparisons.append(comps)
        with open(os.path.join(fold_dir, "xai_comparisons.json"), 'w') as f:
            json.dump(comps, f, indent=4)
            
        logger.debug(f"Confronto XAI - Correlazioni | w-VBM: {comps['corr_w_vbm']:.3f} | a-VBM: {comps['corr_a_vbm']:.3f}")
        
        # 6. Significatività Gaonkar (Punto 3.3)
        p_values, pval_corrected, haufe_masked = calculate_gaonkar_pmap_and_mask(best_svm, haufe_pattern, alpha=0.05)
        
        # (Opzionale: qui inseriresti il salvataggio dei vettori numpy (w, a, haufe_masked) in NIfTI)
        
    return metrics, all_comparisons, best_c_values

def load_real_data_svm(csv_path, mask_path):
    mask_boolean = nib.load(mask_path).get_fdata() > 0
    df = pd.read_csv(csv_path)
    X = np.zeros((len(df), np.sum(mask_boolean)))
    for idx, row in df.iterrows():
        X[idx, :] = nib.load(row['nifti_path']).get_fdata()[mask_boolean]
    return df['subject_id'].values, X, df['label'].values

if __name__ == "__main__":
    run_id = f"run_{int(time.time())}_seed42_SVM"
    base_out_dir = f"results/runs/{run_id}"
    os.makedirs(base_out_dir, exist_ok=True)
    summary_dir = os.path.join(base_out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    logger.add(f"{base_out_dir}/pipeline_svm.log", rotation="10 MB", level="DEBUG")
    
    USE_DUMMY_DATA = True 
    if USE_DUMMY_DATA:
        n_subj, n_vox = 40, 5000
        subjects = np.array([f"sub_{i:03d}" for i in range(n_subj)])
        X_data, y_data = np.random.randn(n_subj, n_vox), np.random.randint(0, 2, n_subj)
    else:
        subjects, X_data, y_data = load_real_data_svm("data/dataset_info.csv", "data/gm_mask_MNI.nii.gz")

    met, xai_comps, C_opt = pipeline_svm_cmepda(subjects, X_data, y_data, base_out_dir, n_splits_ext=5, n_repeats=2)
    
    # --- REPORTING IPERPARAMETRI E CONSENSO ---
    log10_c_values = [math.log10(c) for c in C_opt]
    consensus_log10_c = np.median(log10_c_values)
    
    pd.DataFrame({"fold_idx": range(1, len(C_opt)+1), "C_selected": C_opt, "log10_C": log10_c_values}).to_csv(os.path.join(summary_dir, "hyperparams_distribution.csv"), index=False)
    
    with open(os.path.join(summary_dir, "hyperparams_consensus.json"), 'w') as f:
        json.dump({"consensus_log10_C_median": consensus_log10_c, "consensus_C_value": 10**consensus_log10_c, "frequencies": pd.Series(C_opt).value_counts().to_dict()}, f, indent=4)
        
    # Salva il recap dei confronti XAI di tutti i fold
    pd.DataFrame(xai_comps).to_csv(os.path.join(summary_dir, "xai_comparisons_summary.csv"), index=False)
        
    # NUOVO: Salva il recap delle metriche di tutti i fold in un unico CSV
    df_met = pd.DataFrame(met)
    df_met['Fold'] = [f"outer_F{i}" for i in range(len(df_met))]
    df_met.to_csv(os.path.join(summary_dir, "metrics_all_folds.csv"), index=False)

    # STAMPA A SCHERMO I RISULTATI FINALI MEDI
    logger.success(f"Consenso Iperparametri calcolato: Median log10(C) = {consensus_log10_c:.3f}")
    logger.success("=== RISULTATI FINALI MEDI ===")
    logger.success(f"Balanced Accuracy Media: {np.mean(met['balanced_accuracy']):.3f} ± {np.std(met['balanced_accuracy']):.3f}")
    logger.success(f"Sensitivity Media: {np.mean(met['sensitivity']):.3f} ± {np.std(met['sensitivity']):.3f}")
    logger.success(f"Specificity Media: {np.mean(met['specificity']):.3f} ± {np.std(met['specificity']):.3f}")
    
    # Stampa i risultati medi del confronto XAI
    df_comps = pd.DataFrame(xai_comps)
    logger.success("=== CONFRONTI XAI MEDI ===")
    logger.success(f"Correlazione Haufe(a) vs VBM: {df_comps['corr_a_vbm'].mean():.3f}")
    logger.success(f"Overlap Top-5% Haufe(a) vs VBM: {df_comps['overlap_top5_a_vbm'].mean()*100:.1f}%")