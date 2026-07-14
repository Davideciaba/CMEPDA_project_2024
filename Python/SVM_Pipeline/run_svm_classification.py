"""
SVM Pipeline Orchestrator.

Acts as the absolute entry point for the Linear SVM MLOps architecture.
Coordinates the execution of MATLAB Preprocessing via the MatlabOrchestrator,
loads the serialized data via nibabel, injects structural synchronization 
via the CVManager, and triggers the SVM Double Cross-Validation.
"""
import sys
import pathlib
import pandas as pd
import numpy as np
import joblib
import nibabel as nib

# 1. Resolve absolute path to the current file
current_file_path = pathlib.Path(__file__).resolve()

# 2. Navigate up to the CMEPDA_project_2024 root directory (2 levels up)
project_root = current_file_path.parents[2]

sys.path.append(str(project_root))

#import hashlib

# Internal Module Imports
from Python.utils.py_logger import CustomLogger
from Python.utils.matlab_orchestrator import MatlabOrchestrator, MatlabTask
from Python.utils.cv_manager import CVManager
from Python.Models.svm_classifier import SVMClassifier
from Python.utils.model_renderer import ModelRenderer

"""
def compute_file_hash(filepath: pathlib.Path) -> str:
    
    Computes the SHA-256 cryptographic hash of a file's contents.
    Used for Content-Addressable Caching to detect source code modifications.
    
    if not filepath.exists():
        return ""
    
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()
"""

def run_svm_classification():
    
    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = CURRENT_DIR.parent.parent
    PREPROCESS_DIR = PROJECT_DIR / "MATLAB" / "utils"
    SPM_DIR = pathlib.Path("C:/Users/utente/Desktop/spm")
    SETUP_DIR = PROJECT_DIR / "Python" / "Common_Setup"

    log = CustomLogger(name="SVMPipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log_dir = CURRENT_DIR / "Log_Files"
    log_path = log_dir / "SVMPipeline.log"
    log.add_file_handler(log_path, level="DEBUG")
    log.info("--- Booting Decoupled SVM Engine ---")

    
    preprocess_path = PREPROCESS_DIR / "CommonPreprocess.m"
    preprocess_log_path = log_dir / "CommonPreprocess.log"
    registry_csv_path = SETUP_DIR / "python_registry.csv"
    results_dir = CURRENT_DIR / "Results"
    mask_path = SETUP_DIR / "Mask" / "tpm_mask.nii"
    folds_json_path = SETUP_DIR / "cv_folds_registry.json"
    plots_dir = CURRENT_DIR / "Plots"
    

    """
    # Cache Configuration
    hash_record_path = MATLAB_DIR / "tpm_mask.hash"

    # 1. MATLAB PREPROCESSING (Self-contained dependency with Cryptographic Smart Caching)
    log.info("STEP 1: Checking Spatial Preprocessing Artifacts...")
    
    current_script_hash = compute_file_hash(script_path)
    
    saved_hash = ""
    if hash_record_path.exists():
        with open(hash_record_path, "r", encoding="utf-8") as f:
            saved_hash = f.read().strip()
            
    cache_valid = mask_path.exists() and (current_script_hash == saved_hash) and (current_script_hash != "")

    if not cache_valid:
        if not mask_path.exists():
            log.warning(f"TPM Mask not found at '{mask_path.name}'. Booting MATLAB Engine...")
        else:
            log.warning("MATLAB Source Code modification detected! Purging stale cache and rebooting Engine...")
            # Automatically delete the old mask and hash to prevent overwrite conflicts
            mask_path.unlink(missing_ok=True)
            hash_record_path.unlink(missing_ok=True)
    """        
    preproc_task = MatlabTask(script_path=preprocess_path, log_path=preprocess_log_path)
        
    with MatlabOrchestrator(logger=log, tasks=[preproc_task], include_paths=[SPM_DIR]) as orch:
        orch.run_all()
            
        """with open(hash_record_path, "w", encoding="utf-8") as f:
            f.write(current_script_hash)
            
        log.success("MATLAB Mask generated and cryptographic signature frozen.")
    else:
        log.success(f"Valid Cached TPM Mask found (Signature match). Bypassing MATLAB Engine.")
"""
    # 2. LOAD DATA
    log.info("STEP 2: Ingesting normalized registry and TPM mask...")
    C_DICT = {'C':[1e-4, 1e-3]}
    svm_engine = SVMClassifier(logger=log, param_grid=C_DICT)
    
    subjects, X_full, y_full = svm_engine.load_data(str(registry_csv_path), str(mask_path))
    log.success(f"Data Loaded: {X_full.shape[0]} subjects ready.")

    # 3. LOAD & VALIDATE SSOT
    log.info("STEP 3: Loading Frozen Fold Artifact...")
    cv_splits = CVManager.load_from_json(str(folds_json_path))
    
    for split in cv_splits:
        test_idx = split['outer_test_idx']
        expected_subjects = split['security_test_subjects']
        actual_subjects = subjects[test_idx].tolist()
        
        if expected_subjects != actual_subjects:
            log.critical("FATAL: Memory misalignment detected! The loaded CSV does not match the JSON artifact.")
            log.critical(f"Expected: {expected_subjects}")
            log.critical(f"Actual:   {actual_subjects}")
            sys.exit(1)
            
    log.success("Security Signature validated. Fold integrity is 100% guaranteed.")

    # 4. TRAIN
    log.info("STEP 4: Executing SVM Double Cross Validation...")
    results_df, artifacts = svm_engine.execute_nested_cv(X_full, y_full, subjects, cv_splits)
    
     # 5. RENDER PLOTS & SUMMARY
    log.info("STEP 5: Rendering ROC Curves and Aggregating Metrics...")
    plots_dir.mkdir(parents=True, exist_ok=True)
    renderer = ModelRenderer(logger=log, output_dir=str(plots_dir))
    renderer.plot_roc_curves(artifacts, "Linear SVM", "SVM_ROC.png")
    
    # Stampa i risultati dettagliati di tutti e 5 i fold
    log.info(f"\nDetailed Results per Fold:\n{results_df.to_string(index=False)}")
    
    # Seleziona solo le metriche di cui vogliamo fare la media
    target_cols = ['Accuracy', 'Balanced_Accuracy', 'F1_Score', 'Sensitivity', 'Specificity', 'AUROC']
    summary_dict = {}
    
    # Calcolo mean ± std per le metriche esterne
    for col in target_cols:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        summary_dict[col] = f"{mean_val:.3f} ± {std_val:.3f}"
        
    # Calcolo mean ± std accorpato per la validazione interna
    inner_mean = results_df['Inner_CV_BalAcc_Mean'].mean()
    inner_std = results_df['Inner_CV_BalAcc_Mean'].std()
    summary_dict['Inner_CV_BalAcc'] = f"{inner_mean:.3f} ± {inner_std:.3f}"
    
    summary_df = pd.DataFrame([summary_dict])
    log.info(f"\nFinal Aggregated Metrics across {len(results_df)} Folds:\n{summary_df.to_string(index=False)}")
    
    # Salva su disco IL DATAFRAME GREZZO (numerico) per analisi statistiche future
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out_path = results_dir / "SVM_CV_Metrics.csv"
    results_df.to_csv(str(csv_out_path), index=False)
    log.success(f"Raw Metrics successfully saved to: {csv_out_path.name}")
    
    log.info("Saving trained SVM models to disk for XAI extraction...")

    for artifact in artifacts:
        fold_id = artifact['fold_id']
        model_out_path = results_dir / f"SVM_Model_Fold_{fold_id}.joblib"
        joblib.dump(artifact['model'], str(model_out_path))
        log.debug(f"Saved SVM Pipeline: {model_out_path.name}")

    log.success("--- SVM EXECUTION COMPLETE ---")

if __name__ == "__main__":
    run_svm_classification()