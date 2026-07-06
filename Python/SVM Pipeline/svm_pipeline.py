"""
SVM Pipeline Orchestrator.

Acts as the absolute entry point for the Linear SVM MLOps architecture.
Coordinates the execution of MATLAB Preprocessing via the MatlabOrchestrator,
loads the serialized data via nibabel, injects structural synchronization 
via the CVManager, and triggers the SVM Double Cross-Validation.
"""

import pathlib

# Internal Module Imports
from utils.py_logger import CustomLogger
from utils.matlab_orchestrator import MatlabOrchestrator, MatlabTask
from utils.cv_manager import CVManager
from Models.svm_classifier import SVMClassifier

def svm_pipeline():
    # 1. Initialize Master Logger
    log = CustomLogger(name="SVMPipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log.add_file_handler("svm_pipeline.log", level="TRACE")
    
    log.info("--- Booting SVM Pipeline Orchestrator ---")
    
    # 2. Path Configurations
    # Resolve paths dynamically based on the current script location
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    MATLAB_DIR = BASE_DIR / "MATLAB" / "SVM Pipeline"  # Adjust to your folder structure
    SPM_DIR = pathlib.Path("C:/Users/utente/Desktop/spm")
    
    script_path = MATLAB_DIR / "PreprocessSVM.m"
    log_path = MATLAB_DIR / "Log Files" / "PreprocessSVM.log"
    csv_path = MATLAB_DIR / "covariate_data.csv"
    mask_path = MATLAB_DIR / "tpm_mask.nii"
    
    # --- PHASE 1: MATLAB PREPROCESSING ---
    log.info("PHASE 1: Delegating Spatial Preprocessing to MATLAB...")
    preproc_task = MatlabTask(script_path=script_path, log_path=log_path)
    
    # EAFP Pattern: No generic try/except. We let matlab.engine raise its native 
    # EngineError if the infrastructure fails, preserving the traceback logic.
    with MatlabOrchestrator(logger=log, tasks=[preproc_task], include_paths=[SPM_DIR]) as orch:
        orch.run_all()
            
    log.success("MATLAB Phase Completed Successfully.")

    # --- PHASE 2: DATA INGESTION ---
    log.info("PHASE 2: Ingesting data into Python RAM via nibabel...")
    svm_engine = SVMClassifier(logger=log)
    
    # EAFP Pattern: We attempt to load directly without LBYL (os.path.exists checks).
    # If MATLAB failed to write the mask, this natively raises a FileNotFoundError.
    subjects, X_full, y = svm_engine.load_real_data(str(csv_path), str(mask_path))
    
    log.success(f"Data Loaded: {X_full.shape[0]} subjects, {X_full.shape[1]} flattened features per subject.")

    # --- PHASE 3: FOLD SYNCHRONIZATION (CV Manager) ---
    log.info("PHASE 3: Generating synchronized absolute/relative fold indices...")
    cv_manager = CVManager(outer_folds=5, inner_folds=5, random_state=42)
    
    cv_splits = cv_manager.generate_splits(y)
    
    # Dynamic topology extraction
    num_outer = len(cv_splits)
    num_inner = len(cv_splits[0]['inner_splits_relative'])
    
    log.success(f"Generated {num_outer} structurally synchronized outer folds ({num_inner} inner folds each).")

    # --- PHASE 4: MODEL TRAINING (Double Cross Validation) ---
    log.info("PHASE 4: Executing SVM Double Cross Validation...")
    
    results_df, artifacts = svm_engine.execute_nested_cv(X_full, y, subjects, cv_splits)
    
    # Display Final Summary
    log.success("--- SVM PIPELINE COMPLETE ---")
    log.info(f"\nFinal Aggregated Metrics across {num_outer} Folds:\n{results_df.mean(numeric_only=True).to_frame().T.to_string(index=False)}")
    
    # Save results locally
    results_df.to_csv("SVM_CV_Metrics.csv", index=False)
    log.info("Metrics saved to 'SVM_CV_Metrics.csv'.")

if __name__ == "__main__":
    svm_pipeline()