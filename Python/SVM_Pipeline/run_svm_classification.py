"""
Module: run_svm_classification.py

SVM Pipeline Orchestrator.

Acts as the absolute entry point for the Linear SVM MLOps architecture.
Coordinates the execution of preprocessing, loads the serialized data via nibabel, 
injects structural synchronization via the CVManager, and triggers the SVM Double Cross-Validation.
"""
import sys
from pathlib import Path
import pandas as pd
import joblib
from typing import Optional, List

# Internal Module Imports
from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager
from Python.utils.tpm_mask_generator import TpmMaskGenerator
from Python.Models.svm_classifier import SVMClassifier
from Python.utils.model_renderer import ModelRenderer
from Python.utils.spm_loader import load_spm_environment
from Python.utils.reset_directory import reset_directory


def run_svm_classification(
    enable_file_logging: bool = False, 
    output_dir: Optional[Path] = None, 
    input_dir: Optional[Path] = None,
    csv_name: str = "covariateADCTRLsexAgeTIV.csv",
    bypass_grid: bool = False,
    use_pretrained: bool = False,
    c_values: List[float] = [1e-4],
    outer_folds: int = 5,
    inner_folds: int = 5
) -> None:
    """
    Orchestrator of the Linear SVM for classification.
    
    PURPOSE:
        Connects the analytical logic (SVMClassifier) to the disk operations (loading/saving).
        Ensures strict matching of security signatures defined by cv_setup.
        
    Args:
        enable_file_logging (bool): If True, writes logs to disk.
        output_dir (Optional[Path]): Directory where results will be written.
        input_dir (Optional[Path]): Path to inputs.
        csv_name (str): Expected registry filename.
        bypass_grid (bool): If True, skips search and tries to load previous C params.
        use_pretrained (bool): If True, aborts training and skips to rendering if models exist.
        c_values (List[float]): Grid space for hyperparameter tuning.
        outer_folds (int): Folds for robust model evaluation.
        inner_folds (int): Folds for grid search tuning.
        
    Raises:
        SystemExit: For fatal errors in IO or security signature mismatches.
    """
    
    current_dir = Path(__file__).parent.resolve()
    
    base_out = output_dir.resolve() if output_dir else current_dir
    common_setup_dir = base_out / "Common_Setup_Results"
    registry_csv_path = common_setup_dir / "cohort_registry.csv"
    folds_json_path = common_setup_dir / "cv_folds_registry.json"

    svm_base = base_out / "SVM_Classification_Results"
    results_dir = svm_base / "Results"
    plots_dir = svm_base / "Plots"
    log_dir = svm_base / "Log_Files"
    csv_out_path = results_dir / "SVM_CV_Metrics.csv"

    log = CustomLogger(name="SVMClassification")
    log.add_console_handler(level="DEBUG", use_colors=True)

    if enable_file_logging:
        reset_directory(log_dir, log)
        log_path = log_dir / "SVMPipeline.log"
        
        try:
            log.add_file_handler(str(log_path), level="DEBUG")
            log.success(f"File logging initialized safely at: {log_path.name}")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {log_path.name}")
            log.critical("Pipeline aborted. Ensure you have write permissions.")
            sys.exit(1)
    else:
        svm_base.mkdir(parents=True, exist_ok=True)
        dummy_file = svm_base / ".dummy_write_test"
        try:
            with open(dummy_file, 'w') as f: pass
            dummy_file.unlink()
            log.info("Dummy write test passed. Filesystem allows writing. Operating in console-only mode.")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to the defined output space {svm_base}.")
            log.critical(f"Pipeline aborted. Ensure you have write permissions. Details: {e}")
            sys.exit(1)

    log.info("--- Booting Linear SVM classification ---")
    log.info("Phase 0: Resolving execution state and historical artifacts...")
    active_c_grid = {'C': c_values}

    if bypass_grid:
        if csv_out_path.exists():
            try:
                old_metrics = pd.read_csv(csv_out_path)
                if 'Optimal_C' in old_metrics.columns:
                    loaded_c_list = old_metrics['Optimal_C'].unique().tolist()
                    active_c_grid = {'C': loaded_c_list}
                    log.info(f"GridSearch bypass activated. Loaded historical optimal C values: {loaded_c_list}")
                else:
                    log.warning("GridSearch bypass failed: 'Optimal_C' column missing in old metrics. Using default/parsed values.")
            except Exception as e:
                log.warning(f"Failed to read previous metrics: {e}. Falling back to default/parsed C values.")
        else:
            log.warning("GridSearch bypass requested but no historical metrics found. Falling back to default/parsedlt C values.")
    else:
        log.info(f"Full GridSearch activated. Injecting C values: {c_values}")

    num_combinations = len(active_c_grid['C'])
    log.info(f"GridSearch configuration: {num_combinations} combination(s) per inner fold.")
    log.info(f"Hyperparameters mapped for 'C': {active_c_grid['C']}")

    if folds_json_path.exists():
        cv_splits_temp = CVManager.load_from_json(str(folds_json_path))
        expected_models = [results_dir / f"SVM_Model_Fold_{s['fold']}.joblib" for s in cv_splits_temp]
    else:
        expected_models = [results_dir / f"SVM_Model_Fold_{i+1}.joblib" for i in range(outer_folds)]
        
    models_exist = all(m.exists() for m in expected_models)

    if use_pretrained and models_exist:
        log.success("Pre-trained models found.")
        log.success("Bypassing SVM training and preserving existing artifacts.")
        return # Early exit
    elif use_pretrained and not models_exist:
        log.warning("Existing models and plots will be purged and recreated.")
        reset_directory(results_dir, log)
        reset_directory(plots_dir, log)
    else:
        reset_directory(results_dir, log)
        reset_directory(plots_dir, log)

    if not registry_csv_path.exists() or not folds_json_path.exists():
        log.warning("Common setup missing. Automatically triggering Setup (cv_setup)...")
        try:
            from Python.Common_Setup.cv_setup import cv_setup
            cv_setup(
                enable_file_logging=enable_file_logging, 
                output_dir=base_out, 
                input_dir=input_dir,
                csv_name=csv_name
            )
            log.success("Common setup completed. Resuming SVM classification...")
        except Exception as e:
            log.critical(f"FATAL: Setup failed. Cannot proceed. Details: {e}")
            sys.exit(1)
    else:
        log.success("Valid common setup found. Bypassing setup phase.")

    try:
        spm_dir = load_spm_environment()
        log.success(f"SPM environment loaded successfully mapped at: {spm_dir}")
    except Exception as e:
        log.critical(f"FATAL: Could not resolve SPM dependency. Details: {e}")
        sys.exit(1)

    tpm_source_path = spm_dir / "tpm" / "TPM.nii"
    mask_dir = common_setup_dir / "Mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "tpm_mask.nii"

    log.info("Phase 1: Checking TPM Mask...")
    
    if not mask_path.exists():
        log.warning(f"TPM Mask not found at '{mask_path.name}'. Booting TPM mask generator...")
        mask_generator = TpmMaskGenerator(logger=log)
        try:
            mask_generator.generate_mask(
                registry_csv_path=str(registry_csv_path),
                tpm_nifti_path=str(tpm_source_path),
                output_mask_path=str(mask_path)
            )
        except Exception as e:
            log.critical(f"FATAL: Could not generate TPM mask. Details: {e}")
            sys.exit(1)
    else:
         log.success("Valid cached TPM Mask found. Bypassing mask generation.")
    
    log.info("Phase 2: Loading cohort registry and TPM mask...")
    svm_engine = SVMClassifier(logger=log, param_grid=active_c_grid, inner_folds=inner_folds)
    
    subjects, X_full, y_full = svm_engine.load_data(str(registry_csv_path), str(mask_path))
    log.success(f"Data Loaded: {X_full.shape[0]} subjects ready.")

    log.info("Phase 3: Loading Cross-Validation split registry...")
    cv_splits = CVManager.load_from_json(str(folds_json_path))
    
    for split in cv_splits:
        test_idx = split['outer_test_idx']
        expected_subjects = split['security_test_subjects']
        actual_subjects = subjects[test_idx].tolist()
        
        if expected_subjects != actual_subjects:
            log.critical("FATAL: Memory misalignment detected! The loaded CSV does not match the JSON artifact.")
            sys.exit(1)

    log.info("Phase 4: Executing SVM Double Cross Validation...")
    results_df, artifacts = svm_engine.execute_nested_cv(X_full, y_full, subjects, cv_splits)
    
    log.info("Phase 5: Rendering ROC curves and aggregating metrics...")
    renderer = ModelRenderer(logger=log, output_dir=str(plots_dir))
    renderer.plot_roc_curves(artifacts, "Linear SVM", "SVM_ROC.png")
    
    log.info(f"\nDetailed results per fold:\n{results_df.to_string(index=False)}")
    
    target_cols = ['Accuracy', 'Balanced_Accuracy', 'F1_Score', 'Sensitivity', 'Specificity', 'AUROC']
    summary_dict = {}
    
    for col in target_cols:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        summary_dict[col] = f"{mean_val:.3f} ± {std_val:.3f}"
        
    inner_mean = results_df['Inner_CV_BalAcc_Mean'].mean()
    inner_std = results_df['Inner_CV_BalAcc_Mean'].std()
    summary_dict['Inner_CV_BalAcc'] = f"{inner_mean:.3f} ± {inner_std:.3f}"
    
    summary_df = pd.DataFrame([summary_dict])
    log.info(f"\nFinal aggregated metrics across {len(results_df)} folds:\n{summary_df.to_string(index=False)}")
    
    results_df.to_csv(str(csv_out_path), index=False)
    log.success(f"Metrics successfully saved to: {csv_out_path.name}")
    
    log.info("Saving trained SVM models to disk for XAI extraction...")

    for artifact in artifacts:
        fold_id = artifact['fold_id']
        model_out_path = results_dir / f"SVM_Model_Fold_{fold_id}.joblib"
        joblib.dump(artifact['model'], str(model_out_path))
        log.debug(f"Saved SVM Pipeline: {model_out_path.name}")

    log.success("--- SVM EXECUTION COMPLETE ---")

if __name__ == "__main__":
    run_svm_classification()