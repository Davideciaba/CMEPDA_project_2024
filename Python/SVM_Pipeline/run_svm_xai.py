"""
Module: run_svm_xai.py

SVM XAI Orchestrator.

Standalone execution script for extracting spatial interpretation patterns
(Raw Weights, Haufe, Gaonkar) from pre-trained SVM models saved on disk.
Guarantees decoupling from the training pipeline.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import joblib
from typing import Optional

from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager
from Python.utils.spm_loader import load_spm_environment
from Python.utils.tpm_mask_generator import TpmMaskGenerator
from Python.Models.svm_classifier import SVMClassifier
from Python.utils.model_renderer import ModelRenderer
from Python.utils.xai_svm import SVMExplainer
from Python.utils.reset_directory import reset_directory


def run_svm_xai(
    enable_file_logging: bool = False, 
    output_dir: Optional[Path] = None
) -> None:
    """
    Executes the analytical extraction maps from the frozen Linear SVM pipeline.
    
    PURPOSE:
        Reads pre-trained coefficients from disk and applies matrix algebra to 
        invert them into biologically plausible brain maps (Haufe, Gaonkar). Renders visual slices.
        
    Args:
        enable_file_logging (bool): Writes logs to disk.
        output_dir (Optional[Path]): Directory defining the base output environment.
        
    Raises:
        SystemExit: If the foundational models or JSON topologies are missing.
    """
    
    current_dir = Path(__file__).parent.resolve()
    base_out = output_dir.resolve() if output_dir else current_dir

    common_setup_dir = base_out / "Common_Setup_Results"
    registry_csv_path = common_setup_dir / "cohort_registry.csv"
    folds_json_path = common_setup_dir / "cv_folds_registry.json"
    mask_path = common_setup_dir / "Mask" / "tpm_mask.nii"

    svm_base = base_out / "SVM_Classification_Results"
    svm_results_dir = svm_base / "Results"

    xai_base = base_out / "SVM_XAI_Results"
    xai_results_dir = xai_base / "Results"
    xai_plots_dir = xai_base / "Plots"
    xai_log_dir = xai_base / "Log_Files"

    log = CustomLogger(name="SVM_XAI_Pipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)

    if enable_file_logging:
        reset_directory(xai_log_dir, log)
        log_path = xai_log_dir / "SVM_XAI.log"
        try:
            log.add_file_handler(str(log_path), level="DEBUG")
            log.success(f"File logging safely initialized at: {log_path.name}")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {log_path.name}")
            log.critical("Pipeline aborted. Ensure you have write permissions.")
            sys.exit(1)
    else:
        # Dummy write test for console-only mode
        xai_base.mkdir(parents=True, exist_ok=True)
        dummy_file = xai_base / ".dummy_write_test"
        try:
            with open(dummy_file, 'w') as f: pass
            dummy_file.unlink()
            log.info("Dummy write test passed. Filesystem allows writing. Operating in console-only mode.")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {xai_base}.")
            log.critical(f"Pipeline aborted. Ensure you have write permissions. Details: {e}")
            sys.exit(1)

    log.info("--- Booting Linear SVM XAI engine ---")
    reset_directory(xai_results_dir, log)
    reset_directory(xai_plots_dir, log)

    if not registry_csv_path.exists() or not folds_json_path.exists():
        log.error("FATAL: Common Setup Results are missing.")
        log.error(f"Please run the setup phase first or ensure path is correct: {common_setup_dir}")
        sys.exit(1)

    try:
        cv_splits = CVManager.load_from_json(str(folds_json_path))
        num_outer_folds = len(cv_splits)
        log.info(f"CV Topology read successfully: detected {num_outer_folds} execution folds.")
    except Exception as e:
        log.critical(f"FATAL: Artifact corruption detected. Failed to read CV folds JSON. Details: {e}")
        sys.exit(1)

    expected_models = [svm_results_dir / f"SVM_Model_Fold_{split['fold']}.joblib" for split in cv_splits]
    if not all(m.exists() for m in expected_models):
        log.error("FATAL: Trained SVM Models are missing or incomplete.")
        log.error("XAI Extraction requires all folds to be trained. Please run SVM classification first.")
        sys.exit(1)

    log.info("Phase 1: Validating TPM Mask...")
    if not mask_path.exists():
        log.warning(f"TPM Mask not found at '{mask_path.name}'. Booting TPM Mask generator...")
        try:
            spm_dir = load_spm_environment()
            log.success(f"SPM environment loaded successfully mapped at: {spm_dir}")
        
            tpm_path = spm_dir / "tpm" / "TPM.nii"
            mask_generator = TpmMaskGenerator(logger=log)
        
            mask_generator.generate_mask(
                registry_csv_path=str(registry_csv_path),
                tpm_nifti_path=str(tpm_path),
                output_mask_path=str(mask_path)
            )
        except Exception as e:
            log.critical(f"FATAL: Could not generate TPM Mask. Details: {e}")
            sys.exit(1)
    else:
         log.success("Valid Cached TPM Mask found. Bypassing Generation.")

    log.info("Phase 2: Loading cohort registry and TPM Mask...")
    svm_engine = SVMClassifier(logger=log, param_grid={})
    _, X_full, y_full = svm_engine.load_data(str(registry_csv_path), str(mask_path))
    cv_splits = CVManager.load_from_json(str(folds_json_path))

    log.info("Phase 3: Extracting XAI patterns (Raw, Haufe, Gaonkar) per fold...")
    explainer = SVMExplainer(logger=log)
    renderer = ModelRenderer(logger=log, output_dir=str(xai_plots_dir))

    mask_img = nib.load(str(mask_path))
    mask_bool = mask_img.get_fdata() > 0
    mask_affine = mask_img.affine

    registry_df = pd.read_csv(registry_csv_path)
    ctrl_candidates = registry_df[registry_df['subject_id'].str.contains("CTRL-117")]
    
    bg_path = str(ctrl_candidates.iloc[0]['file_path']) if not ctrl_candidates.empty else None

    for split in cv_splits:
        fold_id = split['fold']
        train_idx = split['outer_train_idx']

        model_path = svm_results_dir / f"SVM_Model_Fold_{fold_id}.joblib"
        
        log.info(f"--- Extracting XAI patterns for Fold {fold_id} ---")
        trained_pipeline = joblib.load(str(model_path))
        
        optimal_c = trained_pipeline.named_steps['svc'].C
        
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        
        X_train_scaled = trained_pipeline.named_steps['scaler'].transform(X_train_fold)
        
        raw_weights = trained_pipeline.named_steps['svc'].coef_[0]
        decision_scores = trained_pipeline.decision_function(X_train_fold)
        
        # 1. RAW WEIGHTS
        raw_weights_top1 = np.where(np.abs(raw_weights) >= np.percentile(np.abs(raw_weights), 99), raw_weights, 0)
        raw_weights_top5 = np.where(np.abs(raw_weights) >= np.percentile(np.abs(raw_weights), 95), raw_weights, 0)

        # 2. HAUFE PATTERNS
        haufe_map = explainer.compute_haufe_patterns(X_train_scaled, decision_scores)
        haufe_map_top1 = np.where(np.abs(haufe_map) >= np.percentile(np.abs(haufe_map), 99), haufe_map, 0)
        haufe_map_top5 = np.where(np.abs(haufe_map) >= np.percentile(np.abs(haufe_map), 95), haufe_map, 0)

        # 3. GAONKAR MAPS
        gaonkar_z_map_thresholded_bonf005, _ = explainer.compute_gaonkar_maps(
            X_train=X_train_scaled, 
            y_train=y_train_fold, 
            svm_weights=raw_weights, 
            C_param=optimal_c, 
            correction='bonferroni', 
            alpha=0.05
        )
        gaonkar_z_map_thresholded_fdr01, _ = explainer.compute_gaonkar_maps(
            X_train=X_train_scaled, 
            y_train=y_train_fold, 
            svm_weights=raw_weights, 
            C_param=optimal_c, 
            correction='fdr_by', 
            alpha=0.1
        )
        
        # NIfTI Definitions
        raw_nii_top1 = str(xai_results_dir / f"SVM_Raw_Weights_Fold_{fold_id}_Top1.nii")
        raw_nii_top5 = str(xai_results_dir / f"SVM_Raw_Weights_Fold_{fold_id}_Top5.nii")
        haufe_nii_top1 = str(xai_results_dir / f"SVM_Haufe_Fold_{fold_id}_Top1.nii")
        haufe_nii_top5 = str(xai_results_dir / f"SVM_Haufe_Fold_{fold_id}_Top5.nii")
        gaonkar_nii_bonf005 = str(xai_results_dir / f"SVM_Gaonkar_Fold_{fold_id}_bonf005.nii")
        gaonkar_nii_fdr01 = str(xai_results_dir / f"SVM_Gaonkar_Fold_{fold_id}_fdr01.nii")

        # Serializing to NIfTI
        explainer.reconstruct_and_save_3d(raw_weights_top1, mask_bool, mask_affine, raw_nii_top1)
        explainer.reconstruct_and_save_3d(raw_weights_top5, mask_bool, mask_affine, raw_nii_top5)
        explainer.reconstruct_and_save_3d(haufe_map_top1, mask_bool, mask_affine, haufe_nii_top1)
        explainer.reconstruct_and_save_3d(haufe_map_top5, mask_bool, mask_affine, haufe_nii_top5)
        explainer.reconstruct_and_save_3d(gaonkar_z_map_thresholded_bonf005, mask_bool, mask_affine, gaonkar_nii_bonf005)
        explainer.reconstruct_and_save_3d(gaonkar_z_map_thresholded_fdr01, mask_bool, mask_affine, gaonkar_nii_fdr01)
        
        slice_config = 3.0
        if bg_path:

            renderer.plot_3d_activation_map(
                bg_path, raw_nii_top1, str(mask_path), f"Raw Weights (Fold {fold_id}) Top 1%", 
                f"SVM_RawWeights_Fold_{fold_id}_Top1.png", slice_config=slice_config
            )

            renderer.plot_3d_activation_map(
                bg_path, raw_nii_top5, str(mask_path), f"Raw Weights (Fold {fold_id}) Top 5%", 
                f"SVM_RawWeights_Fold_{fold_id}_Top5.png", slice_config=slice_config
            )

            renderer.plot_3d_activation_map(
                bg_path, haufe_nii_top1, str(mask_path), f"Haufe Pattern (Fold {fold_id}) Top 1%", 
                f"SVM_Haufe_Fold_{fold_id}_Top1.png", slice_config=slice_config
            )

            renderer.plot_3d_activation_map(
                bg_path, haufe_nii_top5, str(mask_path), f"Haufe Pattern (Fold {fold_id}) Top 5%", 
                f"SVM_Haufe_Fold_{fold_id}_Top5.png", slice_config=slice_config
            )

            renderer.plot_3d_activation_map(
                bg_path, gaonkar_nii_bonf005, str(mask_path), f"Gaonkar Z-Score (Fold {fold_id}) Bonf 0.05", 
                f"SVM_Gaonkar_Fold_{fold_id}_bonf005.png", slice_config=slice_config
            )

            renderer.plot_3d_activation_map(
                bg_path, gaonkar_nii_fdr01, str(mask_path), f"Gaonkar Z-Score (Fold {fold_id}) FDR 0.1", 
                f"SVM_Gaonkar_Fold_{fold_id}_fdr01.png", slice_config=slice_config
            )

    log.success("--- SVM XAI EXTRACTION COMPLETE ---")

if __name__ == "__main__":
    run_svm_xai()