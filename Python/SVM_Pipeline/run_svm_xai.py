"""
SVM XAI Orchestrator.

Standalone execution script for extracting spatial interpretation patterns
(Raw Weights, Haufe, Gaonkar) from pre-trained SVM models saved on disk.
Guarantees decoupling from the training pipeline.
"""
import sys
import pathlib
import pandas as pd
import numpy as np
import nibabel as nib
import joblib

current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager
from Python.utils.spm_loader import load_spm_environment
from Python.utils.tpm_mask_generator import TpmMaskGenerator
from Python.Models.svm_classifier import SVMClassifier
from Python.utils.model_renderer import ModelRenderer
from Python.XAI.xai_svm import SVMExplainer

def run_svm_xai():
    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = CURRENT_DIR.parent.parent
    SETUP_DIR = PROJECT_DIR / "Python" / "Common_Setup"

    log = CustomLogger(name="SVM_XAI_Pipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log.info("--- Booting Decoupled SVM XAI Engine ---")

    registry_csv_path = SETUP_DIR / "python_registry.csv"
    results_dir = CURRENT_DIR / "Results"
    plots_dir = CURRENT_DIR / "Plots"
    mask_path = SETUP_DIR / "Mask" / "tpm_mask.nii"
    folds_json_path = SETUP_DIR / "cv_folds_registry.json"

    if not mask_path.exists():
        log.warning(f"TPM Mask not found at '{mask_path.name}'. Booting TPM Generator...")
        try:
            spm_dir = load_spm_environment()
            log.success(f"SPM environment loaded successfully mapped at: {spm_dir}")
        except Exception as e:
            log.critical(f"FATAL: Could not resolve SPM dependency. Details: {e}")
            sys.exit(1)
        
        tpm_path = spm_dir / "tpm" / "TPM.nii"
        mask_generator = TpmMaskGenerator(logger=log)
        
        try:
            mask_generator.generate_mask(
                registry_csv_path=str(registry_csv_path),
                tpm_nifti_path=str(tpm_path),
                output_mask_path=str(mask_path)
            )
        except Exception as e:
            log.critical(f"FATAL: Could not generate TPM mask natively. Details: {e}")
            sys.exit(1)
    else:
         log.success("Valid Cached TPM Mask found. Bypassing Generation.")

    log.info("Loading Subject Registry and TPM Mask...")
    svm_engine = SVMClassifier(logger=log, param_grid={})
    _, X_full, y_full = svm_engine.load_data(str(registry_csv_path), str(mask_path))
    cv_splits = CVManager.load_from_json(str(folds_json_path))

    log.info("Extracting XAI Spatial Patterns (Raw, Haufe, Gaonkar) PER FOLD...")
    explainer = SVMExplainer(logger=log)
    renderer = ModelRenderer(logger=log, output_dir=str(plots_dir))

    log.info("Reading Affine Matrix and Spatial Geometry from TPM Mask header...")
    mask_img = nib.load(str(mask_path))
    mask_bool = mask_img.get_fdata() > 0
    mask_affine = mask_img.affine

    # Dynamically locate CTRL-117 using the Single Source of Truth (Registry)
    registry_df = pd.read_csv(registry_csv_path)
    ctrl_candidates = registry_df[registry_df['subject_id'].str.contains("CTRL-117")]
    
    bg_path = str(ctrl_candidates.iloc[0]['file_path']) if not ctrl_candidates.empty else None

    for split in cv_splits:
        fold_id = split['fold_id']
        train_idx = split['train_idx']

        # Load pre-trained pipeline
        model_path = results_dir / f"SVM_Model_Fold_{fold_id}.joblib"
        if not model_path.exists():
            log.error(f"Trained model not found at {model_path}. Please run run_svm_pipeline.py first.")
            continue
        
        log.info(f"--- Extracting XAI Patterns for Fold {fold_id} ---")
        trained_pipeline = joblib.load(str(model_path))
        
        # Determine the C parameter used
        optimal_c = trained_pipeline.named_steps['svc'].C
        
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        
        X_train_scaled = trained_pipeline.named_steps['scaler'].transform(X_train_fold)
        
        raw_weights = trained_pipeline.named_steps['svc'].coef_[0]
        decision_scores = trained_pipeline.decision_function(X_train_fold)
        n_support_total = int(np.sum(trained_pipeline.named_steps['svc'].n_support_))
        
        raw_weights_top1 = np.where(np.abs(raw_weights) >= np.percentile(np.abs(raw_weights), 99), raw_weights, 0)
        raw_weights_top5 = np.where(np.abs(raw_weights) >= np.percentile(np.abs(raw_weights), 95), raw_weights, 0)

        haufe_map = explainer.compute_haufe_patterns(X_train_scaled, decision_scores)

        haufe_map_top1 = np.where(np.abs(haufe_map) >= np.percentile(np.abs(haufe_map), 99), haufe_map, 0)
        haufe_map_top5 = np.where(np.abs(haufe_map) >= np.percentile(np.abs(haufe_map), 95), haufe_map, 0)

        gaonkar_z_map_thresholded_bonf005, _ = explainer.compute_gaonkar_maps(
            X_train=X_train_scaled, 
            y_train=y_train_fold, 
            svm_weights=raw_weights, 
            C_param=optimal_c, 
            n_support=n_support_total,
            correction='bonferroni', 
            alpha=0.05
        )
        gaonkar_z_map_thresholded_fdr01, _ = explainer.compute_gaonkar_maps(
            X_train=X_train_scaled, 
            y_train=y_train_fold, 
            svm_weights=raw_weights, 
            C_param=optimal_c, 
            n_support=n_support_total,
            correction='fdr_by', 
            alpha=0.1
        )
        
        raw_nii_top1 = str(results_dir / f"SVM_Raw_Weights_Fold_{fold_id}_Top1.nii")
        raw_nii_top5 = str(results_dir / f"SVM_Raw_Weights_Fold_{fold_id}_Top5.nii")
        haufe_nii_top1 = str(results_dir / f"SVM_Haufe_Fold_{fold_id}_Top1.nii")
        haufe_nii_top5 = str(results_dir / f"SVM_Haufe_Fold_{fold_id}_Top5.nii")
        gaonkar_nii_bonf005 = str(results_dir / f"SVM_Gaonkar_Fold_{fold_id}_bonf005.nii")
        gaonkar_nii_fdr01 = str(results_dir / f"SVM_Gaonkar_Fold_{fold_id}_fdr01.nii")

        
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

if __name__ == "__main__":
    run_svm_xai()
