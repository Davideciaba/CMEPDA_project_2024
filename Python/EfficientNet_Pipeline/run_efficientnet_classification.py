import sys
import pathlib
import pandas as pd
import torch

# 1. Resolve absolute path to the current file
current_file_path = pathlib.Path(__file__).resolve()

# 2. Navigate up to the CMEPDA_project_2024 root directory (2 levels up)
project_root = current_file_path.parents[2]

sys.path.append(str(project_root))

from Python.utils.py_logger import CustomLogger
from Python.Models.efficientnet_classifier import EfficientNetClassifier
from Python.utils.cv_manager import CVManager
from Python.utils.model_renderer import ModelRenderer


def run_efficientnet_classification():

    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = CURRENT_DIR.parent.parent
    SETUP_DIR = PROJECT_DIR / "Python" / "Common_Setup"

    log = CustomLogger(name="EfficientNetPipeline")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log_dir = CURRENT_DIR / "Log_Files"
    log_path = log_dir / "EfficientNetPipeline.log"
    log.add_file_handler(log_path, level="DEBUG")
    log.info("--- Booting 3D EfficientNet Engine ---")
    
    registry_csv_path = SETUP_DIR / "python_registry.csv"
    results_dir = CURRENT_DIR / "Results"
    folds_json_path = SETUP_DIR / "cv_folds_registry.json"
    plots_dir = CURRENT_DIR / "Plots"

    
    # 1. LOAD DATA
    log.info("Ingesting normalized registry for MONAI Dicts...")
    PARAM_DICT = {
        'optimizer': ['adamw'], 
        'scheduler': ['none', 'exp'],
        'lr': [1e-4, 1e-2], 
        'wd': [1e-3]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    efficientnet_engine = EfficientNetClassifier(logger = log, device= device, param_grid=PARAM_DICT)

    subjects, data_dicts, y_full = efficientnet_engine.load_data(str(registry_csv_path), str(project_root))
    log.success(f"Data Loaded: {len(data_dicts)} 3D NIfTI pointers mapped.")

    # 2. LOAD & VALIDATE SSOT
    log.info("Loading Frozen Fold Artifact...")
    cv_splits = CVManager.load_from_json(str(folds_json_path))
    
    # CRITICAL SECURITY GUARD: Verify Artifact Signatures
    for split in cv_splits:
        test_idx = split['outer_test_idx']
        expected_subjects = split['security_test_subjects']
        # Extract the subjects from RAM using the JSON indices
        actual_subjects = subjects[test_idx].tolist()
        
        if expected_subjects != actual_subjects:
            log.critical("FATAL: Memory misalignment detected! The loaded CSV does not match the JSON artifact.")
            log.critical(f"Expected: {expected_subjects}")
            log.critical(f"Actual:   {actual_subjects}")
            sys.exit(1)
            
    log.success("Security Signature validated. Fold integrity is 100% guaranteed.")

    # 3. TRAIN
    log.info("Executing CNN Double Cross Validation...")
    results_df, artifacts = efficientnet_engine.execute_nested_cv(data_dicts, y_full, subjects, cv_splits)

    log.info(f"\nDetailed Results per Fold:\n{results_df.to_string(index=False)}")

    target_cols = ['Accuracy', 'Balanced_Accuracy', 'F1_Score', 'Sensitivity', 'Specificity', 'AUROC']
    summary_dict = {}

    for col in target_cols:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        summary_dict[col] = f"{mean_val:.3f} ± {std_val:.3f}"

    if 'Inner_CV_BalAcc_Mean' in results_df.columns:
        inner_mean = results_df['Inner_CV_BalAcc_Mean'].mean()
        inner_std = results_df['Inner_CV_BalAcc_Mean'].std()
        summary_dict['Inner_CV_BalAcc'] = f"{inner_mean:.3f} ± {inner_std:.3f}"
    
    summary_df = pd.DataFrame([summary_dict])
    log.info(f"\nFinal Aggregated Metrics across {len(results_df)} Folds:\n{summary_df.to_string(index=False)}")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out_path = results_dir / "EfficientNet_CV_Metrics.csv"
    results_df.to_csv(str(csv_out_path), index=False)
    log.success(f"Raw Metrics successfully saved to: {csv_out_path.name}")

    log.info("Saving trained PyTorch models to disk for XAI extraction...")
    for artifact in artifacts:
        fold_id = artifact['fold_id']
        model_out_path = results_dir / f"EfficientNet_Fold_{fold_id}.pth"
        torch.save(artifact['model_state'], str(model_out_path))
        log.debug(f"Saved Network Weights: {model_out_path.name}")

    log.info("Generating performance visualizations...")
    renderer = ModelRenderer(logger=log, output_dir=str(plots_dir))
    renderer.plot_roc_curves(artifacts, "EfficientNet", "EfficientNet_ROC.png")
    renderer.plot_inner_cv_losses(artifacts, "EfficientNet", "EfficientNet_Inner_Losses.png")
    renderer.plot_outer_cv_losses(artifacts, "EfficientNet", "EfficientNet_Outer_Losses.png")

    log.success("--- CNN EXECUTION COMPLETE ---")

if __name__ == "__main__":
    run_efficientnet_classification()