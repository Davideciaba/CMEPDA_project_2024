import sys
import pathlib
import torch
from Python.utils.py_logger import CustomLogger
from Python.Models.efficientnet_classifier import EfficientNetClassifier
from Python.utils.cv_manager import CVManager

def execute_cnn_pipeline():
    log = CustomLogger(name="CNN_Runner")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log.add_file_handler("03_cnn_execution.log", level="TRACE")
    log.info("--- Booting Decoupled 3D EfficientNet Engine ---")
    
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    registry_csv_path = BASE_DIR / "python_registry.csv"
    folds_json_path = BASE_DIR / "cv_folds_registry.json"
    
    # 1. LOAD DATA
    log.info("Ingesting normalized registry for MONAI Dicts...")
    
    # EAFP: Fails natively if 01_freeze_pipeline.py was not run
    subjects, data_dicts, y_cnn = EfficientNetClassifier.load_data_dicts(str(registry_csv_path))
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
            sys.exit(1)
            
    log.success("Security Signature validated. Fold integrity is 100% guaranteed.")

    # 3. TRAIN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"PyTorch Compute Device allocated: {device.type.upper()}")
    
    cnn_engine = EfficientNetClassifier(logger=log, device=device)
    
    log.info("Executing CNN Double Cross Validation...")
    results_df, artifacts = cnn_engine.execute_nested_cv(data_dicts, y_cnn, subjects, cv_splits)
    
    log.success("--- CNN EXECUTION COMPLETE ---")
    log.info(f"\nAggregated Metrics:\n{results_df.mean(numeric_only=True).to_frame().T.to_string(index=False)}")
    results_df.to_csv("CNN_CV_Metrics.csv", index=False)

if __name__ == "__main__":
    execute_cnn_pipeline()