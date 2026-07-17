"""
Pipeline Phase 0: Setup & Freeze Orchestrator.

Responsibilities:
1. Normalizes raw CSV data into a standardized Python Registry.
2. Computes the Nested CV topology and freezes it into a JSON Artifact (SSOT).
No MATLAB or model-specific logic exists here. Pure Universal Setup.
"""
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager
from Python.utils.reset_directory import reset_directory

def cv_setup(
    enable_file_logging: bool = False, 
    output_dir: Optional[Path] = None, 
    input_dir: Optional[Path] = None, 
    csv_name: str = "covariateADCTRLsexAgeTIV.csv",
    outer_folds: int = 5,
    inner_folds: int = 5
) -> None:
    
    # Path Configurations
    current_dir = Path(__file__).parent.resolve()
    project_root_dir = current_dir.parent.parent
    source_dir = input_dir.resolve() if input_dir else project_root_dir / "AD_CTRL"
    csv_path = source_dir / csv_name
    nifti_dir = source_dir / "AD_CTRL_nii" 
    
    base_out = output_dir.resolve() if output_dir else current_dir
    common_setup_dir = base_out / "Common_Setup_Results"
    log_dir = common_setup_dir / "Log_Files"

    registry_csv_path = common_setup_dir / "cohort_registry.csv"
    folds_json_path = common_setup_dir / "cv_folds_registry.json"

    log = CustomLogger(name="CVSetup")
    log.add_console_handler(level="DEBUG", use_colors=True)
    if enable_file_logging:
        reset_directory(log_dir, log)
        log_path = log_dir / "CVSetup.log"
        try:
            log.add_file_handler(str(log_path), level="DEBUG")
            log.success(f"File logging safely initialized at: {log_path.name}")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to {log_path.name}. Details: {e}")
            sys.exit(1)
    else:
        # Dummy write test for console-only mode
        common_setup_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = common_setup_dir / ".dummy_write_test"
        try:
            with open(dummy_file, 'w') as f: pass
            dummy_file.unlink()
            log.info("Dummy write test passed. Filesystem allows writing. Operating in console-only mode.")
        except OSError as e:
            log.critical(f"I/O ERROR: Cannot write to the defined output space {common_setup_dir}.")
            log.critical(f"Pipeline aborted. Ensure you have write permissions. Details: {e}")
            sys.exit(1)

    log.info("--- Data and Cross-Validation Setup ---")

    log.info("Creating a cohort registry containing essential information...")
    
    try:
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        log.critical(f"FATAL: Source CSV not found at {csv_path}")
        sys.exit(1)

    registry_data = []
    
    for _, row in df_raw.iterrows():
        subj_id = str(row['ID'])
        group = str(row['Group']).upper()
        label = 1 if group == 'AD' else 0
        file_path = nifti_dir / f"smwc1{subj_id}.nii"
        
        try:
            with open(file_path, 'rb'): pass
        except FileNotFoundError as e:
            log.critical(f"Integrity Error: missing NIfTI volume for subject {subj_id} at {file_path}")
            raise e
        
        try:
            # Calculate the path relative to the project root
            portable_path = file_path.relative_to(project_root_dir)
        except ValueError:
            # Fallback if input_dir is strictly outside the project root
            log.warning(f"Subject {subj_id} NIfTI is outside the project root. Absolute path will be used.")
            portable_path = file_path

        registry_data.append({
            "subject_id": subj_id,
            "label": label,
            "file_path": portable_path
        })
        
    registry_df = pd.DataFrame(registry_data)
    registry_df.to_csv(registry_csv_path, index=False)
    log.success(f"Registry built: saved {len(registry_df)} validated subjects to '{registry_csv_path.name}'.")

    log.info("Generating and freezing the nested CV topology ({outer_folds} Outer, {inner_folds} Inner)...")
    y_target = registry_df['label'].values
    subjects = registry_df['subject_id'].values
    
    cv_manager = CVManager(outer_folds=outer_folds, inner_folds=inner_folds, random_state=42)
    splits = cv_manager.generate_splits(y_target)
    
    CVManager.save_to_json(splits, subjects, str(folds_json_path))
    log.success(f"Cross-validation topology saved to '{folds_json_path.name}'. Ready for model execution.")