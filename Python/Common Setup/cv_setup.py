"""
Pipeline Phase 0: Setup & Freeze Orchestrator.

Responsibilities:
1. Normalizes raw CSV data into a standardized Python Registry.
2. Computes the Nested CV topology and freezes it into a JSON Artifact (SSOT).
No MATLAB or model-specific logic exists here. Pure Universal Setup.
"""
import sys
import pathlib

# 1. Resolve absolute path to the current file
current_file_path = pathlib.Path(__file__).resolve()

# 2. Navigate up to the CMEPDA_project_2024 root directory (2 levels up)
project_root = current_file_path.parents[2]

sys.path.append(str(project_root))

import pandas as pd
from Python.utils.py_logger import CustomLogger
from Python.utils.cv_manager import CVManager

def cv_setup():
    
    # Path Configurations
    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = CURRENT_DIR.parent.parent
    AD_CTRL_DIR = PROJECT_DIR / "AD_CTRL"
    csv_path = AD_CTRL_DIR / "covariateADCTRLsexAgeTIV.csv"
    nifti_dir = AD_CTRL_DIR / "AD_CTRL_nii"
    
    registry_csv_path = CURRENT_DIR / "python_registry.csv"
    folds_json_path = CURRENT_DIR / "cv_folds_registry.json"
    log_path = CURRENT_DIR / "CVSetup.log"

    log = CustomLogger(name="CVSetup")
    log.add_console_handler(level="DEBUG", use_colors=True)
    log.add_file_handler(log_path, level="DEBUG")

    log.info("--- Data and Cross-Validation Setup ---")

    # 1. STANDARDIZED REGISTRY CREATION
    log.info("Creating a data registry containing essential information...")
    
    # EAFP applies below: pd.read_csv will natively fail if the raw CSV is missing
    df_raw = pd.read_csv(csv_path)
    registry_data = []
    
    for _, row in df_raw.iterrows():
        subj_id = str(row['ID'])
        group = str(row['Group']).upper()
        label = 1 if group == 'AD' else 0
        
        # Expected file pattern based on MATLAB cohort design
        file_path = nifti_dir / f"smwc1{subj_id}.nii"
        
        # EAFP: Physical File Validation. Crash immediately if a NIfTI is missing.
        try:
            with open(file_path, 'rb'): pass
        except FileNotFoundError as e:
            log.critical(f"Integrity Error: missing NIfTI volume for subject {subj_id} at {file_path}")
            raise e
            
        registry_data.append({
            "subject_id": subj_id,
            "label": label,
            "file_path": str(file_path)
        })
        
    registry_df = pd.DataFrame(registry_data)
    registry_df.to_csv(registry_csv_path, index=False)
    log.success(f"Registry built: saved {len(registry_df)} validated subjects to '{registry_csv_path.name}'.")

    # 2. FOLD SYNCHRONIZATION (SSOT)
    log.info("Generating and freezing the cross-validation topology for both the models...")
    y_target = registry_df['label'].values
    subjects = registry_df['subject_id'].values
    
    cv_manager = CVManager(outer_folds=5, inner_folds=5, random_state=42)
    splits = cv_manager.generate_splits(y_target)
    
    CVManager.save_to_json(splits, subjects, str(folds_json_path))
    log.success(f"Cross-validation topology saved to '{folds_json_path.name}'. Ready for model execution.")

if __name__ == "__main__":
    cv_setup()