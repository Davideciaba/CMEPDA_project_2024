"""
Module: spm_loader.py

Bridges Python environments with the Statistical Parametric Mapping (SPM) suite.
Handles dynamic path resolution via environment variables
"""
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

ENV_VAR_NAME = "SPM_DIR"
DOTENV_FILE = ".env"
SPM_SIGNATURE = "spm.m"

def _find_env_file(start_path: Path) -> Optional[Path]:
    """
    Traverses upwards from a given starting path to locate the .env file.
    Halts if the specific project directory is reached.
    """
    current_path = start_path.resolve()
    
    while True:
        potential_env = current_path / DOTENV_FILE

        if potential_env.is_file():
            return potential_env
        
        if current_path.parent == current_path:
            break
            
        current_path = current_path.parent

    return None

def load_spm_environment() -> Path:
    """
    Locates the SPM directory via environment variable or .env file,
    validates its existence, and prioritizes it in the system path.
    """
    # Search upwards from cwd
    cwd_path = Path.cwd()
    env_path = _find_env_file(cwd_path)

    # Search upwards from script location
    if not env_path:
        script_path = Path(__file__).resolve().parent
        env_path = _find_env_file(script_path)

        
    if env_path:
        # Load .env file into environment variables
        load_dotenv(dotenv_path=str(env_path))

    spm_dir_str = os.getenv(ENV_VAR_NAME)

    if not spm_dir_str:
        raise EnvironmentError(
            f"SPM path not found. Set the '{ENV_VAR_NAME}' env variable or "
            f"ensure a '{DOTENV_FILE}' file exists in the project hierarchy.."
        )

    spm_path = Path(spm_dir_str).resolve()

    if not spm_path.is_dir():
        raise FileNotFoundError(
            f"The SPM directory specified does not exist: {spm_path}. "
            "Please verify your configuration."
        )
    
    # Verify the target directory is actually an SPM installation
    if not (spm_path / SPM_SIGNATURE).is_file():
        raise FileNotFoundError(
            f"Invalid SPM directory. Signature '{SPM_SIGNATURE}' missing in {spm_path}."
        )

    spm_path_str = str(spm_path)

    # Identify and remove any existing SPM paths to prevent version conflicts
    paths_to_remove = []
    for path_entry in sys.path:
        entry_path = Path(path_entry)
        # Check if the path itself is the SPM target or contains the signature file
        if path_entry == spm_path_str or (entry_path / SPM_SIGNATURE).is_file():
            paths_to_remove.append(path_entry)

            
    for obsolete_path in paths_to_remove:
        sys.path.remove(obsolete_path)

    # Inject the correct SPM path with maximum priority
    sys.path.insert(0, spm_path_str)

    return spm_path