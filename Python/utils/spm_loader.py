import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ENV_VAR_NAME = "SPM_DIR"
DOTENV_FILE = ".env"


def load_spm_environment() -> Path:
    """
    Locates the SPM directory via environment variable or .env file,
    validates its existence, and prioritizes it in the system path.

    Returns:
        Path: A pathlib.Path object pointing to the validated SPM directory.
    """
    # Load .env file into environment variables
    load_dotenv(dotenv_path=DOTENV_FILE)

    spm_dir_str = os.getenv(ENV_VAR_NAME)

    if not spm_dir_str:
        raise EnvironmentError(
            f"SPM path not found. Please set the '{ENV_VAR_NAME}' "
            f"environment variable or create a '{DOTENV_FILE}' file in the root."
        )

    spm_path = Path(spm_dir_str).resolve()

    if not spm_path.is_dir():
        raise FileNotFoundError(
            f"The SPM directory specified does not exist: {spm_path}. "
            "Please verify your configuration."
        )

    # Ensure the target SPM directory is at the top of sys.path.
    # Remove it first if it exists elsewhere in the path to avoid duplicates.
    spm_path_str = str(spm_path)
    if spm_path_str in sys.path:
        sys.path.remove(spm_path_str)
        
    sys.path.insert(0, spm_path_str)

    return spm_path