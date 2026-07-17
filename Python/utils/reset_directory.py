"""
Module: reset_directory.py

Provides a centralized utility for safely purging and recreating directories.
Used to ensure clean execution states across pipeline runs.
"""
import sys
from pathlib import Path
import shutil
from Python.utils.py_logger import CustomLogger

def reset_directory(dir_path: Path, log: CustomLogger) -> None:
    """
    Purges an existing directory entirely and recreates it.
    
    PURPOSE:
        Acts as a failsafe equivalent to MATLAB's `rmdir('s') / mkdir()`.
        Ensures that old pipeline artifacts do not pollute subsequent executions.
        
    Args:
        dir_path (Path): Pathlib object pointing to the target directory.
        log (CustomLogger): Centralized logging instance for tracking I/O errors.
        
    Raises:
        SystemExit: If the OS denies deletion (e.g., file open in another program).
    """
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            log.critical(f"FATAL: Could not purge directory {dir_path}. Reason: {e}")
            sys.exit(1)
    
    dir_path.mkdir(parents=True, exist_ok=True)