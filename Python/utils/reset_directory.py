"""
Module: reset_directory.py

Provides a centralized utility for safely purging and recreating directories.
"""
import sys
from pathlib import Path
import shutil
from Python.utils.py_logger import CustomLogger

def reset_directory(dir_path: Path, log: CustomLogger) -> None:
    """
    Purges an existing directory entirely and recreates it.
    """
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            log.critical(f"FATAL: Could not purge directory {dir_path}. Reason: {e}")
            sys.exit(1)
    
    dir_path.mkdir(parents=True, exist_ok=True)