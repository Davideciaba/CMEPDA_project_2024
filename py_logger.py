"""
Module: py_logger.py

This module defines the CustomLogger class, which is a customized wrapper around the loguru logger.
"""
import sys
from loguru import logger

class CustomLogger:
    """
    CustomLogger is a customized wrapper around the loguru logger to provide:
    - Dual sinks: one for standard Python logs and another for MATLAB logs (if needed).
    - Configurable log levels and formatting.
    - Optional file logging with rotation and compression.
    - Context manager to attach extra fields (like session_id) to logs within a specific context.
    Usage:
    log = CustomLogger(enable_file_logging=True, level="DEBUG")
    log.info("This is an info message with {placeholder}", placeholder="value")
    with log.context(session_id="SESSION-123"):
        log.debug("This message will have the session_id in its context.")
    """
    def __init__(
        self,
        log_file_path: str = "log_file.log",
        enable_file_logging: bool = False,
        level: str = "DEBUG"
    ):
        """
        Initialize the CustomLogger.
        Parameters:
        - log_file_path: Path to the log file (used if enable_file_logging is True).
        - enable_file_logging: If True, logs will also be written to a file.
        - level: The minimum logging level. 
            Options: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
            Default is DEBUG, which means all levels DEBUG and above will be logged.
        """
        # Reset sinks
        logger.remove()

        # Configure the logger with the specified sinks and formatting
        logger.configure(extra={
            "session_id": "python-default-session",
            "from_matlab": False,  # flag to filter MATLAB logs
            "mt": ""               # timestamp MATLAB string
        })

        # --- Sink A: default (Python) ---
        logger.add(
            sys.stdout,
            level=level.upper(),
            colorize=True,
            filter=lambda r: not r["extra"].get("from_matlab", False),
            format=(
                "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[session_id]}</cyan> | "
                "<dim>{file.name}:{function}:{line}</dim> - "
                "<level>{message}</level>"
            )
        )

        # --- Sink B: MATLAB (replayed) ---
        logger.add(
            sys.stdout,
            level=level.upper(),
            colorize=True,
            filter=lambda r: r["extra"].get("from_matlab", False),
            format=(
                "<dim>{extra[mt]}</dim> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[session_id]}</cyan> | "
                "<dim>{file.name}:{function}:{line}</dim> - "
                "<level>{message}</level>"
            )
        )

        # Optional file sink
        if enable_file_logging:
            logger.add(
                log_file_path,
                level=level.upper(),
                rotation="50 KB",
                compression="zip",
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{extra[session_id]} | " 
                    "{file.name}:{function}:{line} - {message}"
                )
            )
            self.info("Logger started. File logging ENABLED. Writing to {path}", path=log_file_path)
        else:
            self.info("Logger started. File logging DISABLED. Console only.")

    def context(self, **kwargs):
        """Context manager to attach extra fields (e.g., session_id=...)."""
        return logger.contextualize(**kwargs)

    # ---------- Wrappers for each log level ----------
    def trace(self, message, **fmt):
        """Log a TRACE level message with optional formatting."""
        logger.opt(depth=1).log("TRACE", message, **fmt)

    def debug(self, message, **fmt):
        """Log a DEBUG level message with optional formatting."""
        logger.opt(depth=1).log("DEBUG", message, **fmt)

    def info(self, message, **fmt):
        """Log a INFO level message with optional formatting."""
        logger.opt(depth=1).log("INFO", message, **fmt)

    def success(self, message, **fmt):
        """Log a SUCCESS level message with optional formatting."""
        logger.opt(depth=1).log("SUCCESS", message, **fmt)

    def warn(self, message, **fmt):
        """Log a WARNING level message with optional formatting."""
        logger.opt(depth=1).log("WARNING", message, **fmt)

    def error(self, message, **fmt):
        """Log a ERROR level message with optional formatting."""
        logger.opt(depth=1).log("ERROR",  message, **fmt)

    def critical(self, message, **fmt):
        """Log a CRITICAL level message with optional formatting."""
        logger.opt(depth=1).log("CRITICAL", message, **fmt)
