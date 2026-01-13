
import sys
from loguru import logger

class CustomLogger:
    """Custom logger based on Loguru.
    - Convenience methods (info/debug/warning/...) accept **kwargs for formatting.
    - Context manager to attach extra fields (e.g., session_id=...).
    - Separate sinks for Python logs and MATLAB logs (filtered by 'from_matlab' extra field).
    """
    def __init__(
        self,
        log_file_path: str = "log_file.log",
        enable_file_logging: bool = False,
        level: str = "DEBUG"
    ):
        """
        Initialize the logger.

        Args:
            log_file_path (str): The path to the log file.
            enable_file_logging (bool): Whether to enable file logging.
                                        Default is False (console only).
            level (str): The minimum logging level. Default is "DEBUG".
        """
        # Reset sinks
        logger.remove()

        # Default extra fields (keep keys present to avoid KeyError in formats)
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
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[session_id]} | {file.name}:{function}:{line} - {message}"
            )
            self.info("File logging ENABLED. Writing to {path}", path=log_file_path)
        else:
            self.info("File logging DISABLED. Console only.")

    def context(self, **kwargs):
        """Context manager to attach extra fields (e.g., session_id=...)."""
        return logger.contextualize(**kwargs)
   
    
    # ---------- Wrappers for each log level ----------
    def trace(self, message, **fmt): logger.opt(depth=1).log("TRACE", message, **fmt)
    def debug(self, message, **fmt): logger.opt(depth=1).log("DEBUG", message, **fmt)
    def info(self, message, **fmt): logger.opt(depth=1).log("INFO", message, **fmt)
    def success(self, message, **fmt): logger.opt(depth=1).log("SUCCESS", message, **fmt)
    def warn(self, message, **fmt): logger.opt(depth=1).log("WARNING", message, **fmt)
    def error(self, message, **fmt): logger.opt(depth=1).log("ERROR",  message, **fmt)
    def critical(self, message, **fmt): logger.opt(depth=1).log("CRITICAL", message, **fmt)