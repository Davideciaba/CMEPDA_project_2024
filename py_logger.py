"""
Module: py_logger.py

This module defines the CustomLogger class, which is a customized wrapper around the loguru logger.
It is architected to mirror the exact API and functionality of the MATLAB Logger.m class.

KEY FEATURES:
- Log Levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
- Multiple Handlers: Dynamically configure console and file sinks.
- Colored Output: Native support via Loguru.
- File Rotation: Native rotation and compression for file sinks.
- Contextual Data: Dynamically attach extra key-value context to all logs.
        It can wrap logs from MATLAB and Python.
"""
import sys
import os
from typing import Any, Optional, Dict, Union
from loguru import logger
import contextlib

class CustomLogger:
    def __init__(self, name: str = "root" ):
        """
        Initializes a new CustomLogger instance without any active handlers.
        """
        self.name = name
        self.extra_context: Dict[str, Any] = {}

        # Internal state tracking
        self._file_sinks = []
        self._sinks_config = {}

        # Maps log level names to numerical values for manual filtering
        self.levels = {
            "TRACE": 5, "DEBUG": 10, "INFO": 20, "SUCCESS": 25,
            "WARNING": 30, "ERROR": 40, "CRITICAL": 50
        }
        self.current_level = self.levels["DEBUG"]

        # Reset global loguru sinks to prevent duplication
        logger.remove()

        # Patch the logger globally for this instance to inject the context map
        # It is like formatContext in MATLAB Logger.m
        self._logger = logger.patch(self._patch_record)

    def set_level(self, level_name: str):
        """
        Adjusts the global minimum severity threshold for all attached handlers.
        """
        level_name = level_name.upper()
        if level_name in self.levels:
            self.current_level = self.levels[level_name]
        else:
            self.warning(f"Invalid log level: {level_name}. Level not changed.")

    def add_console_handler(self, level: str = "DEBUG", use_colors: bool = False):
        """
        Attaches the standard console output as a log sink.
        """
        self.set_level(level)
        
        sink_id = logger.add(
            sys.stdout,
            level="TRACE",  # "TRACE" prevents loguru from blocking low-level logs
            colorize=use_colors,
            format=self._get_format if use_colors else self._get_plain_format,
            filter=self._level_filter
        )
        self._sinks_config['console'] = sink_id

    def add_file_handler(self, filename: str, level: str = "DEBUG", rotation: Optional[Union[str, int, float]] = "10 KB"):
        """
        Attaches a file on disk as a log sink with optional rotation.
        """
        self.set_level(level)
        
        # Standardize rotation format if bytes are provided numerically
        rot_val = f"{rotation} B" if isinstance(rotation, (int, float)) else rotation

        sink_id = logger.add(
            filename,
            level="TRACE",  # "TRACE" prevents loguru from blocking low-level logs
            colorize=False,
            format=self._get_plain_format,
            rotation=rot_val,
            filter=self._level_filter
        )
        self._file_sinks.append(filename)
        self._sinks_config[f'file_{filename}'] = sink_id

    def add_context(self, key: str, value: Any):
        """Adds or updates a key-value pair in the context map."""
        self.extra_context[key] = value

    def clear_context(self):
        """Resets the entire context map."""
        self.extra_context.clear()

    @contextlib.contextmanager
    def context(self, **kwargs):
        """
        Context manager to temporarily attach extra fields to logs within a specific block.
        It can be used to wrap MATLAB engine executions.
        
        Usage:
            with log.context(Source="MATLAB", Script="preprocessing.m"):
        """
        # Save the current state of the context
        old_context = self.extra_context.copy()
        
        # Update with the new temporary context
        self.extra_context.update(kwargs)
        try:
            yield self
        finally:
            # Restore the original context regardless of exceptions
            self.extra_context = old_context

    # ---- Helper Methods ----

    def _level_filter(self, record: Dict[str, Any]):
        """Filters log entries against the configured global level."""
        return record["level"].no >= self.current_level

    def _patch_record(self, record: Dict[str, Any]):
        """Builds the context string for the current log record."""
        if self.extra_context:
            parts = [f"{k} = {v}" for k, v in self.extra_context.items()]
            record["extra"]["ctx_str"] = " | ".join(parts)
        else:
            record["extra"]["ctx_str"] = ""

    def _get_format(self, record: Dict[str, Any]):
        """Formatter with colored output."""
        fmt = "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | <level>{level: <8}</level> | <dim>{file.name}:{function}:{line}</dim> - <level>{message}</level>"
        if record["extra"].get("ctx_str"):
            fmt += " | <cyan>{extra[ctx_str]}</cyan>"
        fmt += "\n"
        return fmt 
    
    def _get_plain_format(self, record: Dict[str, Any]):
        """Formatter with plain output."""
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file.name}:{function}:{line} - {message}"
        if record["extra"].get("ctx_str"):
            fmt += " | {extra[ctx_str]}"
        fmt += "\n"
        return fmt
    
    # ---------- Wrappers for each log level ----------

    def log(self, level_name: str, message: str, *args, **kwargs) -> None:
        """
        Processes the message and sends it to all configured sinks.
        Fallback method if dynamic level injection is required.
        """
        level_name = level_name.upper()
        if level_name not in self.levels:
            raise ValueError(f"Unknown log level: {level_name}")
        
        # Simulate MATLAB's sprintf behavior if extra formatting arguments are passed
        if args or kwargs:
            try:
                message = message.format(*args, **kwargs)
            except Exception as e:
                self._logger.opt(depth=1).warning(f"Failed to format log message! Reason: {e}")
                # Escape braces to prevent loguru from crashing
                safe_message = message.replace("{", "{{").replace("}", "}}")
                self._logger.opt(depth=1).log(level_name, safe_message)
                return
            
        safe_message = message.replace("{", "{{").replace("}", "}}")
        self._logger.opt(depth=1).log(level_name, safe_message)

    def trace(self, msg: str, *args, **kwargs):
        """Log a TRACE level message with optional formatting."""
        self._logger.opt(depth=1).trace(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log a DEBUG level message with optional formatting."""
        self._logger.opt(depth=1).debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log a INFO level message with optional formatting."""
        self._logger.opt(depth=1).info(msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs):
        """Log a SUCCESS level message with optional formatting."""
        self._logger.opt(depth=1).success(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a WARNING level message with optional formatting."""
        self._logger.opt(depth=1).warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log a ERROR level message with optional formatting."""
        self._logger.opt(depth=1).error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log a CRITICAL level message with optional formatting."""
        self._logger.opt(depth=1).critical(msg, *args, **kwargs)

    # --- Destructor and Garbage Collection ---
    def shutdown(self) -> None:
        """
        Iterates through all sinks to ensure file handles are closed properly
        and performs garbage collection on 0-byte files.
        """
        # Detach all loguru handlers
        logger.remove()
        
        # Garbage Collection
        for filename in self._file_sinks:
            try:
                if os.path.exists(filename) and os.path.getsize(filename) == 0:
                    os.remove(filename)
            except Exception as e:
                print(f"Warning: Failed to delete empty log file: {filename}. OS Reason: {e}")
        
        self._file_sinks.clear()
        self._sinks_config.clear()

    def __del__(self) -> None:
        self.shutdown()