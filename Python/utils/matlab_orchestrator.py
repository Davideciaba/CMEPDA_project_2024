"""
Module: matlab_orchestrator.py

Architectural Orchestrator between Python and MATLAB.
Object-Oriented implementation to manage the MATLAB Engine lifecycle,
execute multiple dynamic tasks, and tail log files in real-time.

Requirements:
    pip install loguru
    pip install matlabengine
"""
import io
import time
import pathlib
import re
from collections import namedtuple
from datetime import datetime
from dataclasses import dataclass
from typing import List, Union, Optional
import matlab.engine
from Python.utils.py_logger import CustomLogger

# OPTIMIZATION: Pre-compile Regex pattern globally for massive performance boost
# Expected format: "YYYY-MM-DD HH:mm:ss.SSS | LEVEL    | caller - message"
MATLAB_LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s*\|\s*([A-Z]+)\s*\|\s*(.*?)\s*-\s*(.*)"
)

@dataclass
class MatlabTask:
    """
    Encapsulates a MATLAB execution task, binding the script to run 
    with its physically generated log file for real-time tailing.
    """
    script_path: Union[str, pathlib.Path]
    log_path: Union[str, pathlib.Path]
    
    def __post_init__(self):
        self.script_path = pathlib.Path(self.script_path).resolve()
        self.log_path = pathlib.Path(self.log_path).resolve()
        # The function name matching the MATLAB file name (without .m)
        self.name = self.script_path.stem


class MatlabOrchestrator:
    """
    Manager class responsible for orchestrating MATLAB execution.
    It manages engine connectivity, dynamically injects required paths,
    executes tasks asynchronously, and handles cross-process log tailing.
    """

    def __init__(
        self, 
        logger: CustomLogger, 
        tasks: List[MatlabTask], 
        include_paths: Optional[List[Union[str, pathlib.Path]]] = None
    ):
        """
        Args:
            logger: The customized CustomLogger instance.
            tasks: A list of MatlabTask objects defining scripts and logs.
            include_paths: Optional directories to add to MATLAB's system path (e.g. SPM dir).
        """
        self.log = logger
        self.tasks = tasks
        
        # Resolve all include paths for security and consistency
        if include_paths:
            self.include_paths = [pathlib.Path(p).resolve() for p in include_paths]
        else:
            self.include_paths = []
            
        self.eng = None

    def __enter__(self):
        """Context manager entry point. Automatically boots the engine."""
        self.start_engine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point. Ensures graceful teardown of the engine."""
        # Detect if the context is exiting due to an unhandled exception
        is_crash = exc_type is not None
        self.stop_engine(is_crash=is_crash)

    def start_engine(self) -> None:
        """Initializes the MATLAB Engine and injects global dependencies."""
        if self.eng is not None:
            return
            
        self.log.info("Booting up MATLAB Engine (this may take a few seconds)...")
        try:
            self.eng = matlab.engine.start_matlab()
            self.log.success("MATLAB Engine connected successfully.")
            
            # Inject global include paths (like SPM)
            for path in self.include_paths:
                if path.exists():
                    self.eng.addpath(str(path), nargout=0)
                    self.log.trace(f"Added to MATLAB Path: {path}")
                else:
                    self.log.warning(f"Include path not found: {path}")
                    
        except matlab.engine.EngineError as e:
            self.log.critical(f"Failed to start MATLAB Engine. Is it installed? Details: {e}")
            raise
        except Exception as e:
            self.log.critical(f"Unexpected system failure during boot: {e}")
            raise

    def stop_engine(self, is_crash: bool = False) -> None:
        """Gracefully terminates the MATLAB Engine. Context-aware of crashes."""
        if self.eng is not None:
            if is_crash:
                self.log.warning("Emergency shutdown of MATLAB Engine triggered...")
            else:
                self.log.info("Shutting down MATLAB Engine...")
                
            self.eng.quit()
            self.eng = None
            
            if is_crash:
                self.log.error("MATLAB Engine terminated abnormally due to an exception.")
            else:
                self.log.success("MATLAB Engine terminated.")


    def run_all(self) -> None:
        """Executes all queued MATLAB tasks sequentially."""
        if not self.eng:
            self.log.error("Cannot run tasks. MATLAB Engine is not running.")
            return

        for task in self.tasks:
            self._execute_task_real_time(task)

    def _parse_and_route_matlab_log(self, log_line: str) -> None:
        """Private parser mapping MATLAB output to Python loguru architecture."""
        clean_line = log_line.replace("<strong>", "").replace("</strong>", "")
        match = MATLAB_LOG_PATTERN.match(clean_line)
        
        if match:
            time_str, raw_level, caller_str, rest_of_message = match.groups()
            
            parts = caller_str.split(":")
            if len(parts) >= 3:
                file_str = f"{parts[0]}.m"
                func_str = parts[1]
                try:
                    line_int = int(parts[2])
                except ValueError:
                    line_int = 0
            else:
                file_str, func_str, line_int = "MATLAB.m", "script", 0
            
            MockFile = namedtuple("MockFile", ["name", "path"])
            matlab_file_obj = MockFile(name=file_str, path=file_str)
            
            try:
                matlab_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                
                def patch_record(r):
                    r["time"] = r["time"].replace(
                        year=matlab_time.year, month=matlab_time.month, day=matlab_time.day,
                        hour=matlab_time.hour, minute=matlab_time.minute, second=matlab_time.second,
                        microsecond=matlab_time.microsecond
                    )
                    r["file"] = matlab_file_obj
                    r["function"] = func_str
                    r["line"] = line_int
                    r["name"] = file_str
                
                patched_logger = self.log._logger.patch(patch_record)
                
                try:
                    patched_logger.log(raw_level, rest_of_message)
                except ValueError:
                    patched_logger.info(rest_of_message) 
                return
                
            except ValueError:
                pass 

        # --- FALLBACK ---
        line_upper = clean_line.upper()
        for target_level in ["CRITICAL", "ERROR", "EXCEPTION", "WARNING", "SUCCESS", "TRACE", "DEBUG"]:
            if target_level in line_upper:
                mapped_level = "ERROR" if target_level == "EXCEPTION" else target_level
                self.log.log(mapped_level, clean_line)
                return
                
        self.log.info(clean_line)

    def _execute_task_real_time(self, task: MatlabTask) -> None:
        """
        Executes a single MATLAB task asynchronously.
        Tails the associated physical log file on disk using Smart Polling.
        """
        last_pos = task.log_path.stat().st_size if task.log_path.exists() else 0

        with self.log.context(Engine="MATLAB", Module=task.name):
            self.log.info(f"Initiating Task [{task.name}] in background thread...")
            self.log.info(f"Tailing physical log file at: {task.log_path}")
            
            try:
                # Add task-specific directory to MATLAB path
                self.eng.addpath(str(task.script_path.parent), nargout=0)
                
                # Trap native MATLAB output in dummy buffers
                dummy_out = io.StringIO()
                dummy_err = io.StringIO()
                
                # OPTIMIZATION: Dynamic attribute call to execute ANY script
                matlab_func = getattr(self.eng, task.name)
                future = matlab_func(nargout=0, background=True, stdout=dummy_out, stderr=dummy_err)
                
                # --- REAL-TIME SMART DISK POLLING LOOP ---
                while not future.done():
                    if task.log_path.exists():
                        try:
                            current_size = task.log_path.stat().st_size
                            
                            if current_size > last_pos:
                                with open(task.log_path, 'r', encoding='utf-8') as f:
                                    f.seek(last_pos)
                                    new_data = f.read()
                                    last_pos = f.tell()
                                
                                for line in new_data.splitlines():
                                    if line.strip():
                                        self._parse_and_route_matlab_log(line.strip())
                                        
                            elif current_size < last_pos:
                                last_pos = 0 
                                
                        except PermissionError:
                            pass 
                            
                    time.sleep(0.2)
                
                # --- FINAL FLUSH ---
                if task.log_path.exists():
                    try:
                        with open(task.log_path, 'r', encoding='utf-8') as f:
                            f.seek(0, 2)
                            if f.tell() < last_pos:
                                last_pos = 0
                            f.seek(last_pos)
                            remaining_out = f.read()
                            
                        for line in remaining_out.splitlines():
                            if line.strip():
                                self._parse_and_route_matlab_log(line.strip())
                    except Exception:
                        pass

                try:
                    future.result()
                    self.log.success(f"Task [{task.name}] executed completely.")
                except Exception as execution_err:
                    self.log.critical(f"MATLAB Execution crashed fatally in [{task.name}]:\n{execution_err}")
                    raise

            except Exception as e:
                self.log.error(f"Python Orchestration error during execution setup for [{task.name}]: {e}")
                raise
            finally:
                dummy_out.close()
                dummy_err.close()