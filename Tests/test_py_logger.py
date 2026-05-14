"""
Module: test_py_logger.py

This test suite uses memory buffers and mock objects to ensure deterministic,
I/O-free testing of logging behavior, context managers, and sink isolation.
"""
import unittest
import sys
import pathlib
import io
from unittest.mock import patch, MagicMock

# Dynamically resolve paths using pathlib
current_dir= pathlib.Path(__file__).resolve().parent
parent_dir= current_dir.parent

# Add the parent directory to sys.path to allow imports from there
sys.path.insert(0, str(parent_dir))

# Import the target module
import py_logger

class TestCustomLogger(unittest.TestCase):
    """
    Test suite for py_logger.CustomLogger.
    
    """

    # Constants for testing
    TEST_LOG_PATH = "mocked_path/dummy.log"
    DEFAULT_SESSION = "python-default-session"
    MATLAB_SESSION = "MATLAB_SESSION"
    MATLAB_TIME = "YYYY-MM-DD HH:mm:ss.SSS"

    # Mock sys.stdout to capture console output (self.info in __init__)
    # with a StringIO object for assertions
    @patch("sys.stdout", new_callable=io.StringIO)
    # Spy on logger.remove to ensure it's called during initialization.
    # We execute it instead to mock it because we want to remove the loguru logger standard sink
    @patch("py_logger.logger.remove", wraps=py_logger.logger.remove)
    def test_initialization(self, mock_logger_remove: MagicMock, mock_stdout: io.StringIO):
        """
        Test that CustomLogger initializes correctly, calls logger.remove(),
        and sets up the logger.
        """
        # Instantiate logger
        py_logger.CustomLogger(enable_file_logging=False)

        # Verify logger.remove() was called exactly once during __init__
        mock_logger_remove.assert_called_once()

        # Ensure the mock_stdout captured the setup message, keeping the console clean
        output = mock_stdout.getvalue()
        self.assertIn("File logging DISABLED", output)

    # No need to capture sys.stdout for this test since we are mocking logger.add for the file
    # No need to wrap logger.add since we want to mock it to prevent file creation
    @patch("py_logger.logger.add")
    def test_file_logging_parameters(self, mock_logger_add: MagicMock):
        """
        Test that the file logging parameters are set correctly when enable_file_logging is True.
        """
        py_logger.CustomLogger(
            log_file_path=self.TEST_LOG_PATH,
            enable_file_logging=True,
            level="DEBUG"
        )

        expected_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[session_id]} | " 
            "{file.name}:{function}:{line} - {message}"
        )

        # Iterate through all calls to logger.add to find the file sink configuration
        file_sink_configured = False
        for call_args in mock_logger_add.call_args_list:
            args, kwargs = call_args
            # Check if this specific call was intended for the file path
            if args and args[0] == self.TEST_LOG_PATH:
                file_sink_configured = True
                self.assertEqual(kwargs.get("rotation"), "50 KB")
                self.assertEqual(kwargs.get("compression"), "zip")
                self.assertEqual(kwargs.get("level"), "DEBUG")
                self.assertEqual(kwargs.get("format"), expected_format)

        self.assertTrue(
            file_sink_configured,
            "The file sink was not configured with the expected log file path."
        )

    # Since that we checked logger.remove() is called, we instantiate the logger now
    # No need to mock logger.add since we want to test the console sinks
    # We need only to mock sys.stdout to capture console output for assertions
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_console_routing(self, mock_stdout: io.StringIO):
        """
        Test that the standard Python logs and MATLAB logs are strictly routed
        to their respective formatting sinks without duplication.
        """
        # Instantiate logger
        log = py_logger.CustomLogger(enable_file_logging=False, level="INFO")

        # Trigger standard log
        standard_msg = "Standard Python Log"
        log.info(standard_msg)

        # Trigger MATLAB log via context manager
        matlab_msg = "MATLAB Engine Log"
        with log.context(session_id=self.MATLAB_SESSION, from_matlab=True, mt=self.MATLAB_TIME):
            log.info(matlab_msg)

        # Extract console output captured by mock_stdout
        output = mock_stdout.getvalue()

        # Positive Assertions
        self.assertIn(standard_msg, output)
        self.assertIn(self.DEFAULT_SESSION, output)
        self.assertIn(matlab_msg, output)
        self.assertIn(self.MATLAB_SESSION, output)
        self.assertIn(self.MATLAB_TIME, output)

        # Negative Assertions
        self.assertEqual(
            output.count(standard_msg), 1,
            "Standard message was duplicated, check filter logic."
        )
        self.assertEqual(
            output.count(matlab_msg), 1,
            "MATLAB message was duplicated, check filter logic."
        )

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_levels(self, mock_stdout: io.StringIO):
        """
        Test that all log levels are correctly captured and formatted in the console output.
        """
        log = py_logger.CustomLogger(enable_file_logging=False, level="TRACE")

        # Execute all wrapper methods with string formatting
        log.trace("Metric {val}", val="TRACE")
        log.debug("Metric {val}", val="DEBUG")
        log.info("Metric {val}", val="INFO")
        log.success("Metric {val}", val="SUCCESS")
        log.warn("Metric {val}", val="WARN")
        log.error("Metric {val}", val="ERROR")
        log.critical("Metric {val}", val="CRITICAL")

        output = mock_stdout.getvalue()

        # Verify every level was captured and correctly interpolated
        levels_to_check = [
            "TRACE", "DEBUG", "INFO", "SUCCESS", 
            "WARN", "ERROR", "CRITICAL"
        ]

        for level in levels_to_check:
            self.assertIn(
                level, output,
                f"Expected interpolated value {level} missing from output."
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)
