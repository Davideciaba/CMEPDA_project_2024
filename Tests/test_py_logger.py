"""
Module: test_py_logger.py

This test suite ensures deterministic, I/O-free testing of logging behavior,
context managers, garbage collection and sink isolation for the CustomLogger class.
"""
import unittest
import sys
import os
import pathlib
import io
import tempfile
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
    def test_initialization_and_context(self):
        """Verify that basic object states and context mapping behave as expected."""
        log = py_logger.CustomLogger(name="test_runner")
        
        self.assertEqual(log.name, "test_runner")
        
        # Add Context
        log.add_context("Session", "12345")
        self.assertIn("Session", log.extra_context)
        self.assertEqual(log.extra_context["Session"], "12345")
        
        # Clear Context
        log.clear_context()
        self.assertEqual(len(log.extra_context), 0)

    # Mock sys.stdout to capture console output
    # with a StringIO object for assertions
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_console_handler_and_levels(self, mock_stdout: io.StringIO):
        """Verify that global filters apply correctly to the console handler."""
        log = py_logger.CustomLogger()
        log.add_console_handler(level="INFO", use_colors=False)
        
        # Should be ignored (filtered out by the level)
        log.debug("Hidden Message")
        
        # Should be captured
        log.info("Visible Message")
        
        # Test runtime level shifting
        log.set_level("TRACE")
        log.trace("Now Visible Trace")

        output = mock_stdout.getvalue()
        
        # Asserts
        self.assertNotIn("Hidden Message", output)
        self.assertIn("Visible Message", output)
        self.assertIn("Now Visible Trace", output)
        self.assertIn("INFO", output)
        self.assertIn("TRACE", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_context_injection(self, mock_stdout: io.StringIO):
        """Verify that the context is appended to logs."""
        log = py_logger.CustomLogger()
        log.add_console_handler(level="DEBUG")
        
        log.add_context("Module", "VBM")
        log.add_context("Status", "Active")
        log.debug("Processing slice")
        
        output = mock_stdout.getvalue()
        
        self.assertIn("Processing slice", output)
        self.assertIn("Module = VBM | Status = Active", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_context_manager(self, mock_stdout: io.StringIO):
        """
        Verify that the context manager temporarily injects and properly 
        cleans up metadata in a with block.
        """
        log = py_logger.CustomLogger()
        log.add_console_handler(level="DEBUG")
        
        log.add_context("Global", "Active")
        
        # Enter context block
        with log.context(Temporary="Yes", Task=1):
            log.info("Inside block")
            
        log.info("Outside block")
        
        output = mock_stdout.getvalue()
        lines = [line for line in output.splitlines() if line.strip()]
        
        # Check inside block log
        inside_log = lines[0]
        self.assertIn("Global = Active | Temporary = Yes | Task = 1", inside_log)
        self.assertIn("Inside block", inside_log)
        
        # Check outside block log
        outside_log = lines[1]
        self.assertIn("Global = Active", outside_log)
        self.assertNotIn("Temporary", outside_log)
        self.assertNotIn("Task", outside_log)
        self.assertIn("Outside block", outside_log)

    # Mock logger.add to prevent file creation and capture parameters
    @patch("py_logger.logger.add")
    def test_rotation_parameter_translation(self, mock_logger_add):
        """
        Verify that numeric rotation parameters are mathematically translated to bytes, 
        while semantic strings are preserved.
        """
        log = py_logger.CustomLogger()
        
        # Test numeric (int/float) bytes translation
        log.add_file_handler("numeric.log", rotation=1024)
        args, kwargs = mock_logger_add.call_args
        self.assertEqual(kwargs.get("rotation"), "1024 B")
        
        # Test string preservation for advanced loguru features
        log.add_file_handler("string.log", rotation="10 MB")
        args, kwargs = mock_logger_add.call_args
        self.assertEqual(kwargs.get("rotation"), "10 MB")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_method_and_formatting_error(self, mock_stdout: io.StringIO):
        """
        Verify the log() method and its exception handling 
        for intentionally malformed format strings.
        """
        log = py_logger.CustomLogger()
        log.add_console_handler(level="TRACE")
        
        # Correct formatting
        log.log("DEBUG", "Value: {}", 42)
        
        # Broken formatting (missing placeholder values)
        log.log("INFO", "Broken string {missing}", 42)
        
        output = mock_stdout.getvalue()
        
        # Asserts
        self.assertIn("Value: 42", output)
        self.assertIn("Failed to format log message! Reason", output)
        self.assertIn("Broken string {{missing}}", output)

    def test_garbage_collection(self):
        """
        Verifies that 0-byte log files are automatically 
        deleted from disk during the shutdown sequence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = pathlib.Path(temp_dir) / "empty_garbage.log"
            
            # Instantiate and immediately destroy
            def create_abandoned_file():
                log = py_logger.CustomLogger()
                log.add_file_handler(str(log_file), level="INFO")
                # Intentionally writing NO logs. File created but remains at 0 bytes.
                log.shutdown() # Force immediate teardown
                
            create_abandoned_file()
            
            # Assert file was collected and destroyed
            self.assertFalse(
                log_file.exists(), 
                "The 0-byte abandoned log file was not destroyed."
            )
    
    def test_file_logging_integration(self):
        """
        Verifies successful writes to disk and that the log file contains the expected content.
        """
        # Create an isolated temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = pathlib.Path(temp_dir) / "integration_test.log"

            log = py_logger.CustomLogger()
            log.add_file_handler(str(log_file), level="INFO")
            log.add_context("Deployment", "TestEnv")
            
            test_message = "REAL_DISK_WRITE_TEST_SUCCESSFUL"
            log.info(test_message)

            # Shutdown the logger to flush and close file handles
            log.shutdown()

            # Assert file exists and it is not empty
            self.assertTrue(log_file.exists())
            self.assertTrue(os.path.getsize(str(log_file)) > 0)

            # Read the file content
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(test_message, content)
            self.assertIn("Deployment = TestEnv", content)   


if __name__ == '__main__':
    unittest.main(verbosity=2)
