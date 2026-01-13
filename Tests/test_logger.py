import sys
import os

# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path to allow imports from there
sys.path.append(parent_dir)

from logging_utils import CustomLogger

def run_console_test():
    print("\n=== TEST 1: Console Logging, Levels & Formatting ===")
    
    # Instantiate the logger (File logging disabled, Level TRACE to see everything)
    # Note: We use () to instantiate the class properly.
    log = CustomLogger(enable_file_logging=False, level="TRACE")
    
    # Test all standard levels
    log.trace("This is a TRACE message (very verbose).")
    log.debug("This is a DEBUG message.")
    log.info("This is a standard INFO message.")
    log.success("Operation completed with SUCCESS.")
    log.warn("Warning: This is a WARNING message.")
    log.error("Error detected: This is an ERROR message.")
    log.critical("Critical failure: This is a CRITICAL message.")

    # Test string formatting using kwargs
    print("\n--- Testing String Formatting ---")
    user_name = "TestUser"
    item_count = 5
    # The logger handles {placeholder} replacement automatically
    log.info("Hello {name}, you have {count} items pending.", name=user_name, count=item_count)

def run_context_test():
    print("\n=== TEST 2: Context Manager (Session ID) ===")
    log = CustomLogger(enable_file_logging=False)

    log.info("Message outside context (default session_id).")

    # Use the context manager to temporarily change the session_id
    with log.context(session_id="TEST-SESSION-XYZ"):
        log.info("Message INSIDE context (modified session_id).")
        log.warn("Another message within the same session context.")

    log.info("Message back outside context (reverted to default session_id).")

def run_file_logging_test():
    print("\n=== TEST 3: File Logging ===")
    log_filename = "test_output.log"
    full_log_path = os.path.join(current_dir, log_filename)

    # Cleanup: remove the file if it already exists
    if os.path.exists(full_log_path):
        try:
            os.remove(full_log_path)
            print(f"Old log file removed.")
        except PermissionError:
            print(f"Warning: Could not remove log file. It might be in use.")

    # Instantiate logger with file logging ENABLED
    log = CustomLogger(log_file_path=full_log_path, enable_file_logging=True)
    
    log.info("This message should appear in console AND in the file.")
    log.error("This error should also be written to the file.")

    # Verify if file was created and contains text
    if os.path.exists(full_log_path):
        print(f"--> Success: The file was created at {full_log_path}")
        print("--> File content preview:")
        print("-" * 30)
        with open(full_log_path, "r") as f:
            print(f.read().strip())
        print("-" * 30)
    else:
        print(f"--> Error: The file was NOT created at {full_log_path}")

if __name__ == "__main__":
    try:
        run_console_test()
        run_context_test()
        run_file_logging_test()
        print("\n=== ALL TESTS COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        print(f"\n!!! Exception occurred during testing: {e}")