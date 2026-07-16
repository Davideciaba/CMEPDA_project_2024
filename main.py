"""
Project Main Entry Point.

Orchestrates the entire Python Machine Learning pipeline sequentially.
Given the previous execution of the MATLAB VBM Analysis, this script manages:
1. Linear SVM Training and Double CV
2. EfficientNet Deep Learning Training and Nested CV
3. Explainable AI (XAI) feature attribution maps generation.
"""
import argparse
import sys
import traceback
import pathlib

project_root = pathlib.Path(__file__).resolve().parent

# Assicurati che sys.path contenga la root per permettere gli import assoluti (es. 'Python.Models...')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Internal Orchestrators
# Note: Ensure these modules are properly located in your PYTHONPATH
from Python.SVM_Pipeline.run_svm_classification import run_svm_classification
from Python.EfficientNet_Pipeline.run_efficientnet_classification import run_efficientnet_classification
from Python.SVM_Pipeline.run_svm_xai import run_svm_xai
from Python.EfficientNet_Pipeline.run_efficientnet_xai import run_efficientnet_xai

from Python.utils.py_logger import CustomLogger

SESSION_ID = "ml-pipeline-session"

def validate_inputs(args):
    problems = []
    if not args.dir_ad.is_dir():
        problems.append(f"dirAD not found or not a directory: {args.dir_ad}")
    if not args.dir_ctrl.is_dir():
        problems.append(f"dirCTRL not found or not a directory: {args.dir_ctrl}")
    if not args.tiv_path.is_file():
        problems.append(f"TIV CSV not found: {args.tiv_path}")
    if problems:
        raise FileNotFoundError("; ".join(problems))

def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses command line arguments to configure the execution flow and I/O parameters.
    """
    parser = argparse.ArgumentParser(
        description="Launch Machine Learning and XAI Pipelines."
    )
    
    parser.add_argument(
        "--enable-file-logging",
        action="store_true",
        help="Globally enable writing .log files to disk for all executed modules."
    )
    
    # Execution Switches
    parser.add_argument(
        "--run-svm",
        action="store_true",
        help="Execute the Linear SVM Classification pipeline."
    )
    parser.add_argument(
        "--run-effnet",
        action="store_true",
        help="Execute the EfficientNet Deep Learning Classification pipeline."
    )
    parser.add_argument(
        "--run-xai",
        action="store_true",
        help="Execute the XAI extraction pipelines for the trained models."
    )
    
    return parser.parse_args(argv)

def main(argv=None) -> int:
    """
    Entry point for the MLOps pipeline.
    """
    args = parse_args(argv)

    # Initialize a master logger purely to track the orchestrator's state
    master_logger = CustomLogger(name="MasterOrchestrator")
    master_logger.add_console_handler(level="INFO", use_colors=True)
    
    if not (args.run_svm or args.run_effnet or args.run_xai):
        master_logger.warning("No execution flags provided. Pass --run-svm, --run-effnet, or --run-xai to begin.")
        master_logger.info("Use 'python main.py --help' for usage instructions.")
        return 0

    return_code = 0

    try:
        with master_logger.context(Session=SESSION_ID):
            master_logger.info(f"Global File Logging Enabled: {args.enable_file_logging}")

            # 1. Execute Linear SVM Pipeline
            if args.run_svm:
                master_logger.info("--- Handing execution over to SVM Orchestrator ---")
                run_svm_classification(enable_file_logging=args.enable_file_logging)
                master_logger.success("SVM Execution returned successfully.")

            # 2. Execute EfficientNet Pipeline
            if args.run_effnet:
                master_logger.info("--- Handing execution over to EfficientNet Orchestrator ---")
                run_efficientnet_classification(enable_file_logging=args.enable_file_logging)
                master_logger.success("EfficientNet Execution returned successfully.")

            # 3. Execute Explainable AI Maps
            if args.run_xai:
                master_logger.info("--- Handing execution over to XAI Extractors ---")
                if args.run_svm:
                    run_svm_xai(enable_file_logging=args.enable_file_logging)
                if args.run_effnet:
                    run_efficientnet_xai(enable_file_logging=args.enable_file_logging)
                master_logger.success("XAI Generation returned successfully.")

            master_logger.success("All requested pipelines completed without fatal errors.")

    except Exception as ex:
        with master_logger.context(Session=SESSION_ID):
            master_logger.critical(f"Unhandled Python-side Orchestration error: {str(ex)}")
            master_logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return_code = 1

    return return_code

if __name__ == "__main__":
    sys.exit(main())
