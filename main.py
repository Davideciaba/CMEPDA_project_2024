"""
Project Main Entry Point.

Orchestrates the entire Python Machine Learning pipeline sequentially.
Given the previous execution of the MATLAB VBM Analysis, this script manages:
0. Setup (Phase 0)
1. Linear SVM Training and Double CV
2. EfficientNet Deep Learning Training and Nested CV
3. Explainable AI (XAI) feature attribution maps generation.
"""
import argparse
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Python.Common_Setup.cv_setup import cv_setup
from Python.SVM_Pipeline.run_svm_classification import run_svm_classification
from Python.SVM_Pipeline.run_svm_xai import run_svm_xai
from Python.utils.py_logger import CustomLogger
from Python.XAI.run_xai_comparison import run_xai_comparison



def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses command line arguments to configure the execution flow and I/O parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            """
            This script performs Alzheimer's Disease vs Healthy Control binary classification
            using Support Vector Machine and tests the interpretative capabilites
            of Haufe transformation and Gaonkar statistics against VBM analysis and SVM's raw weights
            making a qualitative and quantitative analysis.
            """
        )
    )
    
    parser.add_argument(
        "-log",
        "--enable-file-logging",
        action="store_true",
        help="Enable writing .log files to disk for all executed modules (Default: False)."
    )
    
    parser.add_argument(
        "-out",
        "--output-dir", 
        type=Path,
        default=project_root / "Python_Results", 
        help="Target root directory for all outputs (Default: Python_Results)."
    )

    parser.add_argument(
        "-in",
        "--input-dir", 
        type=Path, 
        default=project_root / "AD_CTRL", 
        help="Source directory containing NIfTI and CSV files (Default: AD_CTRL in project root)."
    )

    parser.add_argument(
        "-csv",
        "--csv-name", 
        type=str, 
        default="covariateADCTRLsexAgeTIV.csv", 
        help="Name of the clinical covariate CSV file (Default: covariateADCTRLsexAgeTIV.csv in AD_CTRL directory)."
    )

    parser.add_argument(
        "-set",
        "--run-setup", 
        action="store_true", 
        help="Execute common setup (generates CV folds & cohort registry)."
        )

    parser.add_argument(
        "-svm",
        "--run-svm",
        action="store_true",
        help="Execute the Linear SVM classification pipeline."
    )
    
    parser.add_argument(
        "-xai",
        "--run-xai",
        action="store_true",
        help="Execute the XAI pipeline for the trained Linear SVM."
    )

    parser.add_argument(
        "-up",
        "--use-pretrained", 
        action="store_true", 
        help="If passed, skips training and loads pre-trained models. Default: False (Forces full retraining)."
    )

    parser.add_argument(
        "-bg",
        "--bypass-grid", 
        action="store_true", 
        help="If passed, skips GridSearch and uses historical Optimal_C from CSV. Default: False (Executes full GridSearch)."
    )

    parser.add_argument(
        "-cv",
        "--c-values", 
        nargs='+', 
        type=float, 
        default=[1e-4], 
        help="List of values for the SVM 'C' hyperparameter. Example: -cv 0.0001 0.001 0.01 (Default: [1e-4])."
    )

    parser.add_argument(
        "-xc",
        "--run-compare",
        action="store_true",
        help="Execute the Comparative XAI benchmarking against MATLAB VBM Ground Truth."
    )

    parser.add_argument(
        "-of",
        "--outer-folds", 
        type=int, 
        default=5, 
        help="Number of outer cross-validation folds. Default: 5"
    )

    parser.add_argument(
        "-inf",
        "--inner-folds", 
        type=int, 
        default=5, 
        help="Number of inner cross-validation folds. Default: 5"
    )
    
    return parser.parse_args(argv)

def main(argv=None) -> int:
    """
    Entry point for the MLOps pipeline.
    """
    args = parse_args(argv)
    logger = CustomLogger(name="MainLogger")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    
    if not (args.run_setup or args.run_svm or args.run_xai or args.run_compare):
        logger.warning("No execution flags provided. Pass --run-setup, --run-svm or --run-xai to begin.")
        logger.info("Use 'python main.py --help' for usage instructions.")
        return 0

    active_input_dir = args.input_dir.resolve()
    active_output_dir = args.output_dir.resolve()
    return_code = 0

    try:
        logger.info(f"Global output directory mapped to: {active_output_dir}")
        logger.info(f"Global input directory mapped to: {active_input_dir}")
        logger.info(f"Target covariate CSV: {args.csv_name}")

        # Setup Phase
        if args.run_setup:
            logger.info("--- Handing execution over to Setup Orchestrator ---")
            cv_setup(
                enable_file_logging=args.enable_file_logging, 
                output_dir=active_output_dir,
                input_dir=active_input_dir,
                csv_name=args.csv_name,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds
            )
            logger.success("Setup Phase completed successfully.")

        # Linear SVM Pipeline
        if args.run_svm:
            logger.info("--- Handing execution over to SVM orchestrator ---")
            run_svm_classification(
                enable_file_logging=args.enable_file_logging, 
                output_dir=active_output_dir,
                input_dir=active_input_dir,
                csv_name=args.csv_name,
                bypass_grid=args.bypass_grid,
                use_pretrained=args.use_pretrained,
                c_values=args.c_values,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds
            )
            logger.success("SVM Execution completed successfully.")

        # XAI Pipeline
        if args.run_xai:
            logger.info("--- Handing execution over to XAI extractors ---")
            run_svm_xai(
                enable_file_logging=args.enable_file_logging, 
                output_dir=active_output_dir
            )
            logger.success("XAI Generation completed successfully.")

        if args.run_compare:
            logger.info("--- Handing execution over to Comparative Orchestrator ---")
            run_xai_comparison(
                enable_file_logging=args.enable_file_logging, 
                output_dir=active_output_dir,
                input_dir=active_input_dir
            )
            logger.success("Comparative Analysis completed successfully.")

    except Exception as ex:
        logger.critical(f"Unhandled orchestration error: {str(ex)}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return_code = 1

    return return_code

if __name__ == "__main__":
    sys.exit(main())
