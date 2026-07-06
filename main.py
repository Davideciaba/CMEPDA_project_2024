# main.py - Launches MATLAB Preliminaries with live log tailing
import argparse
import sys
import time
import traceback

from pathlib import Path

from Python.utils.py_logger import CustomLogger
from loguru import logger as _loguru_logger

SESSION_ID = "matlab-preliminaries-session"



def _resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def parse_args(project_root: Path, argv=None):
    parser = argparse.ArgumentParser(
        description="Launch MATLAB Preliminaries with configurable inputs."
    )
    default_tiv = project_root / "AD_CTRL" / "covariateADCTRLsexAgeTIV.csv"
    parser.add_argument(
        "--dir-ad",
        type=Path,
        default=project_root / "AD_CTRL" / "AD_s3",
        help="Directory containing AD NIfTI files (default: %(default)s)",
    )
    parser.add_argument(
        "--dir-ctrl",
        type=Path,
        default=project_root / "AD_CTRL" / "CTRL_s3",
        help="Directory containing CTRL NIfTI files (default: %(default)s)",
    )
    parser.add_argument(
        "--tiv-path",
        type=Path,
        default=default_tiv,
        help="CSV file with TIV values (default: %(default)s)",
    )
    parser.add_argument(
        "--session-id",
        default=SESSION_ID,
        help=f"Session identifier used in Python logs (default: {SESSION_ID})",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Minimum log level for the Python logger (default: DEBUG)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=project_root / "preliminaries.log",
        help="File path for optional Loguru file sink (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-file-logging",
        action="store_true",
        help="Enable the Loguru file sink pointing to --log-file",
    )
    parser.add_argument(
        "--tail-poll",
        type=float,
        default=0.05,
        help="Polling interval (seconds) for the MATLAB live log tailer.",
    )
    args = parser.parse_args(argv)
    args.dir_ad = _resolve_path(args.dir_ad, project_root)
    args.dir_ctrl = _resolve_path(args.dir_ctrl, project_root)
    args.tiv_path = _resolve_path(args.tiv_path, project_root)
    args.log_file = _resolve_path(args.log_file, project_root)
    return args


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



def main(argv=None):
    project_root = Path(__file__).resolve().parent
    args = parse_args(project_root, argv)

    py_logger = CustomLogger(
        log_file_path=str(args.log_file),
        enable_file_logging=args.enable_file_logging,
        level=args.log_level.upper(),
    )

    try:
        validate_inputs(args)
    except FileNotFoundError as exc:
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.error(str(exc))
        return 1

    live_path = None
    rc = 1

    try:

        with py_logger.context(session_id=args.session_id):
            _loguru_logger.debug("Live log file: {}", live_path)
            _loguru_logger.debug("dirAD:   {}", args.dir_ad)
            _loguru_logger.debug("dirCTRL: {}", args.dir_ctrl)
            _loguru_logger.debug("TIV CSV: {}", args.tiv_path)

        

        with py_logger.context(session_id=args.session_id):
            _loguru_logger.success("Preliminaries completed")

    except Exception as ex:
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.error("Python-side error: {}", str(ex))
            _loguru_logger.debug("Traceback:\n{}", traceback.format_exc())
        

    return rc


if __name__ == "__main__":
    sys.exit(main())
