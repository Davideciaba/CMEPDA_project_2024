# main.py - Launches MATLAB Preliminaries with live log tailing
import argparse
import io
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

import matlab.engine
from matlab.engine import MatlabExecutionError
from logging_utils import CustomLogger
from loguru import logger as _loguru_logger

SESSION_ID = "matlab-preliminaries-session"

LINE_RE = re.compile(
    r'^\s*'
    r'(?P<ts_str>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s*\|\s*'
    r'(?P<lvl>TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s*\|\s*'
    r'(?P<name>[^:]+):(?P<func>[^:]+):(?P<line>\d+)\s*-\s*(?P<msg>.*)$'
)
WARN_RE = re.compile(r'^\s*Warning:\s*(?P<msg>.*)$', re.IGNORECASE)
ERR_RE = re.compile(r'^\s*Error using .*?:?\s*(?P<msg>.*)$', re.IGNORECASE)


class FileTailer:
    """Tail a live log written by MATLAB.Logger and replay entries via Loguru."""

    def __init__(self, path: Path, py_logger: CustomLogger,
                 poll=0.05, session_id=SESSION_ID):
        self.path = Path(path)
        self.poll = float(poll)
        self._stop = threading.Event()
        self._thread = None
        self._buf = ""
        self.py_logger = py_logger
        self.session_id = session_id

    def start(self):
        if self._thread and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.join(timeout=3)

    @staticmethod
    def _parse_ts(ts: str):
        ts = (ts or "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        return None

    def _log_message(self, level: str, msg: str, *,
                     ts_str=None, name="MATLAB", func="console", line=0):
        ts_dt = self._parse_ts(ts_str)
        if ts_dt is None:
            ts_dt = datetime.now()
        mt_value = ts_str or ts_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        def _patch(record):
            record["extra"]["from_matlab"] = True
            record["extra"]["mt"] = mt_value
            record["file"] = types.SimpleNamespace(name=name)
            record["function"] = func
            try:
                record["line"] = int(line)
            except Exception:
                record["line"] = 0
            record["time"] = ts_dt

        with self.py_logger.context(session_id=self.session_id):
            _loguru_logger.patch(_patch).opt(depth=0).log(level, msg)

    def _handle_line(self, raw: str):
        s = raw.rstrip("\r\n")
        if not s or s.startswith("__PY_HEARTBEAT__"):
            return
        if (match := LINE_RE.match(s)):
            self._log_message(
                match.group("lvl").upper(),
                match.group("msg"),
                ts_str=match.group("ts_str"),
                name=match.group("name"),
                func=match.group("func"),
                line=match.group("line"),
            )
            return
        if (warn := WARN_RE.match(s)):
            self._log_message("WARNING", warn.group("msg"))
            return
        if (err := ERR_RE.match(s)):
            self._log_message("ERROR", err.group("msg"))
            return
        self._log_message("INFO", s)

    def _run(self):
        with self.py_logger.context(session_id=self.session_id):
            _loguru_logger.debug("Tailer started on {}", self.path)

        while not self.path.exists() and not self._stop.is_set():
            time.sleep(self.poll)

        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                while not self._stop.is_set():
                    where = f.tell()
                    chunk = f.read()
                    if not chunk:
                        time.sleep(self.poll)
                        f.seek(where)
                        continue
                    self._buf += chunk
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        try:
                            self._handle_line(line)
                        except Exception as ex:
                            with self.py_logger.context(session_id=self.session_id):
                                _loguru_logger.error("Tailer line error: {}", str(ex))
                if self._buf:
                    try:
                        self._handle_line(self._buf)
                    except Exception as ex:
                        with self.py_logger.context(session_id=self.session_id):
                            _loguru_logger.error("Tailer flush error: {}", str(ex))
                    self._buf = ""
        except Exception as ex:
            with self.py_logger.context(session_id=self.session_id):
                _loguru_logger.error("Tailer error: {}", str(ex))


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


def configure_matlab_environment(eng, project_root: Path):
    eng.addpath(str(project_root / "Old Code"), nargout=0)
    eng.addpath(str(project_root / "matlab" / "utils"), nargout=0)
    spm_path = r"C:\Program Files\spm"
    if os.path.exists(spm_path):
        eng.addpath(spm_path, nargout=0)
    else:
        pass

def prepare_live_log_file():
    fd, live_str = tempfile.mkstemp(prefix="matlab_live_", suffix=".log")
    os.close(fd)
    path = Path(live_str)
    path.write_text("", encoding="utf-8")
    return path


def assign_live_log_path(eng, live_path: Path):
    live_posix = str(live_path).replace("\\", "/")
    eng.assignin("base", "PY_LIVE_LOG", live_posix, nargout=0)


def run_preliminaries_async(eng, args):
    null_sink = io.StringIO()
    return eng.feval(
        "Preliminaries",
        str(args.dir_ad),
        str(args.dir_ctrl),
        str(args.tiv_path),
        nargout=0,
        stdout=null_sink,
        stderr=null_sink,
        background=True,
    )


def cleanup_live_log(live_path: Path, rc: int, py_logger: CustomLogger, session_id: str):
    if live_path is None:
        return
    if rc == 0:
        try:
            live_path.unlink(missing_ok=True)
        except Exception:
            with py_logger.context(session_id=session_id):
                _loguru_logger.warning("Could not delete live log {}", live_path)
    else:
        failed_path = live_path.with_name(live_path.name + ".failed")
        try:
            live_path.rename(failed_path)
            live_path = failed_path
        except Exception:
            pass
        with py_logger.context(session_id=session_id):
            _loguru_logger.info("Kept live log for debugging: {}", live_path)


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

    eng = None
    tailer = None
    live_path = None
    future = None
    rc = 1

    try:
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.info("Starting MATLAB Engine...")
        eng = matlab.engine.start_matlab()
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.success("MATLAB Engine started")

        configure_matlab_environment(eng, project_root)

        live_path = prepare_live_log_file()
        assign_live_log_path(eng, live_path)

        with py_logger.context(session_id=args.session_id):
            _loguru_logger.debug("Live log file: {}", live_path)
            _loguru_logger.debug("dirAD:   {}", args.dir_ad)
            _loguru_logger.debug("dirCTRL: {}", args.dir_ctrl)
            _loguru_logger.debug("TIV CSV: {}", args.tiv_path)

        tailer = FileTailer(live_path, py_logger, poll=args.tail_poll,
                            session_id=args.session_id)
        tailer.start()

        future = run_preliminaries_async(eng, args)
        future.result()
        rc = 0

        with py_logger.context(session_id=args.session_id):
            _loguru_logger.success("Preliminaries completed")

    except MatlabExecutionError as me:
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.error("MATLAB error: {}", str(me))
            _loguru_logger.debug("Traceback:\n{}", traceback.format_exc())
    except Exception as ex:
        with py_logger.context(session_id=args.session_id):
            _loguru_logger.error("Python-side error: {}", str(ex))
            _loguru_logger.debug("Traceback:\n{}", traceback.format_exc())
    finally:
        if tailer is not None:
            tailer.stop()
            tailer.join(timeout=3)
        if eng is not None:
            try:
                eng.eval("if exist('PY_LIVE_LOG','var'), clear PY_LIVE_LOG; end", nargout=0)
            except Exception:
                pass
            eng.quit()
            with py_logger.context(session_id=args.session_id):
                _loguru_logger.info("MATLAB Engine stopped")
        cleanup_live_log(live_path, rc, py_logger, args.session_id)

    return rc


if __name__ == "__main__":
    sys.exit(main())
