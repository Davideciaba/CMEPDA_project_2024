# tools/run_matlab_engine.py
# Realtime logging da MATLAB a Loguru via file "live" + tail (timestamp MATLAB visualizzato)
import io
import os
import re
import sys
import time
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

import matlab.engine
from matlab.engine import MatlabExecutionError
from loguru import logger as _loguru_logger

from logging_utils import CustomLogger

SESSION_ID = "matlab-preliminaries-session"


LINE_RE = re.compile(
    r"^\s*"
    r"(?P<ts_str>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s*\|\s*"
    r"(?P<lvl>TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s*\|\s*"
    r"(?P<name>[^:]+):(?P<func>[^:]+):(?P<line>\d+)\s*-\s*(?P<msg>.*)$"
)
WARN_RE = re.compile(r"^\s*Warning:\s*(?P<msg>.*)$", re.IGNORECASE)
ERR_RE = re.compile(r"^\s*Error using .*?:?\s*(?P<msg>.*)$", re.IGNORECASE)


class FileTailer:
    def __init__(self, path: Path, py_logger: CustomLogger, poll=0.05, session_id=SESSION_ID):
        self.path = Path(path)
        self.poll = float(poll)
        self._stop = threading.Event()
        self._buf = ""
        self.py_logger = py_logger
        self.session_id = session_id

    def stop(self):
        self._stop.set()

    @staticmethod
    def _parse_ts(ts: str):
        ts = ts.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                pass
        return None

    def _emit(self, lvl: str, ts_dt, ts_str: str, name: str, func: str, line: int, msg: str):
        # Per i log MATLAB usiamo un sink dedicato (from_matlab=True) e passiamo il timestamp MATLAB come extra["mt"]
        def _patch(record):
            # Non tocchiamo record["time"]: il sink MATLAB usa extra["mt"], quello Python usa {time}.
            if isinstance(ts_dt, datetime):
                record["time"] = ts_dt
            record["extra"]["from_matlab"] = True
            record["extra"]["mt"] = ts_str  # stringa "2025-11-12 00:30:12.345"
            record["file"] = type("F", (), {"name": name})()
            record["function"] = func
            try:
                record["line"] = int(line)
            except Exception:
                record["line"] = 0

        with self.py_logger.context(session_id=self.session_id):
            _loguru_logger.patch(_patch).opt(depth=0).log(lvl, msg)

    def _emit_plain(self, lvl: str, msg: str):
        # Messaggi grezzi/console MATLAB (senza header standard): li marchiamo comunque come from_matlab
        def _patch(record):
            record["extra"]["from_matlab"] = True
            record["extra"]["mt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            record["file"] = type("F", (), {"name": "MATLAB"})()
            record["function"] = "console"
            record["line"] = 0

        with self.py_logger.context(session_id=self.session_id):
            _loguru_logger.patch(_patch).opt(depth=0).log(lvl, msg)

    def _handle_line(self, raw: str):
        s = raw.rstrip("\r\n")
        if not s or s.startswith("__PY_HEARTBEAT__"):
            return
        m = LINE_RE.match(s)
        if m:
            ts_str = m.group("ts_str")
            ts_dt = self._parse_ts(ts_str)
            self._emit(
                m.group("lvl").upper(),
                ts_dt,
                ts_str,
                m.group("name"),
                m.group("func"),
                m.group("line"),
                m.group("msg"),
            )
            return
        if (wm := WARN_RE.match(s)):
            self._emit_plain("WARNING", wm.group("msg"))
            return
        if (em := ERR_RE.match(s)):
            self._emit_plain("ERROR", em.group("msg"))
            return
        self._emit_plain("INFO", s)

    def run(self):
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


def run_preliminaries_async(eng, project_root: Path, live_path: Path):
    """
    Legacy runner (kept as-is, only path adjustments).
    Note: Preliminaries is archived in Fase A; this remains a tool, not the pipeline core.
    """
    live_posix = str(live_path).replace("\\", "/")
    eng.assignin("base", "PY_LIVE_LOG", live_posix, nargout=0)

    # MATLAB utils moved here in Fase A
    eng.addpath(str(project_root / "matlab" / "utils"), nargout=0)

    # Legacy preliminaries archived here in Fase A
    eng.addpath(str(project_root / "scripts" / "archive" / "matlab"), nargout=0)

    # Legacy paths (kept to avoid changing behavior in Fase A)
    dirAD = str(project_root / "AD_CTRL" / "AD_s3")
    dirCTRL = str(project_root / "AD_CTRL" / "CTRL_s3")
    tivPath = project_root / "AD_CTRL" / "covariateADCTRLsexAgeTIV.csv"

    null_sink = io.StringIO()  # evita duplicati: niente stdout/err Engine in console
    return eng.feval(
        "Preliminaries",
        dirAD,
        dirCTRL,
        tivPath,
        nargout=0,
        stdout=null_sink,
        stderr=null_sink,
        background=True,
    )


def main():
    # tools/ -> project root
    project_root = Path(__file__).resolve().parents[1]
    py_logger = CustomLogger()

    with py_logger.context(session_id=SESSION_ID):
        _loguru_logger.info("Starting MATLAB Engine…")

    eng = None
    live_path = None
    tailer = None
    t = None
    rc = 1

    try:
        eng = matlab.engine.start_matlab()
        with py_logger.context(session_id=SESSION_ID):
            _loguru_logger.success("MATLAB Engine started")

        fd, live_str = tempfile.mkstemp(prefix="matlab_live_", suffix=".log")
        os.close(fd)
        live_path = Path(live_str)
        with py_logger.context(session_id=SESSION_ID):
            _loguru_logger.debug("Live log file: {}", live_path)

        fut = run_preliminaries_async(eng, project_root, live_path)

        tailer = FileTailer(live_path, py_logger, poll=0.05, session_id=SESSION_ID)
        t = threading.Thread(target=tailer.run, daemon=True)
        t.start()

        fut.result()
        rc = 0

        with py_logger.context(session_id=SESSION_ID):
            _loguru_logger.success("Preliminaries completed")

    except MatlabExecutionError as me:
        with py_logger.context(session_id=SESSION_ID):
            _loguru_logger.error("MATLAB error: {}", str(me))
            _loguru_logger.debug("Traceback:\n{}", traceback.format_exc())
    except Exception as ex:
        with py_logger.context(session_id=SESSION_ID):
            _loguru_logger.error("Python-side error: {}", str(ex))
            _loguru_logger.debug("Traceback:\n{}", traceback.format_exc())
    finally:
        try:
            if tailer is not None:
                tailer.stop()
            if t is not None:
                t.join(timeout=3)
            if eng is not None:
                eng.quit()
                with py_logger.context(session_id=SESSION_ID):
                    _loguru_logger.info("MATLAB Engine stopped")
        finally:
            try:
                if live_path and rc == 0:
                    live_path.unlink(missing_ok=True)
            except Exception:
                pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
