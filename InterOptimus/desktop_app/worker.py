"""
Subprocess entry for the Eqnorm IOMaker workflow: allows the GUI to terminate the child process on cancel.

Run as: ``python -m InterOptimus.desktop_app.worker <config.json>`` (dev) or
``<frozen exe> --interoptimus-worker <config.json>`` (PyInstaller).

- Progress (print/tqdm from dependencies) goes to **stderr** so the GUI can stream it live.
- Final result JSON is one line on stdout **and** (when set) ``INTEROPTIMUS_WORKER_RESULT`` for the GUI parent.

**Critical:** BLAS/OpenMP and ``MPLBACKEND`` are applied in :mod:`InterOptimus._env` via
``import InterOptimus`` (see package ``__init__``). This module imports that package first
so odd launch paths still initialize safely before NumPy/PyTorch.
"""

from __future__ import annotations

import InterOptimus  # noqa: F401

import json
import os
import sys
import traceback
from pathlib import Path


def _write_result_line_to_stdout_pipe(line: str) -> None:
    """
    Write the final JSON line to the OS stdout file descriptor (fd 1).

    In frozen (PyInstaller) subprocesses, ``sys.__stdout__`` may not be the same stream as fd 1
    that the parent ``Popen(..., stdout=PIPE)`` reads. Always use fd 1 so the GUI receives the line.
    """
    data = line.encode("utf-8", errors="replace")
    try:
        n = 0
        while n < len(data):
            k = os.write(1, data[n:])
            if k <= 0:
                break
            n += k
    except OSError:
        # Last resort: best-effort on whatever Python thinks stdout is
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception:
            pass


def _finalize_result_line(line: str) -> None:
    """Emit one JSON line to stdout (fd 1) and optionally to a file for the parent process."""
    _write_result_line_to_stdout_pipe(line)
    outp = (os.environ.get("INTEROPTIMUS_WORKER_RESULT") or "").strip()
    if outp:
        try:
            Path(outp).write_text(line, encoding="utf-8")
        except OSError:
            pass


def _emit(obj: dict, *, rc: int = 0) -> None:
    line = json.dumps(obj, default=str, ensure_ascii=False) + "\n"
    _finalize_result_line(line)
    if rc:
        sys.exit(rc)


def main() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if len(sys.argv) < 2:
        _emit({"ok": False, "error": "missing config path"}, rc=2)

    path = Path(sys.argv[1])
    if not path.is_file():
        _emit({"ok": False, "error": f"config file not found: {path}"}, rc=2)

    try:
        raw = path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except (OSError, UnicodeError) as e:
        _emit({"ok": False, "error": f"failed to read config: {e}"}, rc=2)
    except json.JSONDecodeError as e:
        _emit({"ok": False, "error": f"invalid JSON in config: {e}"}, rc=2)

    try:
        film = Path(cfg["film"])
        sub = Path(cfg["sub"])
        form = cfg["form"]
    except (KeyError, TypeError) as e:
        _emit({"ok": False, "error": f"invalid config keys: {e}"}, rc=2)

    vl = cfg.get("viz_log")
    ve = bool(cfg.get("viz_enable"))
    if ve and isinstance(vl, str) and vl.strip():
        os.environ["INTEROPTIMUS_VIZ_LOG"] = vl.strip()
        os.environ["INTEROPTIMUS_VIZ_ENABLE"] = "1"
    else:
        os.environ.pop("INTEROPTIMUS_VIZ_LOG", None)
        os.environ.pop("INTEROPTIMUS_VIZ_ENABLE", None)

    # Route library stdout → stderr so live logs/tqdm show up on stderr for the GUI.
    _saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from InterOptimus.desktop_app.local_workflow import run_eqnorm_session

        out = run_eqnorm_session(film_cif_path=film, substrate_cif_path=sub, form=form)
    except Exception as e:
        sys.stdout = _saved_stdout
        _emit(
            {
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
            rc=1,
        )
    else:
        sys.stdout = _saved_stdout
        if not isinstance(out, dict):
            _emit(
                {
                    "ok": False,
                    "error": f"run_eqnorm_session returned {type(out).__name__}, expected dict",
                },
                rc=1,
            )
        else:
            _emit(out, rc=0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _finalize_result_line(
            json.dumps(
                {
                    "ok": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
                default=str,
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.exit(1)
