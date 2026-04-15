"""
Subprocess entry for MatRIS workflow: allows the GUI to terminate the child process on cancel.

Run as: python -m InterOptimus.desktop_app.worker <config.json>

- Progress (print/tqdm from dependencies) goes to **stderr** so the GUI can stream it live.
- Final result JSON is written to **real stdout** only (one line), so the parent can parse it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Original process stdout (FD 1) — never redirect this away from _emit
_REAL_STDOUT = getattr(sys, "__stdout__", sys.stdout)


def _emit(obj: dict, *, rc: int = 0) -> None:
    _REAL_STDOUT.write(json.dumps(obj, default=str, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()
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
        from InterOptimus.web.local_workflow import run_matris_session

        out = run_matris_session(film_cif_path=film, substrate_cif_path=sub, form=form)
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
        _emit(out, rc=0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _REAL_STDOUT.write(
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
        _REAL_STDOUT.flush()
        sys.exit(1)
