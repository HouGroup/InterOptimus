"""PyInstaller bootstrap: ``pyinstaller desktop/interoptimus_desktop.spec`` (run from repo root)."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path


def _real_user_home() -> Path:
    """Not :func:`Path.home` — after the app sets ``HOME`` to a portable dir, that follows ``HOME``."""
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path(os.environ.get("HOME") or os.path.expanduser("~"))


def _crash_log_path() -> Path:
    rh = _real_user_home()
    if sys.platform == "darwin":
        return rh / "Library" / "Logs" / "InterOptimus-desktop.log"
    return rh / ".cache" / "InterOptimus" / "desktop.log"


def _write_crash(msg: str) -> None:
    path = _crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(msg, encoding="utf-8")
    except OSError:
        pass


def _alert_macos(message: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        safe = message.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display alert "InterOptimus Desktop" message "{safe}" as critical',
            ],
            check=False,
            timeout=30,
        )
    except OSError:
        pass


def main() -> None:
    # InterOptimus._env sets OMP/MKL/MPLBACKEND before NumPy (via package __init__).
    import InterOptimus  # noqa: F401

    # Required for PyInstaller + multiprocessing (torch/deepmd/jobflow may spawn workers).
    # Without this, a child re-exec of this binary falls through to app_main() and opens a second GUI.
    try:
        import multiprocessing

        multiprocessing.freeze_support()
    except Exception:
        pass
    try:
        from InterOptimus.desktop_app.entry import main as app_main

        app_main()
    except BaseException as exc:
        tb = traceback.format_exc()
        text = f"{exc!r}\n\n{tb}"
        _write_crash(text)
        log = _crash_log_path()
        _alert_macos(f"Failed to start. See: {log}")
        print(f"InterOptimus Desktop failed: {exc!r}\nLog: {log}\n", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
