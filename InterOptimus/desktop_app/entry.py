"""
InterOptimus Desktop: native Tkinter GUI for the MatRIS ``simple_iomaker`` workflow.

Sets a portable ``HOME`` so MatRIS reads checkpoints from ``~/.cache/matris/`` (see ``MatRIS.load``).

**Frozen (PyInstaller):** weights under ``bundled_home`` in the bundle are read-only under Finder;
they are copied into the writable user data directory at startup.

Usage::

    interoptimus-desktop
    # or
    python -m InterOptimus.desktop_app.entry
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

MATRIS_CKPT_NAME = "MatRIS_10M_OAM.pth.tar"


def _real_user_home() -> Path:
    """
    Actual login home directory. Do not use :func:`Path.home` after we set ``HOME`` to the
    portable bundle directory — :func:`Path.home` follows ``HOME`` and breaks log paths.
    """
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path(os.environ.get("HOME") or os.path.expanduser("~"))


def _repo_root() -> Path:
    """Repository root (directory containing ``desktop/`` and ``InterOptimus/``)."""
    return Path(__file__).resolve().parent.parent.parent


def _crash_log_path() -> Path:
    rh = _real_user_home()
    if sys.platform == "darwin":
        return rh / "Library" / "Logs" / "InterOptimus-desktop.log"
    return rh / ".cache" / "InterOptimus" / "desktop.log"


def _log_crash(msg: str) -> None:
    path = _crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(msg, encoding="utf-8")
    except OSError:
        pass


def _writable_home_frozen() -> Path:
    """User-writable directory for HOME when running from a frozen bundle."""
    rh = _real_user_home()
    if sys.platform == "darwin":
        return rh / "Library" / "Application Support" / "InterOptimus" / "Desktop"
    return rh / ".cache" / "InterOptimus" / "desktop"


def _find_bundle_bundled_home() -> Path | None:
    """
    Locate ``bundled_home`` inside the PyInstaller bundle (read-only under Finder).
    Returns None if not found (e.g. broken build).
    """
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    try:
        contents = exe.parent.parent
        if contents.name == "Contents":
            for bh in contents.rglob("bundled_home"):
                if bh.is_dir() and (bh / ".cache" / "matris" / MATRIS_CKPT_NAME).is_file():
                    return bh
    except OSError:
        pass
    for base in (exe.parent, exe.parent.parent):
        cand = base / "bundled_home"
        if (cand / ".cache" / "matris" / MATRIS_CKPT_NAME).is_file():
            return cand
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        mp = Path(meipass)
        for cand in (mp / "bundled_home", mp.parent / "bundled_home"):
            if (cand / ".cache" / "matris" / MATRIS_CKPT_NAME).is_file():
                return cand
    return None


def _sync_matris_checkpoint(bundle_bundled_home: Path, dst_home: Path) -> None:
    """Copy checkpoint from read-only bundle into writable HOME if needed."""
    src = bundle_bundled_home / ".cache" / "matris" / MATRIS_CKPT_NAME
    if not src.is_file():
        return
    dst_dir = dst_home / ".cache" / "matris"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / MATRIS_CKPT_NAME
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def bundled_home_path() -> Path:
    """
    Directory used as fake ``HOME`` / ``USERPROFILE``.

    Unfrozen: ``desktop/bundled_home`` in the repo.
    Frozen: writable Application Support (or cache) dir; weights copied from bundle.
    """
    if getattr(sys, "frozen", False):
        return _writable_home_frozen()
    return _repo_root() / "desktop" / "bundled_home"


def _apply_portable_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    hp = str(home.resolve())
    os.environ["HOME"] = hp
    os.environ["USERPROFILE"] = hp
    if sys.platform == "win32":
        os.environ["HOMEDRIVE"] = str(home.drive) + "\\" if home.drive else ""
        os.environ["HOMEPATH"] = str(home)
    os.environ["INTEROPTIMUS_NO_AUTO_VENV"] = "1"
    os.environ.setdefault("INTEROPTIMUS_WEB_SESSIONS", str(home / ".interoptimus" / "web_sessions"))


def main() -> None:
    if getattr(sys, "frozen", False):
        bundle_bh = _find_bundle_bundled_home()
        home = _writable_home_frozen()
        _apply_portable_home(home)
        if bundle_bh is not None:
            _sync_matris_checkpoint(bundle_bh, home)
    else:
        home = bundled_home_path()
        _apply_portable_home(home)

    from InterOptimus.desktop_app.gui import run_gui

    run_gui()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        tb = traceback.format_exc()
        _log_crash(f"{exc!r}\n\n{tb}")
        print(
            f"InterOptimus Desktop failed: {exc!r}\n"
            f"Details: {_crash_log_path()}\n",
            file=sys.stderr,
            flush=True,
        )
        raise
