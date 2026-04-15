"""
InterOptimus Desktop: native Tkinter GUI for the local ``simple_iomaker`` workflow.

Sets a portable ``HOME`` so MLIP checkpoints resolve from the bundled cache layout.

**Frozen (PyInstaller):** checkpoint files under ``bundled_home`` in the bundle are read-only under Finder;
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
                if bh.is_dir():
                    return bh
    except OSError:
        pass
    for base in (exe.parent, exe.parent.parent):
        cand = base / "bundled_home"
        if cand.is_dir():
            return cand
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        mp = Path(meipass)
        for cand in (mp / "bundled_home", mp.parent / "bundled_home"):
            if cand.is_dir():
                return cand
    return None


def _sync_tree_if_present(src_root: Path, dst_root: Path) -> None:
    """Copy files from *src_root* into *dst_root* when the source exists."""
    if not src_root.exists():
        return
    if src_root.is_file():
        dst_root.parent.mkdir(parents=True, exist_ok=True)
        if dst_root.is_file() and dst_root.stat().st_size == src_root.stat().st_size:
            return
        shutil.copy2(src_root, dst_root)
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue
        except OSError:
            pass
        shutil.copy2(src, dst)


def _sync_bundled_checkpoints(bundle_bundled_home: Path, dst_home: Path) -> None:
    """Copy bundled MLIP checkpoints into writable HOME for frozen desktop runs."""
    _sync_tree_if_present(bundle_bundled_home / ".cache" / "InterOptimus" / "checkpoints", dst_home / ".cache" / "InterOptimus" / "checkpoints")


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


def _prepare_portable_home() -> None:
    if getattr(sys, "frozen", False):
        bundle_bh = _find_bundle_bundled_home()
        home = _writable_home_frozen()
        _apply_portable_home(home)
        if bundle_bh is not None:
            _sync_bundled_checkpoints(bundle_bh, home)
    else:
        home = bundled_home_path()
        _apply_portable_home(home)


def main() -> None:
    # PyInstaller bootloader is not a real CPython: ``sys.executable -m pkg.mod`` does not run
    # ``pkg.mod``. The GUI subprocess must use ``--interoptimus-worker <config.json>`` instead.
    if len(sys.argv) >= 3 and sys.argv[1] == "--interoptimus-worker":
        _prepare_portable_home()
        from InterOptimus.desktop_app.worker import main as worker_main

        sys.argv = [sys.argv[0], sys.argv[2]]
        worker_main()
        return

    _prepare_portable_home()
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
