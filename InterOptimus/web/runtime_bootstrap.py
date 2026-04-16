"""
Self-contained runtime for ``interoptimus-web`` without requiring ``conda activate``.

On first start, creates ``~/.interoptimus/web_venv`` (stdlib :mod:`venv`), runs
``pip install -e ".[web]"`` when the package is checked out next to ``pyproject.toml`` /
``setup.py``, otherwise ``pip install InterOptimus[web]``.

Set ``INTEROPTIMUS_NO_AUTO_VENV=1`` to use the current ``sys.executable`` only (for developers).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_MARKER = ".bootstrap_ok"


def managed_venv_root() -> Path:
    return Path.home() / ".interoptimus" / "web_venv"


def managed_python() -> Path:
    base = managed_venv_root()
    if sys.platform == "win32":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def find_project_root() -> Path | None:
    """Editable checkout root (contains ``pyproject.toml`` or ``setup.py``), if any."""
    import InterOptimus

    p = Path(InterOptimus.__file__).resolve().parent
    for _ in range(10):
        if (p / "pyproject.toml").is_file() or (p / "setup.py").is_file():
            return p
        p = p.parent
    return None


def _venv_sanity_check(py: Path) -> bool:
    if not py.is_file():
        return False
    r = subprocess.run(
        [str(py), "-c", "import fastapi; import InterOptimus"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0


def ensure_managed_venv() -> Path:
    """
    Create or repair ``~/.interoptimus/web_venv`` and install InterOptimus with web extras.

    Returns path to the venv's ``python`` executable.
    """
    root = managed_venv_root()
    py = managed_python()

    if py.is_file() and (root / _MARKER).is_file() and _venv_sanity_check(py):
        return py

    print(
        "InterOptimus: preparing isolated web environment under ~/.interoptimus/web_venv …\n"
        "(first run may take several minutes: PyTorch, pymatgen, …)\n",
        file=sys.stderr,
        flush=True,
    )

    if root.is_dir() and not py.is_file():
        import shutil

        shutil.rmtree(root, ignore_errors=True)

    if not root.is_dir():
        subprocess.run([sys.executable, "-m", "venv", str(root)], check=True)

    subprocess.run([str(py), "-m", "ensurepip", "--upgrade"], capture_output=True)
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip", "wheel"], check=True)

    project = find_project_root()
    if project is not None:
        subprocess.run(
            [str(py), "-m", "pip", "install", "-e", f"{project}[web]"],
            cwd=str(project),
            check=True,
        )
    else:
        subprocess.run([str(py), "-m", "pip", "install", "InterOptimus[web]"], check=True)

    (root / _MARKER).write_text("ok\n", encoding="utf-8")
    return py


def maybe_reexec_into_managed_venv() -> None:
    """
    If not already using the managed interpreter, build the venv and ``exec`` into it.

    Child processes set ``INTEROPTIMUS_WEB_INTERNAL_REEXEC=1`` to avoid recursion.
    """
    if os.environ.get("INTEROPTIMUS_WEB_INTERNAL_REEXEC"):
        return
    if os.environ.get("INTEROPTIMUS_NO_AUTO_VENV", "").lower() in ("1", "true", "yes"):
        return

    mp = managed_python()
    try:
        if mp.is_file() and Path(sys.executable).resolve() == mp.resolve():
            return
    except OSError:
        pass

    py = ensure_managed_venv()
    py = managed_python()
    env = os.environ.copy()
    env["INTEROPTIMUS_WEB_INTERNAL_REEXEC"] = "1"
    argv = [str(py.resolve()), "-m", "InterOptimus.web.app", *sys.argv[1:]]
    os.execve(str(py.resolve()), argv, env)
