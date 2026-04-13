#!/usr/bin/env python3
"""Lightweight install check for the pure-workflow InterOptimus layout."""

import os
import sys
from pathlib import Path

# Allow `python InterOptimus/verify_installation.py` from the repository root.
def _path0_resolved() -> Path:
    if not sys.path or sys.path[0] == "":
        return Path.cwd().resolve()
    return Path(sys.path[0]).resolve()


_pkg_dir = Path(__file__).resolve().parent
# Script launch prepends this dir → `import jobflow` would load InterOptimus/jobflow.py.
if sys.path and _path0_resolved() == _pkg_dir:
    sys.path.pop(0)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def check_file_structure() -> bool:
    print("Checking InterOptimus file layout...")
    root = Path(__file__).resolve().parent
    core = [
        "itworker.py",
        "matching.py",
        "tool.py",
        "mlip.py",
        "CNID.py",
        "equi_term.py",
        "jobflow.py",
    ]
    agents = [
        "agents/simple_iomaker.py",
        "agents/iomaker_job.py",
        "agents/remote_submit.py",
        "agents/server_env.py",
    ]
    ok = True
    for rel in core + agents:
        p = root / rel
        tag = "OK" if p.is_file() else "MISSING"
        if not p.is_file():
            ok = False
        print(f"  {tag:8} {rel}")
    return ok


def check_imports() -> bool:
    print("\nChecking imports...")
    try:
        from InterOptimus.itworker import InterfaceWorker  # noqa: F401

        print("  OK  InterOptimus.itworker")
        from InterOptimus.agents.simple_iomaker import run_simple_iomaker  # noqa: F401

        print("  OK  InterOptimus.agents.simple_iomaker")
        from InterOptimus.agents.iomaker_job import execute_iomaker_from_settings  # noqa: F401

        print("  OK  InterOptimus.agents.iomaker_job")
    except ImportError as e:
        print(f"  FAIL {e}")
        return False
    return True


def main() -> bool:
    os.chdir(Path(__file__).resolve().parent)  # package dir (contains itworker.py)
    a = check_file_structure()
    b = check_imports()
    print("\nDone." if a and b else "\nSome checks failed — see docs/GETTING_STARTED.md.")
    return a and b


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
