#!/usr/bin/env python3
"""
Run before `pyinstaller desktop/interoptimus_desktop.spec`.

Eqnorm pulls in torch_geometric → torch_scatter. If ``torch_scatter`` was built for
another PyTorch ABI, imports can fail; on macOS, importing ``torch_scatter`` before
``libtorch`` is fully initialized can SIGSEGV.

Install scatter (and related PyG wheels) from PyG’s index matching your torch::

  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \\
    -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html"

Then re-run this script; exit code 0 means imports succeeded.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def _ensure_repo_on_path() -> Path:
    """So we can ``import InterOptimus._env`` when run as ``python desktop/verify_torch_stack.py``."""
    repo = Path(__file__).resolve().parents[1]
    rs = str(repo)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    return repo


def _init_torch_before_extensions() -> None:
    """Warm up libtorch before torch_scatter/torch_sparse .so static init (macOS stability)."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Force allocator + ATen init so c10 is ready before extension registration.
    with torch.inference_mode():
        x = torch.randn(4, 4)
        _ = (x @ x).sum().item()


def _run_inner() -> int:
    _ensure_repo_on_path()
    try:
        import InterOptimus._env  # noqa: F401 — OMP/MPL/OBJC before numpy/torch
    except ImportError:
        # Repo not on PYTHONPATH (e.g. only site-packages): still set minimal env.

        for k, v in (
            ("OMP_NUM_THREADS", "1"),
            ("MKL_NUM_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
            ("VECLIB_MAXIMUM_THREADS", "1"),
            ("KMP_DUPLICATE_LIB_OK", "TRUE"),
            ("MPLBACKEND", "Agg"),
        ):
            os.environ.setdefault(k, v)
        if sys.platform == "darwin":
            os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    try:
        import torch

        print(f"torch {torch.__version__}")
    except Exception as e:
        print(f"FAIL: import torch: {e}", file=sys.stderr)
        return 1

    try:
        _init_torch_before_extensions()
    except Exception as e:
        print(f"FAIL: torch warmup: {e}", file=sys.stderr)
        return 1

    try:
        import torch_scatter  # noqa: F401

        print("torch_scatter: import ok", flush=True)
    except Exception as e:
        print(
            "FAIL: torch_scatter — reinstall from https://data.pyg.org/whl/ for this torch build. "
            f"({e})",
            file=sys.stderr,
        )
        return 1

    # torch_geometric pulls pyg_lib / sparse / cluster; mismatched wheels → c10 length_error (uncaught C++).
    for _label, _mod in (
        ("pyg_lib", "pyg_lib"),
        ("torch_sparse", "torch_sparse"),
        ("torch_cluster", "torch_cluster"),
    ):
        try:
            __import__(_mod)
            print(f"{_label}: import ok", flush=True)
        except Exception as e:
            print(f"FAIL: {_label}: {e}", file=sys.stderr)
            return 1

    try:
        import torch_geometric  # noqa: F401

        print("torch_geometric: import ok", flush=True)
    except Exception as e:
        print(f"FAIL: torch_geometric: {e}", file=sys.stderr)
        return 1

    print("OK — PyTorch + PyG stack is importable; safe to run PyInstaller.")
    return 0


def main() -> int:
    # Run in a subprocess so a native SIGSEGV in torch_scatter does not kill the driver
    # (and returns a normal exit code for CI / shell).
    if len(sys.argv) > 1 and sys.argv[1] == "--inner":
        return _run_inner()
    _ensure_repo_on_path()
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--inner"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=os.environ.copy(),
    )
    code = r.returncode
    if code == 0:
        return 0
    _d = Path(__file__).resolve().parent
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
    try:
        from pyg_wheel_url import pyg_find_links_url

        _url = pyg_find_links_url()
    except Exception:
        _url = "https://data.pyg.org/whl/torch-<ver>+cpu.html"

    sigsegv = getattr(signal, "SIGSEGV", 11)
    if code == -sigsegv or code == 128 + sigsegv:
        print(
            "FAIL: a PyG extension crashed with SIGSEGV. Typical: stale wheel vs current torch. Try:\n"
            f"  pip install --force-reinstall --no-build-isolation torch-scatter pyg-lib -f {_url}\n"
            "  python desktop/prep_pyg_stack.py --install",
            file=sys.stderr,
        )
        return 1
    sigabrt = getattr(signal, "SIGABRT", 6)
    if code == -sigabrt or code == 128 + sigabrt:
        print(
            "FAIL: PyG native code aborted (SIGABRT / std::length_error). "
            "Reinstall pyg-lib + torch-scatter from the same PyG index as torch:\n"
            f"  pip install --force-reinstall --no-build-isolation torch-scatter pyg-lib -f {_url}\n"
            "  python desktop/prep_pyg_stack.py --install",
            file=sys.stderr,
        )
        return 1
    return 1 if code is None else code


if __name__ == "__main__":
    raise SystemExit(main())
