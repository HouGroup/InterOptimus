"""
Environment defaults for the desktop **worker** subprocess (PyTorch, OpenMP, jobflow).

On macOS + PyInstaller, unbounded BLAS/OMP threading and fork-related ObjC issues can
manifest as SIGSEGV (subprocess exit -11) with no Python traceback.
"""

from __future__ import annotations

import os
import sys
from typing import MutableMapping


def apply_worker_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """
    Set conservative thread limits and macOS fork-safety hints on *env* (in-place).

    Call on ``os.environ`` at worker startup and on the dict passed to ``Popen(..., env=…)``.
    """
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("MKL_DOMAIN_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
        ("MPLBACKEND", "Agg"),
    ):
        env.setdefault(key, val)
    if sys.platform == "darwin":
        env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    return env
