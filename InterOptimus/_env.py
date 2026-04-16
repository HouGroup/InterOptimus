"""
Process-wide environment for safe BLAS/OpenMP + headless plotting.

Must stay import-light: only ``os`` / ``sys``. Imported from :mod:`InterOptimus`
before any NumPy/PyTorch/Matplotlib load.
"""

from __future__ import annotations

import os
import sys


def apply_interoptimus_process_env() -> None:
    """Idempotent: ``setdefault`` only."""
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("MKL_DOMAIN_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
        # Avoid native macOS GUI backend + PyTorch in the same worker (segfault risk).
        ("MPLBACKEND", "Agg"),
    ):
        os.environ.setdefault(k, v)
    if sys.platform == "darwin":
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


apply_interoptimus_process_env()
