# PyInstaller runtime hook (runs before user main).
#
# macOS crash (SIGSEGV in c10::RegisterOperators) when torch_scatter's _version_cpu.so
# registers with libtorch before libtorch is loaded. Force-import torch first.
# See: torch_scatter static init + dlopen order in frozen one-folder apps.

from __future__ import annotations

import os
import sys


def _thread_env() -> None:
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("MKL_DOMAIN_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ.setdefault(k, v)
    if sys.platform == "darwin":
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


_thread_env()

if getattr(sys, "frozen", False):
    try:
        import torch  # noqa: F401

        # Single-thread native libs before any torch_geometric / torch_scatter extension runs.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass
