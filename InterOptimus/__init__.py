"""InterOptimus: BLAS/thread + Matplotlib env before NumPy / PyTorch (see :mod:`InterOptimus._env`)."""

from __future__ import annotations

from . import _env  # noqa: F401  # side effect: apply_interoptimus_process_env
