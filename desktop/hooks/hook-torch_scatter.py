# PyInstaller hook: torch_scatter (PyTorch Geometric / Eqnorm)
#
# Bundle the full conda/pip package directory so _version_cpu*.so sit next to __init__.py.
# Used with Analysis(noarchive=True) in interoptimus_desktop.spec so bytecode is not split from .so.

from __future__ import annotations

import importlib.util
from pathlib import Path

datas: list = []
binaries: list = []
hiddenimports: list = [
    "torch_scatter",
    "torch_scatter.scatter",
    "torch_scatter.segment",
    "torch_scatter.utils",
]

_spec = importlib.util.find_spec("torch_scatter")
if _spec is not None and _spec.origin:
    _root = Path(_spec.origin).resolve().parent
    if _root.is_dir():
        datas.append((str(_root), "torch_scatter"))
