"""
PyG wheels live under https://data.pyg.org/whl/torch-<torch tag>.html

PyTorch often reports ``2.11.0`` but CPU/macOS wheels are indexed as ``2.11.0+cpu``.
Using ``torch-2.11.0.html`` hits a missing/forbidden index and pip falls back to the wrong
``torch-scatter`` → SIGSEGV at import.
"""

from __future__ import annotations


def pyg_find_links_url() -> str:
    import sys

    import torch

    v = torch.__version__
    if "+" in v:
        return f"https://data.pyg.org/whl/torch-{v}.html"

    # No local tag in __version__ — PyG uses +cpu / +cu126 / …
    if sys.platform == "darwin":
        return f"https://data.pyg.org/whl/torch-{v}+cpu.html"
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return f"https://data.pyg.org/whl/torch-{v}+cpu.html"

    # Linux + CUDA: map e.g. 12.4 -> cu124 (PyG naming)
    cuda = str(torch.version.cuda or "")
    digits = "".join(c for c in cuda if c.isdigit())
    if len(digits) >= 2:
        cu = "cu" + digits[: min(4, len(digits))]  # cu124
        return f"https://data.pyg.org/whl/torch-{v}+{cu}.html"
    return f"https://data.pyg.org/whl/torch-{v}+cpu.html"
