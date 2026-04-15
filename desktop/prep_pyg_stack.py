#!/usr/bin/env python3
"""
Install PyTorch Geometric extension wheels that match the *currently installed* ``torch``.

Eqnorm → torch_geometric → torch_scatter. If ``torch_scatter`` was built for another
PyTorch ABI, macOS can abort inside ``_version_cpu.so`` at import time.

Usage::

    python desktop/prep_pyg_stack.py --install   # pip install from PyG index
    python desktop/prep_pyg_stack.py             # print wheel URL only (dry run)

Then::

    python desktop/verify_torch_stack.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError

_DESKTOP_DIR = Path(__file__).resolve().parent
if str(_DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_DESKTOP_DIR))

from pyg_wheel_url import pyg_find_links_url as _pyg_wheel_index_url


def _pip_install_pyg(pkgs: list[str], url: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-build-isolation",
        *pkgs,
        "-f",
        url,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Align torch_scatter etc. with installed torch.")
    ap.add_argument(
        "--install",
        action="store_true",
        help="Run pip install for PyG native wheels (same interpreter as this script).",
    )
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: import torch failed. Install torch first, e.g. pip install -e .", file=sys.stderr)
        return 1

    url = _pyg_wheel_index_url()
    print(f"PyG wheel index: {url}")

    if not args.install:
        print("Dry run. Pass --install to: pip install torch-scatter ... -f <url>")
        return 0

    # Cached wheels from an older torch cause SIGSEGV / c10 length_error; refresh core PyG natives.
    # torch_geometric needs pyg_lib + scatter + sparse + cluster aligned with this torch.
    print(
        "Reinstalling torch-scatter, pyg-lib, torch-sparse, torch-cluster (--force-reinstall) …",
        flush=True,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-build-isolation",
            "torch-scatter",
            "pyg-lib",
            "torch-sparse",
            "torch-cluster",
            "-f",
            url,
        ]
    )

    # When PyG has no prebuilt wheel for a package (e.g. torch-spline-conv on some torch+macOS
    # combos), pip builds from sdist. Default build isolation hides ``torch``, so setup.py fails
    # with ModuleNotFoundError: torch — use the active env for builds.
    pkgs_full = [
        "torch-spline-conv",
        "torch-geometric",
    ]
    pkgs_min = [
        "torch-geometric",
    ]
    try:
        _pip_install_pyg(pkgs_full, url)
    except CalledProcessError:
        print(
            "WARN: full PyG install failed (often torch-spline-conv: no wheel or missing C++ toolchain). "
            "Retrying without torch-spline-conv — sufficient for most Eqnorm / torch_geometric use.",
            flush=True,
        )
        _pip_install_pyg(pkgs_min, url)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
