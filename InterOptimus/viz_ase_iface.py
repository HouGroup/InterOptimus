"""
Render interface snapshots with ASE ``plot_atoms`` (orthographic view).

Used by the web worker to write preview PNGs next to ``web_viz.jsonl`` so the browser
can display interface images without running ASE in JavaScript.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

# Orthographic view rotation for interface plots (Å axes).
ASE_PLOT_VIEW_ROTATION = "90x,0y,90z"


def _positions_from_event(ev: Dict[str, Any]) -> Optional[Any]:
    """Return an ``(N, 3)`` positions-like object from BO or relax-final payloads."""
    pc = ev.get("positions_cart")
    if isinstance(pc, list):
        return pc
    flat = ev.get("positions")
    nums = ev.get("numbers")
    if not isinstance(flat, list) or not isinstance(nums, list):
        return None
    try:
        import numpy as np

        arr = np.asarray(flat, dtype=float).reshape((-1, 3))
    except Exception:
        return None
    if len(arr) > len(nums):
        arr = arr[: len(nums)]
    return arr


def render_bo_iface_png_to_bytes(ev: Dict[str, Any], *, dpi: int = 110) -> Optional[bytes]:
    """
    Build PNG bytes from one ``phase=bo`` / ``event=sample`` JSONL row.

    Expects BO payload fields ``positions_cart`` / ``lattice`` or relax-final fields
    ``positions`` / ``cell``; optional ``film_substrate`` (tags).
    """
    pc = _positions_from_event(ev)
    nums = ev.get("numbers")
    lat = ev.get("lattice") or ev.get("cell")
    if pc is None or not isinstance(nums, list) or len(pc) != len(nums):
        return None
    if not lat or not isinstance(lat, list):
        return None
    try:
        import numpy as np
        from ase import Atoms
        from ase.data.colors import jmol_colors
        from ase.visualize.plot import plot_atoms
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception:
        return None

    try:
        positions = np.asarray(pc, dtype=float)
        numbers = np.asarray(nums, dtype=int)
        cell = np.asarray(lat, dtype=float)
    except Exception:
        return None

    if positions.ndim != 2 or positions.shape[1] != 3:
        return None
    if cell.shape != (3, 3):
        return None

    try:
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    except Exception:
        return None

    fs = ev.get("film_substrate")
    if isinstance(fs, list) and len(fs) == len(numbers):
        try:
            atoms.set_tags(np.asarray(fs, dtype=int))
        except Exception:
            pass

    z = numbers
    cols = np.zeros((len(z), 3), dtype=float)
    for k in range(len(z)):
        el = int(z[k])
        if 0 <= el < len(jmol_colors):
            cols[k] = np.asarray(jmol_colors[el], dtype=float)
        else:
            cols[k] = (0.55, 0.55, 0.55)
    color_list: List[Any] = [tuple(np.clip(c, 0.0, 1.0)) for c in cols]

    face = "#f8fafc"
    fig = Figure(figsize=(4.7, 3.85), dpi=dpi, facecolor=face)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor=face)
    scale = 0.92
    rotation = ASE_PLOT_VIEW_ROTATION
    plotted = False
    for kwargs in (
        {"scale": scale, "rotation": rotation, "show_unit_cell": 2, "colors": color_list},
        {"scale": scale, "rotation": rotation, "show_unit_cell": 2},
        {"scale": scale, "rotation": rotation, "show_unit_cell": True},
        {"scale": scale, "rotation": rotation},
    ):
        try:
            ax.clear()
            ax.set_facecolor(face)
            plot_atoms(atoms, ax, **kwargs)
            plotted = True
            break
        except Exception:
            continue
    if not plotted:
        return None
    try:
        ax.set_aspect("equal")
    except Exception:
        pass
    ax.set_xlabel("x (Å)", fontsize=9, color="#475569")
    ax.set_ylabel("y (Å)", fontsize=9, color="#475569")
    ax.tick_params(colors="#64748b", labelsize=8)
    try:
        fig.tight_layout(pad=0.35)
    except Exception:
        pass
    buf = io.BytesIO()
    try:
        canvas.print_png(buf)
    except Exception:
        return None
    return buf.getvalue()


def write_bo_iface_png(ev: Dict[str, Any], path: Path) -> bool:
    """Write PNG to ``path``; returns False if ASE/matplotlib unavailable or payload invalid."""
    data = render_bo_iface_png_to_bytes(ev)
    if not data:
        return False
    try:
        path.write_bytes(data)
        return True
    except OSError:
        return False
