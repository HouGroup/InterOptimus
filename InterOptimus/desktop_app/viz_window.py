"""
Live workflow visualization (Bayesian registration + MLIP relaxation) driven by JSONL events.

Requires matplotlib. Structure panels use ASE :func:`ase.visualize.plot.plot_atoms` with
``rotation`` (see ``ASE_PLOT_VIEW_ROTATION``); install ``ase`` for that view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PairKey = Tuple[int, int]

# ASE plot_atoms: view via ``rotation`` only (see ``ase.utils.rotate``).
# - 90° x: edge-on slab (stacking normal in-plane, termination visible).
# - 90° z: extra rotation in the figure plane (matplotlib x–y); long in-plane interfaces
#   read left–right instead of tall/narrow in a wide window layout.
ASE_PLOT_VIEW_ROTATION = "90x,0y,90z"


def _rotate_positions_like_plot_atoms(pos: np.ndarray, rot: str) -> np.ndarray:
    """Fallback scatter: same rotation matrix as ASE :func:`ase.utils.rotate` / ``plot_atoms``."""
    from ase.utils import rotate as ase_rotate

    R = ase_rotate(rot)
    return np.asarray(np.dot(np.asarray(pos, dtype=float), R), dtype=float)


def _empty_bo_pair() -> Dict[str, Any]:
    return {
        "xyz_cart": [],
        "e": [],
        "e_per_atom": [],
        "bo_idx": [],
        "iface_positions": None,
        "iface_numbers": None,
        "iface_side": None,
        "iface_cell": None,
    }

import tkinter as tk
from tkinter import ttk

try:
    import numpy as np
    import matplotlib

    matplotlib.rcParams["axes.unicode_minus"] = False
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Tk / figure chrome (aligned with desktop_app.gui palette)
_VIZ_WIN_BG = "#eef2f7"
_VIZ_FIG_FACE = "#eef2f7"


def _configure_viz_mpl_theme() -> None:
    """Cohesive typography + grid + neutrals for workflow figures."""
    if not _HAS_MPL:
        return
    matplotlib.rcParams.update(
        {
            "figure.facecolor": _VIZ_FIG_FACE,
            "figure.edgecolor": _VIZ_FIG_FACE,
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "axes.labelcolor": "#475569",
            "axes.titlecolor": "#0f172a",
            "text.color": "#334155",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#e2e8f0",
            "legend.fontsize": 7,
        }
    )


def _style_axes_2d(ax: Any) -> None:
    try:
        ax.grid(True, alpha=0.45, color="#e2e8f0", linewidth=0.8, linestyle="-")
        ax.tick_params(axis="both", colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.9)
    except Exception:
        pass


def _style_axes_3d(ax: Any) -> None:
    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.97, 0.98, 0.99))
            axis.pane.set_alpha(0.5)
            axis.pane.set_edgecolor("#e2e8f0")
        ax.tick_params(axis="x", colors="#64748b", labelsize=8)
        ax.tick_params(axis="y", colors="#64748b", labelsize=8)
        ax.tick_params(axis="z", colors="#64748b", labelsize=8)
    except Exception:
        pass


if _HAS_MPL:
    _configure_viz_mpl_theme()


def _frac_to_cart(frac_list: List[List[float]], lattice: List[List[float]]) -> Any:
    M = np.array(lattice, dtype=float)
    P = np.array(frac_list, dtype=float)
    if P.size == 0:
        return P.reshape(0, 3)
    return P @ M


def _ylim_focus_low_band(e: Any) -> tuple[float, float]:
    """Tight y-axis on the low-energy band; high outliers do not stretch the scale."""
    arr = np.asarray(e, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    e_min = float(np.min(arr))
    p90 = float(np.percentile(arr, 90))
    p100 = float(np.max(arr))
    span = max(p90 - e_min, 1e-12)
    if p100 > p90 + 0.25 * span + 1e-9:
        y_max = e_min + 1.25 * span
    else:
        y_max = p100 + 0.04 * max(abs(p100), 1e-9)
    y_min = e_min - 0.03 * max(abs(e_min), 1e-9)
    if y_max <= y_min:
        y_max = y_min + 1e-6
    return (y_min, y_max)


def _iface_atom_colors(
    numbers: np.ndarray, film_side: Optional[List[int]]
) -> np.ndarray:
    """Jmol-like colors with film/substrate tint (optional ASE)."""
    n = len(numbers)
    out = np.zeros((n, 3), dtype=float)
    try:
        from ase.data.colors import jmol_colors

        for k in range(n):
            z = int(numbers[k])
            if 0 <= z < len(jmol_colors):
                base = np.asarray(jmol_colors[z], dtype=float)
            else:
                base = np.array([0.55, 0.55, 0.55])
            if film_side is not None and k < len(film_side):
                if film_side[k] == 0:
                    c = 0.55 * base + 0.45 * np.array([1.0, 0.55, 0.25])
                else:
                    c = 0.55 * base + 0.45 * np.array([0.35, 0.55, 0.95])
            else:
                c = base
            out[k] = np.clip(c, 0.0, 1.0)
    except Exception:
        try:
            cmap = matplotlib.colormaps["tab20"]
        except Exception:
            cmap = matplotlib.cm.get_cmap("tab20")
        for k in range(n):
            out[k] = cmap(k % 20)[:3]
    return out


def _plot_interface_ase_orthographic(
    ax: Any,
    positions: List[List[float]],
    numbers: List[int],
    cell: List[List[float]],
    film_side: Optional[List[int]],
    *,
    title: str,
    show_edge_caption: bool = True,
    show_title: bool = True,
    show_axis_labels: bool = True,
    scale: float = 0.92,
    rotation: str = ASE_PLOT_VIEW_ROTATION,
) -> bool:
    """
    ASE :func:`ase.visualize.plot.plot_atoms` — 2D orthographic projection (not Matplotlib 3D).
    View is controlled only by ``rotation`` (see ``ASE_PLOT_VIEW_ROTATION``).
    """
    try:
        from ase import Atoms
        from ase.visualize.plot import plot_atoms
    except Exception:
        return False
    try:
        atoms = Atoms(
            numbers=np.asarray(numbers, dtype=int),
            positions=np.asarray(positions, dtype=float),
            cell=np.asarray(cell, dtype=float),
            pbc=True,
        )
        if film_side is not None and len(film_side) == len(numbers):
            atoms.set_tags(np.asarray(film_side, dtype=int))
        z = np.asarray(numbers, dtype=int)
        cols = _iface_atom_colors(z, film_side)
        color_list = [tuple(np.clip(c, 0.0, 1.0)) for c in cols]
        last_err: Optional[Exception] = None
        for kwargs in (
            {"scale": scale, "rotation": rotation, "show_unit_cell": 2, "colors": color_list},
            {"scale": scale, "rotation": rotation, "show_unit_cell": 2},
            {"scale": scale, "rotation": rotation, "show_unit_cell": True},
            {"scale": scale, "rotation": rotation},
        ):
            try:
                plot_atoms(atoms, ax, **kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            return False
        try:
            ax.set_aspect("equal")
        except Exception:
            pass
        if show_title:
            ax.set_title(title)
        if show_axis_labels:
            ax.set_xlabel("x (Angstrom)")
            ax.set_ylabel("y (Angstrom)")
        if show_edge_caption:
            try:
                ax.text(
                    0.02,
                    0.98,
                    f"ASE plot_atoms · {rotation}",
                    transform=ax.transAxes,
                    fontsize=5,
                    va="top",
                    color="#94a3b8",
                )
            except Exception:
                pass
        return True
    except Exception:
        return False


def _plot_interface_fallback_2d(
    ax: Any,
    positions: List[List[float]],
    numbers: List[int],
    cell: List[List[float]],
    film_side: Optional[List[int]],
    *,
    title: str,
    show_title: bool = True,
    show_axis_labels: bool = True,
    show_edge_caption: bool = True,
    rotation: str = ASE_PLOT_VIEW_ROTATION,
) -> bool:
    """If ``plot_atoms`` is unavailable, approximate same ``rotation`` + 2D scatter."""
    try:
        from ase import Atoms
    except Exception:
        return False
    try:
        atoms = Atoms(
            numbers=np.asarray(numbers, dtype=int),
            positions=np.asarray(positions, dtype=float),
            cell=np.asarray(cell, dtype=float),
            pbc=True,
        )
        pos = _rotate_positions_like_plot_atoms(np.asarray(atoms.get_positions(), dtype=float), rotation)
        z = np.asarray(numbers, dtype=int)
        cols = _iface_atom_colors(z, film_side)
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=cols,
            s=14,
            linewidths=0.12,
            edgecolors="0.25",
            alpha=0.95,
        )
        ax.set_aspect("equal")
        if show_title:
            ax.set_title(title)
        if show_axis_labels:
            ax.set_xlabel("x (Angstrom)")
            ax.set_ylabel("y (Angstrom)")
        if show_edge_caption:
            ax.text(
                0.02,
                0.98,
                f"Fallback scatter · {rotation}",
                transform=ax.transAxes,
                fontsize=5,
                va="top",
                color="#94a3b8",
            )
        return True
    except Exception:
        return False


class WorkflowVizWindow:
    """Toplevel with matplotlib figures: BO registration (3D), BO interface (ASE 2D), energy, relax."""

    def __init__(
        self,
        master: tk.Tk,
        viz_path: Path,
        *,
        is_active: Callable[[], bool],
        on_closed: Optional[Callable[[], None]] = None,
    ) -> None:
        if not _HAS_MPL:
            raise RuntimeError("matplotlib is required for workflow visualization")
        self._viz_path = Path(viz_path)
        self._fp = 0
        self._is_active = is_active
        self._on_closed = on_closed

        self._bo_pairs: Dict[PairKey, Dict[str, Any]] = {}
        self._pair_order: List[PairKey] = []
        self._active_pair: Optional[PairKey] = None
        self._last_iface_frac: Optional[List[List[float]]] = None
        self._last_lat: Optional[List[List[float]]] = None
        self._relax_by_pair: Dict[PairKey, Dict[str, List[Any]]] = {}
        self._relax_pair_order: List[PairKey] = []
        self._relax_label = ""
        self._relax_final_snapshots: List[Tuple[PairKey, Dict[str, Any]]] = []

        self._cbar_reg: Any = None
        self._relax_twin_ax: Any = None
        self._fig_title_artist: Any = None

        self.win = tk.Toplevel(master)
        self.win.title("InterOptimus · live workflow viz")
        self.win.minsize(980, 720)
        try:
            self.win.configure(bg=_VIZ_WIN_BG)
        except tk.TclError:
            pass

        hdr = tk.Frame(self.win, bg=_VIZ_WIN_BG)
        hdr.pack(fill="x", padx=14, pady=(12, 6))
        _title_f = (".AppleSystemUIFont", 16, "bold") if sys.platform == "darwin" else ("Segoe UI", 14, "bold")
        _hint_f = (".AppleSystemUIFont", 11) if sys.platform == "darwin" else ("Segoe UI", 10)
        tk.Label(
            hdr,
            text="InterOptimus · live workflow",
            bg=_VIZ_WIN_BG,
            fg="#0f172a",
            font=_title_f,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            hdr,
            text=(
                "Row 1 — BO registration (3D) & interface (ASE)  ·  "
                "Row 2 — E/N (active pair & all pairs)  ·  "
                "Row 3 — relax E/N + RMS  ·  "
                "Row 4 — last 6 relaxed structures"
            ),
            bg=_VIZ_WIN_BG,
            fg="#64748b",
            font=_hint_f,
            anchor="w",
            justify="left",
            wraplength=920,
        ).pack(anchor="w", pady=(6, 0))

        self.fig = Figure(figsize=(13, 11.5), dpi=100, facecolor=_VIZ_FIG_FACE)
        gs = self.fig.add_gridspec(
            4, 2, height_ratios=[1.12, 1.08, 1.02, 1.0], hspace=0.58, wspace=0.30
        )
        gs_top = gs[0, :].subgridspec(1, 2, wspace=0.05)
        self.ax_reg = self.fig.add_subplot(gs_top[0, 0], projection="3d")
        self.ax_iface = self.fig.add_subplot(gs_top[0, 1])
        self.ax_ebo = self.fig.add_subplot(gs[1, 0])
        self.ax_ebo_all = self.fig.add_subplot(gs[1, 1])
        self.ax_relax = self.fig.add_subplot(gs[2, :])
        gsf = gs[3, :].subgridspec(1, 6, wspace=0.42)
        self._ax_rf: List[Any] = [self.fig.add_subplot(gsf[0, i]) for i in range(6)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        _cw = self.canvas.get_tk_widget()
        try:
            _cw.configure(bg=_VIZ_WIN_BG, highlightthickness=0)
        except tk.TclError:
            pass
        _cw.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        self._poll_id: Optional[str] = None
        self.win.protocol("WM_DELETE_WINDOW", self._on_user_close)
        self._schedule_poll()

    def _on_user_close(self) -> None:
        if self._poll_id is not None:
            try:
                self.win.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        if self._on_closed:
            self._on_closed()

    def _schedule_poll(self) -> None:
        if self._poll_id is not None:
            try:
                self.win.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        self._poll_id = self.win.after(280, self._poll_tick)

    def _poll_tick(self) -> None:
        try:
            self._read_and_apply()
            self._redraw()
        except tk.TclError:
            return
        if self._is_active():
            self._schedule_poll()
        else:
            try:
                self._read_and_apply()
                self._redraw()
            except Exception:
                pass

    def _read_and_apply(self) -> None:
        if not self._viz_path.is_file():
            return
        try:
            with open(self._viz_path, "rb") as f:
                f.seek(self._fp)
                chunk = f.read()
                self._fp = f.tell()
        except OSError:
            return
        if not chunk:
            return
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._handle_event(ev)

    def _handle_event(self, ev: Dict[str, Any]) -> None:
        ph = ev.get("phase")
        evn = ev.get("event")
        if ph == "bo" and evn == "sample":
            try:
                mk = int(ev.get("match_id", -1))
                tk = int(ev.get("term_id", -1))
            except (TypeError, ValueError):
                mk, tk = -1, -1
            key: PairKey = (mk, tk)
            if key not in self._bo_pairs:
                self._bo_pairs[key] = _empty_bo_pair()
                self._pair_order.append(key)
            b = self._bo_pairs[key]
            self._active_pair = key

            xc = ev.get("xyz_cart")
            if isinstance(xc, list) and len(xc) == 3:
                b["xyz_cart"].append([float(xc[0]), float(xc[1]), float(xc[2])])
            try:
                b["e"].append(float(ev.get("energy", 0.0)))
            except (TypeError, ValueError):
                b["e"].append(0.0)
            try:
                etot = float(b["e"][-1])
                na = ev.get("n_atoms")
                epa = ev.get("energy_per_atom")
                if epa is not None:
                    b["e_per_atom"].append(float(epa))
                elif isinstance(na, (int, float)) and float(na) > 0:
                    b["e_per_atom"].append(etot / float(na))
                else:
                    b["e_per_atom"].append(etot)
            except (TypeError, ValueError):
                b["e_per_atom"].append(0.0)
            try:
                b["bo_idx"].append(int(ev.get("sample_index", len(b["e"]) - 1)))
            except (TypeError, ValueError):
                b["bo_idx"].append(len(b["e"]) - 1)
            iface = ev.get("interface_frac")
            lat = ev.get("lattice")
            if isinstance(iface, list) and isinstance(lat, list):
                self._last_iface_frac = iface
                self._last_lat = lat
            pc = ev.get("positions_cart")
            nums = ev.get("numbers")
            fs = ev.get("film_substrate")
            if (
                isinstance(pc, list)
                and isinstance(nums, list)
                and len(pc) == len(nums)
                and len(pc) > 0
                and isinstance(lat, list)
            ):
                b["iface_positions"] = [[float(a), float(b), float(c)] for a, b, c in pc]
                b["iface_numbers"] = [int(x) for x in nums]
                b["iface_cell"] = [list(map(float, row)) for row in lat]
                if isinstance(fs, list) and len(fs) == len(nums):
                    b["iface_side"] = [int(x) for x in fs]
                else:
                    b["iface_side"] = None
        elif ev.get("event") == "relax_step":
            try:
                mk = int(ev.get("match_id", -1))
                tk = int(ev.get("term_id", -1))
            except (TypeError, ValueError):
                mk, tk = -1, -1
            rk: PairKey = (mk, tk)
            if rk not in self._relax_by_pair:
                self._relax_by_pair[rk] = {"step": [], "e_pa": [], "rms": [], "gamma_j_m2": []}
                self._relax_pair_order.append(rk)
            rb = self._relax_by_pair[rk]
            if "gamma_j_m2" not in rb:
                rb["gamma_j_m2"] = []
            try:
                rb["step"].append(int(ev.get("step", 0)))
                e = float(ev.get("energy", 0.0))
                epa = ev.get("energy_per_atom")
                if epa is not None:
                    rb["e_pa"].append(float(epa))
                else:
                    na = ev.get("n_atoms")
                    if isinstance(na, (int, float)) and float(na) > 0:
                        rb["e_pa"].append(e / float(na))
                    else:
                        rb["e_pa"].append(e)
                rb["rms"].append(float(ev.get("rms_displacement", 0.0)))
                gj = ev.get("interface_gamma_J_m2")
                rb["gamma_j_m2"].append(float(gj) if gj is not None else float("nan"))
            except (TypeError, ValueError):
                pass
            lab = ev.get("match_label")
            if isinstance(lab, str) and lab.strip():
                self._relax_label = lab.strip()
        elif ev.get("event") == "relax_final":
            try:
                mk = int(ev.get("match_id", -1))
                tk = int(ev.get("term_id", -1))
            except (TypeError, ValueError):
                mk, tk = -1, -1
            rk = (mk, tk)
            pos = ev.get("positions")
            nums = ev.get("numbers")
            cell = ev.get("cell")
            if isinstance(pos, list) and isinstance(nums, list) and isinstance(cell, list) and len(nums) >= 1:
                self._relax_final_snapshots.append(
                    (
                        rk,
                        {
                            "positions": pos,
                            "numbers": nums,
                            "cell": cell,
                            "rms_displacement": ev.get("rms_displacement"),
                            "interface_gamma_J_m2": ev.get("interface_gamma_J_m2"),
                            "energy_per_atom": ev.get("energy_per_atom"),
                        },
                    )
                )

    def _strip_plot_artists(self) -> None:
        """Remove colorbars / twinx from last draw so redraw does not stack layers."""
        if self._fig_title_artist is not None:
            try:
                self._fig_title_artist.remove()
            except Exception:
                pass
            self._fig_title_artist = None
        if self._cbar_reg is not None:
            try:
                self._cbar_reg.remove()
            except Exception:
                pass
            self._cbar_reg = None
        if self._relax_twin_ax is not None:
            try:
                self.fig.delaxes(self._relax_twin_ax)
            except Exception:
                pass
            self._relax_twin_ax = None

    def _effective_active_pair(self) -> Optional[PairKey]:
        if self._active_pair is not None and self._active_pair in self._bo_pairs:
            return self._active_pair
        if self._pair_order:
            return self._pair_order[-1]
        return None

    def _active_bo_bucket(self) -> Optional[Dict[str, Any]]:
        k = self._effective_active_pair()
        if k is None:
            return None
        return self._bo_pairs.get(k)

    def _redraw(self) -> None:
        self._strip_plot_artists()

        ak = self._effective_active_pair()
        b = self._active_bo_bucket()
        pair_lbl = ""
        if ak is not None:
            pair_lbl = f"match {ak[0]} · term {ak[1]}"

        iface_title = ""

        # --- BO registration 3D (active pair only; color = per-atom energy) — no axes / colorbar
        self.ax_reg.clear()
        if b and b.get("xyz_cart") and (b.get("e_per_atom") or b.get("e")):
            X = np.array(b["xyz_cart"], dtype=float)
            if len(b["e_per_atom"]) >= len(X):
                E = np.array(b["e_per_atom"][: len(X)], dtype=float)
            elif len(b["e"]) >= len(X):
                E = np.array(b["e"][: len(X)], dtype=float)
            else:
                m = min(len(X), len(b["e"]))
                X, E = X[:m], np.array(b["e"][:m], dtype=float)
            self.ax_reg.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=E,
                cmap="viridis",
                s=36,
                depthshade=True,
            )
            j = int(np.argmin(E))
            self.ax_reg.scatter(
                [X[j, 0]],
                [X[j, 1]],
                [X[j, 2]],
                c="red",
                s=120,
                marker="*",
                depthshade=False,
            )
        try:
            self.ax_reg.set_axis_off()
            self.ax_reg.dist = 12
        except Exception:
            pass

        # --- BO interface: ASE 2D (active pair last sample only); title shared via fig.text
        self.ax_iface.clear()
        si = int(b["bo_idx"][-1]) if b and b.get("bo_idx") else 0
        epa = float(b["e_per_atom"][-1]) if b and b.get("e_per_atom") else float("nan")
        iface_title = f"BO: interface — active {pair_lbl}  sample {si}  E/N={epa:.4f}"
        ipos = b.get("iface_positions") if b else None
        inum = b.get("iface_numbers") if b else None
        icell = b.get("iface_cell") if b else None
        iside = b.get("iface_side") if b else None
        if ipos and inum and icell:
            ok = _plot_interface_ase_orthographic(
                self.ax_iface,
                ipos,
                inum,
                icell,
                iside,
                title=iface_title,
                show_title=False,
                show_edge_caption=False,
            )
            if not ok:
                ok = _plot_interface_fallback_2d(
                    self.ax_iface,
                    ipos,
                    inum,
                    icell,
                    iside,
                    title=iface_title,
                    show_title=False,
                    show_edge_caption=False,
                )
            if not ok:
                self.ax_iface.text(
                    0.5,
                    0.5,
                    "Install ASE to draw the interface (plot_atoms).",
                    ha="center",
                    va="center",
                    transform=self.ax_iface.transAxes,
                    fontsize=9,
                )
        elif self._last_iface_frac and self._last_lat and b:
            iface_title = f"BO: interface (legacy atoms) — {pair_lbl}"
            C = _frac_to_cart(self._last_iface_frac, self._last_lat)
            if C.size:
                self.ax_iface.scatter(
                    C[:, 0], C[:, 1], c="steelblue", s=8, alpha=0.85
                )
                try:
                    self.ax_iface.set_aspect("equal")
                except Exception:
                    pass
            self.ax_iface.set_xlabel("x (Angstrom)")
            self.ax_iface.set_ylabel("y (Angstrom)")
        else:
            iface_title = "BO: interface (waiting for active pair samples)"
            self.ax_iface.set_xlabel("x (Angstrom)")
            self.ax_iface.set_ylabel("y (Angstrom)")

        # --- BO energy vs index — active pair only
        self.ax_ebo.clear()
        if b and b.get("e_per_atom"):
            Epa = np.array(b["e_per_atom"], dtype=float)
            x = list(range(len(Epa)))
            self.ax_ebo.plot(x, Epa, "o-", ms=3, lw=1.2, color="tab:blue")
            self.ax_ebo.set_xlabel("sample # (this pair)")
            self.ax_ebo.set_ylabel("E/N (eV/atom)")
            self.ax_ebo.set_title(f"BO: E/N — active {pair_lbl} (low band)")
            y0, y1 = _ylim_focus_low_band(Epa)
            self.ax_ebo.set_ylim(y0, y1)
            self.ax_ebo.grid(True, alpha=0.3)
        elif b and b.get("e"):
            x = list(range(len(b["e"])))
            self.ax_ebo.plot(x, b["e"], "o-", ms=3, lw=1)
            self.ax_ebo.set_xlabel("sample # (this pair)")
            self.ax_ebo.set_ylabel("E (eV)")
            self.ax_ebo.set_title(f"BO: total E — active {pair_lbl} (no n_atoms)")
            self.ax_ebo.grid(True, alpha=0.3)
        else:
            self.ax_ebo.set_title("BO: E/N — active pair (no samples yet)")
            self.ax_ebo.set_xlabel("sample # (this pair)")
            self.ax_ebo.set_ylabel("E/N (eV/atom)")

        # --- All (match, term) pairs: E/N vs sample index within each pair (history + current)
        self.ax_ebo_all.clear()
        all_vals: List[float] = []
        try:
            cmap = matplotlib.colormaps["tab20"]
        except Exception:
            cmap = matplotlib.cm.get_cmap("tab20")
        for pi, key in enumerate(self._pair_order):
            pd = self._bo_pairs.get(key)
            if not pd or not pd.get("e_per_atom"):
                continue
            ep = np.array(pd["e_per_atom"], dtype=float)
            if ep.size == 0:
                continue
            all_vals.extend(ep.tolist())
            x = np.arange(len(ep))
            label = f"M{key[0]} T{key[1]}"
            is_act = ak is not None and key == ak
            col = cmap(pi % 20)
            self.ax_ebo_all.plot(
                x,
                ep,
                "o-",
                ms=2,
                lw=2.2 if is_act else 1.0,
                alpha=1.0 if is_act else 0.42,
                color=col,
                label=label,
            )
        if all_vals:
            y0, y1 = _ylim_focus_low_band(np.array(all_vals, dtype=float))
            self.ax_ebo_all.set_ylim(y0, y1)
        self.ax_ebo_all.set_xlabel("sample # within pair")
        self.ax_ebo_all.set_ylabel("E/N (eV/atom)")
        self.ax_ebo_all.set_title("BO: all (match, term) pairs — bold = active, faint = others")
        self.ax_ebo_all.grid(True, alpha=0.25)
        hs, ls = self.ax_ebo_all.get_legend_handles_labels()
        if hs:
            nmax = 14
            if len(hs) > nmax:
                hs, ls = hs[-nmax:], ls[-nmax:]
            self.ax_ebo_all.legend(hs, ls, loc="upper right", fontsize=6, framealpha=0.9)

        # --- Relaxation: left = interface energy γ (J/m²) when available, else E/N; right = RMSD
        self.ax_relax.clear()
        rt = self._relax_label or ""
        self.ax_relax.set_title(f"Relax (MLIP)  {rt}" if rt else "Relax (MLIP)")
        try:
            cmap = matplotlib.colormaps["tab20"]
        except Exception:
            cmap = matplotlib.cm.get_cmap("tab20")
        any_relax = False
        use_gamma_axis = False
        for key in self._relax_pair_order:
            rb = self._relax_by_pair.get(key)
            if not rb or not rb.get("e_pa"):
                continue
            ep = np.array(rb["e_pa"], dtype=float)
            st = np.array(rb["step"], dtype=int)
            gjm = np.asarray(rb.get("gamma_j_m2") or [], dtype=float)
            if ep.size == 0 or st.size == 0:
                continue
            n = min(ep.size, st.size)
            if gjm.size >= n:
                gjm = gjm[:n]
                if np.any(np.isfinite(gjm)):
                    use_gamma_axis = True
                    break
        for pi, key in enumerate(self._relax_pair_order):
            rb = self._relax_by_pair.get(key)
            if not rb or not rb.get("e_pa"):
                continue
            ep = np.array(rb["e_pa"], dtype=float)
            st = np.array(rb["step"], dtype=int)
            gjm = np.asarray(rb.get("gamma_j_m2") or [], dtype=float)
            if ep.size == 0 or st.size == 0:
                continue
            n = min(ep.size, st.size)
            ep, st = ep[:n], st[:n]
            if gjm.size >= n:
                gjm = gjm[:n]
            else:
                gjm = np.array([], dtype=float)
            any_relax = True
            label = f"M{key[0]} T{key[1]}"
            if use_gamma_axis and gjm.size == n and np.any(np.isfinite(gjm)):
                y = gjm
            else:
                y = ep
            self.ax_relax.plot(
                st,
                y,
                "o-",
                ms=2,
                lw=1.2,
                color=cmap(pi % 20),
                label=label,
            )
        if any_relax:
            self.ax_relax.set_xlabel("step (within this relax)")
            if use_gamma_axis:
                self.ax_relax.set_ylabel(r"Interface energy $\gamma$ (J/m$^2$)", color="b")
            else:
                self.ax_relax.set_ylabel("E/N (eV/atom)", color="b")
            self.ax_relax.tick_params(axis="y", labelcolor="b")
            ax2 = self.ax_relax.twinx()
            self._relax_twin_ax = ax2
            for pi, key in enumerate(self._relax_pair_order):
                rb = self._relax_by_pair.get(key)
                if not rb or not rb.get("rms"):
                    continue
                rms = np.array(rb["rms"], dtype=float)
                st = np.array(rb["step"], dtype=int)
                if rms.size == 0:
                    continue
                n = min(rms.size, st.size)
                ax2.plot(
                    st[:n],
                    rms[:n],
                    "-",
                    lw=0.8,
                    alpha=0.45,
                    color=cmap(pi % 20),
                )
            ax2.set_ylabel("RMSD (Å)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            self.ax_relax.grid(True, alpha=0.3)
            hs, ls = self.ax_relax.get_legend_handles_labels()
            if hs:
                nmax = 12
                if len(hs) > nmax:
                    hs, ls = hs[-nmax:], ls[-nmax:]
                self.ax_relax.legend(hs, ls, loc="upper right", fontsize=6, framealpha=0.92)
        else:
            self.ax_relax.text(
                0.5,
                0.5,
                "Waiting for relax phase…",
                ha="center",
                va="center",
                transform=self.ax_relax.transAxes,
            )

        # --- Relaxed structures: last 6 finals, ASE 2D (same edge-on style as interface)
        snaps = self._relax_final_snapshots[-6:]
        for i, axp in enumerate(self._ax_rf):
            axp.clear()
            axp.set_axis_on()
            if i >= len(snaps):
                axp.set_axis_off()
                continue
            key, data = snaps[i]
            title = f"relax final  M{key[0]} T{key[1]}"
            flat = data.get("positions") or []
            nums = data.get("numbers") or []
            cell = data.get("cell") or []
            if not flat or not nums or not cell:
                axp.set_title(title)
                axp.text(0.5, 0.5, "no coords", ha="center", va="center", transform=axp.transAxes)
                continue
            ncoord = len(flat) // 3
            n = min(ncoord, len(nums))
            if n < 1:
                axp.set_title(title)
                continue
            pos = [[float(flat[3 * j + k]) for k in range(3)] for j in range(n)]
            nums_i = [int(nums[j]) for j in range(n)]
            ok = _plot_interface_ase_orthographic(
                axp,
                pos,
                nums_i,
                [list(map(float, row)) for row in cell],
                None,
                title=title,
                show_title=False,
                show_axis_labels=False,
                show_edge_caption=False,
                scale=0.62,
            )
            if not ok:
                ok = _plot_interface_fallback_2d(
                    axp,
                    pos,
                    nums_i,
                    [list(map(float, row)) for row in cell],
                    None,
                    title=title,
                    show_title=False,
                    show_axis_labels=False,
                    show_edge_caption=False,
                )
            axp.tick_params(labelsize=6)
            parts: List[str] = []
            ig = data.get("interface_gamma_J_m2")
            if ig is not None:
                try:
                    parts.append(f"γ = {float(ig):.6g} J/m²")
                except (TypeError, ValueError):
                    pass
            rms = data.get("rms_displacement")
            if rms is not None:
                try:
                    parts.append(f"RMSD = {float(rms):.4f} Å")
                except (TypeError, ValueError):
                    pass
            if parts:
                axp.text(
                    0.5,
                    -0.18,
                    "  ·  ".join(parts),
                    transform=axp.transAxes,
                    ha="center",
                    fontsize=5.5,
                    color="#475569",
                )

        try:
            if self.ax_reg.get_axis_on():
                _style_axes_3d(self.ax_reg)
        except Exception:
            pass
        for ax in (self.ax_iface, self.ax_ebo, self.ax_ebo_all, self.ax_relax):
            _style_axes_2d(ax)
        for axp in self._ax_rf:
            if axp.get_visible():
                _style_axes_2d(axp)

        self.fig.tight_layout(pad=1.35, h_pad=1.05, w_pad=0.75)
        if iface_title:
            pos = self.ax_iface.get_position()
            self._fig_title_artist = self.fig.text(
                0.5,
                min(0.992, pos.y1 + 0.034),
                iface_title,
                transform=self.fig.transFigure,
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#0f172a",
                fontweight="600",
            )
        self.canvas.draw_idle()

    def destroy(self) -> None:
        self._on_user_close()
