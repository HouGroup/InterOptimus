"""
CIF helpers for the web UI's structure viewer.

Materials-Project-style rendering wants every atom inside the cell box plus
periodic-image copies on faces / edges / corners.  3Dmol.js's ``doAssembly``
flag instead applies symmetry operations to the asymmetric unit and frequently
emits atoms outside ``[0, 1)`` fractional coords. To avoid that, we pre-process
the CIF on the server with pymatgen:

1. Parse the CIF (non-primitive: keep the conventional cell as written).
2. Wrap fractional coordinates into ``[0, 1)``.
3. For atoms sitting on a cell face (``frac_coord ≈ 0`` along one or more axes),
   add their ``+1`` images along those axes (so faces show their atoms, edges
   their two extra copies, corners their seven extra copies).
4. Strip oxidation-state decorators (``Ni1.33+``) so 3Dmol's CIF parser sees
   plain element symbols.
5. Re-emit a ``P1`` CIF with explicit atoms.

Falls back to the original CIF text on any failure (missing pymatgen, corrupt
CIF, etc.).
"""

from __future__ import annotations

from itertools import product
from typing import List

import numpy as np


def _parse_structure(cif_text: str):
    try:
        from pymatgen.io.cif import CifParser
    except ImportError:
        return None

    parser = None
    try:
        parser = CifParser.from_str(cif_text)
    except AttributeError:
        # Older pymatgen
        try:
            parser = CifParser.from_string(cif_text)  # type: ignore[attr-defined]
        except Exception:
            return None
    except Exception:
        return None

    structures = None
    for method in ("parse_structures", "get_structures"):
        fn = getattr(parser, method, None)
        if fn is None:
            continue
        for kwargs in ({"primitive": False}, {}):
            try:
                structures = fn(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                structures = None
                break
        if structures:
            break
    if not structures:
        return None
    return structures[0]


def _strip_oxidation(structure):
    try:
        structure.remove_oxidation_states()
    except Exception:
        pass
    return structure


def _compute_bonds(structure, *, max_strategies=("CrystalNN", "MinimumDistanceNN")):
    """
    Run pymatgen local-environment analysis on the wrapped (un-expanded)
    structure to produce a list of directed bonds:

        (i, j, image, distance)

    where ``i`` / ``j`` are site indices and ``image = (dx, dy, dz)`` is the
    integer translation applied to site ``j`` so that the bond ``i → j+image``
    is the actual bond drawn.  Mirrors Materials Project's CrystalNN-based
    bonding for a faithful look (with a JmolNN/MinimumDistanceNN fallback for
    structures CrystalNN cannot resolve, e.g. very ionic or radical cases).
    """
    bonds: List = []
    try:
        from pymatgen.analysis import local_env as le
    except Exception:
        return bonds

    strategies = []
    for name in max_strategies:
        try:
            cls = getattr(le, name)
            if name == "MinimumDistanceNN":
                strategies.append((name, cls(tol=0.4)))
            else:
                strategies.append((name, cls()))
        except Exception:
            continue

    for name, nn in strategies:
        try:
            collected: List = []
            for i in range(len(structure)):
                try:
                    info = nn.get_nn_info(structure, i)
                except Exception:
                    continue
                for nb in info:
                    j = int(nb["site_index"])
                    img = tuple(int(round(x)) for x in nb["image"])
                    if (i, j, img) in collected:
                        continue
                    if (j, i, tuple(-x for x in img)) in collected:
                        continue
                    try:
                        dist = float(structure.get_distance(i, j, jimage=list(img)))
                    except Exception:
                        try:
                            dist = float(nb.get("dist", 0.0)) or float(np.linalg.norm(
                                structure.lattice.get_cartesian_coords(
                                    structure[j].frac_coords + np.asarray(img) - structure[i].frac_coords
                                )
                            ))
                        except Exception:
                            continue
                    if dist <= 0.05 or dist > 5.0:
                        continue
                    collected.append((i, j, img, dist))
            if collected:
                return collected
        except Exception:
            continue
    return bonds


def _bond_label_lookup(species_list, frac_coords_list, *, tol=2e-3):
    """
    Build a quick spatial index that maps a fractional coordinate (any cell shift)
    to the atom-site label (``Li0`` / ``Li1`` / ...) used in the emitted CIF
    when image atoms are added to the expanded structure.
    """
    labels: List[str] = []
    counters: dict = {}
    for sp in species_list:
        sym = getattr(sp, "symbol", str(sp))
        counters[sym] = counters.get(sym, 0)
        labels.append(f"{sym}{counters[sym]}")
        counters[sym] += 1

    def find(target_frac):
        target = np.asarray(target_frac, dtype=float)
        best_idx = -1
        best_d = tol
        for k, c in enumerate(frac_coords_list):
            d = np.linalg.norm(np.asarray(c) - target)
            if d < best_d:
                best_d = d
                best_idx = k
        return labels[best_idx] if best_idx >= 0 else None

    return labels, find


def _format_bond_loop(bond_pairs):
    """Format a list of (label1, label2, distance) into a CIF ``_geom_bond`` loop."""
    if not bond_pairs:
        return ""
    lines = [
        "loop_",
        " _geom_bond_atom_site_label_1",
        " _geom_bond_atom_site_label_2",
        " _geom_bond_distance",
        " _geom_bond_site_symmetry_2",
    ]
    for l1, l2, d in bond_pairs:
        lines.append(f"  {l1}  {l2}  {d:.5f}  .")
    return "\n".join(lines) + "\n"


def expand_cif_for_view(cif_text: str, *, tol: float = 1e-3, with_bonds: bool = True) -> str:
    """
    Expand a CIF for the manage-page structure viewer (see module docstring).

    With ``with_bonds=True`` (the default), CrystalNN-derived bond pairs are
    written into the emitted CIF as a ``_geom_bond`` loop so that 3Dmol.js draws
    the same bonds Materials Project would (instead of relying on its built-in
    distance-only auto-bonding which mis-handles ionic crystals like Li2S).
    """
    s = _parse_structure(cif_text)
    if s is None:
        return cif_text
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
    except ImportError:
        return cif_text

    try:
        species = list(s.species)
        fc = np.mod(np.asarray(s.frac_coords, dtype=float), 1.0)
        wrapped = Structure(s.lattice, species, fc, coords_are_cartesian=False)
        bonds = _compute_bonds(wrapped) if with_bonds else []

        new_species: List = list(species)
        new_coords: List = [list(c) for c in fc]
        for sp, coord in zip(species, fc):
            near_zero = [i for i in range(3) if coord[i] < tol]
            if not near_zero:
                continue
            axis_options = [(0, 1) if i in near_zero else (0,) for i in range(3)]
            for shift in product(*axis_options):
                if shift == (0, 0, 0):
                    continue
                new_species.append(sp)
                new_coords.append(list(np.asarray(coord) + np.asarray(shift, dtype=float)))

        s2 = Structure(s.lattice, new_species, new_coords, coords_are_cartesian=False)
        s2 = _strip_oxidation(s2)
        try:
            writer = CifWriter(s2, symprec=None)
        except TypeError:
            writer = CifWriter(s2)
        cif_out = str(writer)

        if not bonds:
            return cif_out

        # Build a (label, find-by-frac-coord) map from the EXPANDED structure so
        # we can resolve a bond's "neighbor at image (dx,dy,dz)" to a real
        # atom-site label that exists in the emitted CIF.
        labels, find_label = _bond_label_lookup(new_species, new_coords)

        bond_pairs: List = []
        for (i, j, img, dist) in bonds:
            target_frac = np.asarray(fc[j]) + np.asarray(img, dtype=float)
            l1 = labels[i]
            # Self-image bond (same site index, non-zero image): only useful if the
            # destination point sits inside the expanded cell.
            l2 = find_label(target_frac)
            if l2 is None:
                # Look for the back-direction (j → i) too: keep the bond if EITHER
                # endpoint maps into the cell; the other endpoint is then off-cell
                # but we still draw the half-bond by faking a label that exists.
                l2 = labels[j] if all(0.0 - tol <= target_frac[k] <= 1.0 + tol for k in range(3)) else None
            if l2 is None:
                continue
            if l1 == l2:
                continue
            bond_pairs.append((l1, l2, dist))

        if bond_pairs:
            return cif_out + _format_bond_loop(bond_pairs)
        return cif_out
    except Exception:
        return cif_text


def poscar_to_cif(poscar_text: str) -> str:
    """
    Convert a VASP POSCAR / CONTCAR string to a viewer-ready P1 CIF.

    Used by the manage detail page so the relaxed interface POSCARs under
    ``pairs_best_it/match_*_term_*/best_it_POSCAR`` can be loaded into the
    same VESTA-style 3Dmol viewer as the bulk CIFs (with face/edge image
    atoms and CrystalNN bonds).
    """
    try:
        from pymatgen.core.structure import Structure
    except ImportError:
        return ""
    try:
        s = Structure.from_str(poscar_text, fmt="poscar")
    except Exception:
        return ""
    try:
        s = _strip_oxidation(s)
        cif_str = s.to(fmt="cif")
    except Exception:
        return ""
    return expand_cif_for_view(cif_str)


def cell_metadata(cif_text: str) -> dict:
    """Quick metadata for the side caption (formula, spacegroup, lattice params)."""
    s = _parse_structure(cif_text)
    if s is None:
        return {}
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        SpacegroupAnalyzer = None  # type: ignore
    out: dict = {}
    try:
        comp = s.composition
        out["formula"] = comp.reduced_formula
        out["formula_full"] = comp.formula
        out["num_sites"] = int(s.num_sites)
        latt = s.lattice
        out["a"] = round(float(latt.a), 4)
        out["b"] = round(float(latt.b), 4)
        out["c"] = round(float(latt.c), 4)
        out["alpha"] = round(float(latt.alpha), 3)
        out["beta"] = round(float(latt.beta), 3)
        out["gamma"] = round(float(latt.gamma), 3)
        out["volume"] = round(float(latt.volume), 3)
        out["density"] = round(float(s.density), 4)
    except Exception:
        pass
    if SpacegroupAnalyzer is not None:
        try:
            sg = SpacegroupAnalyzer(s, symprec=0.05)
            out["spacegroup_symbol"] = sg.get_space_group_symbol()
            out["spacegroup_number"] = int(sg.get_space_group_number())
            out["crystal_system"] = sg.get_crystal_system()
        except Exception:
            pass
    return out
