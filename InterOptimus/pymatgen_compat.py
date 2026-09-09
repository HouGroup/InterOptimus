"""Compatibility helpers for the pinned pymatgen/pymatgen-core pair."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
from pymatgen.analysis.interfaces import CoherentInterfaceBuilder, SubstrateAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def coherent_interface_builder_for_match(
    *,
    substrate_structure: Any,
    film_structure: Any,
    match: Any,
    termination_ftol: float | tuple[float, float] = 0.25,
    label_index: bool = True,
    filter_out_sym_slabs: bool = False,
) -> CoherentInterfaceBuilder:
    """Build a CIB for exactly one existing SubstrateAnalyzer match."""

    builder = CoherentInterfaceBuilder(
        substrate_structure=substrate_structure,
        film_structure=film_structure,
        film_miller=tuple(int(value) for value in match.film_miller),
        substrate_miller=tuple(int(value) for value in match.substrate_miller),
        zslgen=SubstrateAnalyzer(max_area=100),
        termination_ftol=termination_ftol,
        label_index=label_index,
        filter_out_sym_slabs=filter_out_sym_slabs,
    )
    # CIB's own search validates against a separately reduced surface basis and
    # rejects some valid external SubstrateAnalyzer matches for non-orthogonal
    # cells. Preserve InterOptimus's established selected-match behavior here,
    # in one version-tested compatibility boundary.
    film_vectors = zsl_vectors_in_crystal_frame(
        film_structure,
        match.film_miller,
        np.vstack((match.film_vectors, match.film_sl_vectors)),
    )
    substrate_vectors = zsl_vectors_in_crystal_frame(
        substrate_structure,
        match.substrate_miller,
        np.vstack((match.substrate_vectors, match.substrate_sl_vectors)),
    )
    builder.zsl_matches = [
        replace(
            match,
            film_vectors=film_vectors[:2],
            film_sl_vectors=film_vectors[2:],
            substrate_vectors=substrate_vectors[:2],
            substrate_sl_vectors=substrate_vectors[2:],
        )
    ]
    return builder


def zsl_vectors_in_crystal_frame(
    structure: Any,
    miller_index: tuple[int, ...],
    vectors: Any,
) -> np.ndarray:
    """Undo the slab display rotation applied during ZSL surface generation."""

    generator = SlabGenerator(
        structure,
        miller_index,
        min_slab_size=20,
        min_vacuum_size=15,
        primitive=False,
    )
    crystal_lattice = generator.oriented_unit_cell.lattice.matrix
    displayed_lattice = generator.get_slab().oriented_unit_cell.lattice.matrix
    rotation = np.linalg.solve(crystal_lattice, displayed_lattice)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise RuntimeError("pymatgen slab reorientation is not a rigid rotation")
    return np.asarray(vectors, dtype=float) @ np.linalg.inv(rotation)


def symmetrically_equivalent_millers(
    structure: Any,
    miller_index: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    """Return equivalent hkl planes in the structure's current lattice basis.

    Pymatgen's surface helper changes trigonal structures to a primitive
    standard basis but does not transform the supplied Miller index with it.
    Applying public fractional symmetry operations directly keeps indices and
    symmetry matrices in the same basis, including hexagonal R settings.
    """

    miller = np.asarray(
        (miller_index[0], miller_index[1], miller_index[-1]),
        dtype=int,
    )
    equivalent: list[tuple[int, int, int]] = []
    for operation in SpacegroupAnalyzer(structure).get_symmetry_operations(
        cartesian=False
    ):
        transformed = np.linalg.solve(operation.rotation_matrix.T, miller)
        integral = np.rint(transformed).astype(int)
        if not np.allclose(transformed, integral, atol=1e-8):
            raise RuntimeError(
                "A crystallographic symmetry operation produced a non-integral "
                f"Miller index: {transformed}"
            )
        result = tuple(int(value) for value in integral)
        if result not in equivalent:
            equivalent.append(result)
    return equivalent


def termination_shift_map(builder: CoherentInterfaceBuilder) -> dict[Any, tuple[float, float]]:
    """Reconstruct public termination labels to slab shifts without ``_terminations``."""

    film_sg = SlabGenerator(
        builder.film_structure,
        builder.film_miller,
        min_slab_size=1,
        min_vacuum_size=3,
        in_unit_planes=True,
        center_slab=True,
        primitive=True,
        reorient_lattice=False,
    )
    substrate_sg = SlabGenerator(
        builder.substrate_structure,
        builder.substrate_miller,
        min_slab_size=1,
        min_vacuum_size=3,
        in_unit_planes=True,
        center_slab=True,
        primitive=True,
        reorient_lattice=False,
    )
    if isinstance(builder.termination_ftol, tuple):
        film_ftol, substrate_ftol = builder.termination_ftol
    else:
        film_ftol = substrate_ftol = builder.termination_ftol

    film_shifts = [
        float(slab.shift)
        for slab in film_sg.get_slabs(
            ftol=film_ftol,
            filter_out_sym_slabs=builder.filter_out_sym_slabs,
        )
    ]
    substrate_shifts = [
        float(slab.shift)
        for slab in substrate_sg.get_slabs(
            ftol=substrate_ftol,
            filter_out_sym_slabs=builder.filter_out_sym_slabs,
        )
    ]
    shift_pairs = list(product(film_shifts, substrate_shifts))
    if len(builder.terminations) != len(shift_pairs):
        raise RuntimeError(
            "pymatgen returned inconsistent termination labels and slab shifts "
            f"({len(builder.terminations)} labels, {len(shift_pairs)} shift pairs)"
        )
    return dict(zip(builder.terminations, shift_pairs, strict=True))


def slab_projected_height(slab_generator: SlabGenerator) -> float:
    """Compute the oriented-cell height using public lattice geometry."""

    lattice = slab_generator.oriented_unit_cell.lattice
    base_area = float(np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1])))
    if base_area <= 0:
        raise ValueError("The oriented unit cell has zero in-plane area")
    return float(lattice.volume / base_area)
