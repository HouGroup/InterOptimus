"""Regression tests for the supported pymatgen 2026.5/2026.8 pair."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import SlabGenerator

from InterOptimus.CNID import calculate_cnid_in_supercell
from InterOptimus.equi_term import pair_fit
from InterOptimus.itworker import InterfaceWorker
from InterOptimus.pymatgen_compat import (
    coherent_interface_builder_for_match,
    slab_projected_height,
    symmetrically_equivalent_millers,
    termination_shift_map,
)
from InterOptimus.tool import get_film_length


def _simple_cubic_structure() -> Structure:
    return Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])


class TestPymatgenCompatibility(unittest.TestCase):
    def test_public_miller_equivalence_always_returns_hkl(self) -> None:
        millers = symmetrically_equivalent_millers(
            _simple_cubic_structure(),
            (1, 0, 0),
        )

        self.assertTrue(millers)
        self.assertTrue(all(len(miller) == 3 for miller in millers))
        self.assertIn((1, 0, 0), millers)

    def test_projected_height_uses_oriented_cell_geometry(self) -> None:
        slab_generator = SlabGenerator(
            _simple_cubic_structure(),
            (1, 1, 1),
            min_slab_size=1,
            min_vacuum_size=3,
            primitive=True,
        )
        lattice = slab_generator.oriented_unit_cell.lattice
        expected = lattice.volume / np.linalg.norm(
            np.cross(lattice.matrix[0], lattice.matrix[1])
        )

        projected_height = slab_projected_height(slab_generator)
        self.assertAlmostEqual(projected_height, expected)
        self.assertAlmostEqual(projected_height, slab_generator._proj_height)

    def test_single_match_builder_and_public_termination_mapping(self) -> None:
        structure = _simple_cubic_structure()
        analyzer = SubstrateAnalyzer(
            max_area=20,
            max_length_tol=0.05,
            max_angle_tol=0.05,
            film_max_miller=1,
            substrate_max_miller=1,
        )
        match = next(
            analyzer.calculate(
                film=structure,
                substrate=structure,
                film_millers=[(0, 0, 1)],
                substrate_millers=[(0, 0, 1)],
            )
        )

        builder = coherent_interface_builder_for_match(
            film_structure=structure,
            substrate_structure=structure,
            match=match,
            termination_ftol=0.15,
        )
        shifts = termination_shift_map(builder)

        self.assertEqual(len(builder.zsl_matches), 1)
        self.assertEqual(builder.zsl_matches[0].film_miller, match.film_miller)
        np.testing.assert_allclose(
            builder.zsl_matches[0].film_transformation,
            match.film_transformation,
        )
        self.assertEqual(set(shifts), set(builder.terminations))
        self.assertTrue(
            all(np.isfinite(value) for pair in shifts.values() for value in pair)
        )
        for termination, shift_pair in shifts.items():
            np.testing.assert_allclose(
                shift_pair,
                builder._terminations[termination],
                atol=1e-12,
            )

    def test_nonorthogonal_substrate_match_is_accepted_by_builder(self) -> None:
        film = Structure(
            Lattice.from_parameters(14.59, 5.86, 5.10, 71.68, 71.85, 58.62),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        substrate = Structure(
            Lattice.cubic(10.25),
            ["Si", "Si"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        match = next(
            SubstrateAnalyzer(
                max_area=300,
                max_length_tol=0.1,
                max_angle_tol=0.1,
                film_max_miller=3,
                substrate_max_miller=3,
            ).calculate(
                film=film,
                substrate=substrate,
                film_millers=[(0, 0, 1)],
            )
        )

        builder = coherent_interface_builder_for_match(
            film_structure=film,
            substrate_structure=substrate,
            match=match,
            termination_ftol=0.15,
        )

        self.assertEqual(len(builder.zsl_matches), 1)
        self.assertEqual(builder.zsl_matches[0].film_miller, match.film_miller)
        self.assertTrue(builder.terminations)

    def test_generated_interface_preserves_modeling_invariants(self) -> None:
        film = Structure(
            Lattice.cubic(4.21),
            ["Mg", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        substrate = Structure(
            Lattice.cubic(3.905),
            ["Sr", "Ti", "O", "O", "O"],
            [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ],
        )
        matches = list(
            SubstrateAnalyzer(
                max_area=100,
                max_length_tol=0.1,
                max_angle_tol=0.05,
                film_max_miller=1,
                substrate_max_miller=1,
            ).calculate(
                film=film,
                substrate=substrate,
                film_millers=[(0, 0, 1)],
                substrate_millers=[(0, 0, 1)],
            )
        )
        match = min(
            matches,
            key=lambda item: (item.von_mises_strain, item.match_area),
        )
        builder = coherent_interface_builder_for_match(
            film_structure=film,
            substrate_structure=substrate,
            match=match,
            termination_ftol=0.15,
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(len(builder.terminations), 4)
        for termination in builder.terminations:
            interface = next(
                builder.get_interfaces(
                    termination=termination,
                    substrate_thickness=2,
                    film_thickness=2,
                    vacuum_over_film=10,
                    gap=2,
                    in_layers=True,
                )
            )
            film_ids = {int(index) for index in interface.film_indices}
            substrate_ids = {int(index) for index in interface.substrate_indices}
            film_z = interface.cart_coords[sorted(film_ids), 2]
            substrate_z = interface.cart_coords[sorted(substrate_ids), 2]
            cnid = np.asarray(calculate_cnid_in_supercell(interface)[0])

            self.assertFalse(film_ids.intersection(substrate_ids))
            self.assertEqual(film_ids.union(substrate_ids), set(range(len(interface))))
            self.assertEqual(len(interface.film), len(film_ids))
            self.assertEqual(len(interface.substrate), len(substrate_ids))
            self.assertAlmostEqual(float(np.min(film_z) - np.max(substrate_z)), 2.0)
            self.assertGreater(interface.volume, 0)
            self.assertTrue(np.isfinite(cnid).all())
            self.assertEqual(np.linalg.matrix_rank(cnid), 2)

    def test_no_matches_fail_before_plotting_or_termination_generation(self) -> None:
        worker = InterfaceWorker(
            _simple_cubic_structure(),
            _simple_cubic_structure(),
        )
        with patch(
            "InterOptimus.itworker.interface_searching",
            return_value=([], [], [], [], []),
        ):
            with self.assertRaisesRegex(ValueError, "No lattice matches"):
                worker.lattice_matching()

    def test_interface_worker_end_to_end_without_notebook_widgets(self) -> None:
        structure = _simple_cubic_structure()
        worker = InterfaceWorker(structure, structure)
        worker.lattice_matching(
            max_area=10,
            max_length_tol=0.05,
            max_angle_tol=0.05,
            film_max_miller=1,
            substrate_max_miller=1,
            film_millers=[(0, 0, 1)],
            substrate_millers=[(0, 0, 1)],
        )
        worker.parse_interface_structure_params(
            termination_ftol=0.15,
            film_thickness=8,
            substrate_thickness=8,
            vacuum_over_film=5,
        )

        interface = worker.get_specified_interface(0, 0)

        self.assertEqual(len(worker.unique_matches), 1)
        self.assertEqual([len(items) for items in worker.all_unique_terminations], [1])
        self.assertEqual(len(interface), 6)
        self.assertEqual(len(interface.film_indices), 3)
        self.assertEqual(len(interface.substrate_indices), 3)

    def test_substrate_equivalence_uses_cartesian_rotation(self) -> None:
        class SlabStub:
            def __init__(self) -> None:
                self.lattice = SimpleNamespace(matrix=np.eye(3))

            def __eq__(self, other) -> bool:
                return self is other

        film = SlabStub()
        substrate_reference = SlabStub()
        substrate_comparison = SlabStub()
        matcher = MagicMock()
        matcher.get_transformation.return_value = (
            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            None,
            None,
        )
        identity = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
        analyzer = MagicMock()
        analyzer.get_point_group_operations.return_value = [identity]

        with patch(
            "InterOptimus.equi_term.get_rotation_from_match",
            return_value=np.eye(3),
        ), patch(
            "InterOptimus.equi_term.SpacegroupAnalyzer",
            return_value=analyzer,
        ):
            self.assertTrue(
                pair_fit(
                    film,
                    substrate_reference,
                    film,
                    substrate_comparison,
                    matcher,
                    c_periodic=True,
                )
            )

    def test_polyatomic_film_length_counts_unit_cells_not_atomic_planes(self) -> None:
        u = 0.375
        wurtzite = Structure(
            Lattice.hexagonal(3, 5),
            ["Zn", "Zn", "O", "O"],
            [
                [0, 0, 0],
                [2 / 3, 1 / 3, 0.5],
                [0, 0, u],
                [2 / 3, 1 / 3, 0.5 + u],
            ],
        )
        match = min(
            SubstrateAnalyzer(
                max_area=10,
                max_length_tol=0.03,
                max_angle_tol=0.01,
                film_max_miller=1,
                substrate_max_miller=1,
            ).calculate(
                film=wurtzite,
                substrate=wurtzite,
                film_millers=[(0, 0, 1)],
                substrate_millers=[(0, 0, 1)],
            ),
            key=lambda item: item.match_area,
        )
        builder = coherent_interface_builder_for_match(
            film_structure=wurtzite,
            substrate_structure=wurtzite,
            match=match,
            termination_ftol=0.15,
        )
        interface = next(
            builder.get_interfaces(
                termination=builder.terminations[0],
                substrate_thickness=1,
                film_thickness=1,
                vacuum_over_film=10,
                gap=2,
                in_layers=True,
            )
        )

        self.assertEqual(interface.film_layers, 2)
        self.assertAlmostEqual(get_film_length(match, wurtzite, interface), 5.0)

    def test_trigonal_conventional_miller_end_to_end(self) -> None:
        film = Structure.from_spacegroup(
            "R-3m",
            Lattice.hexagonal(2.815, 14.05),
            ["Li", "Co", "O"],
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.241]],
        )
        substrate = Structure.from_spacegroup(
            "R-3m",
            Lattice.hexagonal(2.84, 14.15),
            ["Li", "Co", "O"],
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.241]],
        )

        self.assertEqual(
            set(symmetrically_equivalent_millers(film, (0, 0, 1))),
            {(0, 0, 1), (0, 0, -1)},
        )

        worker = InterfaceWorker(film, substrate)
        worker.lattice_matching(
            max_area=10,
            max_length_tol=0.08,
            max_angle_tol=0.03,
            film_max_miller=1,
            substrate_max_miller=1,
            film_millers=[(0, 0, 1)],
            substrate_millers=[(0, 0, 1)],
        )
        np.testing.assert_array_equal(
            worker.unique_matches_indices_data[0]["film_conventional_miller"],
            [0, 0, 1],
        )
        worker.parse_interface_structure_params(
            termination_ftol=0.15,
            film_thickness=14,
            substrate_thickness=14,
            vacuum_over_film=12,
        )
        interface = worker.get_specified_interface(0, 0)
        film_indices = set(interface.film_indices)
        substrate_indices = set(interface.substrate_indices)

        self.assertEqual(len(worker.unique_matches), 1)
        self.assertTrue(worker.all_unique_terminations[0])
        self.assertFalse(film_indices & substrate_indices)
        self.assertEqual(
            film_indices | substrate_indices,
            set(range(len(interface))),
        )
        self.assertTrue(np.isfinite(interface.lattice.matrix).all())


if __name__ == "__main__":
    unittest.main()
