from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.core import Lattice, Structure

from InterOptimus.CNID import _cnid_float_to_rational
from InterOptimus.itworker import (
    BO_CLASH_PENALTY_EV,
    InterfaceWorker,
    bounded_gradient_step,
    gradient_descend,
)
from InterOptimus.jobflow import check_it_phase_stability
from InterOptimus.matching import equi_match_identifier
from InterOptimus.mlip import MlipCalc, get_optimizer


class TestWorkflowRobustness(unittest.TestCase):
    def test_cnid_rationalization_supports_commensurate_large_denominator(self) -> None:
        self.assertEqual(_cnid_float_to_rational(1 / 97), "1/97")

    def test_clashing_bo_sample_is_recorded_with_high_penalty(self) -> None:
        worker = InterfaceWorker.__new__(InterfaceWorker)
        worker.match_id_now = 0
        worker.term_id_now = 0
        worker.discut = 0.8
        worker.opt_results = {(0, 0): {"sampled_interfaces": []}}
        interface = SimpleNamespace()
        worker.get_specified_interface = MagicMock(return_value=interface)
        worker.get_interface_atom_indices = MagicMock(return_value=[0])
        worker.mc = MagicMock()

        with patch(
            "InterOptimus.itworker.get_min_nb_distance",
            return_value=0.2,
        ):
            energy = worker.sample_xyz_energy([0.0, 0.0, 1.0])

        self.assertEqual(energy, BO_CLASH_PENALTY_EV)
        self.assertEqual(worker.opt_results[(0, 0)]["sampled_interfaces"], [interface])
        worker.mc.calculate.assert_not_called()

    def test_unknown_optimizer_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Supported optimizers"):
            get_optimizer("not-an-optimizer")

    def test_layer_thickness_rejects_match_without_terminations(self) -> None:
        worker = InterfaceWorker.__new__(InterfaceWorker)
        worker.get_specified_match_cib = MagicMock(
            return_value=SimpleNamespace(terminations=[])
        )

        with self.assertRaisesRegex(ValueError, "no slab terminations"):
            worker.get_film_substrate_layer_thickness(3)

    def test_layer_thickness_probe_is_bounded(self) -> None:
        worker = InterfaceWorker.__new__(InterfaceWorker)
        builder = MagicMock()
        builder.terminations = [("film", "substrate")]
        builder.get_interfaces.side_effect = lambda **kwargs: iter(
            [SimpleNamespace(lattice=SimpleNamespace(c=10.0))]
        )
        worker.get_specified_match_cib = MagicMock(return_value=builder)

        with self.assertRaisesRegex(RuntimeError, "after 20 probes"):
            worker.get_film_substrate_layer_thickness(0)

        self.assertEqual(builder.get_interfaces.call_count, 40)

    def test_all_clashing_bo_samples_fail_with_context(self) -> None:
        worker = InterfaceWorker.__new__(InterfaceWorker)

        def fake_minimizer(active_worker, n_calls, z_range):
            active_worker.opt_results[(0, 0)]["sampled_interfaces"] = [
                SimpleNamespace() for _ in range(10)
            ]
            return SimpleNamespace(
                x_iters=[[0.0, 0.0, 1.0] for _ in range(10)],
                func_vals=np.full(10, BO_CLASH_PENALTY_EV),
            )

        with patch(
            "InterOptimus.itworker.registration_minimizer",
            side_effect=fake_minimizer,
        ):
            with self.assertRaisesRegex(ValueError, "All sampled registrations"):
                worker.optimize_specified_interface_by_mlip(0, 0, n_calls=10)

        self.assertEqual(
            worker.opt_results[(0, 0)]["failure"]["stage"],
            "registration",
        )

    def test_flat_gradient_descent_is_bounded(self) -> None:
        xs, energies, steps = gradient_descend(
            sampling_function=lambda position, **kwargs: 1.0,
            dx=0.05,
            dim=3,
            tol=1e-6,
            initial_r=0.1,
            initial_xy=[np.zeros(3), 1.0],
            min_steps=5,
            max_steps=3,
        )

        self.assertEqual(len(xs), 4)
        self.assertEqual(len(energies), 4)
        self.assertTrue(np.isfinite(steps).all())

    def test_gradient_translation_step_is_norm_bounded(self) -> None:
        step = bounded_gradient_step([100.0, 0.0, 0.0], scale=0.5, max_displacement=0.2)
        self.assertAlmostEqual(float(np.linalg.norm(step)), 0.2)
        self.assertLess(step[0], 0)

    def test_match_comparison_with_no_termination_is_not_equivalent(self) -> None:
        identifier = equi_match_identifier.__new__(equi_match_identifier)
        identifier.film = MagicMock()
        identifier.substrate = MagicMock()

        with patch(
            "InterOptimus.matching.coherent_interface_builder_for_match",
            return_value=SimpleNamespace(terminations=[]),
        ):
            self.assertFalse(identifier.identify_by_stct_matching(object(), object()))

    def test_explicit_sevennet_checkpoint_failure_does_not_fallback(self) -> None:
        calculator_module = types.ModuleType("sevenn.calculator")

        class BrokenSevenNetCalculator:
            def __init__(self, *args, **kwargs) -> None:
                raise ValueError("bad checkpoint")

        calculator_module.SevenNetCalculator = BrokenSevenNetCalculator
        sevenn_module = types.ModuleType("sevenn")
        sevenn_module.calculator = calculator_module

        with patch.dict(
            sys.modules,
            {
                "sevenn": sevenn_module,
                "sevenn.calculator": calculator_module,
            },
        ), patch("InterOptimus.mlip._patch_torch_jit_for_frozen_bundle"):
            with self.assertRaisesRegex(RuntimeError, "refusing to silently substitute"):
                MlipCalc(
                    "sevenn",
                    {"device": "cpu", "ckpt_path": "/tmp/broken-sevennet.pth"},
                )

    def test_phase_stability_uses_supported_worker_arguments(self) -> None:
        structure = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
        worker = MagicMock()
        worker.phase_stability_evaluation.return_value = (
            structure,
            structure,
            0.0,
            0.0,
        )

        with patch(
            "InterOptimus.jobflow.InterfaceWorker",
            return_value=worker,
        ), patch(
            "InterOptimus.jobflow.resolve_mlip_checkpoint",
            return_value="/tmp/model.pth",
        ):
            result = check_it_phase_stability.original(
                structure,
                structure,
                calc="dpa",
            )

        structure_kwargs = worker.parse_interface_structure_params.call_args.kwargs
        optimization_kwargs = worker.parse_optimization_params.call_args.kwargs
        self.assertTrue(structure_kwargs["double_interface"])
        self.assertNotIn("c_periodic", structure_kwargs)
        self.assertNotIn("shift_to_bottom", structure_kwargs)
        self.assertNotIn("do", optimization_kwargs)
        self.assertNotIn("fix_in_layers", optimization_kwargs)
        self.assertEqual(result["rms_cart"], 0.0)


if __name__ == "__main__":
    unittest.main()
