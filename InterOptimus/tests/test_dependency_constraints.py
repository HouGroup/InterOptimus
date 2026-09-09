"""Keep package and deployment dependency constraints valid and synchronized."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement

from InterOptimus import deploy_jobflow_stack
from InterOptimus.deploy_jobflow_stack import INTEROPTIMUS_CORE_PIP


def _setup_install_requires() -> list[str]:
    setup_path = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "install_requires":
                return list(ast.literal_eval(keyword.value))
    raise AssertionError("setup.py does not define install_requires")


def _setup_python_requires() -> str:
    setup_path = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup":
            for keyword in node.keywords:
                if keyword.arg == "python_requires":
                    return str(ast.literal_eval(keyword.value))
    raise AssertionError("setup.py does not define python_requires")


class TestDependencyConstraints(unittest.TestCase):
    def test_core_dependency_lists_are_synchronized(self) -> None:
        self.assertEqual(_setup_install_requires(), INTEROPTIMUS_CORE_PIP)

    def test_all_core_requirements_are_valid_pep440(self) -> None:
        for requirement in INTEROPTIMUS_CORE_PIP:
            with self.subTest(requirement=requirement):
                Requirement(requirement)

    def test_pymatgen_packages_are_pinned_as_a_tested_pair(self) -> None:
        requirements = {
            Requirement(item).name: item for item in INTEROPTIMUS_CORE_PIP
        }
        self.assertEqual(requirements["pymatgen"], "pymatgen==2026.5.4")
        self.assertEqual(requirements["pymatgen-core"], "pymatgen-core==2026.8.30")

    def test_python_requirement_matches_pymatgen(self) -> None:
        self.assertEqual(_setup_python_requires(), ">=3.11,<3.13")

    def test_vasp_parser_packages_match_new_pymatgen_potcar_api(self) -> None:
        requirements = {
            Requirement(item).name: item for item in INTEROPTIMUS_CORE_PIP
        }
        self.assertEqual(requirements["atomate2"], "atomate2>=0.1.5,<0.2")
        self.assertEqual(requirements["emmet-core"], "emmet-core>=0.87.2,<0.88")

    def test_config_installer_reuses_core_dependency_constraints(self) -> None:
        with patch.object(deploy_jobflow_stack, "_run") as run:
            deploy_jobflow_stack._pip_install(skip=False, upgrade=False)

        run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", *INTEROPTIMUS_CORE_PIP]
        )


if __name__ == "__main__":
    unittest.main()
