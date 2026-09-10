"""Regression tests for the ``itom config --interactive`` entry point."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from InterOptimus import deploy_jobflow_stack


class InteractiveInvoked(Exception):
    """Stop ``main`` immediately after argument parsing."""


class TestConfigInteractive(unittest.TestCase):
    def test_interactive_does_not_require_mongo_db_at_parse_time(self) -> None:
        with patch.object(
            deploy_jobflow_stack,
            "_apply_interactive_config",
            side_effect=InteractiveInvoked,
        ):
            with self.assertRaises(InteractiveInvoked):
                deploy_jobflow_stack.main(["--interactive"])

    def test_misspelled_interacive_alias_is_supported(self) -> None:
        with patch.object(
            deploy_jobflow_stack,
            "_apply_interactive_config",
            side_effect=InteractiveInvoked,
        ):
            with self.assertRaises(InteractiveInvoked):
                deploy_jobflow_stack.main(["--interacive"])

    def test_noninteractive_still_requires_mongo_db(self) -> None:
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                deploy_jobflow_stack.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_secret_default_is_not_displayed(self) -> None:
        with patch("getpass.getpass", return_value="") as prompt:
            value = deploy_jobflow_stack._prompt_text(
                "password",
                default="do-not-print",
                secret=True,
            )
        self.assertEqual(value, "do-not-print")
        prompt.assert_called_once_with("password: ")

    def test_pip_install_does_not_require_local_source_checkout(self) -> None:
        with patch.object(
            deploy_jobflow_stack.metadata,
            "version",
            return_value="0.1.3",
        ):
            install_arg, source_root, description = (
                deploy_jobflow_stack._resolve_interoptimus_install_source(None)
            )

        self.assertEqual(install_arg, "InterOptimus==0.1.3")
        self.assertIsNone(source_root)
        self.assertIn("pip", description)

    def test_explicit_source_checkout_still_uses_editable_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "pyproject.toml").write_text(
                "[build-system]\nrequires=[]\n",
                encoding="utf-8",
            )
            install_arg, source_root, _ = (
                deploy_jobflow_stack._resolve_interoptimus_install_source(source)
            )

        self.assertEqual(install_arg, str(source.resolve()))
        self.assertEqual(source_root, source.resolve())

    def test_mlip_env_uses_normal_pip_install_for_pypi_source(self) -> None:
        with patch.object(deploy_jobflow_stack, "_conda_run_pip") as run:
            deploy_jobflow_stack._install_interoptimus_in_env(
                "orb",
                "InterOptimus==0.1.3",
                None,
            )

        run.assert_called_once_with(
            "orb",
            ["install", "InterOptimus==0.1.3", "--no-deps"],
        )


if __name__ == "__main__":
    unittest.main()
