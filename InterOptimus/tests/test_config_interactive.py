"""Regression tests for the ``itom config --interactive`` entry point."""

from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main()
