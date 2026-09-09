"""Checkpoint discovery accepts supported model-version filename families."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from InterOptimus.checkpoints import CHECKPOINTS, checkpoint_status, missing_checkpoint_specs
from InterOptimus.mlip import _prepare_matris_checkpoint, resolve_mlip_checkpoint


class TestCheckpointCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.tempdir.name)
        self.env = patch.dict(
            os.environ,
            {"INTEROPTIMUS_CHECKPOINT_DIR": str(self.checkpoint_dir)},
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    def _write(self, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        path.write_bytes(b"checkpoint")
        return path.resolve()

    def test_legacy_sevennet_checkpoint_is_ready(self) -> None:
        expected = self._write("checkpoint_sevennet_mf_ompa.pth")
        spec = next(item for item in CHECKPOINTS if item.key == "sevenn")

        self.assertEqual(checkpoint_status(spec), (True, expected))
        self.assertNotIn(spec, missing_checkpoint_specs([spec]))

    def test_alternate_versions_resolve_for_each_backend(self) -> None:
        cases = {
            "orb-models": "orb-v3-conservative-20-mpa-20250101.ckpt",
            "sevenn": "checkpoint_sevennet_custom.pth",
            "dpa": "dpa-2.2.pb",
            "matris": "MatRIS_10M_MP.pth.tar",
        }
        for calculator, filename in cases.items():
            with self.subTest(calculator=calculator):
                expected = self._write(filename)
                self.assertEqual(resolve_mlip_checkpoint(calculator), str(expected))
                expected.unlink()

    def test_empty_compatible_file_is_not_ready(self) -> None:
        (self.checkpoint_dir / "checkpoint_sevennet_old.pth").touch()
        spec = next(item for item in CHECKPOINTS if item.key == "sevenn")

        self.assertEqual(checkpoint_status(spec), (False, None))

    def test_matris_compatible_checkpoint_is_exposed_to_upstream_cache(self) -> None:
        source = self._write("MatRIS-release-10M_MP.pth.tar")
        with patch("InterOptimus.mlip.Path.home", return_value=self.checkpoint_dir):
            model = _prepare_matris_checkpoint(str(source), "matris_10m_oam")

        target = self.checkpoint_dir / ".cache" / "matris" / "MatRIS_10M_MP.pth.tar"
        self.assertEqual(model, "matris_10m_mp")
        self.assertTrue(target.is_file())
        self.assertEqual(target.resolve(), source)


if __name__ == "__main__":
    unittest.main()
