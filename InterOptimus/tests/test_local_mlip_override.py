"""local MLIP-done override when remote job_state lags behind fetched_results."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from InterOptimus.web_app.task_store import (
    _apply_local_mlip_done_progress_override,
    local_mlip_stage_done_on_disk,
)


class TestLocalMlipOverride(unittest.TestCase):
    def test_override_when_fetched_opt_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tid = "test-session-123"
            root = Path(td) / tid
            fr = root / "fetched_results"
            fr.mkdir(parents=True)
            (fr / "opt_results_summary.json").write_text(
                json.dumps({"pairs": list(range(30))}),
                encoding="utf-8",
            )
            from InterOptimus.web_app import task_store as ts

            prev = ts.sessions_root

            def _tmp_root() -> Path:
                return Path(td)

            try:
                ts.sessions_root = _tmp_root  # type: ignore[method-assign]
                self.assertTrue(local_mlip_stage_done_on_disk(tid))
                prog = {
                    "success": True,
                    "mode": "mlip_and_vasp",
                    "current_phase": "mlip",
                    "job_state": "READY",
                    "expanded_vasp_job_counts": {"total": 0},
                }
                out = _apply_local_mlip_done_progress_override(tid, prog)
                self.assertEqual(out["current_phase"], "vasp")
                self.assertTrue(out.get("mlip_local_done_override"))
            finally:
                ts.sessions_root = prev  # type: ignore[method-assign]

    def test_no_override_when_expanded_vasp_known(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tid = "test-session-456"
            root = Path(td) / tid
            fr = root / "fetched_results"
            fr.mkdir(parents=True)
            (fr / "opt_results_summary.json").write_text('{"x": 1}', encoding="utf-8")
            from InterOptimus.web_app import task_store as ts

            prev = ts.sessions_root

            def _tmp_root() -> Path:
                return Path(td)

            try:
                ts.sessions_root = _tmp_root  # type: ignore[method-assign]
                prog = {
                    "success": True,
                    "mode": "mlip_and_vasp",
                    "current_phase": "mlip",
                    "expanded_vasp_job_counts": {"total": 2},
                }
                out = _apply_local_mlip_done_progress_override(tid, prog)
                self.assertEqual(out["current_phase"], "mlip")
            finally:
                ts.sessions_root = prev  # type: ignore[method-assign]


if __name__ == "__main__":
    unittest.main()
