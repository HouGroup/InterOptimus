"""Regression tests for MLIP vs VASP phase detection (jobflow-remote states)."""

from __future__ import annotations

import unittest

from InterOptimus.agents.remote_submit import (
    _job_state_bucket,
    _mlip_done_for_phase_transition,
    _select_mlip_root_row,
)


class TestProgressPhase(unittest.TestCase):
    def test_job_state_bucket_run_finished_is_running_bucket(self) -> None:
        self.assertEqual(_job_state_bucket("RUN_FINISHED"), "running")
        self.assertEqual(_job_state_bucket("DOWNLOADED"), "running")
        self.assertEqual(_job_state_bucket("JobState.RUN_FINISHED"), "running")

    def test_mlip_done_treats_post_run_pipeline_as_finished(self) -> None:
        self.assertTrue(_mlip_done_for_phase_transition("RUN_FINISHED", None))
        self.assertTrue(_mlip_done_for_phase_transition("DOWNLOADED", None))
        self.assertTrue(_mlip_done_for_phase_transition("COMPLETED", None))
        self.assertFalse(_mlip_done_for_phase_transition("RUNNING", None))
        self.assertFalse(_mlip_done_for_phase_transition("WAITING", None))

    def test_mlip_done_end_time_without_completed_state(self) -> None:
        row = {"end_time": "2026-05-08T12:00:00+00:00"}
        self.assertTrue(_mlip_done_for_phase_transition("RUNNING", row))

    def test_select_mlip_root_excludes_planned_vasp_uuids(self) -> None:
        planned = {"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
        flow = [
            {
                "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "index": 0,
                "state": "WAITING",
            },
            {
                "uuid": "11111111-2222-3333-4444-555555555555",
                "index": 1,
                "state": "RUN_FINISHED",
            },
        ]
        row = _select_mlip_root_row(flow, "", planned)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["uuid"], "11111111-2222-3333-4444-555555555555")

    def test_select_mlip_root_prefers_explicit_ref(self) -> None:
        planned = set()
        flow = [
            {"uuid": "aaa", "index": 1, "state": "RUNNING"},
            {"uuid": "bbb", "index": 0, "state": "COMPLETED"},
        ]
        row = _select_mlip_root_row(flow, "aaa", planned)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["uuid"], "aaa")


if __name__ == "__main__":
    unittest.main()
