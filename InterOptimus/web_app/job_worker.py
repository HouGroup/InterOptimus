"""
Subprocess entry: runs :func:`~InterOptimus.session_workflow.run_iomaker_session`
and writes ``web_result.json`` under the session directory.

Optional live viz: when ``session_form.json`` sets ``viz_enable`` = true, sets
``INTEROPTIMUS_VIZ_LOG`` → ``web_viz.jsonl``.

Started by the web app as::

    python -m InterOptimus.web_app.job_worker <session_id>
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from InterOptimus.session_workflow import run_iomaker_session, sessions_root
from InterOptimus.web_app.session_artifacts import enrich_ok_payload_artifacts
from InterOptimus.web_app.task_store import get_task_store, refresh_remote_progress, refresh_task_indexes, sync_submission_refs_from_result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_relax_final_iface_png(workdir: Path) -> None:
    """Best-effort fallback: render missing ``relax_final`` ASE PNGs from ``web_viz.jsonl``."""
    viz_path = workdir / "web_viz.jsonl"
    if not viz_path.is_file():
        return
    by_pair: dict[tuple[int, int] | None, dict] = {}
    try:
        for line in viz_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"event": "relax_final"' not in line and '"event":"relax_final"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "relax_final":
                continue
            mid, tid = row.get("match_id"), row.get("term_id")
            if mid is not None and tid is not None:
                try:
                    key = (int(mid), int(tid))
                except (TypeError, ValueError):
                    key = None
                by_pair[key] = row
            else:
                by_pair[None] = row
    except OSError:
        return
    if not by_pair:
        return
    try:
        from InterOptimus.viz_ase_iface import write_bo_iface_png
    except Exception:
        return
    for key, row in by_pair.items():
        if key is None:
            out_path = workdir / "web_relax_final_iface.png"
        else:
            m, t = key
            out_path = workdir / f"web_relax_final_iface_m{m}_t{t}.png"
        if out_path.is_file():
            continue
        try:
            write_bo_iface_png(row, out_path)
        except Exception:
            pass


def _write_result(workdir: Path, payload: dict) -> None:
    p = workdir / "web_result.json"
    try:
        p.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError:
        pass


def _task_event(session_id: str, event: str, message: str = "", **data: object) -> None:
    try:
        get_task_store().append_event(session_id, event, message, **data)
    except Exception:
        pass


def _task_update(session_id: str, **updates: object) -> None:
    try:
        get_task_store().update_task(session_id, updates)
    except Exception:
        pass


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m InterOptimus.web_app.job_worker <session_id>", file=sys.stderr)
        raise SystemExit(2)
    session_id = sys.argv[1].strip()
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise SystemExit(2)

    workdir = sessions_root() / session_id
    if not workdir.is_dir():
        _write_result(
            workdir,
            {
                "ok": False,
                "session_id": session_id,
                "error": "Session workdir missing",
                "workdir": str(workdir),
            },
        )
        raise SystemExit(1)

    film = workdir / "film.cif"
    sub = workdir / "substrate.cif"
    form_path = workdir / "session_form.json"
    if not film.is_file() or not sub.is_file() or not form_path.is_file():
        _write_result(
            workdir,
            {
                "ok": False,
                "session_id": session_id,
                "error": "Missing film.cif, substrate.cif, or session_form.json",
                "workdir": str(workdir.resolve()),
            },
        )
        raise SystemExit(1)

    try:
        form = json.loads(form_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _write_result(
            workdir,
            {
                "ok": False,
                "session_id": session_id,
                "error": f"Invalid session_form.json: {e}",
                "workdir": str(workdir.resolve()),
            },
        )
        raise SystemExit(1)

    film_bytes = film.read_bytes()
    substrate_bytes = sub.read_bytes()
    _task_update(session_id, status="mlip_running", phase="mlip", started_at=_utc_now())
    _task_event(session_id, "worker_started", "Worker started", workdir=str(workdir.resolve()))

    def _viz_enabled_from_form(f: dict) -> bool:
        v = f.get("viz_enable", "")
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return int(v) != 0
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    viz_on = _viz_enabled_from_form(form) if isinstance(form, dict) else False
    if viz_on:
        viz_path = workdir / "web_viz.jsonl"
        try:
            viz_path.write_text("", encoding="utf-8")
        except OSError:
            pass
        try:
            from InterOptimus.viz_runtime import emit_event, pin_web_viz_session

            pin_web_viz_session(str(viz_path.resolve()))
        except Exception:
            os.environ["INTEROPTIMUS_VIZ_LOG"] = str(viz_path.resolve())
            os.environ["INTEROPTIMUS_VIZ_ENABLE"] = "1"
        print(f"[InterOptimus] web viz JSONL enabled: {viz_path.resolve()}", flush=True)
        _task_event(session_id, "viz_enabled", "Live visualization event log enabled", path=str(viz_path.resolve()))
        try:
            from InterOptimus.viz_runtime import emit_event

            emit_event(
                {
                    "event": "viz_log_ready",
                    "phase": "web",
                    "session_id": session_id,
                    "viz_path": str(viz_path.resolve()),
                }
            )
        except Exception:
            pass
    else:
        os.environ.pop("INTEROPTIMUS_VIZ_LOG", None)
        os.environ.pop("INTEROPTIMUS_VIZ_ENABLE", None)

    try:
        result = run_iomaker_session(
            film_bytes=film_bytes,
            substrate_bytes=substrate_bytes,
            form=form if isinstance(form, dict) else {},
            session_id=session_id,
        )
    except BaseException as e:
        tb = traceback.format_exc()
        result = {
            "ok": False,
            "session_id": session_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb,
            "workdir": str(workdir.resolve()),
        }
        _task_update(session_id, status="failed", phase="failed", error=str(e))
        _task_event(session_id, "failed", str(e), error_type=type(e).__name__)
    if isinstance(result, dict):
        _ensure_relax_final_iface_png(workdir)
        enrich_ok_payload_artifacts(workdir, result)
        inner = result.get("result") if isinstance(result.get("result"), dict) else {}
        server_submission = inner.get("server_submission") if isinstance(inner.get("server_submission"), dict) else {}
        is_remote_submit = bool(server_submission.get("success") or inner.get("mlip_job_uuid") or inner.get("flow_uuid"))
        if result.get("ok") and is_remote_submit:
            refs = {
                "submit_jf_job_id": str(server_submission.get("job_id") or "").strip() or None,
                "mlip_job_uuid": str(inner.get("mlip_job_uuid") or server_submission.get("mlip_job_uuid") or "").strip() or None,
                "flow_uuid": str(inner.get("flow_uuid") or server_submission.get("flow_uuid") or "").strip() or None,
                "vasp_job_uuids": list(inner.get("vasp_job_uuids") or server_submission.get("vasp_job_uuids") or []),
                "submit_workdir": str(server_submission.get("submit_workdir") or inner.get("submit_workdir") or "").strip() or None,
                "interoptimus_task_serial": str(inner.get("interoptimus_task_serial") or "").strip() or None,
            }
            _task_update(
                session_id,
                status="mlip_running",
                phase="mlip",
                remote_submitted_at=_utc_now(),
                completed_at=None,
                **{k: v for k, v in refs.items() if v not in (None, "", [])},
            )
            _task_event(session_id, "remote_submitted", "Submitted to jobflow-remote")
            try:
                sync_submission_refs_from_result(session_id)
                refresh_remote_progress(session_id)
            except Exception:
                pass
        elif result.get("ok"):
            _task_update(session_id, status="completed", phase="completed", completed_at=_utc_now())
            _task_event(session_id, "completed", "Task completed")
        else:
            _task_update(
                session_id,
                status="failed",
                phase="failed",
                error=str(result.get("error") or "Task failed"),
            )
            _task_event(session_id, "failed", str(result.get("error") or "Task failed"))
    _write_result(workdir, result)
    try:
        refresh_task_indexes(session_id)
    except Exception:
        pass


if __name__ == "__main__":
    main()
