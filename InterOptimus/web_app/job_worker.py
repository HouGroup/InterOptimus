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
from pathlib import Path

from InterOptimus.session_workflow import run_iomaker_session, sessions_root
from InterOptimus.web_app.session_artifacts import enrich_ok_payload_artifacts


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
    if isinstance(result, dict):
        _ensure_relax_final_iface_png(workdir)
        enrich_ok_payload_artifacts(workdir, result)
    _write_result(workdir, result)


if __name__ == "__main__":
    main()
