"""
Local web UI: upload film / substrate CIFs, run the simple_iomaker pipeline with **MatRIS** MLIP,
then show ``io_report.txt`` and stereographic figures from the run directory.

Run::

    pip install '.[web]'
    interoptimus-web

Or::

    uvicorn InterOptimus.web.app:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from InterOptimus.agents.simple_iomaker import run_simple_iomaker

def _sessions_root() -> Path:
    env = os.environ.get("INTEROPTIMUS_WEB_SESSIONS", "").strip()
    root = Path(env) if env else (Path.home() / ".interoptimus" / "web_sessions")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_session_dir(session_id: str) -> Path:
    try:
        sid = uuid.UUID(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid session id") from e
    d = _sessions_root() / str(sid)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    return d.resolve()


def _build_config(
    *,
    workflow_name: str,
    cost_preset: str,
    double_interface: bool,
    execution: str,
    cluster: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "workflow_name": workflow_name.strip() or "IO_web_matris",
        "IO_workflow_config": {
            "cost_preset": cost_preset,
            "bulk_cifs": {
                "film_cif": "film.cif",
                "substrate_cif": "substrate.cif",
            },
            "lattice_matching_settings": {},
            "structure_settings": {"double_interface": double_interface},
            # ``calc`` is merged into global_minimization_settings by simple_iomaker
            "optimization_settings": {"calc": "matris"},
            "vasp_settings": {"do_vasp": False},
        },
        "execution": execution,
    }
    if cluster:
        cfg["cluster"] = cluster
    return cfg


def _run_in_workdir(workdir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    old = os.getcwd()
    try:
        os.chdir(workdir)
        return run_simple_iomaker(config)
    finally:
        os.chdir(old)


def _json_safe_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON entries (e.g. flow_dict) for API responses."""
    skip = {"flow_dict", "settings"}
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            try:
                import json

                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = f"<{type(v).__name__}>"
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = str(v)[:500]
    return out


app = FastAPI(
    title="InterOptimus simple_iomaker (MatRIS)",
    description="Upload substrate and film CIFs; run MLIP workflow; view report and stereographic plots.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if html_path.is_file():
        return html_path.read_text(encoding="utf-8")
    return "<p>Missing static/index.html</p>"


@app.post("/api/run")
async def api_run(
    film_cif: UploadFile = File(..., description="Film crystal CIF"),
    substrate_cif: UploadFile = File(..., description="Substrate crystal CIF"),
    workflow_name: str = Form("IO_web_matris"),
    cost_preset: str = Form("medium"),
    double_interface: str = Form("false"),
    execution: str = Form("local"),
) -> JSONResponse:
    """
    Save CIFs to a new session directory, ``chdir`` there, and call :func:`run_simple_iomaker`
    with MatRIS as ``global_minimization_settings.calc``.

    Default ``execution`` is ``local`` (full jobflow run on this machine). Use ``server`` only if
    jobflow-remote is configured on this host (same as CLI).
    """
    if cost_preset not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="cost_preset must be low, medium, or high")
    ex = execution.strip().lower().replace("-", "_")
    if ex not in ("local", "server"):
        raise HTTPException(status_code=400, detail="execution must be local or server")
    di = str(double_interface).lower() in ("1", "true", "yes", "on")

    sid = str(uuid.uuid4())
    workdir = _sessions_root() / sid
    workdir.mkdir(parents=True, exist_ok=True)

    film_path = workdir / "film.cif"
    sub_path = workdir / "substrate.cif"

    try:
        film_path.write_bytes(await film_cif.read())
        sub_path.write_bytes(await substrate_cif.read())
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}") from e

    cluster = None
    if ex == "server":
        # Minimal nested cluster; user can override via env / full CLI for production.
        cluster = {
            "mlip": {
                "slurm_partition": os.environ.get("INTEROPTIMUS_SLURM_PARTITION", "interactive"),
            },
            "vasp": {},
        }

    config = _build_config(
        workflow_name=workflow_name,
        cost_preset=cost_preset,
        double_interface=di,
        execution=ex,
        cluster=cluster,
    )

    try:
        result = _run_in_workdir(workdir, config)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "session_id": sid,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )

    wd = result.get("local_workdir") or str(workdir)
    report_path = result.get("report_path")
    report_text = ""
    if report_path and os.path.isfile(report_path):
        try:
            report_text = Path(report_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            report_text = ""

    stereo = os.path.join(wd, "stereographic.jpg")
    stereo_url = f"/api/sessions/{sid}/artifact/stereographic.jpg" if os.path.isfile(stereo) else None
    stereo_html = os.path.join(wd, "stereographic_interactive.html")
    stereo_html_url = (
        f"/api/sessions/{sid}/artifact/stereographic_interactive.html"
        if os.path.isfile(stereo_html)
        else None
    )
    project_jpg = os.path.join(wd, "project.jpg")
    project_url = f"/api/sessions/{sid}/artifact/project.jpg" if os.path.isfile(project_jpg) else None

    payload = {
        "ok": True,
        "session_id": sid,
        "result": _json_safe_result(result),
        "report_text": report_text,
        "artifacts": {
            "stereographic_jpg": stereo_url,
            "stereographic_interactive_html": stereo_html_url,
            "project_jpg": project_url,
            "local_workdir": wd,
        },
    }
    return JSONResponse(content=payload)


@app.get("/api/sessions/{session_id}/artifact/{name}")
async def get_artifact(session_id: str, name: str) -> FileResponse:
    """Serve files from the session directory (only basenames allowed)."""
    if name != os.path.basename(name) or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    allowed = {
        "stereographic.jpg",
        "stereographic_interactive.html",
        "project.jpg",
        "unique_matches.jpg",
        "io_report.txt",
        "pairs_summary.txt",
    }
    if name not in allowed:
        raise HTTPException(status_code=400, detail="Artifact not exposed")
    d = _safe_session_dir(session_id)
    path = d / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media = "text/html" if name.endswith(".html") else None
    return FileResponse(path, filename=name, media_type=media)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="InterOptimus web UI (MatRIS simple_iomaker)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true", help="Dev only: auto-reload on code changes")
    args = p.parse_args()
    try:
        import uvicorn
    except ImportError:
        print("Install web extras: pip install 'InterOptimus[web]' or pip install uvicorn[standard]", file=sys.stderr)
        sys.exit(1)
    uvicorn.run(
        "InterOptimus.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
