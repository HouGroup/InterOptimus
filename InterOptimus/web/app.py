"""
Local web UI: upload film / substrate CIFs, run the simple_iomaker pipeline with **MatRIS** MLIP,
then show ``io_report.txt``, stereographic figures, and materialized pair POSCARs.

For a **native desktop GUI** (no browser), use ``interoptimus-desktop`` or
``python -m InterOptimus.desktop_app.gui``.

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
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from InterOptimus.web.local_workflow import run_matris_session, sessions_root


def _safe_session_dir(session_id: str) -> Path:
    try:
        sid = uuid.UUID(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid session id") from e
    d = sessions_root() / str(sid)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    return d.resolve()


def _artifacts_with_web_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turn local paths in ``artifacts`` into ``/api/sessions/...`` URLs for the HTML client."""
    if not payload.get("ok"):
        return payload
    sid = payload["session_id"]
    art = dict(payload.get("artifacts") or {})
    wd = art.get("local_workdir") or ""

    def _u(local_path: str | None, name: str) -> str | None:
        if not local_path or not os.path.isfile(local_path):
            return None
        return f"/api/sessions/{sid}/artifact/{name}"

    stereo = os.path.join(wd, "stereographic.jpg") if wd else None
    stereo_html = os.path.join(wd, "stereographic_interactive.html") if wd else None
    project_jpg = os.path.join(wd, "project.jpg") if wd else None

    art["stereographic_jpg"] = _u(stereo, "stereographic.jpg")
    art["stereographic_interactive_html"] = _u(stereo_html, "stereographic_interactive.html")
    art["project_jpg"] = _u(project_jpg, "project.jpg")

    zpath = art.get("poscars_zip_path")
    if zpath and os.path.isfile(zpath):
        art["poscars_zip"] = f"/api/sessions/{sid}/poscars.zip"
    else:
        art["poscars_zip"] = None
    art.pop("poscars_zip_path", None)

    payload = dict(payload)
    payload["artifacts"] = art
    return payload


app = FastAPI(
    title="InterOptimus simple_iomaker (MatRIS)",
    description="Upload substrate and film CIFs; run MLIP workflow; view report, stereographic plots, and POSCARs.",
    version="0.2.0",
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
    double_interface: str = Form("true"),
    execution: str = Form("local"),
    lm_max_area: str = Form("20"),
    lm_max_length_tol: str = Form("0.03"),
    lm_max_angle_tol: str = Form("0.03"),
    lm_film_max_miller: str = Form("3"),
    lm_substrate_max_miller: str = Form("3"),
    st_film_thickness: str = Form("10"),
    st_substrate_thickness: str = Form("10"),
    st_termination_ftol: str = Form("0.15"),
    st_vacuum_over_film: str = Form("5"),
    opt_device: str = Form("cpu"),
    opt_steps: str = Form("500"),
    do_mlip_gd: str = Form("false"),
    relax_in_ratio: str = Form("true"),
    relax_in_layers: str = Form("false"),
    fix_film_fraction: str = Form("0.5"),
    fix_substrate_fraction: str = Form("0.5"),
    set_relax_film_ang: str = Form("0"),
    set_relax_substrate_ang: str = Form("0"),
    adv_ckpt_path: str = Form(""),
    adv_matris_model: str = Form(""),
    adv_matris_task: str = Form(""),
    adv_fmax: str = Form(""),
    adv_discut: str = Form(""),
    adv_gd_tol: str = Form(""),
    adv_n_calls_density: str = Form(""),
    adv_strain_E_correction: str = Form(""),
    adv_term_screen_tol: str = Form(""),
    adv_z_range_lo: str = Form(""),
    adv_z_range_hi: str = Form(""),
    adv_bo_coord_bin: str = Form(""),
    adv_bo_energy_bin: str = Form(""),
    adv_bo_rms_bin: str = Form(""),
) -> JSONResponse:
    film_bytes = await film_cif.read()
    sub_bytes = await substrate_cif.read()

    form: Dict[str, Any] = {
        "workflow_name": workflow_name,
        "cost_preset": cost_preset,
        "double_interface": double_interface,
        "execution": execution,
        "lm_max_area": lm_max_area,
        "lm_max_length_tol": lm_max_length_tol,
        "lm_max_angle_tol": lm_max_angle_tol,
        "lm_film_max_miller": lm_film_max_miller,
        "lm_substrate_max_miller": lm_substrate_max_miller,
        "st_film_thickness": st_film_thickness,
        "st_substrate_thickness": st_substrate_thickness,
        "st_termination_ftol": st_termination_ftol,
        "st_vacuum_over_film": st_vacuum_over_film,
        "opt_device": opt_device,
        "opt_steps": opt_steps,
        "do_mlip_gd": do_mlip_gd,
        "relax_in_ratio": relax_in_ratio,
        "relax_in_layers": relax_in_layers,
        "fix_film_fraction": fix_film_fraction,
        "fix_substrate_fraction": fix_substrate_fraction,
        "set_relax_film_ang": set_relax_film_ang,
        "set_relax_substrate_ang": set_relax_substrate_ang,
        "adv_ckpt_path": adv_ckpt_path,
        "adv_matris_model": adv_matris_model,
        "adv_matris_task": adv_matris_task,
        "adv_fmax": adv_fmax,
        "adv_discut": adv_discut,
        "adv_gd_tol": adv_gd_tol,
        "adv_n_calls_density": adv_n_calls_density,
        "adv_strain_E_correction": adv_strain_E_correction,
        "adv_term_screen_tol": adv_term_screen_tol,
        "adv_z_range_lo": adv_z_range_lo,
        "adv_z_range_hi": adv_z_range_hi,
        "adv_bo_coord_bin": adv_bo_coord_bin,
        "adv_bo_energy_bin": adv_bo_energy_bin,
        "adv_bo_rms_bin": adv_bo_rms_bin,
    }

    payload = run_matris_session(film_bytes=film_bytes, substrate_bytes=sub_bytes, form=form)
    payload = _artifacts_with_web_urls(payload)

    if not payload.get("ok"):
        return JSONResponse(status_code=500, content=payload)

    return JSONResponse(content=payload)


@app.get("/api/sessions/{session_id}/poscars.zip")
async def download_poscars_zip(session_id: str) -> FileResponse:
    d = _safe_session_dir(session_id)
    zpath = d / "pairs_poscars.zip"
    if not zpath.is_file():
        raise HTTPException(status_code=404, detail="pairs_poscars.zip not found; run local MLIP first.")
    return FileResponse(zpath, filename="pairs_poscars.zip", media_type="application/zip")


@app.get("/api/sessions/{session_id}/artifact/{name}")
async def get_artifact(session_id: str, name: str) -> FileResponse:
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

    try:
        from InterOptimus.web.runtime_bootstrap import maybe_reexec_into_managed_venv

        maybe_reexec_into_managed_venv()
    except Exception as e:
        print(
            f"InterOptimus: managed venv bootstrap failed ({e}); continuing with {sys.executable}\n",
            file=sys.stderr,
            flush=True,
        )

    p = argparse.ArgumentParser(description="InterOptimus web UI (MatRIS simple_iomaker)")
    p.add_argument(
        "--host",
        default=os.environ.get("INTEROPTIMUS_WEB_HOST", "0.0.0.0"),
        help="Bind address (default 0.0.0.0 for Cursor port-forward / LAN; set INTEROPTIMUS_WEB_HOST=127.0.0.1 to listen on localhost only).",
    )
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
