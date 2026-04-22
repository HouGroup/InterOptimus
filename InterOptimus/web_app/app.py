"""
FastAPI application: browser front-end for :func:`InterOptimus.session_workflow.run_iomaker_session`.

Form POST → ``session_form.json`` + CIFs → subprocess worker. Long runs execute in a
**subprocess** so the server can handle ``SIGINT``/``SIGTERM`` and :func:`POST /api/cancel`.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from InterOptimus.session_workflow import sessions_root
from InterOptimus.web_app.session_artifacts import find_pairs_best_it_dir, pick_artifact_file

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

_jobs_lock = threading.Lock()
# session_id -> subprocess.Popen (child runs job_worker)
_jobs: Dict[str, subprocess.Popen] = {}


def _terminate_all_jobs() -> None:
    with _jobs_lock:
        sids = list(_jobs.keys())
    for sid in sids:
        cancel_job(sid, reason="server_shutdown")


def cancel_job(session_id: str, *, reason: str = "cancelled") -> bool:
    """
    Send SIGTERM to the worker process; SIGKILL if still alive after a short wait.
    Writes a minimal ``web_result.json`` if the job was running.
    """
    with _jobs_lock:
        proc = _jobs.pop(session_id, None)
    if proc is None:
        return False
    workdir = sessions_root() / session_id
    try:
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
            else:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    result_path = workdir / "web_result.json"
    if not result_path.is_file():
        payload = {
            "ok": False,
            "session_id": session_id,
            "error": reason,
            "cancelled": True,
            "workdir": str(workdir.resolve()),
        }
        try:
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    _terminate_all_jobs()


app = FastAPI(title="InterOptimus IOMaker", version="0.2", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("INTEROPTIMUS_WEB_CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_artifact_under_session(session_id: str, basename: str) -> Optional[Path]:
    """Locate ``basename`` anywhere under the session directory (nested runs / ``result/``)."""
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        return None
    root = sessions_root() / session_id
    if not root.is_dir():
        return None
    return pick_artifact_file(root, basename)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.post("/api/run")
async def api_run(request: Request) -> JSONResponse:
    """
    Start a background job and return immediately with ``session_id``.
    Poll ``GET /api/status/{session_id}`` or call ``POST /api/cancel/{session_id}``.
    """
    raw = await request.form()
    film = raw.get("film_cif")
    sub = raw.get("substrate_cif")
    if not hasattr(film, "read") or not hasattr(sub, "read"):
        raise HTTPException(status_code=400, detail="Upload film_cif and substrate_cif (multipart files).")
    film_bytes = await film.read()  # type: ignore[union-attr]
    substrate_bytes = await sub.read()  # type: ignore[union-attr]
    if not film_bytes or not substrate_bytes:
        raise HTTPException(status_code=400, detail="Empty CIF upload.")

    form: Dict[str, str] = {}
    for key in raw:
        if key in ("film_cif", "substrate_cif"):
            continue
        val = raw[key]
        if hasattr(val, "read"):
            continue
        form[key] = str(val)

    sid = str(uuid.uuid4())
    workdir = sessions_root() / sid
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        (workdir / "film.cif").write_bytes(film_bytes)
        (workdir / "substrate.cif").write_bytes(substrate_bytes)
        (workdir / "session_form.json").write_text(
            json.dumps(form, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write session files: {e}") from e

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout/stderr so web_job.log updates live
        "-m",
        "InterOptimus.web_app.job_worker",
        sid,
    ]
    log_path = workdir / "web_job.log"
    try:
        log_f = open(log_path, "ab", buffering=0)  # noqa: SIM115
    except OSError:
        log_f = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(workdir),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg"),
            },
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to start worker: {e}") from e

    with _jobs_lock:
        _jobs[sid] = proc

    return JSONResponse(
        content={
            "started": True,
            "session_id": sid,
            "status": "running",
            "poll_url": f"/api/status/{sid}",
            "cancel_url": f"/api/cancel/{sid}",
            "log_hint": str(log_path),
        }
    )


@app.get("/api/status/{session_id}")
async def api_status(session_id: str) -> JSONResponse:
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    workdir = sessions_root() / session_id
    result_path = workdir / "web_result.json"

    with _jobs_lock:
        proc = _jobs.get(session_id)

    if proc is not None:
        code = proc.poll()
        if code is None:
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "status": "running",
                    "pid": proc.pid,
                }
            )
        with _jobs_lock:
            _jobs.pop(session_id, None)
        if not result_path.is_file():
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "status": "failed",
                    "error": f"Worker exited with code {code} before writing web_result.json",
                }
            )

    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Bad web_result.json: {e}") from e
        return JSONResponse(
            content={
                "session_id": session_id,
                "status": "completed" if data.get("ok") else "failed",
                "result": data,
            }
        )

    return JSONResponse(
        content={
            "session_id": session_id,
            "status": "unknown",
            "detail": "No job and no result file (session may be invalid or cleaned up).",
        },
        status_code=404,
    )


@app.post("/api/cancel/{session_id}")
async def api_cancel(session_id: str) -> JSONResponse:
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    ok = cancel_job(session_id, reason="cancelled_by_user")
    return JSONResponse(
        content={
            "session_id": session_id,
            "cancelled": ok,
            "detail": "Job terminated" if ok else "No running job for this session",
        }
    )


@app.get("/api/session/{session_id}/viz_log")
async def session_viz_log(session_id: str, offset: int = 0) -> JSONResponse:
    """
    Incremental read of ``web_viz.jsonl`` (optional relax-step events when viz is enabled).
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    workdir = sessions_root() / session_id
    log_path = workdir / "web_viz.jsonl"
    if not log_path.is_file():
        return JSONResponse(
            content={
                "session_id": session_id,
                "chunk": "",
                "next_offset": 0,
                "file_size": 0,
                "eof": True,
                "exists": False,
            }
        )
    try:
        size = log_path.stat().st_size
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    off = max(0, offset)
    if off > size:
        off = size
    max_bytes = min(256 * 1024, size - off)
    try:
        with open(log_path, "rb") as f:
            f.seek(off)
            raw = f.read(max_bytes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    next_off = off + len(raw)
    text = raw.decode("utf-8", errors="replace")
    return JSONResponse(
        content={
            "session_id": session_id,
            "chunk": text,
            "next_offset": next_off,
            "file_size": size,
            "eof": next_off >= size,
            "exists": True,
        }
    )


@app.get("/api/session/{session_id}/bo_iface.png")
async def session_bo_iface_png(session_id: str) -> FileResponse:
    """
    Latest BO interface snapshot rendered server-side with ASE ``plot_atoms`` (``web_bo_iface.png``).
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    path = sessions_root() / session_id / "web_bo_iface.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="BO interface image not available")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/session/{session_id}/job_log")
async def session_job_log(session_id: str, offset: int = 0) -> JSONResponse:
    """
    Incremental read of ``web_job.log`` (worker stdout/stderr) for live log in the browser.

    Poll with ``offset`` = previous ``next_offset`` until ``eof`` is true or job ends.
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    workdir = sessions_root() / session_id
    log_path = workdir / "web_job.log"
    if not log_path.is_file():
        return JSONResponse(
            content={
                "session_id": session_id,
                "chunk": "",
                "next_offset": 0,
                "file_size": 0,
                "eof": True,
                "exists": False,
            }
        )
    try:
        size = log_path.stat().st_size
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    off = max(0, offset)
    if off > size:
        off = size
    max_bytes = min(512 * 1024, size - off)
    try:
        with open(log_path, "rb") as f:
            f.seek(off)
            raw = f.read(max_bytes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    next_off = off + len(raw)
    text = raw.decode("utf-8", errors="replace")
    return JSONResponse(
        content={
            "session_id": session_id,
            "chunk": text,
            "next_offset": next_off,
            "file_size": size,
            "eof": next_off >= size,
            "exists": True,
        }
    )


@app.get("/api/session/{session_id}/file")
async def session_file(session_id: str, name: str) -> FileResponse:
    """
    Serve a single artifact by basename (e.g. ``stereographic_interactive.html``, ``stereographic.jpg``).
    Uses ``content_disposition_type=inline`` so HTML opens in the browser / iframe.
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    path = _find_artifact_under_session(session_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    media = "application/octet-stream"
    low = name.lower()
    if low.endswith(".html"):
        media = "text/html; charset=utf-8"
    elif low.endswith(".jpg") or low.endswith(".jpeg"):
        media = "image/jpeg"
    elif low.endswith(".txt"):
        media = "text/plain; charset=utf-8"
    return FileResponse(
        path,
        filename=name,
        media_type=media,
        content_disposition_type="inline",
    )


@app.get("/api/session/{session_id}/structures.zip")
async def session_structures_zip(session_id: str) -> Response:
    """
    ZIP of ``pairs_best_it/`` (relaxed best-interface POSCAR trees), for the given session.
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    workdir = sessions_root() / session_id
    pdir = find_pairs_best_it_dir(workdir)
    if pdir is None or not pdir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="pairs_best_it not found (local run only, or job did not materialize POSCARs).",
        )
    buf = io.BytesIO()
    root_label = Path("pairs_best_it")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pdir.rglob("*")):
            if f.is_file():
                arc = root_label / f.relative_to(pdir)
                zf.write(f, arcname=str(arc).replace("\\", "/"))
    data = buf.getvalue()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="pairs_best_it.zip"',
        },
    )


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
