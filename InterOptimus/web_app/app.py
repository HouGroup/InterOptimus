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
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from InterOptimus.session_workflow import sessions_root, vasp_runtime_effective_from_form
from InterOptimus.web_app.cif_view import cell_metadata, expand_cif_for_view, poscar_to_cif
from InterOptimus.web_app.cluster_info import query_cluster_info
from InterOptimus.web_app.jfremote_workers import (
    auto_select_mlip_worker,
    list_workers,
    probe_mlip_in_worker,
    query_module_avail,
)
from InterOptimus.web_app.session_artifacts import find_pairs_best_it_dir, pick_artifact_file
from InterOptimus.web_app.task_store import (
    _load_web_result,
    create_initial_task,
    get_task_store,
    is_safe_task_id,
    refresh_remote_progress,
    refresh_task_indexes,
    sync_submission_refs_from_result,
    task_has_remote_refs,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

_jobs_lock = threading.Lock()
# session_id -> subprocess.Popen (child runs job_worker)
_jobs: Dict[str, subprocess.Popen] = {}


def _task_with_config_effective(task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach ``config_effective`` from ``web_result.json`` when present, else compute
    ``vasp_runtime_effective`` from submit form so the detail page can show merged INCAR.
    """
    task_out = dict(task)
    wr = _load_web_result(task_id)
    ce_wr = wr.get("config_effective") if isinstance(wr.get("config_effective"), dict) else None
    if ce_wr:
        task_out["config_effective"] = ce_wr
    elif isinstance(task_out.get("form"), dict) and str(task_out.get("mode")) == "mlip+vasp":
        eff = vasp_runtime_effective_from_form(task_out["form"])
        if eff:
            task_out["config_effective"] = {"vasp_runtime_effective": eff}
    return task_out


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
    try:
        store = get_task_store()
        store.update_task(session_id, status="cancelled", phase="cancelled", error=reason)
        store.append_event(session_id, "cancelled", reason)
        refresh_task_indexes(session_id)
    except Exception:
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
        "portal.html",
        {
            "request": request,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.get("/local", response_class=HTMLResponse)
async def local_gui(request: Request) -> Any:
    return templates.TemplateResponse(
        "local.html",
        {
            "request": request,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.get("/hub", response_class=HTMLResponse)
async def flow_hub(request: Request) -> Any:
    return templates.TemplateResponse(
        "hub_simple.html",
        {
            "request": request,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.get("/manage", response_class=HTMLResponse)
async def manage_page(request: Request) -> Any:
    """Task management dashboard: submission form + list of submitted tasks."""
    return templates.TemplateResponse(
        "manage.html",
        {
            "request": request,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.get("/manage/{task_id}", response_class=HTMLResponse)
async def manage_task_page(request: Request, task_id: str) -> Any:
    """Task detail page: structure visualization, CIF download, and phase-aware progress."""
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    return templates.TemplateResponse(
        "manage_detail.html",
        {
            "request": request,
            "task_id": task_id,
            "default_host": os.environ.get("INTEROPTIMUS_WEB_PUBLIC_HOST", ""),
        },
    )


@app.get("/api/tasks/{task_id}/cif")
async def api_task_cif(task_id: str, which: str = "film") -> FileResponse:
    """Download the original ``film.cif`` or ``substrate.cif`` for a given task."""
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    which = (which or "film").strip().lower()
    if which not in ("film", "substrate"):
        raise HTTPException(status_code=400, detail="which must be 'film' or 'substrate'")
    workdir = sessions_root() / task_id
    target = workdir / f"{which}.cif"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{which}.cif not found for this task")
    return FileResponse(
        target,
        filename=f"{task_id}_{which}.cif",
        media_type="chemical/x-cif",
        content_disposition_type="attachment",
    )


@app.get("/api/tasks/{task_id}/cif_view")
async def api_task_cif_view(task_id: str, which: str = "film") -> Response:
    """
    Return a viewer-ready (P1, conventional cell, image atoms on cell faces) CIF
    for ``film`` / ``substrate``.  Used by the manage-detail page so 3Dmol does
    not have to run symmetry assembly client-side (which produces atoms outside
    the cell).  Falls back to the raw CIF if pymatgen is unavailable.
    """
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    which = (which or "film").strip().lower()
    if which not in ("film", "substrate"):
        raise HTTPException(status_code=400, detail="which must be 'film' or 'substrate'")
    workdir = sessions_root() / task_id
    target = workdir / f"{which}.cif"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{which}.cif not found for this task")
    raw = target.read_text(encoding="utf-8", errors="replace")
    expanded = expand_cif_for_view(raw)
    return Response(
        content=expanded,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/tasks/{task_id}/cif_meta")
async def api_task_cif_meta(task_id: str, which: str = "film") -> JSONResponse:
    """Return a small metadata dict (formula, spacegroup, lattice) for the side caption."""
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    which = (which or "film").strip().lower()
    if which not in ("film", "substrate"):
        raise HTTPException(status_code=400, detail="which must be 'film' or 'substrate'")
    workdir = sessions_root() / task_id
    target = workdir / f"{which}.cif"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{which}.cif not found for this task")
    raw = target.read_text(encoding="utf-8", errors="replace")
    return JSONResponse(content=cell_metadata(raw))


@app.post("/api/run")
async def api_run(request: Request) -> JSONResponse:
    """
    Start a background job and return immediately with ``session_id``.
    Poll ``GET /api/status/{session_id}`` or call ``POST /api/cancel/{session_id}``.
    """
    try:
        return await _api_run_impl(request)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[api_run] unhandled error: {tb}", file=sys.stderr, flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "error_type": type(e).__name__,
            },
        )


async def _api_run_impl(request: Request) -> JSONResponse:
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
    store = get_task_store()

    try:
        (workdir / "film.cif").write_bytes(film_bytes)
        (workdir / "substrate.cif").write_bytes(substrate_bytes)
        (workdir / "session_form.json").write_text(
            json.dumps(form, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write session files: {e}") from e

    film_name = str(getattr(film, "filename", "") or "film.cif")
    substrate_name = str(getattr(sub, "filename", "") or "substrate.cif")
    try:
        task = create_initial_task(
            task_id=sid,
            workdir=workdir,
            form=form,
            film_name=film_name,
            substrate_name=substrate_name,
        )
        store.create_task(task)
        store.append_event(sid, "submitted", "Task submitted", film_name=film_name, substrate_name=substrate_name)
        refresh_task_indexes(sid)
    except Exception as e:
        # Surface DB write failures so the user does not get an "OK" response while
        # the task is missing from /api/tasks (which used to silently swallow this).
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register task in store: {type(e).__name__}: {e}",
        ) from e

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
        try:
            store.update_task(sid, status="failed", phase="failed", error=f"Failed to start worker: {e}")
            store.append_event(sid, "failed", f"Failed to start worker: {e}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to start worker: {e}") from e

    with _jobs_lock:
        _jobs[sid] = proc
    try:
        store.update_task(sid, status="mlip_running", phase="mlip", pid=proc.pid)
        store.append_event(sid, "process_started", "Worker process started", pid=proc.pid)
    except Exception:
        pass

    return JSONResponse(
        content={
            "started": True,
            "session_id": sid,
            "task_id": sid,
            "status": "running",
            "poll_url": f"/api/status/{sid}",
            "detail_url": f"/api/tasks/{sid}",
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
    store = get_task_store()
    task = store.get_task(session_id)

    with _jobs_lock:
        proc = _jobs.get(session_id)

    if proc is not None:
        code = proc.poll()
        if code is None:
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "task_id": session_id,
                    "status": "running",
                    "pid": proc.pid,
                    "task": task,
                }
            )
        with _jobs_lock:
            _jobs.pop(session_id, None)
        if not result_path.is_file():
            store.update_task(
                session_id,
                status="failed",
                phase="failed",
                error=f"Worker exited with code {code} before writing web_result.json",
            )
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "task_id": session_id,
                    "status": "failed",
                    "error": f"Worker exited with code {code} before writing web_result.json",
                    "task": store.get_task(session_id),
                }
            )

    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Bad web_result.json: {e}") from e
        try:
            refresh_task_indexes(session_id)
            result_inner = data.get("result") if isinstance(data.get("result"), dict) else {}
            sync_submission_refs_from_result(session_id)
            if data.get("ok") and task_has_remote_refs(store.get_task(session_id), result_inner):
                refresh_remote_progress(session_id)
            else:
                store.update_task(
                    session_id,
                    status="completed" if data.get("ok") else "failed",
                    phase="completed" if data.get("ok") else "failed",
                    error=None if data.get("ok") else str(data.get("error") or "Task failed"),
                )
        except Exception:
            pass
        task_now = store.get_task(session_id)
        status_now = (task_now or {}).get("status") or ("completed" if data.get("ok") else "failed")
        return JSONResponse(
            content={
                "session_id": session_id,
                "task_id": session_id,
                "status": status_now,
                "result": data,
                "task": task_now,
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


@app.get("/api/tasks")
async def api_tasks(limit: int = 100) -> JSONResponse:
    store = get_task_store()
    return JSONResponse(
        content={
            "backend": store.backend,
            "tasks": store.list_tasks(limit=limit),
        }
    )


@app.get("/api/tasks/{task_id}")
async def api_task_detail(task_id: str) -> JSONResponse:
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    store = get_task_store()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        refresh_task_indexes(task_id)
        sync_submission_refs_from_result(task_id)
        refresh_remote_progress(task_id)
    except Exception:
        pass
    task = store.get_task(task_id) or task
    task_out = _task_with_config_effective(task_id, task)

    return JSONResponse(
        content={
            "task": task_out,
            "events": store.list_events(task_id, limit=500),
            "artifacts": store.list_artifacts(task_id),
            "match_terms": store.list_match_terms(task_id),
        }
    )


@app.post("/api/tasks/{task_id}/refresh")
async def api_task_refresh(task_id: str) -> JSONResponse:
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    store = get_task_store()
    if store.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    refresh_task_indexes(task_id)
    sync_submission_refs_from_result(task_id)
    progress = refresh_remote_progress(task_id)
    t = store.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(
        content={
            "task": _task_with_config_effective(task_id, t),
            "remote_progress": progress,
            "artifacts": store.list_artifacts(task_id),
            "match_terms": store.list_match_terms(task_id),
        }
    )


@app.get("/api/tasks/{task_id}/events")
async def api_task_events(task_id: str, limit: int = 500) -> JSONResponse:
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    return JSONResponse(content={"events": get_task_store().list_events(task_id, limit=limit)})


@app.get("/api/tasks/{task_id}/artifacts")
async def api_task_artifacts(task_id: str) -> JSONResponse:
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    try:
        refresh_task_indexes(task_id)
    except Exception:
        pass
    store = get_task_store()
    return JSONResponse(
        content={
            "artifacts": store.list_artifacts(task_id),
            "match_terms": store.list_match_terms(task_id),
        }
    )


def _media_type_for_path(path: Path) -> str:
    low = path.name.lower()
    if low.endswith(".html") or low == "stereographic_interactive":
        return "text/html; charset=utf-8"
    if low.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if low.endswith(".png"):
        return "image/png"
    if low.endswith((".txt", ".log", ".json", ".jsonl", ".cif", ".vasp", ".poscar")):
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


@app.get("/api/tasks/{task_id}/artifact")
async def api_task_artifact_file(task_id: str, path: str) -> FileResponse:
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    root = (sessions_root() / task_id).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Artifact path escapes task directory") from e
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(
        target,
        filename=target.name,
        media_type=_media_type_for_path(target),
        content_disposition_type="inline",
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


@app.get("/api/session/{session_id}/relax_final_iface.png")
async def session_relax_final_iface_png(
    session_id: str,
    match_id: Optional[int] = None,
    term_id: Optional[int] = None,
) -> FileResponse:
    """
    MLIP relaxation final state rendered server-side with ASE ``plot_atoms``.

    With ``match_id`` and ``term_id``, serves ``web_relax_final_iface_m{m}_t{t}.png``
    (one file per relaxed interface pair). Without query params, serves legacy
    ``web_relax_final_iface.png`` when present.
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    root = sessions_root() / session_id
    if match_id is not None and term_id is not None:
        path = root / f"web_relax_final_iface_m{match_id}_t{term_id}.png"
    else:
        path = root / "web_relax_final_iface.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Relax-final interface image not available")
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
    if path is None and name == "stereographic_interactive.html":
        path = _find_artifact_under_session(session_id, "stereographic_interactive")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    media = _media_type_for_path(path)
    return FileResponse(
        path,
        filename=name,
        media_type=media,
        content_disposition_type="inline",
    )


@app.get("/api/session/{session_id}/structures.zip")
async def session_structures_zip(session_id: str, under: Optional[str] = None) -> Response:
    """
    ZIP relaxed interface structures:

    * **Legacy** — full ``pairs_best_it/`` POSCAR tree.
    * **Minimal export** — all ``match_*_term_*.cif`` inside a results directory
      (e.g. ``fetched_results/mlip_results``).

    * With ``under``: session-relative path to ``pairs_best_it`` or to a folder containing interface CIFs.
    * Default: first ``pairs_best_it`` in the session tree (legacy).
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    workdir = (sessions_root() / session_id).resolve()
    if not workdir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")

    pdir: Optional[Path] = None
    zip_mode = "pairs_best_it"
    dl_name = "pairs_best_it.zip"
    raw_under = (under or "").strip().replace("\\", "/")
    if raw_under:
        if raw_under.startswith("/") or ".." in Path(raw_under).parts:
            raise HTTPException(status_code=400, detail="Invalid under path")
        cand = (workdir / raw_under).resolve()
        try:
            cand.relative_to(workdir)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail="under path escapes session directory"
            ) from e
        if not cand.is_dir():
            raise HTTPException(status_code=404, detail="under path is not a directory")
        if cand.name == "pairs_best_it":
            pdir = cand
            zip_mode = "pairs_best_it"
            dl_name = "pairs_best_it.zip"
        else:
            cifs = sorted(cand.glob("match_*_term_*.cif"))
            if not cifs:
                raise HTTPException(
                    status_code=404,
                    detail="directory has no match_*_term_*.cif files (and is not pairs_best_it)",
                )
            pdir = cand
            zip_mode = "interface_cifs"
            dl_name = "interface_structures.zip"
    else:
        found = find_pairs_best_it_dir(workdir)
        if found is None or not found.is_dir():
            raise HTTPException(
                status_code=404,
                detail="pairs_best_it not found (pass under=… to a folder with interface CIFs).",
            )
        pdir = found
        zip_mode = "pairs_best_it"
        dl_name = "pairs_best_it.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if zip_mode == "pairs_best_it":
            root_label = Path("pairs_best_it")
            for f in sorted(pdir.rglob("*")):
                if f.is_file():
                    arc = root_label / f.relative_to(pdir)
                    zf.write(f, arcname=str(arc).replace("\\", "/"))
        else:
            root_label = Path(pdir.name)
            for f in sorted(pdir.glob("match_*_term_*.cif")):
                arc = root_label / f.name
                zf.write(f, arcname=str(arc).replace("\\", "/"))
    data = buf.getvalue()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{dl_name}"',
        },
    )


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/cluster/info")
async def api_cluster_info(refresh: bool = False) -> JSONResponse:
    """Return parsed ``sinfo`` snapshot (partitions, nodes, queue) for the manage UI."""
    return JSONResponse(content=query_cluster_info(force=bool(refresh)))


@app.get("/api/jfremote/workers")
async def api_jfremote_workers(project: Optional[str] = None) -> JSONResponse:
    """List jobflow-remote workers configured for ``project`` (default: $JFREMOTE_PROJECT)."""
    return JSONResponse(content=list_workers(project=project))


@app.post("/api/jfremote/probe_mlip")
async def api_jfremote_probe_mlip(request: Request) -> JSONResponse:
    """
    Find (and validate) a jobflow-remote worker that can run the requested MLIP.

    Body (JSON or form-encoded):
        - ``mlip``: required; one of ``orb-models``, ``sevenn``, ``matris``, ``dpa``
        - ``project``: optional jfremote project name
        - ``worker``: optional explicit worker name to test first
        - ``timeout``: optional probe timeout (seconds, default 240)
        - ``stop_at_first``: optional; if exactly JSON ``true`` or truthy strings
          (``"true"``, ``"1"``, …), stop after the first fully validated worker.
          Other values (including ``1`` as a number) are ignored so clients that
          enumerate workers still receive the full ``validated_workers`` list unless
          they explicitly opt into early exit.
        - ``device``: optional Si-probe device for :class:`MlipCalc`: ``cpu`` (default),
          ``cuda``, or ``gpu`` (normalized to ``cuda``). Should match the form's
          MLIP ``opt_device`` when validating workers for production runs.
        - ``probe_node``: optional Slurm hostname for ``srun -w`` on the **full**
          Si probe (requires ``srun`` and shared ``$HOME`` for scratch files).
        - ``probe_partition``: optional ``srun -p`` for the **full** Si probe. If set
          without ``probe_node``, the probe still runs under ``srun`` on a node in
          that partition (not on the web/SSH host shell).
    """
    payload: Dict[str, Any] = {}
    try:
        ctype = (request.headers.get("content-type") or "").lower()
        if "json" in ctype:
            payload = await request.json()
        else:
            form = await request.form()
            payload = {k: form.get(k) for k in form.keys()}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    mlip = str(payload.get("mlip") or "").strip()
    if not mlip:
        raise HTTPException(status_code=400, detail="missing 'mlip'")
    project = (payload.get("project") or None)
    if isinstance(project, str):
        project = project.strip() or None
    explicit_worker = (payload.get("worker") or None)
    if isinstance(explicit_worker, str):
        explicit_worker = explicit_worker.strip() or None
    try:
        timeout = int(payload.get("timeout") or 240)
    except (TypeError, ValueError):
        timeout = 240
    timeout = max(20, min(timeout, 900))
    # Only treat explicit JSON true / common truthy strings as stop-at-first.
    # ``bool(1)`` would otherwise enable early return and the manage UI would
    # only list the first passing worker in the dropdown.
    raw_stop = payload.get("stop_at_first")
    if isinstance(raw_stop, str):
        stop_at_first = raw_stop.strip().lower() in ("1", "true", "yes", "on")
    else:
        stop_at_first = raw_stop is True

    raw_dev = payload.get("device")
    if isinstance(raw_dev, str) and raw_dev.strip():
        probe_device = raw_dev.strip()
    else:
        probe_device = "cpu"

    raw_pn = payload.get("probe_node")
    probe_node = raw_pn.strip() if isinstance(raw_pn, str) and raw_pn.strip() else None
    raw_pp = payload.get("probe_partition")
    probe_partition = raw_pp.strip() if isinstance(raw_pp, str) and raw_pp.strip() else None

    result = auto_select_mlip_worker(
        mlip,
        project=project,
        explicit_worker=explicit_worker,
        timeout=timeout,
        stop_at_first=stop_at_first,
        probe_device=probe_device,
        probe_node=probe_node,
        probe_partition=probe_partition,
    )
    return JSONResponse(content=result)


@app.get("/api/jfremote/probe_mlip")
async def api_jfremote_probe_mlip_get(
    mlip: str,
    project: Optional[str] = None,
    worker: Optional[str] = None,
    timeout: int = 240,
    device: str = "cpu",
    probe_node: Optional[str] = None,
    probe_partition: Optional[str] = None,
) -> JSONResponse:
    """GET variant of :func:`api_jfremote_probe_mlip` for convenience."""
    pn = probe_node.strip() if isinstance(probe_node, str) and probe_node.strip() else None
    pp = probe_partition.strip() if isinstance(probe_partition, str) and probe_partition.strip() else None
    if worker:
        info = list_workers(project=project)
        target: Optional[Dict[str, Any]] = None
        for w in info.get("workers") or []:
            if w.get("name") == worker:
                target = w
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"worker {worker!r} not found in project")
        probe = probe_mlip_in_worker(
            mlip,
            target,
            timeout=max(20, min(int(timeout), 900)),
            device=device,
            probe_node=pn,
            probe_partition=pp,
        )
        return JSONResponse(
            content={
                "ok": bool(probe.get("ok")),
                "mlip": mlip,
                "project": project or "",
                "selected": target if probe.get("ok") else None,
                "candidates_tried": [worker],
                "probe": probe,
                "errors": [] if probe.get("ok") else [probe.get("error") or "probe failed"],
            }
        )
    return JSONResponse(
        content=auto_select_mlip_worker(
            mlip,
            project=project,
            explicit_worker=None,
            timeout=max(20, min(int(timeout), 900)),
            probe_device=device,
            probe_node=pn,
            probe_partition=pp,
        )
    )


@app.get("/api/modules/avail")
async def api_modules_avail(pattern: str = "VASP", timeout: int = 20) -> JSONResponse:
    """Run ``module avail <pattern>`` and return the parsed module specifiers."""
    return JSONResponse(
        content=query_module_avail(pattern=pattern, timeout=max(5, min(int(timeout), 60)))
    )


@app.get("/api/tasks/{task_id}/poscar_view")
async def api_task_poscar_view(task_id: str, path: str) -> Response:
    """
    Convert a relaxed interface structure (POSCAR / CONTCAR / CIF) under the task workdir
    to a P1 CIF for 3Dmol.

    ``path`` is a session-relative artifact path returned by
    :func:`scan_artifacts` (``relative_path`` field).
    """
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    root = (sessions_root() / task_id).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Path escapes task directory") from e
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Structure file not found")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    suf = target.suffix.lower()
    if suf == ".cif":
        cif = expand_cif_for_view(text, with_bonds=True)
        if not cif:
            cif = text
        return Response(
            content=cif,
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )
    cif = poscar_to_cif(text)
    if not cif:
        raise HTTPException(
            status_code=415,
            detail="POSCAR could not be parsed (pymatgen unavailable or malformed file)",
        )
    return Response(
        content=cif,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/tasks/{task_id}/fetch_results")
async def api_task_fetch_results(task_id: str) -> JSONResponse:
    """
    Pull final ``mlip_results`` / ``vasp_results`` artifacts for a remote
    task into ``<session>/fetched_results/`` so the manage detail page can
    show them. For local tasks this is a no-op (artifacts live on disk
    already) and the call simply re-scans the session tree.
    """
    if not is_safe_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    workdir = sessions_root() / task_id
    if not workdir.is_dir():
        raise HTTPException(status_code=404, detail="Task workdir missing")

    payload: Dict[str, Any] = {
        "task_id": task_id,
        "ok": False,
        "skipped_reason": "",
        "fetched_results_dir": "",
        "fetch_summary": None,
    }
    result_path = workdir / "web_result.json"
    web_result: Dict[str, Any] = {}
    if result_path.is_file():
        try:
            web_result = json.loads(result_path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            web_result = {}
    inner = web_result.get("result") if isinstance(web_result.get("result"), dict) else {}
    mlip_uuid = str(
        inner.get("mlip_job_uuid")
        or (inner.get("interoptimus_task_record") or {}).get("mlip_job_uuid")
        or ""
    ).strip()

    is_remote = bool(
        inner.get("flow_uuid")
        or inner.get("mlip_job_uuid")
        or inner.get("vasp_job_uuids")
        or (
            isinstance(inner.get("server_submission"), dict)
            and inner["server_submission"].get("success")
        )
    )

    if not is_remote:
        payload["ok"] = True
        payload["skipped_reason"] = "local task — artifacts are produced on disk"
        try:
            refresh_task_indexes(task_id)
        except Exception as e:
            payload["refresh_error"] = str(e)
        return JSONResponse(content=payload)

    try:
        from InterOptimus.agents.remote_submit import iomaker_fetch_results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"iomaker_fetch_results unavailable: {e}",
        ) from e

    dest_dir = workdir / "fetched_results"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"cannot create fetched_results: {e}") from e

    try:
        fr = iomaker_fetch_results(
            str(dest_dir),
            inner,
            ref=mlip_uuid,
            jf_job_id=mlip_uuid,
            include_progress=True,
            verbose=False,
        )
    except Exception as e:
        # Catch broken-pipe / jf-remote / Mongo errors — but propagate KeyboardInterrupt
        # / SystemExit so a `kill` actually stops the server.
        payload["error"] = str(e)
        payload["error_type"] = type(e).__name__
        return JSONResponse(content=payload, status_code=200)

    payload["ok"] = bool(fr and fr.get("success"))
    payload["fetched_results_dir"] = str(dest_dir.resolve())
    payload["fetch_summary"] = {
        k: fr.get(k)
        for k in (
            "success",
            "error",
            "run_dir",
            "export_dir",
            "vasp_results_dir",
            "vasp_io_report_path",
            "vasp_pairs_summary_path",
            "report_markdown_path",
            "task_ref_used",
        )
        if isinstance(fr, dict)
    }
    try:
        refresh_task_indexes(task_id)
    except Exception as e:
        payload["refresh_error"] = str(e)
    return JSONResponse(content=payload)
