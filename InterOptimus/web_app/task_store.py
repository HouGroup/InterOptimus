"""Task metadata store for the InterOptimus web UI.

The web UI should remain usable on a laptop without MongoDB, while cluster
deployments should persist task metadata in the same MongoDB configured for
jobflow.  This module provides a small store abstraction with MongoDB-first
``auto`` mode and a JSON fallback under the web session root.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from InterOptimus.iomaker_minimal_export import (
    FN_ALL_MATCH_INFO,
    FN_SELECTED_CSV,
    FN_STEREO_INTERACTIVE,
    FN_STEREO_JPG,
)
from InterOptimus.session_workflow import sessions_root

TASK_COLLECTION = "io_tasks"
EVENT_COLLECTION = "io_task_events"
ARTIFACT_COLLECTION = "io_artifacts"
MATCH_TERM_COLLECTION = "io_match_terms"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_safe_task_id(task_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id or ""))


def task_db_root() -> Path:
    root = Path(os.environ.get("INTEROPTIMUS_TASK_DB_DIR", "")).expanduser()
    if not str(root) or str(root) == ".":
        root = sessions_root() / "_task_db"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_safe(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        if isinstance(v, dict):
            return {str(k): _json_safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        return str(v)


def _clean_mongo_doc(doc: Optional[dict]) -> Optional[dict]:
    if not doc:
        return None
    out = dict(doc)
    out.pop("_id", None)
    return out


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _jobflow_mongo_config() -> Optional[dict]:
    uri = (os.environ.get("INTEROPTIMUS_TASK_MONGO_URI") or "").strip()
    db = (os.environ.get("INTEROPTIMUS_TASK_MONGO_DB") or "").strip()
    if uri and db:
        return {"uri": uri, "database": db}

    cfg_path = Path(
        os.environ.get("JOBFLOW_CONFIG_FILE") or (Path.home() / ".jobflow.yaml")
    ).expanduser()
    cfg = _load_yaml(cfg_path)
    store = ((cfg.get("JOB_STORE") or {}).get("docs_store") or {})
    if not isinstance(store, dict):
        return None
    database = store.get("database")
    if not database:
        return None
    out = {
        "host": store.get("host", "localhost"),
        "port": int(store.get("port", 27017)),
        "database": str(database),
    }
    if store.get("username"):
        out["username"] = str(store.get("username"))
        out["password"] = str(store.get("password", ""))
    return out


class TaskStore(Protocol):
    backend: str

    def create_task(self, task: dict) -> dict: ...

    def update_task(self, task_id: str, updates: dict) -> dict: ...

    def get_task(self, task_id: str) -> Optional[dict]: ...

    def list_tasks(self, limit: int = 100) -> List[dict]: ...

    def append_event(self, task_id: str, event: str, message: str = "", **data: Any) -> dict: ...

    def list_events(self, task_id: str, limit: int = 500) -> List[dict]: ...

    def upsert_artifacts(self, task_id: str, artifacts: List[dict]) -> None: ...

    def list_artifacts(self, task_id: str) -> List[dict]: ...

    def upsert_match_terms(self, task_id: str, terms: List[dict]) -> None: ...

    def list_match_terms(self, task_id: str) -> List[dict]: ...

    def delete_task(self, task_id: str) -> bool: ...


class FileTaskStore:
    backend = "file"

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or task_db_root()
        self.path = self.root / "tasks_db.json"
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write({"tasks": {}, "events": [], "artifacts": [], "match_terms": {}})

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        data.setdefault("tasks", {})
        data.setdefault("events", [])
        data.setdefault("artifacts", [])
        data.setdefault("match_terms", {})
        return data

    def _write(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="tasks_db.", suffix=".tmp", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, self.path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def create_task(self, task: dict) -> dict:
        now = utc_now()
        doc = {**_json_safe(task), "created_at": now, "updated_at": now}
        with self._lock:
            data = self._read()
            data["tasks"][doc["task_id"]] = doc
            self._write(data)
        return dict(doc)

    def update_task(self, task_id: str, updates: dict) -> dict:
        with self._lock:
            data = self._read()
            task = dict(data["tasks"].get(task_id) or {"task_id": task_id, "created_at": utc_now()})
            task.update(_json_safe(updates))
            task["updated_at"] = utc_now()
            data["tasks"][task_id] = task
            self._write(data)
        return task

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._read()["tasks"].get(task_id) or {}) or None

    def list_tasks(self, limit: int = 100) -> List[dict]:
        with self._lock:
            tasks = [dict(x) for x in self._read()["tasks"].values()]
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks[: max(1, min(limit, 1000))]

    def append_event(self, task_id: str, event: str, message: str = "", **data: Any) -> dict:
        row = {
            "task_id": task_id,
            "event": event,
            "message": message,
            "created_at": utc_now(),
            "ts": time.time(),
            **_json_safe(data),
        }
        with self._lock:
            db = self._read()
            db["events"].append(row)
            task = dict(db["tasks"].get(task_id) or {"task_id": task_id, "created_at": utc_now()})
            task["updated_at"] = row["created_at"]
            db["tasks"][task_id] = task
            self._write(db)
        return dict(row)

    def list_events(self, task_id: str, limit: int = 500) -> List[dict]:
        with self._lock:
            rows = [dict(x) for x in self._read()["events"] if x.get("task_id") == task_id]
        rows.sort(key=lambda x: x.get("ts", 0))
        return rows[-max(1, min(limit, 5000)) :]

    def upsert_artifacts(self, task_id: str, artifacts: List[dict]) -> None:
        with self._lock:
            db = self._read()
            rest = [x for x in db["artifacts"] if x.get("task_id") != task_id]
            db["artifacts"] = rest + [_json_safe({**x, "task_id": task_id}) for x in artifacts]
            self._write(db)

    def list_artifacts(self, task_id: str) -> List[dict]:
        with self._lock:
            return [dict(x) for x in self._read()["artifacts"] if x.get("task_id") == task_id]

    def upsert_match_terms(self, task_id: str, terms: List[dict]) -> None:
        key = str(task_id)
        with self._lock:
            db = self._read()
            db["match_terms"][key] = [_json_safe(x) for x in terms]
            self._write(db)

    def list_match_terms(self, task_id: str) -> List[dict]:
        with self._lock:
            return [dict(x) for x in self._read()["match_terms"].get(str(task_id), [])]

    def delete_task(self, task_id: str) -> bool:
        """Drop the task and all associated events / artifacts / match_terms."""
        key = str(task_id)
        with self._lock:
            db = self._read()
            existed = key in db["tasks"]
            db["tasks"].pop(key, None)
            db["events"] = [x for x in db["events"] if x.get("task_id") != key]
            db["artifacts"] = [x for x in db["artifacts"] if x.get("task_id") != key]
            db["match_terms"].pop(key, None)
            self._write(db)
        return existed


class MongoTaskStore:
    backend = "mongo"

    def __init__(self, config: dict) -> None:
        from pymongo import ASCENDING, DESCENDING, MongoClient

        if config.get("uri"):
            self.client = MongoClient(config["uri"], serverSelectionTimeoutMS=3000)
        else:
            kwargs: Dict[str, Any] = {
                "host": config.get("host", "localhost"),
                "port": int(config.get("port", 27017)),
                "serverSelectionTimeoutMS": 3000,
            }
            if config.get("username"):
                kwargs["username"] = config["username"]
                kwargs["password"] = config.get("password", "")
            self.client = MongoClient(**kwargs)
        self.client.admin.command("ping")
        self.db = self.client[str(config["database"])]
        self.tasks = self.db[TASK_COLLECTION]
        self.events = self.db[EVENT_COLLECTION]
        self.artifacts = self.db[ARTIFACT_COLLECTION]
        self.match_terms = self.db[MATCH_TERM_COLLECTION]
        self.tasks.create_index([("task_id", ASCENDING)], unique=True)
        self.tasks.create_index([("created_at", DESCENDING)])
        self.events.create_index([("task_id", ASCENDING), ("ts", ASCENDING)])
        self.artifacts.create_index([("task_id", ASCENDING), ("name", ASCENDING)])
        self.match_terms.create_index([("task_id", ASCENDING), ("match_id", ASCENDING), ("term_id", ASCENDING)])

    def create_task(self, task: dict) -> dict:
        now = utc_now()
        doc = {**_json_safe(task), "created_at": now, "updated_at": now}
        self.tasks.update_one({"task_id": doc["task_id"]}, {"$setOnInsert": doc}, upsert=True)
        return self.get_task(doc["task_id"]) or doc

    def update_task(self, task_id: str, updates: dict) -> dict:
        doc = {**_json_safe(updates), "updated_at": utc_now()}
        self.tasks.update_one(
            {"task_id": task_id},
            {"$set": doc, "$setOnInsert": {"task_id": task_id, "created_at": utc_now()}},
            upsert=True,
        )
        return self.get_task(task_id) or {"task_id": task_id, **doc}

    def get_task(self, task_id: str) -> Optional[dict]:
        return _clean_mongo_doc(self.tasks.find_one({"task_id": task_id}))

    def list_tasks(self, limit: int = 100) -> List[dict]:
        cur = self.tasks.find({}, {"_id": 0}).sort("created_at", -1).limit(max(1, min(limit, 1000)))
        return [dict(x) for x in cur]

    def append_event(self, task_id: str, event: str, message: str = "", **data: Any) -> dict:
        row = {
            "task_id": task_id,
            "event": event,
            "message": message,
            "created_at": utc_now(),
            "ts": time.time(),
            **_json_safe(data),
        }
        self.events.insert_one(dict(row))
        self.tasks.update_one({"task_id": task_id}, {"$set": {"updated_at": row["created_at"]}})
        return row

    def list_events(self, task_id: str, limit: int = 500) -> List[dict]:
        cur = (
            self.events.find({"task_id": task_id}, {"_id": 0})
            .sort("ts", 1)
            .limit(max(1, min(limit, 5000)))
        )
        return [dict(x) for x in cur]

    def upsert_artifacts(self, task_id: str, artifacts: List[dict]) -> None:
        self.artifacts.delete_many({"task_id": task_id})
        if artifacts:
            self.artifacts.insert_many([_json_safe({**x, "task_id": task_id}) for x in artifacts])

    def list_artifacts(self, task_id: str) -> List[dict]:
        return [dict(x) for x in self.artifacts.find({"task_id": task_id}, {"_id": 0})]

    def upsert_match_terms(self, task_id: str, terms: List[dict]) -> None:
        self.match_terms.delete_many({"task_id": task_id})
        if terms:
            self.match_terms.insert_many([_json_safe({**x, "task_id": task_id}) for x in terms])

    def list_match_terms(self, task_id: str) -> List[dict]:
        cur = self.match_terms.find({"task_id": task_id}, {"_id": 0}).sort([("match_id", 1), ("term_id", 1)])
        return [dict(x) for x in cur]

    def delete_task(self, task_id: str) -> bool:
        """Drop the task and all associated events / artifacts / match_terms across collections."""
        res = self.tasks.delete_one({"task_id": task_id})
        existed = bool(getattr(res, "deleted_count", 0))
        self.events.delete_many({"task_id": task_id})
        self.artifacts.delete_many({"task_id": task_id})
        self.match_terms.delete_many({"task_id": task_id})
        return existed


_STORE: Optional[TaskStore] = None
_STORE_LOCK = threading.Lock()


def get_task_store() -> TaskStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            return _STORE
        mode = (os.environ.get("INTEROPTIMUS_TASK_STORE") or "auto").strip().lower()
        if mode not in ("auto", "mongo", "file"):
            mode = "auto"
        if mode in ("auto", "mongo"):
            cfg = _jobflow_mongo_config()
            if cfg:
                try:
                    _STORE = MongoTaskStore(cfg)
                    return _STORE
                except Exception:
                    if mode == "mongo":
                        raise
        _STORE = FileTaskStore()
        return _STORE


_CLUSTER_FORM_KEYS = (
    "slurm_partition",
    "cpus_per_gpu",
    "mlip_cpus_per_task",
    "gpus_per_job",
    "mlip_scheduler_kwargs",
    "mlip_worker",
    "mlip_project",
    "server_pre_cmd",
    "server_run_parent",
    "server_python",
    "server_jf_bin",
    "vasp_slurm_partition",
    "vasp_nodes",
    "vasp_processes_per_node",
    "vasp_worker",
    "vasp_pre_run",
)
_VASP_FORM_KEYS = (
    "vasp_pair_selection",
    "do_vasp_gd",
    "vasp_dipole_correction",
    "relax_user_incar_settings",
    "relax_user_kpoints_settings",
    "relax_user_potcar_settings",
    "relax_user_potcar_functional",
    "static_user_incar_settings",
    "static_user_kpoints_settings",
    "static_user_potcar_settings",
    "static_user_potcar_functional",
    "vasp_gd_kwargs",
    "vasp_relax_settings",
    "vasp_static_settings",
)


def _summarize_form_section(form: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        v = form.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out[k] = s
    return out


def create_initial_task(
    *,
    task_id: str,
    workdir: Path,
    form: Dict[str, Any],
    film_name: str,
    substrate_name: str,
) -> dict:
    mode = "mlip+vasp" if str(form.get("do_vasp", "")).lower() in ("1", "true", "yes", "on") else "mlip"
    cluster_summary = _summarize_form_section(form, _CLUSTER_FORM_KEYS)
    vasp_summary = _summarize_form_section(form, _VASP_FORM_KEYS)
    return {
        "task_id": task_id,
        "session_id": task_id,
        "name": str(form.get("workflow_name") or "IO_web"),
        "status": "queued",
        "phase": "queued",
        "mode": mode,
        "mlip_calc": str(form.get("mlip_calc") or "orb-models"),
        "execution": str(form.get("execution") or "local"),
        "ui_mode": str(form.get("ui_mode") or ""),
        "film_name": film_name,
        "substrate_name": substrate_name,
        "workdir": str(workdir.resolve(strict=False)),
        "form": _json_safe(form),
        "cluster_summary": cluster_summary,
        "vasp_summary": vasp_summary,
    }


def _phase_for_relpath(rel: str) -> str:
    """
    Tag an artifact's phase based on its location under the session tree.

    - Anything under a ``vasp_results/`` folder -> "vasp"
    - Otherwise -> "mlip"
    """
    parts = rel.replace("\\", "/").split("/")
    return "vasp" if "vasp_results" in parts else "mlip"


def _results_skip_vasp_subtree(rel_posix: str) -> bool:
    """When scanning MLIP export root, ignore paths inside ``vasp_results/``."""
    return "vasp_results/" in rel_posix or rel_posix.startswith("vasp_results/")


def _result_scan_roots(workdir: Path) -> List[tuple[str, Path, bool]]:
    """
    Return ``(phase, root_dir, skip_vasp_subtree)`` tuples.

    Only ``mlip_results/``, ``vasp_results/``, and remote ``fetched_results/``
    (with its MLIP-at-root convention) are indexed — matching the manage UI.
    """
    roots: List[tuple[str, Path, bool]] = []
    wr = workdir.resolve(strict=False)
    mlip_top = wr / "mlip_results"
    vasp_top = wr / "vasp_results"
    if mlip_top.is_dir():
        roots.append(("mlip", mlip_top.resolve(), False))
    if vasp_top.is_dir():
        roots.append(("vasp", vasp_top.resolve(), False))

    fr = wr / "fetched_results"
    if fr.is_dir():
        fr = fr.resolve()
        mlip_sub = fr / "mlip_results"
        vasp_sub = fr / "vasp_results"
        if mlip_sub.is_dir():
            roots.append(("mlip", mlip_sub.resolve(), False))
        else:
            roots.append(("mlip", fr, True))
        if vasp_sub.is_dir():
            roots.append(("vasp", vasp_sub.resolve(), False))
    return roots


# Public artifacts at each MLIP / VASP minimal export root (plus optional legacy ``pairs_best_it/``).
_CORE_RESULT_FILES: tuple[tuple[str, str], ...] = (
    (FN_ALL_MATCH_INFO, "stereo_table"),
    (FN_STEREO_JPG, "stereo_plot_jpg"),
    (FN_STEREO_INTERACTIVE, "stereo_interactive_html"),
    (FN_SELECTED_CSV, "selected_interfaces_csv"),
)


def scan_artifacts(task_id: str) -> List[dict]:
    """
    Index for the manage UI:

    * **MLIP / VASP** — under each export root: ``all_match_info``, ``stereographic.jpg``,
      ``stereographic_interactive``, ``selected_interfaces.csv`` (CIF text in the last column).
    * **Legacy** — ``pairs_best_it/`` with POSCAR trees is still scanned when present.

    See :func:`_result_scan_roots` for path resolution (``mlip_results/``, ``fetched_results/``, …).
    """
    workdir = sessions_root() / task_id
    if not workdir.is_dir():
        return []
    artifacts: List[dict] = []
    seen_paths: set = set()
    seen_pbi: set = set()
    wr = workdir.resolve()

    def add_file(
        path: Path,
        kind: str,
        name: str,
        phase: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not path.is_file():
            return
        try:
            stat = path.stat()
            rel = str(path.resolve().relative_to(wr))
        except (OSError, ValueError):
            return
        rel_posix = rel.replace("\\", "/")
        if rel_posix in seen_paths:
            return
        seen_paths.add(rel_posix)
        item: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "path": str(path.resolve(strict=False)),
            "relative_path": rel_posix,
            "size": stat.st_size,
            "updated_at": utc_now(),
            "phase": phase,
        }
        if extra:
            item.update(extra)
        artifacts.append(item)

    def add_pairs_best_it(pbi: Path, phase: str) -> None:
        if not pbi.is_dir() or pbi.name != "pairs_best_it":
            return
        try:
            rel_posix = str(pbi.resolve().relative_to(wr)).replace("\\", "/")
        except (OSError, ValueError):
            return
        if rel_posix in seen_pbi:
            return
        seen_pbi.add(rel_posix)
        artifacts.append(
            {
                "name": "pairs_best_it",
                "kind": "structure_dir",
                "path": str(pbi.resolve(strict=False)),
                "relative_path": rel_posix,
                "updated_at": utc_now(),
                "phase": phase,
            }
        )
        try:
            for sub in sorted(pbi.iterdir()):
                if not sub.is_dir():
                    continue
                m = re.match(r"^match[_-]?(\d+).*?term[_-]?(\d+)\s*$", sub.name)
                if not m:
                    continue
                mid, tid = int(m.group(1)), int(m.group(2))
                for cand in ("best_it_POSCAR", "POSCAR", "CONTCAR"):
                    pf = sub / cand
                    if pf.is_file():
                        add_file(
                            pf,
                            "interface_poscar",
                            f"match {mid} · term {tid}",
                            phase,
                            extra={"match_id": mid, "term_id": tid, "poscar_kind": cand},
                        )
                        break
        except OSError:
            pass

    try:
        for phase, root, skip_vasp in _result_scan_roots(workdir):
            try:
                root_r = root.resolve()
            except OSError:
                continue

            for fname, kind in _CORE_RESULT_FILES:
                p = root_r / fname
                if not p.is_file():
                    continue
                try:
                    rel_posix = str(p.resolve().relative_to(wr)).replace("\\", "/")
                except (OSError, ValueError):
                    continue
                if skip_vasp and _results_skip_vasp_subtree(rel_posix):
                    continue
                add_file(p, kind, fname, phase)

            pbi = root_r / "pairs_best_it"
            if pbi.is_dir():
                try:
                    rel_pbi = str(pbi.resolve().relative_to(wr)).replace("\\", "/")
                except (OSError, ValueError):
                    rel_pbi = ""
                if rel_pbi and skip_vasp and _results_skip_vasp_subtree(rel_pbi):
                    continue
                add_pairs_best_it(pbi, phase)
    except OSError:
        pass

    fr = workdir / "fetched_results"
    vasp_md = fr / "vasp_interface_energy_report.md"
    if vasp_md.is_file():
        add_file(vasp_md, "vasp_interface_energy_md", vasp_md.name, "vasp")
    vasp_plot = fr / "vasp_interface_energy_plot.jpg"
    if vasp_plot.is_file():
        add_file(vasp_plot, "vasp_interface_energy_plot", vasp_plot.name, "vasp")

    # ``pairs_best_it/`` is written at the ``fetched_results/`` root by remote
    # fetch (legacy convention), not under ``fetched_results/mlip_results/``. The
    # ``_result_scan_roots`` loop above only checks each export root, so without
    # this fallback the 3D interface viewer never gets ``interface_poscar``
    # entries and stays at its empty placeholder.
    if fr.is_dir():
        try:
            fr_resolved = fr.resolve()
            pbi_root = fr_resolved / "pairs_best_it"
            if pbi_root.is_dir():
                add_pairs_best_it(pbi_root, "mlip")
        except OSError:
            pass

    return artifacts


def scan_match_terms(task_id: str) -> List[dict]:
    from InterOptimus.web_app.session_artifacts import find_pairs_best_it_dir

    workdir = sessions_root() / task_id
    terms: Dict[tuple[int, int], dict] = {}
    viz = workdir / "web_viz.jsonl"
    if viz.is_file():
        try:
            lines = viz.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                mid = int(ev.get("match_id"))
                tid = int(ev.get("term_id"))
            except (TypeError, ValueError):
                continue
            row = terms.setdefault(
                (mid, tid),
                {"match_id": mid, "term_id": tid, "status": "mlip_running", "relax_steps": 0},
            )
            if ev.get("event") == "relax_step":
                row["status"] = "mlip_running"
                row["relax_steps"] = max(int(row.get("relax_steps") or 0), int(ev.get("step") or 0))
            elif ev.get("event") == "relax_final":
                row["status"] = "mlip_done"
                for key in ("energy", "energy_per_atom", "interface_gamma_J_m2"):
                    if key in ev:
                        row[key] = ev[key]
                img = workdir / f"web_relax_final_iface_m{mid}_t{tid}.png"
                if img.is_file():
                    row["image"] = img.name

    pdir = find_pairs_best_it_dir(workdir)
    if pdir and pdir.is_dir():
        flat_re = re.compile(r"^match[_-]?(\d+).*?term[_-]?(\d+)\s*$")
        try:
            for sub in pdir.iterdir():
                if not sub.is_dir():
                    continue
                fm = flat_re.match(sub.name)
                if fm:
                    mid, tid = int(fm.group(1)), int(fm.group(2))
                    row = terms.setdefault((mid, tid), {"match_id": mid, "term_id": tid})
                    row.setdefault("status", "mlip_done")
                    row["structure_dir"] = str(sub.resolve(strict=False))
                    continue
                mm = re.search(r"match[_-]?(\d+)", sub.name)
                if not mm:
                    continue
                mid = int(mm.group(1))
                for tdir in sub.iterdir():
                    if not tdir.is_dir():
                        continue
                    tm = re.search(r"term[_-]?(\d+)", tdir.name)
                    if not tm:
                        continue
                    tid = int(tm.group(1))
                    row = terms.setdefault((mid, tid), {"match_id": mid, "term_id": tid})
                    row.setdefault("status", "mlip_done")
                    row["structure_dir"] = str(tdir.resolve(strict=False))
        except OSError:
            pass
    return [terms[k] for k in sorted(terms)]


def local_mlip_stage_done_on_disk(task_id: str) -> bool:
    """
    True when MLIP outputs are present under the session directory (including
    ``fetched_results/`` after the user pulls remote artifacts).

    ``jf job info`` / Mongo can still show READY/WAITING while the worker has
    already finished and results were copied locally — the UI should not stay
    on “MLIP 阶段进行中” in that case.
    """
    root = sessions_root() / task_id
    if not root.is_dir():
        return False
    fr = root / "fetched_results"
    try:
        opt = fr / "opt_results_summary.json"
        if opt.is_file() and opt.stat().st_size > 100:
            return True
        ps = fr / "pairs_summary.txt"
        if ps.is_file() and ps.stat().st_size > 30:
            return True
        aml = fr / "mlip_results" / "all_match_info"
        if aml.exists():
            return True
    except OSError:
        pass
    try:
        for sub in root.iterdir():
            if not sub.is_dir() or sub.name == "fetched_results":
                continue
            if (sub / "pairs_summary.txt").is_file():
                return True
            pkl = sub / "opt_results.pkl"
            if pkl.is_file() and pkl.stat().st_size > 32:
                return True
    except OSError:
        pass
    return False


def _apply_local_mlip_done_progress_override(task_id: str, progress: dict) -> dict:
    if not progress.get("success"):
        return progress
    if str(progress.get("mode") or "") != "mlip_and_vasp":
        return progress
    if str(progress.get("current_phase") or "") != "mlip":
        return progress
    if not local_mlip_stage_done_on_disk(task_id):
        return progress
    counts = progress.get("expanded_vasp_job_counts") or {}
    if int(counts.get("total") or 0) > 0:
        return progress
    out = dict(progress)
    out["current_phase"] = "vasp"
    out["is_finished"] = False
    out["mlip_local_done_override"] = True
    js = out.get("job_state") or "UNKNOWN"
    out["stage_summary"] = (
        "当前为 VASP 阶段。已在会话目录检测到 MLIP 结果（如 fetched_results）；"
        f"队列中根作业状态可能尚未同步（job_state={js}）。真实 VASP 子任务展开后计数会更新。"
    )
    return out


def _interoptimus_registry_path() -> Path:
    """Path to ``~/.interoptimus/iomaker_tasks.json`` (overridable via env var).

    Mirrors ``InterOptimus.agents.remote_submit._interoptimus_registry_path`` so
    the web UI can read entries written by ``run_simple_iomaker`` directly,
    without having to import ``remote_submit`` at module load time (it pulls in
    pymatgen / jobflow on import, which we want to defer).
    """
    base = os.environ.get("INTEROPTIMUS_TASK_REGISTRY_DIR") or str(Path.home() / ".interoptimus")
    return Path(base) / "iomaker_tasks.json"


def _load_simple_iomaker_registry() -> Dict[str, Any]:
    p = _interoptimus_registry_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _safe_simple_iomaker_task_id(serial_id: str) -> str:
    """Coerce a registry serial_id into a value that passes ``is_safe_task_id``."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(serial_id or "").strip())
    s = s.lstrip("._-") or "task"
    return s[:128]


def import_simple_iomaker_tasks_into_store() -> Dict[str, Any]:
    """Mirror ``run_simple_iomaker`` registry entries into the web TaskStore.

    For each ``serial_id`` in ``~/.interoptimus/iomaker_tasks.json`` not yet
    present in the TaskStore, insert a stub task with ``mlip_job_uuid`` /
    ``flow_uuid`` / ``vasp_job_uuids`` / ``submit_workdir`` populated so the
    standard ``/manage`` listing + ``refresh_remote_progress`` pipeline works
    against simple-iomaker submissions exactly the same as web-form submissions.

    Idempotent + de-duplicating:
        * Skip serial_ids that already have a TaskStore row.
        * Skip registry entries whose ``mlip_job_uuid`` / ``flow_uuid`` /
          ``jf_job_id`` already matches an existing row (avoids creating a
          duplicate when the GUI form already produced a UUID-shaped task_id).
        * Skip entries with no remote handles at all (nothing the UI can do
          with them — no progress, no fetch, no cancel).

    Returns ``{imported: [...], skipped: int, errors: [...]}``.
    """
    out: Dict[str, Any] = {"imported": [], "skipped": 0, "errors": []}
    reg = _load_simple_iomaker_registry()
    if not reg:
        return out
    store = get_task_store()
    existing = store.list_tasks(limit=1000)
    known_uuids: set[str] = set()
    for t in existing:
        for k in ("mlip_job_uuid", "flow_uuid", "submit_jf_job_id", "task_id"):
            v = t.get(k)
            if isinstance(v, str) and v.strip():
                known_uuids.add(v.strip().lower())
        for u in t.get("vasp_job_uuids") or []:
            if isinstance(u, str) and u.strip():
                known_uuids.add(u.strip().lower())
    for serial_id, rec in reg.items():
        if not isinstance(rec, dict):
            continue
        tid = _safe_simple_iomaker_task_id(rec.get("serial_id") or serial_id)
        if not is_safe_task_id(tid):
            out["errors"].append(f"unsafe_serial:{serial_id}")
            continue
        if store.get_task(tid):
            out["skipped"] += 1
            continue
        rec_uuids = {
            (rec.get("mlip_job_uuid") or "").strip().lower(),
            (rec.get("flow_uuid") or "").strip().lower(),
            (rec.get("jf_job_id") or "").strip().lower(),
        }
        rec_uuids.discard("")
        if rec_uuids and rec_uuids & known_uuids:
            out["skipped"] += 1
            continue
        if not rec_uuids:
            out["skipped"] += 1
            continue
        do_vasp = bool(rec.get("do_vasp"))
        vasp_uuids = list(rec.get("vasp_job_uuids") or [])
        try:
            doc = {
                "task_id": tid,
                "session_id": tid,
                "name": str(rec.get("task_name") or "simple_iomaker"),
                "status": "remote_running",
                "phase": "remote",
                "mode": "mlip+vasp" if do_vasp else "mlip",
                "mlip_calc": "",
                "execution": "server",
                "ui_mode": "simple_iomaker",
                "source": "simple_iomaker",
                "film_name": "",
                "substrate_name": "",
                "workdir": str(rec.get("submit_workdir") or ""),
                "submit_workdir": str(rec.get("submit_workdir") or ""),
                "submit_jf_job_id": str(rec.get("jf_job_id") or "").strip() or None,
                "mlip_job_uuid": str(rec.get("mlip_job_uuid") or "").strip() or None,
                "flow_uuid": str(rec.get("flow_uuid") or "").strip() or None,
                "vasp_job_uuids": [str(x).strip() for x in vasp_uuids if str(x).strip()],
                "interoptimus_task_serial": str(rec.get("serial_id") or serial_id),
                "interoptimus_task_record": rec,
                "stage_summary": "(simple_iomaker submission, refresh for status)",
            }
            doc_created = store.create_task(doc)
            store.append_event(
                tid,
                "imported",
                "Imported from run_simple_iomaker registry",
                source="simple_iomaker",
            )
            out["imported"].append({"task_id": tid, "serial_id": doc_created.get("interoptimus_task_serial")})
            for k in rec_uuids:
                if k:
                    known_uuids.add(k)
        except Exception as e:
            out["errors"].append(f"{serial_id}:{e}")
    return out


def refresh_task_indexes(task_id: str) -> None:
    store = get_task_store()
    workdir = sessions_root() / task_id
    try:
        from InterOptimus.iomaker_minimal_export import ensure_mlip_csv_from_summary_fallback

        ensure_mlip_csv_from_summary_fallback(str(workdir.resolve(strict=False)))
    except Exception:
        pass
    artifacts = scan_artifacts(task_id)
    terms = scan_match_terms(task_id)
    store.upsert_artifacts(task_id, artifacts)
    store.upsert_match_terms(task_id, terms)
    done = sum(1 for t in terms if str(t.get("status")) in ("mlip_done", "vasp_done", "done"))
    if terms:
        store.update_task(
            task_id,
            {
                "progress": {
                    "match_terms_total": len(terms),
                    "match_terms_done": done,
                }
            },
        )


def _load_web_result(task_id: str) -> dict:
    p = sessions_root() / task_id / "web_result.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _server_result_payload(task_id: str) -> dict:
    payload = _load_web_result(task_id)
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


def _submission_refs_from_result(result: dict) -> dict:
    ss = result.get("server_submission") if isinstance(result.get("server_submission"), dict) else {}
    rec = result.get("interoptimus_task_record") if isinstance(result.get("interoptimus_task_record"), dict) else {}
    vasp = result.get("vasp_job_uuids")
    if vasp is None:
        vasp = ss.get("vasp_job_uuids")
    if vasp is None:
        vasp = rec.get("vasp_job_uuids")
    if not isinstance(vasp, (list, tuple)):
        vasp = []
    return {
        "submit_jf_job_id": str(ss.get("job_id") or rec.get("jf_job_id") or "").strip() or None,
        "mlip_job_uuid": str(result.get("mlip_job_uuid") or ss.get("mlip_job_uuid") or rec.get("mlip_job_uuid") or "").strip() or None,
        "flow_uuid": str(result.get("flow_uuid") or ss.get("flow_uuid") or rec.get("flow_uuid") or "").strip() or None,
        "vasp_job_uuids": [str(x).strip() for x in vasp if str(x).strip()],
        "submit_workdir": str(ss.get("submit_workdir") or result.get("submit_workdir") or rec.get("submit_workdir") or "").strip() or None,
        "interoptimus_task_serial": str(result.get("interoptimus_task_serial") or rec.get("serial_id") or "").strip() or None,
        "interoptimus_task_record": rec or None,
        "server_submission": ss or None,
    }


def sync_submission_refs_from_result(task_id: str) -> dict:
    """Persist submit-time UUIDs from ``web_result.json`` into ``io_tasks``."""
    result = _server_result_payload(task_id)
    refs = _submission_refs_from_result(result)
    clean = {k: v for k, v in refs.items() if v not in (None, "", [])}
    if clean:
        get_task_store().update_task(task_id, clean)
    return refs


def task_has_remote_refs(task: Optional[dict], result: Optional[dict] = None) -> bool:
    t = task or {}
    refs = _submission_refs_from_result(result or {}) if result is not None else {}
    return bool(
        t.get("mlip_job_uuid")
        or t.get("submit_jf_job_id")
        or t.get("flow_uuid")
        or refs.get("mlip_job_uuid")
        or refs.get("submit_jf_job_id")
        or refs.get("flow_uuid")
    )


def _state_bucket(state: Any) -> str:
    s = str(state or "").strip().upper()
    if "." in s:
        s = s.split(".")[-1]
    if s in {"COMPLETED", "COMPLETE", "DONE", "FINISHED"}:
        return "completed"
    if s in {"FAILED", "ERROR", "REMOTE_ERROR", "CANCELLED", "CANCELED", "REMOVED"}:
        return "failed"
    if s in {"RUNNING", "ACTIVE", "STARTED", "READY", "SUBMITTED", "RESERVED", "WAITING"}:
        return "running"
    if s in {"PENDING", "QUEUED"}:
        return "pending"
    return "unknown"


def _status_from_remote_progress(progress: dict) -> str:
    if not progress.get("success"):
        return "remote_unknown"
    phase = str(progress.get("current_phase") or "")
    if phase == "mlip":
        bucket = _state_bucket(progress.get("job_state"))
        if bucket == "failed":
            return "failed"
        if progress.get("mode") == "mlip_only" and progress.get("is_finished"):
            return "completed" if bucket == "completed" else "failed"
        return "mlip_running"
    if phase == "vasp":
        return "vasp_running"
    if phase == "done":
        counts = progress.get("expanded_vasp_job_counts") or {}
        return "failed" if int(counts.get("failed") or 0) else "completed"
    return "remote_running"


def _parse_match_term_from_text(text: str) -> tuple[Optional[int], Optional[int]]:
    m = re.search(r"(?:^|[^0-9])(\d+)[_-](\d+)[_-](?:it|sfilm|film|substrate)", text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _remote_vasp_terms(progress: dict) -> List[dict]:
    rows: List[dict] = []
    for kind, source in (
        ("planned_root", progress.get("planned_vasp_roots") or []),
        ("expanded_job", progress.get("vasp_sub_jobs") or progress.get("expanded_vasp_jobs") or []),
    ):
        for idx, row in enumerate(source):
            if not isinstance(row, dict):
                continue
            text = " ".join(
                str(row.get(k) or "") for k in ("name", "run_dir", "uuid", "note")
            )
            mid, tid = _parse_match_term_from_text(text)
            state = row.get("state")
            out = {
                "role": "vasp",
                "vasp_kind": kind,
                "status": _state_bucket(state),
                "state": state,
                "uuid": row.get("uuid"),
                "name": row.get("name"),
                "run_dir": row.get("run_dir"),
                "worker": row.get("worker"),
                "index": row.get("index", idx),
            }
            if mid is not None and tid is not None:
                out["match_id"] = mid
                out["term_id"] = tid
            else:
                out["match_id"] = -1
                out["term_id"] = int(row.get("index") if row.get("index") is not None else idx)
                out["label"] = row.get("name") or row.get("uuid") or f"VASP job {idx + 1}"
            rows.append(out)
    return rows


def refresh_remote_progress(task_id: str, *, jf_bin: str = "jf") -> Optional[dict]:
    """Query jobflow-remote for a submitted server task and persist summary fields."""
    store = get_task_store()
    result = _server_result_payload(task_id)
    refs = sync_submission_refs_from_result(task_id)
    task = store.get_task(task_id) or {}
    if not task_has_remote_refs(task, result):
        return None

    ref = (
        refs.get("mlip_job_uuid")
        or task.get("mlip_job_uuid")
        or refs.get("submit_jf_job_id")
        or task.get("submit_jf_job_id")
        or refs.get("flow_uuid")
        or task.get("flow_uuid")
    )
    if not ref:
        return None
    rec = refs.get("interoptimus_task_record")
    hint: Dict[str, Any] = dict(rec) if isinstance(rec, dict) else {}
    hint.setdefault("jf_job_id", refs.get("submit_jf_job_id") or task.get("submit_jf_job_id") or ref)
    hint.setdefault("mlip_job_uuid", refs.get("mlip_job_uuid") or task.get("mlip_job_uuid"))
    hint.setdefault("flow_uuid", refs.get("flow_uuid") or task.get("flow_uuid"))
    hint.setdefault("submit_workdir", refs.get("submit_workdir") or task.get("submit_workdir"))
    hint.setdefault("do_vasp", bool((refs.get("vasp_job_uuids") or task.get("vasp_job_uuids") or [])))
    hint.setdefault("vasp_job_uuids", refs.get("vasp_job_uuids") or task.get("vasp_job_uuids") or [])
    try:
        from InterOptimus.agents.remote_submit import query_interoptimus_task_progress

        progress = query_interoptimus_task_progress(
            str(ref),
            jf_bin=jf_bin,
            python_head_lines=80,
            task_record_hint=hint,
        )
    except Exception as e:
        progress = {
            "success": False,
            "error": str(e),
            "task_ref_used": str(ref),
        }

    progress = _apply_local_mlip_done_progress_override(str(task_id), progress)

    old_status = task.get("status")
    old_phase = task.get("phase")
    status = _status_from_remote_progress(progress)
    phase = str(progress.get("current_phase") or old_phase or "remote")
    updates: Dict[str, Any] = {
        "status": status,
        "phase": phase,
        "remote_progress": _json_safe(progress),
        "last_remote_refresh_at": utc_now(),
        "stage_summary": progress.get("stage_summary") or progress.get("summary_line"),
    }
    for key in ("mlip_job_uuid", "flow_uuid", "submit_workdir", "vasp_job_uuids"):
        if progress.get(key):
            updates[key] = progress[key]
    if progress.get("submit_jf_job_id"):
        updates["submit_jf_job_id"] = progress["submit_jf_job_id"]
    counts = progress.get("expanded_vasp_job_counts") or progress.get("vasp_job_counts") or {}
    if counts:
        updates["vasp_progress"] = counts
    store.update_task(task_id, updates)
    if status != old_status or phase != old_phase:
        store.append_event(task_id, "remote_progress", str(updates.get("stage_summary") or ""), status=status, phase=phase)

    local_terms = scan_match_terms(task_id)
    vasp_terms = _remote_vasp_terms(progress)
    if vasp_terms:
        store.upsert_match_terms(task_id, local_terms + vasp_terms)
    return progress
