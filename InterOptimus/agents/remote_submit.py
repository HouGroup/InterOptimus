#!/usr/bin/env python3
"""
jobflow-remote helpers: submit InterOptimus ``io_flow.json`` on the **current** machine
(e.g. cluster login node) and poll ``jf job info``.

There is no SSH or cross-host submission in this module.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import shlex
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_FLOW_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _collect_uuid_strings(o: Any, acc: set) -> None:
    if isinstance(o, dict):
        for v in o.values():
            _collect_uuid_strings(v, acc)
    elif isinstance(o, list):
        for x in o:
            _collect_uuid_strings(x, acc)
    elif isinstance(o, str):
        s = o.strip()
        if _FLOW_UUID_RE.match(s):
            acc.add(s.lower())


def _apply_uuid_mapping(o: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(o, dict):
        return {k: _apply_uuid_mapping(v, mapping) for k, v in o.items()}
    if isinstance(o, list):
        return [_apply_uuid_mapping(x, mapping) for x in o]
    if isinstance(o, str):
        s = o.strip()
        if _FLOW_UUID_RE.match(s):
            new_id = mapping.get(s.lower())
            if new_id is not None:
                return new_id
        return o
    return o


def regenerate_flow_uuids_in_json_obj(obj: Any) -> Any:
    """
    Replace **every** UUID-shaped string anywhere in the JSON tree with a new ``uuid4``.

    Jobflow stores flow id, job ids, and cross-references as UUID strings. Only rewriting keys named
    ``uuid`` is not enough: jobflow-remote can still raise
    ``Job ... (xxxxxxxx) already belongs to another flow`` when inner job ids collide with MongoDB.
    """
    acc: set = set()
    _collect_uuid_strings(obj, acc)
    if not acc:
        return obj
    mapping = {u: str(uuid.uuid4()) for u in acc}
    return _apply_uuid_mapping(obj, mapping)


_QRESOURCES_FIELD_NAMES = frozenset(
    {
        "queue_name",
        "job_name",
        "memory_per_thread",
        "nodes",
        "processes",
        "processes_per_node",
        "threads_per_process",
        "gpus_per_job",
        "time_limit",
        "account",
        "qos",
        "priority",
        "output_filepath",
        "error_filepath",
        "process_placement",
        "email_address",
        "rerunnable",
        "project",
        "njobs",
        "scheduler_kwargs",
    }
)


def _normalize_scheduler_kwargs_keys(sk: Any) -> Any:
    """
    qtoolkit SlurmIO expects underscore keys (e.g. ``cpus_per_task``), not Slurm CLI spellings
    like ``cpus-per-task``; the latter raises ValueError in ``generate_header``.
    """
    if not isinstance(sk, dict):
        return sk
    out = _strip_monty_meta_keys_deep(sk)
    h2u = {
        "cpus-per-task": "cpus_per_task",
        "cpus-per-gpu": "cpus_per_gpu",
        "ntasks-per-node": "ntasks_per_node",
    }
    for hyp, usc in h2u.items():
        if hyp in out:
            if usc not in out:
                out[usc] = out[hyp]
            del out[hyp]
    return out


def _strip_monty_meta_keys_deep(obj: Any) -> Any:
    """Remove Monty ``@module`` / ``@class`` keys at every dict level (recursive)."""
    if isinstance(obj, dict):
        return {
            k: _strip_monty_meta_keys_deep(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("@"))
        }
    if isinstance(obj, list):
        return [_strip_monty_meta_keys_deep(x) for x in obj]
    return obj


def _coerce_qresources_kwargs_plain(d: dict) -> dict:
    """Keep only valid ``QResources`` constructor keys; fix enum-like dicts Monty may leave behind."""
    d = _strip_monty_meta_keys_deep(d)
    out: dict = {}
    for k, v in d.items():
        if k not in _QRESOURCES_FIELD_NAMES:
            continue
        if isinstance(v, Enum):
            v = v.value
        elif isinstance(v, Path):
            v = str(v)
        if k == "process_placement" and isinstance(v, dict):
            v = v.get("value") or v.get("name")
            if isinstance(v, dict):
                continue
        if k == "scheduler_kwargs":
            if not isinstance(v, dict):
                continue
            out[k] = _normalize_scheduler_kwargs_keys(v)
            continue
        out[k] = v
    return out


def _merge_jfr_resources_kwargs(user: Optional[dict]) -> dict:
    """Merge user QResources kwargs with the same defaults used for IOMaker demos."""
    default = {
        "nodes": 1,
        "processes_per_node": 1,
        "scheduler_kwargs": {"partition": "standard", "cpus_per_task": 10},
    }
    if not user:
        return _coerce_qresources_kwargs_plain(dict(default))
    merged = {**default, **user}
    if "scheduler_kwargs" in user and isinstance(user.get("scheduler_kwargs"), dict):
        merged["scheduler_kwargs"] = {
            **default["scheduler_kwargs"],
            **user["scheduler_kwargs"],
        }
    return _coerce_qresources_kwargs_plain(merged)


def _jfr_submit_payload(
    worker: str,
    project: str,
    resources_kwargs: Optional[dict],
) -> dict:
    return {
        "worker": worker,
        "project": project,
        "resources": _merge_jfr_resources_kwargs(resources_kwargs),
    }


def qresources_to_plain_dict(res: Any) -> dict:
    """
    Convert a qtoolkit ``QResources`` instance (or similar) to a JSON-serializable dict
    for ``jfr_submit.json``.
    """
    if res is None:
        return {}
    if isinstance(res, dict):
        return _coerce_qresources_kwargs_plain(dict(res))
    as_dict = getattr(res, "as_dict", None)
    if callable(as_dict):
        d = as_dict()
        if isinstance(d, dict):
            return _coerce_qresources_kwargs_plain(d)
    try:
        import dataclasses

        d = dataclasses.asdict(res)
        if isinstance(d, dict):
            return _coerce_qresources_kwargs_plain(d)
    except Exception:
        pass
    out: dict = {}
    for k in (
        "nodes",
        "processes",
        "processes_per_node",
        "scheduler_kwargs",
        "time_limit",
        "gpus_per_job",
        "queue_name",
        "job_name",
        "process_placement",
    ):
        if hasattr(res, k):
            v = getattr(res, k)
            if v is not None:
                out[k] = v
    return _coerce_qresources_kwargs_plain(out)


def _normalize_submit_stdout_for_job_id(text: str) -> str:
    if not text:
        return ""
    lines_out = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if "Enter passphrase for key" in line:
            continue
        lines_out.append(line)
    return "\n".join(lines_out)


def extract_interoptimus_flow_plan(flow_dict: Any) -> Dict[str, Any]:
    """
    Read UUIDs from a serialized jobflow ``Flow`` dict (as written to ``io_flow.json``).

    InterOptimus IOMaker flows list the MLIP job first, then optional VASP follow-up job(s).

    Returns:
        ``flow_uuid``, ``mlip_job_uuid``, ``vasp_job_uuids`` (list), ``ordered_job_uuids``.
    """
    out: Dict[str, Any] = {
        "flow_uuid": None,
        "mlip_job_uuid": None,
        "vasp_job_uuids": [],
        "ordered_job_uuids": [],
    }
    if not isinstance(flow_dict, dict):
        return out
    fu = flow_dict.get("uuid")
    if isinstance(fu, str) and _FLOW_UUID_RE.match(fu.strip()):
        out["flow_uuid"] = fu.strip()
    jobs = flow_dict.get("jobs")
    if not isinstance(jobs, list):
        return out
    ordered: List[str] = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        ju = j.get("uuid")
        if isinstance(ju, str) and _FLOW_UUID_RE.match(ju.strip()):
            ordered.append(ju.strip())
    out["ordered_job_uuids"] = ordered
    if ordered:
        out["mlip_job_uuid"] = ordered[0]
        out["vasp_job_uuids"] = ordered[1:]
    return out


def _extract_job_id_from_submit_stdout(output: str) -> Optional[str]:
    """Parse jobflow-remote / submit_flow stdout for a job id (UUID preferred, else numeric)."""
    if not output:
        return None
    for line in output.splitlines():
        if "__INTEROPTIMUS_JSON__" not in line:
            continue
        jpart = line.split("__INTEROPTIMUS_JSON__", 1)[1].strip()
        try:
            d = json.loads(jpart)
            uid = d.get("uuid") or (d.get("uuids") or [None])[0]
            if uid:
                return str(uid)
        except Exception:
            pass
    # Any UUID in combined output before falling back to numeric db_id / pid
    u_any = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        output,
        re.I,
    )
    if u_any:
        return u_any.group(1)
    clean = _normalize_submit_stdout_for_job_id(output)
    u = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        clean,
        re.I,
    )
    if u:
        return u.group(1)
    m = re.search(r'job[_-]?id["\']?\s*[:=]\s*["\']?(\d+)', clean, re.I)
    if m:
        return m.group(1)
    m2 = re.search(r"\[\s*['\"]?(\d+)['\"]?\s*\]", clean)
    if m2:
        return m2.group(1)
    return None


def _strip_ansi(text: str) -> str:
    """Remove ANSI color / Rich markup escape sequences from captured CLI output."""
    if not text:
        return text
    s = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    s = re.sub(r"\x1b\][^\x07]*\x07", "", s)
    return s


def _strip_rich_table_chars(text: str) -> str:
    """Remove Rich table / box-drawing characters that break ``run_dir`` and UUID strings."""
    if not text:
        return text
    return re.sub(r"[\u2500-\u257F│┃]", "", text)


def _job_info_via_jobflow_remote_api(job_uuid: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Same data as ``jf job info`` (Mongo via JobController), without parsing Rich text.

    jobflow-remote renders ``jf job info`` with Rich ``Pretty()`` (``state = RUNNING``),
    not ``state = 'RUNNING'``; regex on subprocess output often fails. The API path is
    reliable when ``jobflow_remote`` is installed and the project config is available.
    """
    try:
        from jobflow_remote.cli.utils import get_job_controller
    except ImportError:
        return None
    try:
        jc = get_job_controller()
        ref = job_uuid.strip()
        ji = None
        if _FLOW_UUID_RE.match(ref):
            ji = jc.get_job_info(job_id=ref)
        if ji is None:
            ji = jc.get_job_info(db_id=ref)
        if ji is None and ref.isdigit():
            try:
                ji = jc.get_job_info_by_pid(int(ref))
            except Exception:
                ji = None
        if ji is None:
            return None
        raw_state = ji.state
        if hasattr(raw_state, "name"):
            st = str(raw_state.name)
        else:
            st = str(raw_state)
        ju = getattr(ji, "uuid", None)
        return {
            "state": st,
            "run_dir": ji.run_dir,
            "db_id": str(ji.db_id) if getattr(ji, "db_id", None) is not None else None,
            "worker": str(ji.worker) if getattr(ji, "worker", None) is not None else None,
            "job_uuid": str(ju) if ju else None,
        }
    except Exception:
        return None


def _extract_first_uuid_from_text(text: str) -> Optional[str]:
    m = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        text or "",
        re.I,
    )
    return m.group(1) if m else None


def _parse_jf_job_info_stdout(stdout: str) -> Dict[str, Optional[str]]:
    """
    Parse ``jf job info`` **text** output (fallback). Prefer :func:`_job_info_via_jobflow_remote_api`.

    Handles Rich ``scope`` style ``state = RUNNING`` (no quotes), repr strings, and ANSI.
    """
    parsed: Dict[str, Optional[str]] = {
        "state": None,
        "run_dir": None,
        "db_id": None,
        "worker": None,
    }
    if not stdout:
        return parsed
    text = _strip_rich_table_chars(_strip_ansi(stdout))

    def _first_match(patterns: Tuple[str, ...]):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                return m
        return None

    # jobflow-remote Rich: "state = RUNNING" or "state = JobState.RUNNING" (Pretty)
    ms = _first_match(
        (
            r"state\s*=\s*'([^']*)'",
            r'state\s*=\s*"([^"]*)"',
            r"state\s*=\s*JobState\.(\w+)",
            r"state\s*=\s*(\w+)",
            r"(?im)^\s*state\s*[:]\s*(\S+)",
            r"(?im)^\s*status\s*[:]\s*(\S+)",
            r"JobState\.(\w+)",
        )
    )
    if ms:
        parsed["state"] = ms.group(1).strip()

    mr = _first_match(
        (
            r"run_dir\s*=\s*'([^']*)'",
            r'run_dir\s*=\s*"([^"]*)"',
            r"run_dir\s*=\s*(\S+)",
            r"(?im)^\s*run[_\s]?dir\s*[:]\s*(\S+)",
        )
    )
    if mr:
        rd = "".join(mr.group(1).split())
        parsed["run_dir"] = _strip_rich_table_chars(rd)

    md = _first_match((r"db_id\s*=\s*'([^']*)'", r'db_id\s*=\s*"([^"]*)"', r"db_id\s*=\s*(\S+)"))
    if md:
        parsed["db_id"] = md.group(1).strip()

    mw = _first_match((r"worker\s*=\s*'([^']*)'", r'worker\s*=\s*"([^"]*)"', r"(?im)^\s*worker\s*[:]\s*(\S+)"))
    if mw:
        parsed["worker"] = mw.group(1).strip()

    if not parsed.get("job_uuid"):
        ju = _extract_first_uuid_from_text(text)
        if ju:
            parsed["job_uuid"] = ju

    return parsed


def _submit_flow_python_snippet() -> str:
    """Python run after ``io_flow.json`` and ``jfr_submit.json`` are in cwd."""
    return r"""
from jobflow_remote import submit_flow
from jobflow import Flow
from qtoolkit.core.data_objects import QResources
import json
import re as _re

with open("io_flow.json", "r") as f:
    flow_json = json.load(f)

flow = Flow.from_dict(flow_json)

with open("jfr_submit.json", "r") as f:
    jfr = json.load(f)

res_kw = jfr.get("resources") or {}
resources = QResources(**res_kw)

def _submit_result_payload(r):
    out = {"repr": repr(r)}
    try:
        if hasattr(r, "uuid") and r.uuid is not None:
            out["uuid"] = str(r.uuid)
            return out
        if isinstance(r, (list, tuple)):
            uids = []
            for x in r:
                if hasattr(x, "uuid") and getattr(x, "uuid", None) is not None:
                    uids.append(str(x.uuid))
                elif isinstance(x, str) and len(x) >= 32 and x.count("-") >= 4:
                    uids.append(x)
            if uids:
                out["uuids"] = uids
                out["uuid"] = uids[0]
        if "uuid" not in out:
            mu = _re.search(
                r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                repr(r),
                _re.I,
            )
            if mu:
                out["uuid"] = mu.group(1)
    except Exception as ex:
        out["ex"] = str(ex)
    return out

_worker = jfr.get("worker") or "std_worker"
_project = jfr.get("project") or "std"
_r = submit_flow(flow, worker=_worker, resources=resources, project=_project)
print("__INTEROPTIMUS_JSON__ " + json.dumps(_submit_result_payload(_r), ensure_ascii=False, separators=(",", ":")))
""".strip()


def local_jf_job_info(
    job_uuid: str,
    *,
    jf_bin: str = "jf",
    timeout_sec: float = 120.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run ``jf job info <uuid>`` on the local machine."""
    job_uuid = (job_uuid or "").strip()
    if not job_uuid:
        return {
            "success": False,
            "stdout": "",
            "stderr": "empty job uuid",
            "exit_code": -1,
            "parsed": _parse_jf_job_info_stdout(""),
        }

    api_parsed = _job_info_via_jobflow_remote_api(job_uuid)
    if api_parsed is not None and api_parsed.get("state") is not None:
        if debug:
            print(f"[local_jf_job_info] source=api state={api_parsed.get('state')!r}", flush=True)
        return {
            "success": True,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "parsed": api_parsed,
            "parsed_source": "jobflow_remote_api",
        }

    try:
        proc = subprocess.run(
            [jf_bin, "job", "info", job_uuid],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        # Rich / Typer often render the job table on stderr; parsing stdout alone yields state=None.
        combined = _strip_rich_table_chars(
            _strip_ansi((out + ("\n" + err if err else "")).strip())
        )
        if debug:
            print(f"[local_jf_job_info] source=subprocess exit={proc.returncode}\n{combined[:4000]}", flush=True)
        parsed = _parse_jf_job_info_stdout(combined)
        if api_parsed:
            for k, v in api_parsed.items():
                if v is not None and not parsed.get(k):
                    parsed[k] = v
        if not parsed.get("job_uuid"):
            ju = _extract_first_uuid_from_text(combined)
            if ju:
                parsed["job_uuid"] = ju
        return {
            "success": proc.returncode == 0,
            "stdout": out,
            "stderr": err,
            "exit_code": proc.returncode,
            "parsed": parsed,
            "parsed_source": "jf_subprocess",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"executable not found: {jf_bin!r} (is jobflow-remote installed?)",
            "exit_code": -1,
            "parsed": _parse_jf_job_info_stdout(""),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stdout": e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
            "stderr": e.stderr.decode("utf-8", errors="replace") if e.stderr else "timeout",
            "exit_code": -1,
            "parsed": _parse_jf_job_info_stdout(""),
        }


def poll_local_job_until(
    job_uuid: str,
    *,
    jf_bin: str = "jf",
    interval_sec: float = 30.0,
    timeout_sec: float = 3600.0 * 72,
    terminal_states: Tuple[str, ...] = (
        "COMPLETED",
        "COMPLETE",
        "DONE",
        "FINISHED",
        "FAILED",
        "ERROR",
        "REMOTE_ERROR",
        "CANCELLED",
        "CANCELED",
        "REMOVED",
    ),
    on_tick: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Poll ``jf job info`` locally until a terminal state or timeout."""
    terminal_norm = {s.upper() for s in terminal_states}
    deadline = time.monotonic() + timeout_sec
    last: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        last = local_jf_job_info(job_uuid, jf_bin=jf_bin)
        if on_tick:
            on_tick(last)
        st_raw = (last.get("parsed") or {}).get("state")
        if st_raw is not None:
            st_key = st_raw.strip().upper().replace("-", "_")
            # Enum / repr variants
            if st_key.startswith("JOBSTATE."):
                st_key = st_key.split(".", 1)[-1]
            if st_key in terminal_norm:
                return {
                    "finished": True,
                    "state": st_raw.strip(),
                    "last_query": last,
                }
        time.sleep(interval_sec)
    return {"finished": False, "timeout": True, "last_query": last}


def submit_io_flow_locally(
    local_io_flow_json: str | Path,
    *,
    workdir: Optional[str] = None,
    remote_python: str = "python",
    pre_cmd: str = "",
    submit_flow_worker: str = "std_worker",
    submit_flow_project: str = "std",
    submit_flow_resources_kwargs: Optional[dict] = None,
    submit_flow_fresh_uuids: bool = True,
) -> Dict[str, Any]:
    """
    Submit ``io_flow.json`` to jobflow-remote using ``submit_flow`` on the **current** host.

    Writes ``jfr_submit.json`` and ``io_flow.json`` under *workdir*, then runs the embedded
    submit script (same working directory).
    """
    local_io_flow_json = Path(local_io_flow_json).resolve()
    if not local_io_flow_json.is_file():
        raise FileNotFoundError(f"Flow JSON not found: {local_io_flow_json}")

    wd = Path(workdir).resolve() if workdir else local_io_flow_json.parent
    wd.mkdir(parents=True, exist_ok=True)

    jfr_payload = _jfr_submit_payload(
        submit_flow_worker,
        submit_flow_project,
        submit_flow_resources_kwargs,
    )
    jfr_path = wd / "jfr_submit.json"
    jfr_path.write_text(json.dumps(jfr_payload, ensure_ascii=False), encoding="utf-8")

    with open(local_io_flow_json, encoding="utf-8") as f:
        _flow_data = json.load(f)
    if submit_flow_fresh_uuids:
        _flow_data = regenerate_flow_uuids_in_json_obj(_flow_data)
    flow_plan = extract_interoptimus_flow_plan(_flow_data)
    flow_dest = wd / "io_flow.json"
    flow_dest.write_text(json.dumps(_flow_data, ensure_ascii=False), encoding="utf-8")

    script_body = _submit_flow_python_snippet()
    source_root = Path(__file__).resolve().parents[2]
    inner = f"""set -e
cd {shlex.quote(str(wd))}
{pre_cmd}
export PYTHONPATH={shlex.quote(str(source_root))}:${{PYTHONPATH:-}}
{shlex.quote(remote_python)} - <<'PY'
{script_body}
PY
"""
    try:
        proc = subprocess.run(
            ["bash", "-lc", inner],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        combined = stdout_text + ("\n" + stderr_text if stderr_text else "")
        job_id = _extract_job_id_from_submit_stdout(combined)
        mlip_uid = flow_plan.get("mlip_job_uuid")
        flow_uid = flow_plan.get("flow_uuid")
        submit_id_note = None
        if job_id and mlip_uid and str(job_id).strip().lower() != str(mlip_uid).strip().lower():
            if flow_uid and str(job_id).strip().lower() == str(flow_uid).strip().lower():
                submit_id_note = (
                    "submit_flow returned Flow uuid; use mlip_job_uuid for MLIP job status."
                )
            else:
                submit_id_note = (
                    f"submit job_id {job_id!r} differs from planned mlip_job_uuid {mlip_uid!r}; "
                    "prefer mlip_job_uuid for MLIP tracking."
                )
        flow_submit = {
            "exit_status": proc.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "job_id": job_id,
        }
        return {
            "success": proc.returncode == 0,
            "stdout": combined,
            "stderr": stderr_text,
            "job_id": job_id,
            "error": None if proc.returncode == 0 else "submit_failed",
            "flow_submission": flow_submit,
            "submit_workdir": str(wd),
            "flow_json_used": str(flow_dest),
            "flow_plan": flow_plan,
            "mlip_job_uuid": flow_plan.get("mlip_job_uuid"),
            "vasp_job_uuids": list(flow_plan.get("vasp_job_uuids") or []),
            "flow_uuid": flow_plan.get("flow_uuid"),
            "submit_id_note": submit_id_note,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "timeout",
            "job_id": None,
            "error": "submit_timeout",
            "flow_submission": None,
            "submit_workdir": str(wd),
            "flow_plan": flow_plan,
            "mlip_job_uuid": flow_plan.get("mlip_job_uuid"),
            "vasp_job_uuids": list(flow_plan.get("vasp_job_uuids") or []),
            "flow_uuid": flow_plan.get("flow_uuid"),
            "submit_id_note": None,
        }


def summarize_interoptimus_run_dir(run_dir: str, *, tail_lines: int = 40) -> Dict[str, Any]:
    """
    Best-effort summary of a jobflow ``run_dir`` after an InterOptimus flow completes.

    Looks for ``io_report.txt``, ``pairs_best_it/pairs_summary.txt`` (or root ``pairs_summary.txt``),
    ``io_remote_summary.md`` / ``.json``, ``project.jpg``, ``stereographic.jpg``, ``unique_matches.jpg``.
    """
    run_dir = os.path.abspath(run_dir)
    out: Dict[str, Any] = {"run_dir": run_dir, "artifacts": {}}
    if not os.path.isdir(run_dir):
        out["error"] = "not_a_directory"
        return out

    for name in (
        "io_report.txt",
        "io_remote_summary.md",
        "io_remote_summary.json",
        "project.jpg",
        "stereographic.jpg",
        "unique_matches.jpg",
    ):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            out["artifacts"][name] = p

    p_sum = _find_pairs_summary_path(run_dir)
    if p_sum:
        out["artifacts"]["pairs_summary.txt"] = p_sum

    for key, fname in (("io_report", "io_report.txt"),):
        p = os.path.join(run_dir, fname)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                out[f"{key}_tail"] = "".join(lines[-tail_lines:])
            except OSError as e:
                out[f"{key}_error"] = str(e)

    if p_sum:
        try:
            with open(p_sum, encoding="utf-8", errors="replace") as f:
                plines = f.readlines()
            out["pairs_summary_tail"] = "".join(plines[-tail_lines:])
        except OSError as e:
            out["pairs_summary_error"] = str(e)

    pairs_dir = os.path.join(run_dir, "pairs_best_it")
    if os.path.isdir(pairs_dir):
        out["pairs_best_it"] = pairs_dir
    return out


def _find_pairs_summary_path(run_dir: str) -> Optional[str]:
    """``pairs_best_it/pairs_summary.txt`` (IOMaker) or legacy root ``pairs_summary.txt``."""
    for rel in ("pairs_best_it/pairs_summary.txt", "pairs_summary.txt"):
        p = os.path.join(run_dir, rel.replace("/", os.sep))
        if os.path.isfile(p):
            return p
    return None


def _read_text_file(path: Optional[str], *, max_chars: Optional[int] = None) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            t = f.read()
    except OSError:
        return None
    if max_chars is not None and len(t) > max_chars:
        return t[:max_chars] + "\n\n... [truncated]\n"
    return t


def collect_interoptimus_run_dir_results(
    run_dir: Optional[str],
    *,
    io_report_max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load full text reports and image paths from a completed InterOptimus ``run_dir``.

    Matches worker outputs from :class:`InterOptimus.jobflow.IOMaker` (``io_report.txt``,
    ``io_remote_summary.*``, ``pairs_summary.txt`` in the run root or under legacy
    ``pairs_best_it/``) and figures:

    - ``project.jpg``: lattice-match stereographic plot (HKL / orientations).
    - ``stereographic.jpg``: compact static stereographic summary plot.
    - ``stereographic_interactive.html``: larger interactive stereographic plot with hover labels.
    - ``unique_matches.jpg``: unique-match overview.
    - ``opt_results.pkl``: full optimization payload; :func:`fetch_interoptimus_task_results`
      materializes ``pairs_best_it/`` locally from this file.
    - ``opt_results_summary.json``: JSON summary of ``opt_results`` (no pymatgen blobs).
    - ``area_strain``: stereographic energy-plot input data when present.
    """
    out: Dict[str, Any] = {"run_dir": run_dir, "success": False}
    if not run_dir or not os.path.isdir(run_dir):
        out["error"] = "run_dir_missing_or_inaccessible"
        return out
    rd = os.path.abspath(run_dir)
    out["run_dir"] = rd
    out["success"] = True

    out["image_paths"] = {
        "project_jpg": p if os.path.isfile(p := os.path.join(rd, "project.jpg")) else None,
        "stereographic_jpg": p if os.path.isfile(p := os.path.join(rd, "stereographic.jpg")) else None,
        "unique_matches_jpg": p if os.path.isfile(p := os.path.join(rd, "unique_matches.jpg")) else None,
    }

    out["artifact_paths"] = {
        "pairs_best_it_dir": p if os.path.isdir(p := os.path.join(rd, "pairs_best_it")) else None,
        "opt_results_pkl": p if os.path.isfile(p := os.path.join(rd, "opt_results.pkl")) else None,
        "opt_results_summary_json": p
        if os.path.isfile(p := os.path.join(rd, "opt_results_summary.json"))
        else None,
        "area_strain": p if os.path.isfile(p := os.path.join(rd, "area_strain")) else None,
        "stereographic_interactive_html": p
        if os.path.isfile(p := os.path.join(rd, "stereographic_interactive.html"))
        else None,
    }

    out["io_report_text"] = _read_text_file(os.path.join(rd, "io_report.txt"), max_chars=io_report_max_chars)

    psp = _find_pairs_summary_path(rd)
    out["pairs_summary_path"] = psp
    out["pairs_summary_text"] = _read_text_file(psp, max_chars=io_report_max_chars)

    jpath = os.path.join(rd, "io_remote_summary.json")
    out["io_remote_summary"] = None
    if os.path.isfile(jpath):
        try:
            out["io_remote_summary"] = json.loads(Path(jpath).read_text(encoding="utf-8"))
        except Exception as e:
            out["io_remote_summary_error"] = str(e)

    out["io_remote_summary_markdown"] = _read_text_file(os.path.join(rd, "io_remote_summary.md"))

    pairs: List[Dict[str, Any]] = []
    if isinstance(out.get("io_remote_summary"), dict):
        raw_p = out["io_remote_summary"].get("pairs")
        if isinstance(raw_p, list):
            pairs = raw_p
    out["pairs"] = pairs

    out["report_markdown"] = _build_interoptimus_results_markdown(out)
    return out


def _build_interoptimus_results_markdown(collected: Dict[str, Any]) -> str:
    """Single markdown document: remote summary + pairs table + io_report."""
    parts: List[str] = []
    parts.append("# InterOptimus 计算结果\n\n")
    if collected.get("io_remote_summary_markdown"):
        parts.append(collected["io_remote_summary_markdown"])
        parts.append("\n\n")
    elif collected.get("pairs"):
        parts.append("## 各 HKL 匹配界面能量\n\n")
        parts.append(
            "| match | term | film Miller | sub Miller | energy type | energy | film atoms | sub atoms | match_area | strain |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for item in collected["pairs"]:
            if "interface_energy" in item:
                et, ev = "interface_energy", item.get("interface_energy")
            else:
                et, ev = "cohesive_energy", item.get("cohesive_energy")
            row = [
                str(item.get("match_id")),
                str(item.get("term_id")),
                str(item.get("film_conventional_miller")),
                str(item.get("substrate_conventional_miller")),
                str(et),
                str(ev),
                str(item.get("film_atom_count")),
                str(item.get("substrate_atom_count")),
                str(item.get("match_area")),
                str(item.get("strain")),
            ]
            parts.append("| " + " | ".join(row) + " |\n")
        parts.append("\n")
    if collected.get("pairs_summary_text"):
        parts.append("## pairs_summary.txt\n\n```text\n")
        parts.append(collected["pairs_summary_text"].rstrip() + "\n```\n\n")
    if collected.get("io_report_text"):
        parts.append("---\n\n## io_report.txt（与 worker 一致）\n\n```text\n")
        parts.append(collected["io_report_text"].rstrip() + "\n```\n")
    return "".join(parts)


def wait_and_summarize_local_job(
    job_uuid: str,
    *,
    jf_bin: str = "jf",
    interval_sec: float = 30.0,
    timeout_sec: float = 3600.0 * 72,
    on_tick: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Poll until terminal state, then return ``run_dir`` and :func:`summarize_interoptimus_run_dir`."""
    poll = poll_local_job_until(
        job_uuid,
        jf_bin=jf_bin,
        interval_sec=interval_sec,
        timeout_sec=timeout_sec,
        on_tick=on_tick,
    )
    if not poll.get("finished"):
        return {
            "success": False,
            "phase": "poll",
            "error": "timeout_or_no_terminal_state",
            "poll": poll,
            "summary": None,
        }
    st = poll.get("state")
    last = poll.get("last_query") or {}
    run_dir = (last.get("parsed") or {}).get("run_dir")
    summary = summarize_interoptimus_run_dir(run_dir) if run_dir else None
    return {
        "success": True,
        "phase": "done",
        "state": st,
        "poll": poll,
        "run_dir": run_dir,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# InterOptimus task serial + progress (non-blocking submit; query on demand)
# ---------------------------------------------------------------------------

_TASK_JSON_NAME = "io_interoptimus_task.json"


def _interoptimus_registry_base_dir() -> Path:
    base = os.environ.get("INTEROPTIMUS_TASK_REGISTRY_DIR") or str(Path.home() / ".interoptimus")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _interoptimus_registry_path() -> Path:
    return _interoptimus_registry_base_dir() / "iomaker_tasks.json"


def _iomaker_job_meta_dir() -> Path:
    d = _interoptimus_registry_base_dir() / "iomaker_job_meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iomaker_job_aliases_path() -> Path:
    return _interoptimus_registry_base_dir() / "iomaker_job_id_aliases.json"


def _load_iomaker_job_aliases() -> Dict[str, str]:
    path = _iomaker_job_aliases_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            kk, vv = k.strip().lower(), v.strip().lower()
            if kk and vv:
                out[kk] = vv
    return out


def _save_iomaker_job_aliases(aliases: Dict[str, str]) -> None:
    path = _iomaker_job_aliases_path()
    path.write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")


def _iomaker_meta_file_stem(record: Dict[str, Any]) -> str:
    """
    Stable filename stem for ``iomaker_job_meta/<stem>.json``.

    Prefer MLIP / submit job UUID so one file per submission; fall back to *serial_id*.
    """
    for fld in ("mlip_job_uuid", "jf_job_id"):
        v = (record.get(fld) or "").strip()
        if v and _FLOW_UUID_RE.match(v):
            return v.lower()
    sid = (record.get("serial_id") or "").strip()
    if sid:
        return _safe_task_name(sid).lower()
    return "unknown"


def _persist_iomaker_job_index(record: Dict[str, Any]) -> None:
    """
    Write ``iomaker_job_meta/<stem>.json`` and merge ``iomaker_job_id_aliases.json``.

    Any job or flow UUID seen at submit time (MLIP, submit id, flow id, planned VASP ids)
    maps to the same metadata blob so :func:`resolve_interoptimus_task_record` does not
    depend on a full registry scan or a single ``jf`` id type.
    """
    stem = _iomaker_meta_file_stem(record)
    if not stem or stem == "unknown":
        return
    meta_path = _iomaker_job_meta_dir() / f"{stem}.json"
    try:
        meta_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        return

    alias_keys: List[str] = []
    for fld in ("jf_job_id", "mlip_job_uuid", "flow_uuid", "serial_id"):
        v = record.get(fld)
        if isinstance(v, str) and v.strip():
            alias_keys.append(v.strip().lower())
    for u in record.get("vasp_job_uuids") or []:
        if isinstance(u, str) and u.strip():
            alias_keys.append(u.strip().lower())
    aliases = _load_iomaker_job_aliases()
    target = stem.lower()
    for k in alias_keys:
        if k:
            aliases[k] = target
    try:
        _save_iomaker_job_aliases(aliases)
    except Exception:
        pass


def _load_iomaker_job_meta_by_stem(stem: str) -> Optional[Dict[str, Any]]:
    if not stem:
        return None
    safe = stem.replace("/", "_").replace("..", "_")
    path = _iomaker_job_meta_dir() / f"{safe}.json"
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(d) if isinstance(d, dict) else None


def _resolve_interoptimus_task_record_from_index(needle: str) -> Optional[Dict[str, Any]]:
    """Resolve *needle* via ``iomaker_job_id_aliases.json`` / ``iomaker_job_meta``."""
    raw = (needle or "").strip()
    n = raw.lower()
    if not n:
        return None
    aliases = _load_iomaker_job_aliases()
    stem = aliases.get(n)
    if stem:
        meta = _load_iomaker_job_meta_by_stem(stem)
        if meta:
            return meta
    if _FLOW_UUID_RE.match(raw):
        meta = _load_iomaker_job_meta_by_stem(n)
        if meta:
            return meta
    return None


def persist_iomaker_task_index_fallback(
    *,
    task_name: str,
    jf_job_id: str,
    submit_workdir: str,
    do_vasp: bool,
    mlip_job_uuid: Optional[str] = None,
    vasp_job_uuids: Optional[List[str]] = None,
    flow_uuid: Optional[str] = None,
) -> None:
    """
    If :func:`register_interoptimus_server_task` fails (e.g. registry IO), still persist
    meta + aliases so progress queries recover ``do_vasp`` / ``vasp_job_uuids``.
    """
    if not (jf_job_id or "").strip():
        return
    wd = os.path.abspath(submit_workdir)
    serial_id = make_interoptimus_serial_id(task_name, jf_job_id, wd)
    _vasp = list(vasp_job_uuids) if vasp_job_uuids else []
    record: Dict[str, Any] = {
        "serial_id": serial_id,
        "task_name": task_name,
        "jf_job_id": jf_job_id,
        "submit_workdir": wd,
        "do_vasp": bool(do_vasp),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mlip_job_uuid": (str(mlip_job_uuid).strip() if mlip_job_uuid else None),
        "vasp_job_uuids": _vasp,
        "flow_uuid": (str(flow_uuid).strip() if flow_uuid else None),
    }
    _persist_iomaker_job_index(record)


def _safe_task_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", (name or "IO").strip()).strip("_")
    return (s[:40] if s else "IO")


def make_interoptimus_serial_id(task_name: str, jf_job_id: str, submit_workdir: str) -> str:
    """
    Stable, human-readable id: ``{task_name}_{10hex}`` (hash derived from job id + workdir).
    """
    base = _safe_task_name(task_name)
    h = hashlib.sha256(f"{jf_job_id}\n{os.path.abspath(submit_workdir)}".encode()).hexdigest()[:10]
    return f"{base}_{h}"


def _load_task_registry() -> Dict[str, Any]:
    path = _interoptimus_registry_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_task_registry(reg: Dict[str, Any]) -> None:
    path = _interoptimus_registry_path()
    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def register_interoptimus_server_task(
    *,
    task_name: str,
    jf_job_id: str,
    submit_workdir: str,
    do_vasp: bool,
    mlip_job_uuid: Optional[str] = None,
    vasp_job_uuids: Optional[List[str]] = None,
    flow_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    After a successful ``submit_flow``, register a serial id and persist metadata.

    Writes ``~/.interoptimus/iomaker_tasks.json`` (override with
    ``INTEROPTIMUS_TASK_REGISTRY_DIR``), ``{submit_workdir}/io_interoptimus_task.json``, and
    ``iomaker_job_meta/*.json`` plus ``iomaker_job_id_aliases.json`` so any submit-time UUID
    resolves to the same metadata.

    *mlip_job_uuid* / *vasp_job_uuids* / *flow_uuid* come from :func:`extract_interoptimus_flow_plan`
    (saved before ``submit_flow`` runs); use *mlip_job_uuid* with :func:`query_interoptimus_task_progress`
    to track the MLIP root job explicitly.
    """
    if not jf_job_id:
        raise ValueError("jf_job_id is required")
    wd = os.path.abspath(submit_workdir)
    serial_id = make_interoptimus_serial_id(task_name, jf_job_id, wd)
    _vasp = list(vasp_job_uuids) if vasp_job_uuids else []
    record: Dict[str, Any] = {
        "serial_id": serial_id,
        "task_name": task_name,
        "jf_job_id": jf_job_id,
        "submit_workdir": wd,
        "do_vasp": bool(do_vasp),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mlip_job_uuid": (str(mlip_job_uuid).strip() if mlip_job_uuid else None),
        "vasp_job_uuids": _vasp,
        "flow_uuid": (str(flow_uuid).strip() if flow_uuid else None),
    }
    reg = _load_task_registry()
    reg[serial_id] = {k: v for k, v in record.items()}
    _save_task_registry(reg)
    meta_path = Path(wd) / _TASK_JSON_NAME
    Path(wd).mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    _persist_iomaker_job_index(record)
    return record


def _load_task_record_from_disk(serial_or_path: str) -> Optional[Dict[str, Any]]:
    try:
        p = Path(serial_or_path).expanduser().resolve()
    except Exception:
        p = Path(serial_or_path).expanduser()
    if p.is_dir():
        jf = p / _TASK_JSON_NAME
        if jf.is_file():
            try:
                return json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def interoptimus_submit_dir_from_flow_json(flow_json_path: str | Path) -> str:
    """
    Absolute path to the submit workdir (parent directory of ``io_flow.json``).

    After submit, this directory contains ``io_interoptimus_task.json``. For
    :func:`query_interoptimus_task_progress` / :func:`fetch_interoptimus_task_results` prefer
    passing the **MLIP job UUID** (``result["mlip_job_uuid"]``); this helper is only for
    filesystem paths when copying artifacts.
    """
    return str(Path(flow_json_path).resolve().parent)


def _env_interoptimus_job_id() -> str:
    """Kernel-restart fallback: ``INTEROPTIMUS_JOB_ID`` or ``INTEROPTIMUS_MLIP_JOB_UUID``."""
    return (
        os.environ.get("INTEROPTIMUS_JOB_ID", "").strip()
        or os.environ.get("INTEROPTIMUS_MLIP_JOB_UUID", "").strip()
    )


def iomaker_task_ref_for_query(
    result: Optional[Dict[str, Any]] = None,
    *,
    jf_job_id: str = "",
) -> str:
    """
    Build the job-flow **ID** string (UUID) for :func:`query_interoptimus_task_progress` and
    :func:`fetch_interoptimus_task_results`.

    Uses **IDs only** (no registration serial, no directory path):

    1. *jf_job_id* — explicit override (MLIP job UUID, or ``submit_flow`` id if you pass it).
    2. ``result["mlip_job_uuid"]`` / ``interoptimus_task_record["mlip_job_uuid"]``.
    3. ``result["server_submission"]["job_id"]``.
    """
    r = result or {}
    rec = r.get("interoptimus_task_record") if isinstance(r.get("interoptimus_task_record"), dict) else {}
    return (
        (jf_job_id or "").strip()
        or str(r.get("mlip_job_uuid") or rec.get("mlip_job_uuid") or "").strip()
        or str(((r.get("server_submission") or {}).get("job_id")) or "").strip()
    )


def iomaker_resolve_ref(
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    use_env: bool = True,
) -> str:
    """
    Resolve the job **ID** (UUID) for progress / fetch APIs.

    Order: *ref* → :func:`iomaker_task_ref_for_query` → ``INTEROPTIMUS_JOB_ID`` /
    ``INTEROPTIMUS_MLIP_JOB_UUID`` (when *use_env* is True).

    After a kernel restart, export ``INTEROPTIMUS_JOB_ID=<mlip-uuid>`` or pass ``ref=``.
    """
    r = result or {}
    s = (ref or "").strip()
    if s:
        return s
    s = iomaker_task_ref_for_query(r, jf_job_id=jf_job_id)
    if s:
        return s
    if use_env:
        return _env_interoptimus_job_id()
    return ""


def _iomaker_task_record_hint_from_result(
    result: Optional[Dict[str, Any]],
    *,
    task_ref: str = "",
    jf_job_id: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Best-effort metadata hint carried by notebook / CLI ``result``.

    This keeps ``iomaker_progress(result)`` on a single code path even before the registry
    or on-disk task record is available: VASP UUIDs, ``do_vasp``, and ``submit_workdir``
    from the current submit result are passed through to
    :func:`query_interoptimus_task_progress`.
    """
    r = result or {}
    rec = (
        dict(r.get("interoptimus_task_record"))
        if isinstance(r.get("interoptimus_task_record"), dict)
        else {}
    )
    ss = r.get("server_submission") if isinstance(r.get("server_submission"), dict) else {}
    if not rec and not r and not ss:
        return None
    out: Dict[str, Any] = dict(rec)
    if not out.get("jf_job_id"):
        out["jf_job_id"] = (
            (jf_job_id or "").strip()
            or str(ss.get("job_id") or "").strip()
            or str(task_ref or "").strip()
        )
    if not out.get("mlip_job_uuid"):
        out["mlip_job_uuid"] = str(r.get("mlip_job_uuid") or ss.get("mlip_job_uuid") or "").strip() or None
    if not out.get("flow_uuid"):
        out["flow_uuid"] = str(r.get("flow_uuid") or ss.get("flow_uuid") or "").strip() or None
    if not out.get("submit_workdir"):
        sw = str(
            r.get("submit_workdir")
            or ss.get("submit_workdir")
            or r.get("local_workdir")
            or ""
        ).strip()
        out["submit_workdir"] = sw or None
    if "do_vasp" not in out:
        rv = r.get("do_vasp")
        if rv is None and isinstance(r.get("settings"), dict):
            rv = r["settings"].get("do_vasp")
        if rv is not None:
            out["do_vasp"] = bool(rv)
    if not out.get("task_name"):
        out["task_name"] = str(r.get("task_name") or r.get("name") or "").strip() or None
    if not out.get("serial_id"):
        out["serial_id"] = str(r.get("interoptimus_task_serial") or "").strip() or None
    if not isinstance(out.get("vasp_job_uuids"), list) or not out.get("vasp_job_uuids"):
        raw_vasp = r.get("vasp_job_uuids")
        if raw_vasp is None:
            raw_vasp = ss.get("vasp_job_uuids")
        if isinstance(raw_vasp, (list, tuple)):
            out["vasp_job_uuids"] = [
                str(x).strip()
                for x in raw_vasp
                if isinstance(x, str) and str(x).strip()
            ]
    if out.get("jf_job_id") or out.get("mlip_job_uuid") or out.get("flow_uuid"):
        return out
    return None


def _merge_interoptimus_task_record(
    base: Optional[Dict[str, Any]],
    hint: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge persistent record with a submit-time hint, preferring richer values."""
    if not base and not hint:
        return None
    if not base:
        return dict(hint or {})
    if not hint:
        return dict(base)
    out = dict(base)
    for key in (
        "serial_id",
        "task_name",
        "jf_job_id",
        "submit_workdir",
        "do_vasp",
        "created_utc",
        "mlip_job_uuid",
        "flow_uuid",
    ):
        hv = hint.get(key)
        if hv not in (None, "", []):
            out[key] = hv
    bv = out.get("vasp_job_uuids")
    hv = hint.get("vasp_job_uuids")
    base_list = list(bv) if isinstance(bv, (list, tuple)) else []
    hint_list = list(hv) if isinstance(hv, (list, tuple)) else []
    if len(hint_list) > len(base_list):
        out["vasp_job_uuids"] = hint_list
    elif "vasp_job_uuids" not in out:
        out["vasp_job_uuids"] = base_list
    return out


def _print_iomaker_progress_brief(prog: Dict[str, Any]) -> None:
    tr = prog.get("task_ref_used", "")
    print("job_id (query ref):", tr or "(empty)")
    if not prog.get("success"):
        print(" error:", prog.get("error"))
        if prog.get("hint"):
            h = str(prog["hint"])
            if len(h) > 600:
                h = h[:600] + "…"
            print(" hint:", h)
        return
    print(
        " job_state:",
        prog.get("job_state"),
        "| mode:",
        prog.get("mode"),
    )
    if prog.get("run_dir"):
        print(" run_dir:", prog.get("run_dir"))
    if prog.get("mlip_job_uuid"):
        print(" mlip_job_uuid:", prog.get("mlip_job_uuid"))
    sj = prog.get("submit_jf_job_id")
    if sj and sj != prog.get("mlip_job_uuid"):
        print(" submit_jf_job_id:", sj)
    pv = prog.get("planned_vasp_job_uuids") or []
    if pv:
        print(" planned VASP UUIDs (" + str(len(pv)) + "):", pv)
    vsj = prog.get("vasp_sub_jobs") or []
    if vsj:
        print(" vasp_sub_jobs:")
        for row in vsj:
            if not isinstance(row, dict):
                continue
            extra = row.get("note") or ""
            print(
                "  -",
                row.get("uuid"),
                "state=",
                row.get("state"),
                ("  " + extra) if extra else "",
            )
    if prog.get("summary_line"):
        print(" ", prog["summary_line"])
    if prog.get("eta_note"):
        print(" eta:", prog["eta_note"])
    if prog.get("tracking_note"):
        print(" note:", prog["tracking_note"])
    if prog.get("do_vasp_note"):
        print(" ", prog["do_vasp_note"])
    if prog.get("mode_note"):
        print(" mode_note:", prog["mode_note"])


def _job_state_bucket(st: Optional[str]) -> str:
    x = _norm_job_state_str(st)
    if not x:
        return "pending"
    if x in {"COMPLETED", "COMPLETE", "DONE", "FINISHED"}:
        return "completed"
    if x in {"FAILED", "ERROR", "REMOTE_ERROR", "CANCELLED", "CANCELED", "REMOVED"}:
        return "failed"
    if x in {
        "RUNNING",
        "ACTIVE",
        "STARTED",
        "READY",
        "SUBMITTED",
        "RESERVED",
        "WAITING",
    }:
        return "running"
    if x in {"PENDING", "QUEUED"}:
        return "pending"
    return "unknown"


def _summarize_job_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "total": 0,
        "completed": 0,
        "running": 0,
        "pending": 0,
        "failed": 0,
        "unknown": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        counts["total"] += 1
        counts[_job_state_bucket(row.get("state"))] += 1
    counts["finished"] = counts["completed"] + counts["failed"]
    counts["unfinished"] = counts["total"] - counts["finished"]
    return counts


def _split_vasp_tracking(
    flow_jobs: List[Dict[str, Any]],
    *,
    mlip_uuid_key: str,
    planned_vasp_root_uuids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split VASP tracking into:

    - ``planned_roots``: top-level dynamic VASP wrapper jobs known at submit time
    - ``expanded_jobs``: actual VASP jobs materialized after MLIP finishes
    """
    ml = (mlip_uuid_key or "").strip().lower()
    planned_rows: List[Dict[str, Any]] = []
    expanded_rows: List[Dict[str, Any]] = []
    by_u = {(str(r.get("uuid") or "")).strip().lower(): r for r in flow_jobs if r.get("uuid")}
    planned_seen: set = set()
    for uid in planned_vasp_root_uuids:
        u = str(uid).strip()
        if not u:
            continue
        row = by_u.get(u.lower())
        if row:
            d = dict(row)
            d["tracking_role"] = "vasp"
            d["vasp_kind"] = "planned_root"
            planned_rows.append(d)
            planned_seen.add(u.lower())
        else:
            planned_rows.append(
                {
                    "uuid": u,
                    "tracking_role": "vasp",
                    "vasp_kind": "planned_root",
                    "state": None,
                    "name": None,
                    "run_dir": None,
                    "worker": None,
                    "index": None,
                    "note": "planned_root_not_expanded_yet",
                }
            )
            planned_seen.add(u.lower())
    for row in flow_jobs:
        u = (str(row.get("uuid") or "")).strip().lower()
        if not u or (ml and u == ml) or u in planned_seen:
            continue
        d = dict(row)
        d["tracking_role"] = "vasp"
        d["vasp_kind"] = "expanded_job"
        expanded_rows.append(d)
    return {
        "planned_roots": planned_rows,
        "expanded_jobs": expanded_rows,
    }


def _phase_label_from_progress(prog: Dict[str, Any]) -> str:
    phase = str(prog.get("current_phase") or "").strip()
    mapping = {
        "mlip": "MLIP 阶段",
        "vasp": "VASP 阶段",
        "done": "全部结束",
        "unknown": "状态未知",
    }
    return mapping.get(phase, phase or "状态未知")


def _print_iomaker_status_brief(prog: Dict[str, Any]) -> None:
    tr = prog.get("task_ref_used", "")
    print("job_id:", tr or "(empty)")
    if not prog.get("success"):
        print(" error:", prog.get("error"))
        if prog.get("hint"):
            print(" hint:", prog.get("hint"))
        return
    print(" phase:", _phase_label_from_progress(prog), "| mode:", prog.get("mode"))
    if prog.get("job_state"):
        print(" mlip_state:", prog.get("job_state"))
    if prog.get("stage_summary"):
        print(" summary:", prog.get("stage_summary"))
    expanded = prog.get("expanded_vasp_job_counts") or {}
    planned = prog.get("planned_vasp_root_counts") or {}
    if expanded.get("total"):
        print(
            " vasp_jobs:",
            f"{expanded.get('finished', 0)}/{expanded.get('total', 0)} finished",
            f"(completed={expanded.get('completed', 0)}, running={expanded.get('running', 0)}, "
            f"pending={expanded.get('pending', 0)}, failed={expanded.get('failed', 0)})",
        )
    elif planned.get("total"):
        print(" planned_vasp_roots:", planned.get("total"), "(dynamic wrapper job(s), not expanded yet)")
    rows = prog.get("vasp_sub_jobs") or []
    if rows:
        print(" vasp_sub_jobs:")
        for row in rows:
            if not isinstance(row, dict):
                continue
            extra = row.get("note") or ""
            print(
                "  -",
                row.get("uuid"),
                "state=",
                row.get("state"),
                ("  " + extra) if extra else "",
            )
    elif prog.get("planned_vasp_roots"):
        print(" planned_vasp_roots:")
        for row in prog.get("planned_vasp_roots") or []:
            if not isinstance(row, dict):
                continue
            extra = row.get("note") or ""
            print(
                "  -",
                row.get("uuid"),
                "state=",
                row.get("state"),
                ("  " + extra) if extra else "",
            )


def iomaker_progress(
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    verbose: bool = True,
    jf_bin: str = "jf",
    python_head_lines: int = 100,
) -> Dict[str, Any]:
    """
    Single entry point for notebook / CLI: resolve job **ID** (UUID), query jobflow-remote,
    return the same dict as :func:`query_interoptimus_task_progress` plus ``task_ref_used``.

    Typical usage::

        prog = iomaker_progress(result)  # after run_simple_iomaker
        prog = iomaker_progress(ref=\"530a6c27-...\")  # MLIP job UUID after kernel restart
        export INTEROPTIMUS_JOB_ID=...  # optional restart fallback

    Set ``verbose=False`` to skip printing; inspect ``prog`` for full fields
    (e.g. ``python_output_head``, ``vasp_flow_jobs``).
    """
    if isinstance(result, str) and not ref.strip():
        ref = result.strip()
        result = None
    task_ref = iomaker_resolve_ref(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        use_env=True,
    )
    if not task_ref:
        err: Dict[str, Any] = {
            "success": False,
            "error": "missing_task_ref",
            "hint": (
                "Pass result from submit, ref=<MLIP-uuid>, jf_job_id=..., or set "
                "INTEROPTIMUS_JOB_ID (or INTEROPTIMUS_MLIP_JOB_UUID) to the MLIP job UUID."
            ),
            "task_ref_used": "",
        }
        if verbose:
            _print_iomaker_progress_brief(err)
        return err
    rec_hint = _iomaker_task_record_hint_from_result(
        result,
        task_ref=task_ref,
        jf_job_id=jf_job_id,
    )
    prog = query_interoptimus_task_progress(
        task_ref,
        jf_bin=jf_bin,
        python_head_lines=python_head_lines,
        task_record_hint=rec_hint,
    )
    prog = dict(prog)
    prog["task_ref_used"] = task_ref
    if verbose:
        _print_iomaker_progress_brief(prog)
    return prog


def iomaker_status(
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    verbose: bool = True,
    jf_bin: str = "jf",
    python_head_lines: int = 100,
) -> Dict[str, Any]:
    """
    One-line status API for notebooks / scripts.

    Returns the same progress dict as :func:`iomaker_progress`, plus stable summary fields such as
    ``current_phase``, ``stage_summary``, ``vasp_job_counts``, and ``is_finished``.

    After a notebook restart you may pass the MLIP UUID directly, e.g.
    ``iomaker_status("530a6c27-...")``.
    """
    prog = iomaker_progress(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        verbose=False,
        jf_bin=jf_bin,
        python_head_lines=python_head_lines,
    )
    if verbose:
        _print_iomaker_status_brief(prog)
    return prog


def _iomaker_get_job_controller():
    from jobflow_remote.cli.utils import initialize_config_manager, get_job_controller

    initialize_config_manager()
    return get_job_controller()


def _iomaker_read_incar_snapshot(run_dir: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rd = str(run_dir or "").strip()
    if not rd:
        return out
    path = os.path.join(rd, "INCAR")
    if not os.path.isfile(path):
        return out
    out["incar_path"] = path
    try:
        out["incar_text"] = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        out["incar_read_error"] = str(e)
    try:
        from pymatgen.io.vasp.inputs import Incar

        incar = Incar.from_file(path)
        out["incar_dict"] = {str(k): v for k, v in incar.items()}
    except Exception as e:
        out["incar_parse_error"] = str(e)
    return out


def _iomaker_read_job_doc_snapshot(job_uuid: str, job_index: Optional[int] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    uid = str(job_uuid or "").strip()
    if not uid:
        return out
    try:
        jc = _iomaker_get_job_controller()
        db_doc = jc.jobs.find_one({"uuid": uid, **({"index": job_index} if job_index is not None else {})})
    except Exception as e:
        out["job_doc_read_error"] = str(e)
        return out
    if not isinstance(db_doc, dict):
        return out
    out["job_doc"] = db_doc
    raw_job = db_doc.get("job") if isinstance(db_doc.get("job"), dict) else {}
    function_args = list(raw_job.get("function_args") or []) if isinstance(raw_job, dict) else []
    structure_arg = function_args[0] if function_args and isinstance(function_args[0], dict) else None
    out["structure_dict"] = structure_arg
    func = raw_job.get("function") if isinstance(raw_job, dict) else {}
    bound = func.get("@bound") if isinstance(func, dict) else {}
    input_gen = bound.get("input_set_generator") if isinstance(bound, dict) else {}
    user_incar = input_gen.get("user_incar_settings") if isinstance(input_gen, dict) else None
    if isinstance(user_incar, dict):
        out["user_incar_settings"] = dict(user_incar)
    return out


def _iomaker_apply_incar_updates(
    base_incar: Optional[Dict[str, Any]],
    incar_updates: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(base_incar or {})
    for key, value in dict(incar_updates or {}).items():
        k = str(key).strip()
        if not k:
            continue
        if value is None:
            out.pop(k, None)
        else:
            out[k] = value
    return out


def _iomaker_struct_site_label(site: Dict[str, Any]) -> str:
    if not isinstance(site, dict):
        return ""
    lab = site.get("label")
    if isinstance(lab, str) and lab.strip():
        return lab.strip()
    species = site.get("species")
    if isinstance(species, list) and species:
        first = species[0]
        if isinstance(first, dict):
            el = first.get("element")
            if isinstance(el, str) and el.strip():
                return el.strip()
    return ""


def _iomaker_make_magmom_compatible_for_rerun(
    user_incar_settings: Dict[str, Any],
    structure_dict: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    """
    Normalize MAGMOM so pymatgen/atomate2 can regenerate inputs on rerun.

    ``MPRelaxSet.user_incar_settings["MAGMOM"]`` expects a species-keyed dict, while a live INCAR
    may contain a per-site list. When the list length matches the job structure, move the values to
    per-site ``properties.magmom`` and remove ``MAGMOM`` from ``user_incar_settings``. This keeps
    the full site-resolved moments without discarding the information.
    """
    out_incar = dict(user_incar_settings or {})
    out_struct = dict(structure_dict) if isinstance(structure_dict, dict) else structure_dict
    notes: List[str] = []
    magmom = out_incar.get("MAGMOM")
    if not isinstance(magmom, (list, tuple)):
        return out_incar, out_struct, notes

    mags = list(magmom)
    sites = out_struct.get("sites") if isinstance(out_struct, dict) else None
    if isinstance(sites, list) and len(sites) == len(mags):
        patched_sites: List[Dict[str, Any]] = []
        for site, mag in zip(sites, mags):
            sd = dict(site) if isinstance(site, dict) else {"properties": {}}
            props = dict(sd.get("properties") or {})
            props["magmom"] = mag
            sd["properties"] = props
            patched_sites.append(sd)
        out_struct["sites"] = patched_sites
        out_incar.pop("MAGMOM", None)
        notes.append("magmom_list_moved_to_structure_site_properties")
        return out_incar, out_struct, notes

    species_map: Dict[str, Any] = {}
    seen_species: Dict[str, Any] = {}
    if isinstance(sites, list) and len(sites) == len(mags):
        for site, mag in zip(sites, mags):
            label = _iomaker_struct_site_label(site)
            if not label:
                continue
            if label in seen_species and seen_species[label] != mag:
                raise ValueError(
                    "MAGMOM list contains different values for the same species and cannot be "
                    "collapsed into user_incar_settings. Keep the structure information available "
                    "so site-resolved magmoms can be reapplied."
                )
            seen_species[label] = mag
            species_map[label] = mag
    if species_map:
        out_incar["MAGMOM"] = species_map
        notes.append("magmom_list_collapsed_to_species_map")
        return out_incar, out_struct, notes

    raise ValueError(
        "MAGMOM list is not compatible with the serialized job structure; cannot prepare rerun input."
    )


def iomaker_failed_vasp_jobs(
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    verbose: bool = True,
    jf_bin: str = "jf",
) -> Dict[str, Any]:
    """
    Inspect failed expanded VASP sub-jobs and expose their current ``INCAR``.

    Accepts the same *result* / UUID string / ``ref=`` inputs as :func:`iomaker_status`.
    """
    if isinstance(result, str) and not ref.strip():
        ref = result.strip()
        result = None
    prog = iomaker_status(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        verbose=False,
        jf_bin=jf_bin,
    )
    out: Dict[str, Any] = {
        "success": bool(prog.get("success")),
        "task_ref_used": prog.get("task_ref_used"),
        "progress": prog,
        "failed_jobs": [],
    }
    if not prog.get("success"):
        if verbose:
            print("job_id:", out.get("task_ref_used") or "(empty)")
            print(" error:", prog.get("error"))
            if prog.get("hint"):
                print(" hint:", prog.get("hint"))
        return out

    failed_jobs: List[Dict[str, Any]] = []
    for row in prog.get("vasp_sub_jobs") or []:
        if not isinstance(row, dict):
            continue
        if _job_state_bucket(row.get("state")) != "failed":
            continue
        item = dict(row)
        incar_snapshot = _iomaker_read_incar_snapshot(item.get("run_dir"))
        if not incar_snapshot.get("incar_dict"):
            doc_snapshot = _iomaker_read_job_doc_snapshot(
                str(item.get("uuid") or ""),
                item.get("index") if isinstance(item.get("index"), int) else None,
            )
            if isinstance(doc_snapshot.get("user_incar_settings"), dict):
                incar_snapshot["incar_dict"] = dict(doc_snapshot["user_incar_settings"])
                incar_snapshot["incar_source"] = "job_doc.user_incar_settings"
            if doc_snapshot.get("job_doc_read_error") and not incar_snapshot.get("incar_read_error"):
                incar_snapshot["incar_read_error"] = doc_snapshot.get("job_doc_read_error")
        item["current_incar_path"] = incar_snapshot.get("incar_path")
        item["current_incar_text"] = incar_snapshot.get("incar_text")
        item["current_incar"] = incar_snapshot.get("incar_dict")
        item["current_incar_source"] = incar_snapshot.get("incar_source") or (
            "run_dir:INCAR" if incar_snapshot.get("incar_path") else None
        )
        if incar_snapshot.get("incar_parse_error"):
            item["current_incar_parse_error"] = incar_snapshot.get("incar_parse_error")
        if incar_snapshot.get("incar_read_error"):
            item["current_incar_read_error"] = incar_snapshot.get("incar_read_error")
        failed_jobs.append(item)

    out["failed_jobs"] = failed_jobs
    out["failed_job_count"] = len(failed_jobs)
    if verbose:
        print("job_id:", out.get("task_ref_used") or "(empty)")
        print(" failed_vasp_jobs:", len(failed_jobs))
        for item in failed_jobs:
            print(
                "  -",
                item.get("uuid"),
                "index=",
                item.get("index"),
                "run_dir=",
                item.get("run_dir"),
            )
            if item.get("current_incar_path"):
                print("    INCAR:", item.get("current_incar_path"))
    return out


def iomaker_rerun_failed_vasp(
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    failed_job_uuid: str,
    failed_job_index: Optional[int] = None,
    incar_updates: Optional[Dict[str, Any]] = None,
    delete_files: bool = True,
    wait: Optional[int] = 30,
    break_lock: bool = False,
    verbose: bool = True,
    jf_bin: str = "jf",
) -> Dict[str, Any]:
    """
    Patch the failed VASP job's ``user_incar_settings`` and rerun that same job in-place.

    *incar_updates* should only contain the INCAR tags that need to change. Set a value to
    ``None`` to remove that override before rerun.
    """
    if isinstance(result, str) and not ref.strip():
        ref = result.strip()
        result = None
    failed_info = iomaker_failed_vasp_jobs(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        verbose=False,
        jf_bin=jf_bin,
    )
    if not failed_info.get("success"):
        return failed_info

    target_uuid = str(failed_job_uuid or "").strip()
    if not target_uuid:
        out = dict(failed_info)
        out.update({"success": False, "error": "missing_failed_job_uuid"})
        return out
    if incar_updates is not None and not isinstance(incar_updates, dict):
        out = dict(failed_info)
        out.update({"success": False, "error": "invalid_incar_updates"})
        return out

    failed_jobs = failed_info.get("failed_jobs") or []
    target: Optional[Dict[str, Any]] = None
    for item in failed_jobs:
        if not isinstance(item, dict):
            continue
        if str(item.get("uuid") or "").strip() != target_uuid:
            continue
        if failed_job_index is not None and item.get("index") != failed_job_index:
            continue
        target = dict(item)
        break
    if target is None:
        out = dict(failed_info)
        out.update(
            {
                "success": False,
                "error": "failed_job_not_found",
                "failed_job_uuid": target_uuid,
            }
        )
        return out

    base_incar = target.get("current_incar")
    if not isinstance(base_incar, dict) or not base_incar:
        out = dict(failed_info)
        out.update(
            {
                "success": False,
                "error": "missing_current_incar",
                "failed_job_uuid": target_uuid,
                "hint": "Current run_dir/INCAR is required so the rerun can inherit the failed job input.",
            }
        )
        return out

    merged_incar = _iomaker_apply_incar_updates(base_incar, incar_updates)
    job_index = failed_job_index if failed_job_index is not None else target.get("index")

    try:
        jc = _iomaker_get_job_controller()
        db_doc = jc.jobs.find_one({"uuid": target_uuid, **({"index": job_index} if job_index is not None else {})})
        if not isinstance(db_doc, dict):
            return {
                "success": False,
                "error": "job_doc_not_found",
                "failed_job_uuid": target_uuid,
                "job_index": job_index,
            }

        raw_job = db_doc.get("job") if isinstance(db_doc.get("job"), dict) else {}
        function_args = list(raw_job.get("function_args") or []) if isinstance(raw_job, dict) else []
        structure_arg = function_args[0] if function_args and isinstance(function_args[0], dict) else None
        compatible_incar, compatible_structure, compatibility_notes = _iomaker_make_magmom_compatible_for_rerun(
            merged_incar,
            structure_arg,
        )

        if break_lock:
            try:
                jc.unlock_jobs(job_ids=[(target_uuid, int(job_index or 1))])
            except Exception:
                pass
            flow_uuid = str(((failed_info.get("progress") or {}).get("flow_uuid")) or "").strip()
            if flow_uuid:
                try:
                    jc.unlock_flows(flow_ids=[flow_uuid])
                except Exception:
                    pass

        stored_data = dict(db_doc.get("stored_data") or {})
        stored_data["interoptimus_failed_rerun"] = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "incar_updates": dict(incar_updates or {}),
            "compatibility_notes": compatibility_notes,
        }

        update_values: Dict[str, Any] = {
            "job.function.@bound.input_set_generator.user_incar_settings": compatible_incar,
            "stored_data": stored_data,
        }
        if compatible_structure is not None and function_args:
            update_values["job.function_args.0"] = compatible_structure

        jc.set_job_doc_properties(
            update_values,
            job_id=target_uuid,
            job_index=job_index,
            wait=wait,
            break_lock=break_lock,
        )
        modified_job_db_ids = jc.rerun_job(
            job_id=target_uuid,
            job_index=job_index,
            delete_files=delete_files,
            wait=wait,
            break_lock=break_lock,
        )
    except Exception as e:
        return {
            "success": False,
            "error": "rerun_failed",
            "failed_job_uuid": target_uuid,
            "job_index": job_index,
            "hint": str(e),
            "merged_incar": merged_incar,
            "compatible_incar": compatible_incar if 'compatible_incar' in locals() else None,
            "wait": wait,
            "break_lock": bool(break_lock),
        }

    out = dict(failed_info)
    out.update(
        {
            "success": True,
            "failed_job_uuid": target_uuid,
            "job_index": job_index,
            "applied_incar_updates": dict(incar_updates or {}),
            "merged_incar": merged_incar,
            "compatible_incar": compatible_incar,
            "compatibility_notes": compatibility_notes,
            "rerun_modified_job_db_ids": modified_job_db_ids,
            "delete_files": bool(delete_files),
            "wait": wait,
            "break_lock": bool(break_lock),
        }
    )
    refreshed = iomaker_status(
        ref=target_uuid,
        verbose=False,
        jf_bin=jf_bin,
    )
    if isinstance(refreshed, dict):
        out["failed_job_status_after_rerun"] = refreshed
    if verbose:
        print("job_id:", failed_info.get("task_ref_used") or ref or "(empty)")
        print(" rerun_failed_job:", target_uuid, "index=", job_index)
        print(" applied_incar_updates:", dict(incar_updates or {}))
        print(" rerun_modified_job_db_ids:", modified_job_db_ids)
    return out


def _default_iomaker_results_dir(result: Optional[Dict[str, Any]], task_ref: str) -> str:
    r = result or {}
    submit_workdir = str(r.get("submit_workdir") or r.get("local_workdir") or "").strip()
    task_name = str(r.get("task_name") or r.get("name") or "").strip()
    if submit_workdir:
        sw = Path(submit_workdir).resolve()
        stem = task_name or sw.name or "iomaker"
        return str(sw.parent / f"{_safe_task_name(stem)}_results")
    stem = task_name or (task_ref.strip()[:8] if task_ref else "iomaker")
    return str(Path.cwd() / f"{_safe_task_name(stem)}_results")


def _iomaker_bulk_cif_paths_from_result(result: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    r = result or {}
    sm = r.get("structures_meta") if isinstance(r.get("structures_meta"), dict) else {}
    bulk = r.get("settings", {}).get("bulk") if isinstance(r.get("settings"), dict) else {}
    film_cif = str(sm.get("film_cif") or bulk.get("film_cif") or "").strip()
    substrate_cif = str(sm.get("substrate_cif") or bulk.get("substrate_cif") or "").strip()
    return film_cif, substrate_cif


def _iomaker_existing_path(path_str: str, *, base_dir: str = "") -> str:
    s = str(path_str or "").strip()
    if not s:
        return ""
    candidates: List[str] = []
    if base_dir and not os.path.isabs(s):
        candidates.append(os.path.join(base_dir, s))
    candidates.append(s)
    for cand in candidates:
        p = os.path.abspath(os.path.expanduser(cand))
        if os.path.isfile(p):
            return p
    return ""


def _iomaker_bulk_cif_paths_from_submit_dir(submit_workdir: str) -> Tuple[str, str]:
    sw = os.path.abspath(os.path.expanduser((submit_workdir or "").strip()))
    if not sw or not os.path.isdir(sw):
        return "", ""

    report_path = os.path.join(sw, "io_report.txt")
    if os.path.isfile(report_path):
        try:
            film_cif = ""
            substrate_cif = ""
            for raw_line in Path(report_path).read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if line.startswith("Film CIF:"):
                    film_cif = line.split(":", 1)[1].strip()
                elif line.startswith("Substrate CIF:"):
                    substrate_cif = line.split(":", 1)[1].strip()
            film_path = _iomaker_existing_path(film_cif, base_dir=sw)
            substrate_path = _iomaker_existing_path(substrate_cif, base_dir=sw)
            if film_path and substrate_path:
                return film_path, substrate_path
        except Exception:
            pass

    film_guess = _iomaker_existing_path("film.cif", base_dir=sw)
    substrate_guess = _iomaker_existing_path("substrate.cif", base_dir=sw)
    return film_guess, substrate_guess


def _iomaker_bulk_cif_paths(
    task_ref: str,
    result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    film_cif, substrate_cif = _iomaker_bulk_cif_paths_from_result(result)
    film_path = _iomaker_existing_path(film_cif)
    substrate_path = _iomaker_existing_path(substrate_cif)
    if film_path and substrate_path:
        return film_path, substrate_path

    rec_hint = _iomaker_task_record_hint_from_result(result, task_ref=task_ref)
    rec = _merge_interoptimus_task_record(
        resolve_interoptimus_task_record(task_ref),
        rec_hint,
    )
    submit_workdir = str((rec or {}).get("submit_workdir") or "").strip()
    if submit_workdir:
        disk_film, disk_substrate = _iomaker_bulk_cif_paths_from_submit_dir(submit_workdir)
        if disk_film and disk_substrate:
            return disk_film, disk_substrate

    return film_path or film_cif, substrate_path or substrate_cif


def _deep_find_energy(obj: Any, depth: int = 0) -> Optional[float]:
    if depth > 14:
        return None
    if obj is None:
        return None
    if isinstance(obj, (float, int)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        for key in ("energy", "final_energy", "e_fr_energy", "e_wo_entrp"):
            if key in obj:
                v = _deep_find_energy(obj[key], depth + 1)
                if v is not None:
                    return v
        for v in obj.values():
            v = _deep_find_energy(v, depth + 1)
            if v is not None:
                return v
    if isinstance(obj, (list, tuple)):
        for x in obj:
            v = _deep_find_energy(x, depth + 1)
            if v is not None:
                return v
    return None


def _job_tag_from_doc(doc: Any) -> str:
    try:
        md = getattr(doc, "metadata", None) or {}
        if isinstance(md, dict):
            val = md.get("job") or md.get("name") or ""
            return _normalize_jobflow_job_tag(val)
        if hasattr(md, "get"):
            val = md.get("job") or md.get("name") or ""
            return _normalize_jobflow_job_tag(val)
    except Exception:
        pass
    return ""


def _iomaker_load_remote_job_store_doc(run_dir: str) -> Dict[str, Any]:
    rd = os.path.abspath(os.path.expanduser((run_dir or "").strip()))
    if not rd or not os.path.isdir(rd):
        return {}
    path = os.path.join(rd, "remote_job_data.json")
    if not os.path.isfile(path):
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    if isinstance(raw, dict):
        return raw
    return {}


def _iomaker_deep_get_mapping(obj: Any, path: Tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _normalize_jobflow_job_tag(val: Any) -> str:
    """``metadata.job`` may be a str or a one-element list (e.g. ``['film']``) depending on jf-remote version."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _lookup_tag_energy(energies_by_tag: Dict[str, float], tag: str) -> Optional[float]:
    """Resolve ``film`` / ``substrate`` / … with case-insensitive key fallback."""
    if not energies_by_tag:
        return None
    v = energies_by_tag.get(tag)
    if v is not None:
        return float(v)
    tlow = tag.lower()
    for k, ev in energies_by_tag.items():
        if str(k).strip().lower() == tlow:
            return float(ev)
    return None


def _job_tag_from_store_doc(doc: Dict[str, Any]) -> str:
    if not isinstance(doc, dict):
        return ""
    for path in (
        ("metadata", "job"),
        ("metadata", "name"),
        ("additional_json", "jfremote_in", "job", "metadata", "job"),
        ("additional_json", "jfremote_in", "job", "metadata", "name"),
    ):
        raw = _iomaker_deep_get_mapping(doc, path)
        tag = _normalize_jobflow_job_tag(raw)
        if tag:
            return tag
    return ""


def _energy_from_store_doc(doc: Dict[str, Any]) -> Optional[float]:
    if not isinstance(doc, dict):
        return None
    for path in (
        ("output", "output", "energy"),
        ("output", "energy"),
        ("output", "entry", "energy"),
        ("entry", "energy"),
    ):
        val = _iomaker_deep_get_mapping(doc, path)
        if isinstance(val, (float, int)) and not isinstance(val, bool):
            return float(val)
    calcs = _iomaker_deep_get_mapping(doc, ("calcs_reversed",))
    if isinstance(calcs, list):
        for calc in calcs:
            if not isinstance(calc, dict):
                continue
            val = _iomaker_deep_get_mapping(calc, ("output", "energy"))
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                return float(val)
    return None


def _iomaker_collect_vasp_energies_by_tag(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    energies_by_tag: Dict[str, float] = {}
    for row in rows:
        run_dir = str((row or {}).get("run_dir") or "").strip()
        if not run_dir:
            continue
        store_doc = _iomaker_load_remote_job_store_doc(run_dir)
        tag = _job_tag_from_store_doc(store_doc)
        energy = _energy_from_store_doc(store_doc)
        if tag and energy is not None:
            energies_by_tag[tag] = float(energy)
    return energies_by_tag


def _structure_from_store_doc(doc: Dict[str, Any]) -> Optional["Structure"]:
    if not isinstance(doc, dict):
        return None
    try:
        from pymatgen.core.structure import Structure
    except Exception:
        return None
    for path in (
        ("output", "output", "structure"),
        ("output", "structure"),
        ("output", "output", "input", "structure"),
    ):
        st = _iomaker_deep_get_mapping(doc, path)
        if isinstance(st, dict):
            try:
                return Structure.from_dict(st)
            except Exception:
                continue
    return None


def _iomaker_collect_vasp_store_docs_by_tag(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    docs_by_tag: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        run_dir = str((row or {}).get("run_dir") or "").strip()
        if not run_dir:
            continue
        store_doc = _iomaker_load_remote_job_store_doc(run_dir)
        tag = _job_tag_from_store_doc(store_doc)
        if tag:
            docs_by_tag[tag] = store_doc
    return docs_by_tag


def _iomaker_required_vasp_tags(mi: int, ti: int, *, double_it: bool) -> List[str]:
    """VASP job name tags (``metadata['job']``) required to assemble one pair's interface energy."""
    if double_it:
        return ["film", "substrate", f"{mi}_{ti}_it"]
    return [f"{mi}_{ti}_it", f"{mi}_{ti}_film_slab", f"{mi}_{ti}_substrate_slab"]


# Interface relax jobs use metadata job == "{match_id}_{term_id}_it" (see InterfaceWorker.patch_jobflow_jobs).
_VASP_INTERFACE_IT_TAG_RE = re.compile(r"^(\d+)_(\d+)_it$")


def _scheduled_interface_pairs_from_vasp_tags(
    job_by_tag: Dict[str, Dict[str, Any]],
    energies_by_tag: Optional[Dict[str, float]] = None,
) -> set[Tuple[int, int]]:
    """
    Pairs ``(match_id, term_id)`` that have a VASP interface job tag ``"{m}_{t}_it"`` in the
    expanded jobflow-remote rows.

    **Stereographic ``area_strain``** (see ``InterfaceWorker.visualize_minimization_results``) still uses
    one winner per Miller bin (``only_lowest_energy_each_plane``), while **default VASP** uses
    ``vasp_pair_selection='each_match_lowest'`` (one interface relax per match). So ``area_strain`` rows
    need not line up 1:1 with submitted ``*_it`` tags unless you set ``vasp_pair_selection='each_plane'``.
    Pairs with an ``*_it`` tag in the expanded job rows are treated as DFT-scheduled; others are ``mlip_only``.

    Separately, ``opt_results.pkl`` still lists **every** MLIP-optimized termination; pairs with no
    ``*_it`` tag in the fetch expansion are labeled ``mlip_only`` when merging so they are not
    mistaken for incomplete DFT.
    """
    out: set[Tuple[int, int]] = set()
    for mapping in (job_by_tag, energies_by_tag or {}):
        if not isinstance(mapping, dict):
            continue
        for tag in mapping.keys():
            m = _VASP_INTERFACE_IT_TAG_RE.match(str(tag).strip())
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
    return out


def _iomaker_vasp_job_row_by_tag(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map VASP store tag -> latest jobflow row (for state / run_dir)."""
    by_tag: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rd = str((row.get("run_dir") or "")).strip()
        if not rd:
            continue
        doc = _iomaker_load_remote_job_store_doc(rd)
        tag = _job_tag_from_store_doc(doc)
        if tag:
            by_tag[tag] = row
    return by_tag


def _iomaker_pair_vasp_job_rollup(
    mi: int,
    ti: int,
    *,
    double_it: bool,
    job_by_tag: Dict[str, Dict[str, Any]],
) -> str:
    """
    ``pending`` — missing job row or store tag, or any required job not completed;
    ``failed`` — any required job in a terminal failure state;
    ``ready`` — all required jobs completed (energies may still be missing from stores).
    """
    for tg in _iomaker_required_vasp_tags(mi, ti, double_it=double_it):
        row = job_by_tag.get(tg)
        if row is None:
            return "pending"
        b = _job_state_bucket(row.get("state"))
        if b == "failed":
            return "failed"
        if b != "completed":
            return "pending"
    return "ready"


def _iomaker_collect_vasp_energy_rows(
    dest_dir: str,
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    jf_bin: str = "jf",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"success": False, "dest_dir": os.path.abspath(dest_dir)}
    task_ref = iomaker_resolve_ref(result, ref=ref, jf_job_id=jf_job_id, use_env=True)
    if not task_ref:
        out["error"] = "missing_task_ref"
        return out

    dest_root = os.path.abspath(dest_dir)
    pkl_path = os.path.join(dest_root, "opt_results.pkl")
    if not os.path.isfile(pkl_path):
        out["error"] = "missing_opt_results_pkl"
        out["hint"] = "Run iomaker_fetch_results(...) first so opt_results.pkl is exported."
        return out

    film_cif, substrate_cif = _iomaker_bulk_cif_paths(task_ref, result)
    if not film_cif or not substrate_cif:
        out["error"] = "missing_bulk_cif_paths"
        out["hint"] = (
            "Need film/substrate CIF paths from the submit result or the registered submit directory "
            "to build VASP-derived exports."
        )
        return out

    prog = iomaker_status(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        verbose=False,
        jf_bin=jf_bin,
    )
    rows = prog.get("vasp_sub_jobs") or []
    if not rows:
        out["error"] = "expanded_vasp_jobs_unavailable"
        out["hint"] = "VASP dynamic root is known, but real VASP sub-jobs have not expanded yet."
        out["progress"] = prog
        return out

    try:
        from InterOptimus.jobflow import load_opt_results_pickle_payload
        from pymatgen.core.structure import Structure
    except Exception as e:
        out["error"] = "missing_runtime_dependency"
        out["hint"] = str(e)
        return out

    energies_by_tag = _iomaker_collect_vasp_energies_by_tag(rows)
    docs_by_tag = _iomaker_collect_vasp_store_docs_by_tag(rows)
    job_by_tag = _iomaker_vasp_job_row_by_tag(rows)
    scheduled_pairs = _scheduled_interface_pairs_from_vasp_tags(job_by_tag, energies_by_tag)
    payload = load_opt_results_pickle_payload(str(pkl_path))
    opt_results = payload["opt_results"]
    double_it = bool(payload.get("double_interface", True))

    film_prim = Structure.from_file(film_cif).get_primitive_structure()
    sub_prim = Structure.from_file(substrate_cif).get_primitive_structure()
    n_f0 = max(1, len(film_prim))
    n_s0 = max(1, len(sub_prim))
    film_formula = getattr(film_prim.composition, "reduced_formula", "film")
    substrate_formula = getattr(sub_prim.composition, "reduced_formula", "substrate")

    j_per_m2 = 16.02176634
    e_f0 = _lookup_tag_energy(energies_by_tag, "film")
    e_s0 = _lookup_tag_energy(energies_by_tag, "substrate")
    rows_out: List[Dict[str, Any]] = []
    for (mi, ti), od in sorted(opt_results.items(), key=lambda x: (x[0][0], x[0][1])):
        rb = od.get("relaxed_best_interface") or {}
        st = rb.get("structure")
        if not st:
            continue
        struct = Structure.from_dict(json.loads(st) if isinstance(st, str) else st)
        area = float(struct.lattice.a * struct.lattice.b)
        film_atoms = od.get("film_atom_count")
        sub_atoms = od.get("substrate_atom_count")
        if film_atoms is None:
            fi = od.get("film_indices") or rb.get("film_indices") or []
            film_atoms = len(fi) if isinstance(fi, (list, tuple)) else None
        if sub_atoms is None:
            si = od.get("substrate_indices") or rb.get("substrate_indices") or []
            sub_atoms = len(si) if isinstance(si, (list, tuple)) else None

        tag_it = f"{mi}_{ti}_it"
        e_it = energies_by_tag.get(tag_it)
        vasp_energy = None
        mlip_energy = od.get("relaxed_min_it_E") if double_it else od.get("relaxed_min_bd_E")
        energy_label = "interface_energy" if double_it else "cohesive_energy"

        if double_it:
            if (
                e_it is not None
                and e_f0 is not None
                and e_s0 is not None
                and area > 0
                and film_atoms
                and sub_atoms
            ):
                film_scale = float(film_atoms) / float(n_f0)
                sub_scale = float(sub_atoms) / float(n_s0)
                vasp_energy = (e_it - film_scale * e_f0 - sub_scale * e_s0) / area * j_per_m2 / 2.0
        else:
            e_film = energies_by_tag.get(f"{mi}_{ti}_film_slab")
            if e_film is None:
                e_film = energies_by_tag.get(f"{mi}_{ti}_film")
            e_sub = energies_by_tag.get(f"{mi}_{ti}_substrate_slab")
            if e_sub is None:
                e_sub = energies_by_tag.get(f"{mi}_{ti}_substrate")
            if e_it is not None and e_film is not None and e_sub is not None and area > 0:
                vasp_energy = (e_it - e_film - e_sub) / area * j_per_m2

        if (mi, ti) not in scheduled_pairs:
            # MLIP optimized this termination, but no interface VASP job was submitted for it.
            vasp_pair_status = "mlip_only"
        else:
            rollup = _iomaker_pair_vasp_job_rollup(mi, ti, double_it=double_it, job_by_tag=job_by_tag)
            if rollup == "failed":
                vasp_pair_status = "failed"
            elif rollup == "pending":
                vasp_pair_status = "pending"
            elif vasp_energy is not None:
                vasp_pair_status = "complete"
            elif rollup == "ready":
                # Interface VASP finished but γ could not be assembled (missing film/sub bulk E, area, …).
                # Do not mark as hard failed for stereographic: merge uses MLIP binding like mlip_only.
                vasp_pair_status = "mlip_gamma_fallback"
            else:
                vasp_pair_status = "failed"

        rows_out.append(
            {
                "pair": (mi, ti),
                "match_id": mi,
                "term_id": ti,
                "A_A2": area,
                "film_atom_count": film_atoms,
                "substrate_atom_count": sub_atoms,
                "energy_label": energy_label,
                "mlip_energy": mlip_energy,
                "vasp_energy": vasp_energy,
                "vasp_pair_status": vasp_pair_status,
                "E_it_eV": e_it,
                "match_area": od.get("match_area"),
                "strain": od.get("strain"),
            }
        )

    out.update(
        {
            "success": True,
            "task_ref_used": task_ref,
            "progress": prog,
            "pkl_path": pkl_path,
            "payload": payload,
            "film_cif": film_cif,
            "substrate_cif": substrate_cif,
            "film_formula": film_formula,
            "substrate_formula": substrate_formula,
            "double_interface": double_it,
            "energies_by_tag": energies_by_tag,
            "docs_by_tag": docs_by_tag,
            "vasp_job_by_tag": job_by_tag,
            "vasp_scheduled_pairs": sorted(scheduled_pairs),
            "rows": rows_out,
        }
    )
    return out


def iomaker_build_vasp_mlip_style_results(
    dest_dir: str,
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    jf_bin: str = "jf",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Rebuild MLIP-style exported artifacts using finished VASP energies.

    Writes a ``vasp_results/`` directory under *dest_dir* containing
    ``io_report.txt``, ``pairs_summary.txt``, ``area_strain``,
    ``opt_results_summary.json``, ``stereographic.jpg``, and
    ``stereographic_interactive.html``.
    """
    out = _iomaker_collect_vasp_energy_rows(
        dest_dir,
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        jf_bin=jf_bin,
    )
    if not out.get("success"):
        return out

    try:
        from InterOptimus.matching import (
            parse_area_strain_records,
            plot_binding_energy_analysis,
            plot_binding_energy_analysis_interactive,
        )
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        return {
            "success": False,
            "error": "missing_runtime_dependency",
            "hint": str(e),
            "dest_dir": os.path.abspath(dest_dir),
        }

    dest_root = os.path.abspath(dest_dir)
    vasp_root = os.path.join(dest_root, "vasp_results")
    os.makedirs(vasp_root, exist_ok=True)

    rows_out_list = out.get("rows") or []
    if not rows_out_list:
        return {
            "success": False,
            "error": "no_pair_rows",
            "hint": "No interface rows were produced from opt_results.pkl for VASP merge.",
            "dest_dir": os.path.abspath(dest_dir),
            "task_ref_used": out.get("task_ref_used"),
            "energies_by_tag": out.get("energies_by_tag"),
        }

    pair_row: Dict[Tuple[int, int], Dict[str, Any]] = {
        (int(row["match_id"]), int(row["term_id"])): row for row in rows_out_list
    }
    rows_by_pair_complete = {
        pair: row
        for pair, row in pair_row.items()
        if row.get("vasp_pair_status") == "complete" and row.get("vasp_energy") is not None
    }
    partial_note = ""
    dft_eligible = len([r for r in rows_out_list if r.get("vasp_pair_status") != "mlip_only"])
    if len(rows_by_pair_complete) < dft_eligible:
        partial_note = (
            f"Partial DFT: {len(rows_by_pair_complete)}/{dft_eligible} scheduled VASP pairs have a finished "
            "interface energy γ; others are pending/failed. "
            f"{len(rows_out_list) - dft_eligible} extra (match, term) keys in opt_results.pkl have no ``*_it`` "
            "job in the expanded fetch list (labeled mlip)—stereographic area_strain rows should still map to "
            "scheduled pairs when tags match."
        )

    summary_src = os.path.join(dest_root, "opt_results_summary.json")
    summary_data: Dict[str, Any] = {}
    if os.path.isfile(summary_src):
        try:
            summary_data = json.loads(Path(summary_src).read_text(encoding="utf-8"))
        except Exception:
            summary_data = {}
    for item in summary_data.values():
        if not isinstance(item, dict):
            continue
        pair = (int(item.get("match_id", -1)), int(item.get("term_id", -1)))
        hit = rows_by_pair_complete.get(pair)
        if not hit:
            continue
        key = "relaxed_min_it_E" if out.get("double_interface") else "relaxed_min_bd_E"
        item[key] = float(hit["vasp_energy"])
    vasp_summary_path = os.path.join(vasp_root, "opt_results_summary.json")
    Path(vasp_summary_path).write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ``area_strain`` from the MLIP worker: per stereographic bin, (match_id, term_id) is the winning pair
    # among competing matches using the same per-plane selection as DFT (see itworker.visualize_minimization_results).
    area_src = os.path.join(dest_root, "area_strain")
    records = parse_area_strain_records(area_src) if os.path.isfile(area_src) else []
    vasp_area_records: List[Dict[str, Any]] = []
    for record in records:
        pair = (int(record["match_id"]), int(record["term_id"]))
        prow = pair_row.get(pair)
        updated = dict(record)
        if prow and prow.get("vasp_pair_status") == "complete" and prow.get("vasp_energy") is not None:
            updated["binding_energy"] = float(prow["vasp_energy"])
            updated["dft_status"] = "complete"
        elif prow and prow.get("vasp_pair_status") in ("mlip_only", "mlip_gamma_fallback"):
            updated["binding_energy"] = float(record["binding_energy"])
            updated["dft_status"] = "mlip"
        elif prow:
            updated["binding_energy"] = float(record["binding_energy"])
            updated["dft_status"] = str(prow.get("vasp_pair_status") or "pending")
        else:
            updated["binding_energy"] = float(record["binding_energy"])
            updated["dft_status"] = "pending"
        vasp_area_records.append(updated)
    vasp_area_path = os.path.join(vasp_root, "area_strain")
    lines: List[str] = []
    for record in vasp_area_records:
        p1 = tuple(int(x) for x in record["material1_plane"])
        p2 = tuple(int(x) for x in record["material2_plane"])
        line = (
            f"({p1[0]} {p1[1]} {p1[2]}) "
            f"({p2[0]} {p2[1]} {p2[2]}) "
            f"{float(record['area']):.4f} {float(record['strain']):.4f} "
            f"{float(record['binding_energy']):.4f} {int(record['match_id'])} {int(record['term_id'])}"
        )
        st = str(record.get("dft_status") or "").strip().lower()
        if st and st != "complete":
            line += f" {st}"
        lines.append(line)
    Path(vasp_area_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    pairs_summary_path = os.path.join(vasp_root, "pairs_summary.txt")
    pair_summary_rows: List[Dict[str, Any]] = []
    first_record_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for record in records:
        pair = (int(record["match_id"]), int(record["term_id"]))
        first_record_by_pair.setdefault(pair, record)
    for pair in sorted(pair_row.keys()):
        hit = pair_row[pair]
        # Table is for DFT workflow only; MLIP-only terminations stay in opt_results.pkl / area_strain.
        if hit.get("vasp_pair_status") == "mlip_only":
            continue
        sample = first_record_by_pair.get(pair, {})
        ev = hit.get("vasp_energy")
        st = str(hit.get("vasp_pair_status") or "pending")
        if st == "complete" and ev is not None:
            evs = str(float(ev))
        else:
            evs = "N/A"
        pair_summary_rows.append(
            {
                "match_id": pair[0],
                "term_id": pair[1],
                "film_conventional_miller": [int(x) for x in sample.get("material1_plane", [])],
                "substrate_conventional_miller": [int(x) for x in sample.get("material2_plane", [])],
                "energy_type": hit["energy_label"],
                "energy_value": evs,
                "film_atom_count": hit.get("film_atom_count"),
                "substrate_atom_count": hit.get("substrate_atom_count"),
                "vasp_dft_status": st,
                "match_area": hit.get("match_area"),
                "strain": hit.get("strain"),
            }
        )
    with open(pairs_summary_path, "w", encoding="utf-8") as f:
        f.write(
            "# DFT-scheduled pairs only (mlip_only terminations omitted; see opt_results.pkl).\n"
        )
        f.write(
            "\t".join(
                [
                    "match_id",
                    "term_id",
                    "film_conventional_miller",
                    "substrate_conventional_miller",
                    "energy_type",
                    "energy_value",
                    "film_atom_count",
                    "substrate_atom_count",
                    "vasp_dft_status",
                    "match_area",
                    "strain",
                ]
            )
            + "\n"
        )
        for row in pair_summary_rows:
            f.write(
                "\t".join(
                    [
                        str(row["match_id"]),
                        str(row["term_id"]),
                        str(row["film_conventional_miller"]),
                        str(row["substrate_conventional_miller"]),
                        str(row["energy_type"]),
                        str(row["energy_value"]),
                        str(row["film_atom_count"]),
                        str(row["substrate_atom_count"]),
                        str(row["vasp_dft_status"]),
                        str(row["match_area"]),
                        str(row["strain"]),
                    ]
                )
                + "\n"
            )

    vasp_pairs_best_it_dir = os.path.join(vasp_root, "pairs_best_it")
    os.makedirs(vasp_pairs_best_it_dir, exist_ok=True)
    docs_by_tag = out.get("docs_by_tag") or {}
    materialized_vasp_pairs: List[str] = []
    for pair in sorted(rows_by_pair_complete):
        mi, ti = pair
        pair_dir = os.path.join(vasp_pairs_best_it_dir, f"match_{mi}_term_{ti}")
        os.makedirs(pair_dir, exist_ok=True)
        wrote_any = False

        best_it = _structure_from_store_doc(docs_by_tag.get(f"{mi}_{ti}_it", {}))
        if best_it is not None:
            best_it.to(fmt="poscar", filename=os.path.join(pair_dir, "best_it_CONTCAR"))
            wrote_any = True

        sfilm = _structure_from_store_doc(docs_by_tag.get(f"{mi}_{ti}_sfilm", {}))
        if sfilm is not None:
            sfilm.to(fmt="poscar", filename=os.path.join(pair_dir, "sfilm_POSCAR"))
            wrote_any = True

        film_slab = _structure_from_store_doc(docs_by_tag.get(f"{mi}_{ti}_film_slab", {})) or _structure_from_store_doc(
            docs_by_tag.get(f"{mi}_{ti}_film", {})
        )
        if film_slab is not None:
            film_slab.to(fmt="poscar", filename=os.path.join(pair_dir, "film_slab_POSCAR"))
            wrote_any = True

        substrate_slab = _structure_from_store_doc(
            docs_by_tag.get(f"{mi}_{ti}_substrate_slab", {})
        ) or _structure_from_store_doc(docs_by_tag.get(f"{mi}_{ti}_substrate", {}))
        if substrate_slab is not None:
            substrate_slab.to(fmt="poscar", filename=os.path.join(pair_dir, "substrate_slab_POSCAR"))
            wrote_any = True

        if wrote_any:
            materialized_vasp_pairs.append(pair_dir)

    stereographic_jpg = os.path.join(vasp_root, "stereographic.jpg")
    stereographic_html = os.path.join(vasp_root, "stereographic_interactive.html")
    if vasp_area_records:
        material1_planes = np.array([r["material1_plane"] for r in vasp_area_records])
        material2_planes = np.array([r["material2_plane"] for r in vasp_area_records])
        dft_status_plot = [str(r.get("dft_status") or "pending").lower() for r in vasp_area_records]
        # ``mlip`` = no DFT for this termination; still plot MLIP binding energy (must stay finite).
        # Using NaN for mlip made ``create_stereographic_plot`` treat points as failed (×).
        binding_energies = np.array(
            [
                float(r["binding_energy"]) if st in ("complete", "mlip") else float("nan")
                for r, st in zip(vasp_area_records, dft_status_plot)
            ],
            dtype=float,
        )
        pair_labels = [f"({int(r['match_id'])}, {int(r['term_id'])})" for r in vasp_area_records]

        fig = plot_binding_energy_analysis(
            material1_planes,
            material2_planes,
            binding_energies,
            str(out.get("film_formula") or "film"),
            str(out.get("substrate_formula") or "substrate"),
            "Interface Energy" if out.get("double_interface") else "Cohesive Energy",
            dft_status=dft_status_plot,
        )
        fig.savefig(stereographic_jpg, dpi=600, bbox_inches="tight", format="jpg")
        plt.close(fig)

        try:
            import plotly.io as pio

            interactive_fig = plot_binding_energy_analysis_interactive(
                material1_planes,
                material2_planes,
                binding_energies,
                pair_labels,
                pair_labels,
                str(out.get("film_formula") or "film"),
                str(out.get("substrate_formula") or "substrate"),
                "Interface Energy" if out.get("double_interface") else "Cohesive Energy",
                dft_status=dft_status_plot,
            )
            interactive_div = pio.to_html(
                interactive_fig,
                include_plotlyjs=True,
                full_html=False,
                default_width="100%",
                default_height="700px",
            )
            html = (
                "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
                "  <meta charset=\"utf-8\">\n"
                "  <title>Stereographic Results</title>\n"
                "  <style>body { font-family: Arial, sans-serif; margin: 24px; background: #fff; color: #111; }</style>\n"
                "</head>\n<body>\n"
                f"  {interactive_div}\n"
                "</body>\n</html>\n"
            )
            Path(stereographic_html).write_text(html, encoding="utf-8")
        except Exception:
            stereographic_html = ""

    io_report_path = os.path.join(vasp_root, "io_report.txt")
    report_lines = [
        "=" * 80,
        "InterOptimus IO Report",
        "Energy source: rebuilt from VASP jobs (includes partial / failed runs when applicable)",
        "=" * 80,
        "",
        "Structures",
        "-" * 80,
        f"Film CIF: {out.get('film_cif')}",
        f"Substrate CIF: {out.get('substrate_cif')}",
        f"Film: {out.get('film_formula')}",
        f"Substrate: {out.get('substrate_formula')}",
        "",
        "Outputs",
        "-" * 80,
        f"Export directory: {vasp_root}",
        f"Pairs summary: {pairs_summary_path}",
        f"VASP pairs_best_it: {vasp_pairs_best_it_dir}",
        f"Stereographic plot: {stereographic_jpg}",
        f"Interactive stereographic plot: {stereographic_html or '(not written)'}",
        "",
    ]
    if partial_note:
        report_lines.extend(["Notes", "-" * 80, partial_note, ""])
    report_lines.extend(
        [
            "Pairs (DFT-scheduled only; MLIP-only terminations are in area_strain / opt_results.pkl)",
            "(interface energy column: VASP J/m^2 when complete, else N/A; vasp_dft_status column)",
            "-" * 80,
        ]
    )
    report_lines.extend(Path(pairs_summary_path).read_text(encoding="utf-8").splitlines())
    Path(io_report_path).write_text("\n".join(report_lines), encoding="utf-8")

    out.update(
        {
            "success": True,
            "vasp_results_dir": vasp_root,
            "vasp_io_report_path": io_report_path,
            "vasp_pairs_summary_path": pairs_summary_path,
            "vasp_area_strain_path": vasp_area_path,
            "vasp_opt_results_summary_path": vasp_summary_path,
            "vasp_pairs_best_it_dir": vasp_pairs_best_it_dir if materialized_vasp_pairs else None,
            "vasp_stereographic_path": stereographic_jpg if os.path.isfile(stereographic_jpg) else None,
            "vasp_stereographic_interactive_path": (
                stereographic_html if stereographic_html and os.path.isfile(stereographic_html) else None
            ),
            "vasp_partial_export": bool(partial_note),
        }
    )
    # Copy merged stereographic outputs to the export root (same filenames as ``iomaker_fetch_results``).
    # The first artifact pull leaves MLIP-only ``stereographic.jpg`` at *dest_root*; without this step,
    # opening *dest_root*/stereographic.jpg still shows the old plot even after a successful VASP rebuild
    # under *dest_root*/vasp_results/.
    try:
        for src_name, dst_name in (
            ("stereographic.jpg", "stereographic.jpg"),
            ("stereographic_interactive.html", "stereographic_interactive.html"),
            ("area_strain", "area_strain"),
        ):
            src = os.path.join(vasp_root, src_name)
            dst = os.path.join(dest_root, dst_name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        out["merged_stereographic_to_export_root"] = True
    except Exception as e:
        out["merged_stereographic_to_export_root_error"] = str(e)
    if verbose:
        print("VASP rebuilt results:", vasp_root)
        print(" VASP io_report:", io_report_path)
        if out.get("vasp_stereographic_path"):
            print(" VASP stereographic:", out["vasp_stereographic_path"])
        if out.get("merged_stereographic_to_export_root"):
            print(
                " (also copied stereographic.jpg / stereographic_interactive.html / area_strain to export root)"
            )
    return out


def iomaker_build_vasp_interface_report(
    dest_dir: str,
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    jf_bin: str = "jf",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build the VASP interface-energy markdown report and comparison plot in *dest_dir*.

    Requires exported ``opt_results.pkl`` and access to expanded VASP jobs in jobflow-remote.
    """
    out: Dict[str, Any] = {"success": False, "dest_dir": os.path.abspath(dest_dir)}
    task_ref = iomaker_resolve_ref(result, ref=ref, jf_job_id=jf_job_id, use_env=True)
    if not task_ref:
        out["error"] = "missing_task_ref"
        return out

    dest_root = os.path.abspath(dest_dir)
    pkl_path = os.path.join(dest_root, "opt_results.pkl")
    if not os.path.isfile(pkl_path):
        out["error"] = "missing_opt_results_pkl"
        out["hint"] = "Run iomaker_fetch_results(...) first so opt_results.pkl is exported."
        return out

    film_cif, substrate_cif = _iomaker_bulk_cif_paths(task_ref, result)
    if not film_cif or not substrate_cif:
        out["error"] = "missing_bulk_cif_paths"
        out["hint"] = (
            "Need film/substrate CIF paths from the submit result or the registered submit directory "
            "to build the VASP report."
        )
        return out

    prog = iomaker_status(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        verbose=False,
        jf_bin=jf_bin,
    )
    rows = prog.get("vasp_sub_jobs") or []
    if not rows:
        out["error"] = "expanded_vasp_jobs_unavailable"
        out["hint"] = "VASP dynamic root is known, but real VASP sub-jobs have not expanded yet."
        out["progress"] = prog
        return out

    try:
        from jobflow_remote.cli.utils import initialize_config_manager, get_job_controller
        from InterOptimus.jobflow import load_opt_results_pickle_payload
        from pymatgen.core.structure import Structure
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        out["error"] = "missing_runtime_dependency"
        out["hint"] = str(e)
        return out

    try:
        initialize_config_manager()
        jc = get_job_controller()
    except Exception as e:
        out["error"] = "jobflow_remote_unavailable"
        out["hint"] = str(e)
        return out

    energies_by_tag: Dict[str, float] = {}
    for row in rows:
        uid = (row or {}).get("uuid")
        if not uid:
            continue
        try:
            doc = jc.get_job_doc(job_id=str(uid))
        except Exception:
            doc = None
        if doc is None:
            continue
        tag = _job_tag_from_doc(doc)
        raw = doc.model_dump() if hasattr(doc, "model_dump") else doc
        en = _deep_find_energy(raw)
        if en is not None and tag:
            energies_by_tag[tag] = float(en)

    # Which (match, term) had interface VASP jobs is inferred from tags (matches whatever IOMaker submitted).
    scheduled_pairs = _scheduled_interface_pairs_from_vasp_tags({}, energies_by_tag)

    payload = load_opt_results_pickle_payload(str(pkl_path))
    opt_results = payload["opt_results"]
    double_it = bool(payload.get("double_interface", True))

    film_prim = Structure.from_file(film_cif).get_primitive_structure()
    sub_prim = Structure.from_file(substrate_cif).get_primitive_structure()
    n_f0 = max(1, len(film_prim))
    n_s0 = max(1, len(sub_prim))

    j_per_m2 = 16.02176634
    e_f0 = _lookup_tag_energy(energies_by_tag, "film")
    e_s0 = _lookup_tag_energy(energies_by_tag, "substrate")
    rows_out: List[Dict[str, Any]] = []
    for (mi, ti), od in sorted(opt_results.items(), key=lambda x: (x[0][0], x[0][1])):
        if (mi, ti) not in scheduled_pairs:
            continue
        rb = od.get("relaxed_best_interface") or {}
        st = rb.get("structure")
        if not st:
            continue
        struct = Structure.from_dict(json.loads(st) if isinstance(st, str) else st)
        area = float(struct.lattice.a * struct.lattice.b)
        film_atoms = od.get("film_atom_count")
        sub_atoms = od.get("substrate_atom_count")
        if film_atoms is None:
            fi = od.get("film_indices") or rb.get("film_indices") or []
            film_atoms = len(fi) if isinstance(fi, (list, tuple)) else None
        if sub_atoms is None:
            si = od.get("substrate_atom_count") or od.get("substrate_indices") or rb.get("substrate_indices") or []
            sub_atoms = len(si) if isinstance(si, (list, tuple)) else None
        tag_it = f"{mi}_{ti}_it"
        e_it = energies_by_tag.get(tag_it)
        gamma = None
        if (
            e_it is not None
            and e_f0 is not None
            and e_s0 is not None
            and area > 0
            and film_atoms
            and sub_atoms
        ):
            film_scale = float(film_atoms) / float(n_f0)
            sub_scale = float(sub_atoms) / float(n_s0)
            gamma = (e_it - film_scale * e_f0 - sub_scale * e_s0) / area * j_per_m2 / 2.0
        rows_out.append(
            {
                "pair": (mi, ti),
                "E_it_eV": e_it,
                "gamma_J_m2": gamma,
                "A_A2": area,
                "mlip_relaxed_min_it_E": od.get("relaxed_min_it_E"),
                "film_atom_count": film_atoms,
                "substrate_atom_count": sub_atoms,
            }
        )

    report_md = os.path.join(dest_root, "vasp_interface_energy_report.md")
    lines = [
        "# VASP 界面能（由 Flow 输出能量 + CIF primitive 原子数估算）",
        "",
        f"- 双界面: {double_it}",
        f"- opt_results: `{pkl_path}`",
        f"- primitive 原子数: N_f0={n_f0}, N_s0={n_s0}（来自 {film_cif} / {substrate_cif}）",
        f"- 已从 jobflow 解析的标签能量键: {sorted(energies_by_tag.keys())}",
        "",
        "| match | term | E_it (eV) | γ_VASP (J/m²) | γ_MLIP (J/m²) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows_out:
        mi, ti = row["pair"]
        lines.append(
            f"| {mi} | {ti} | {row.get('E_it_eV')} | "
            f"{row.get('gamma_J_m2') if row.get('gamma_J_m2') is not None else '—'} | "
            f"{row.get('mlip_relaxed_min_it_E') if row.get('mlip_relaxed_min_it_E') is not None else '—'} |"
        )
    Path(report_md).write_text("\n".join(lines), encoding="utf-8")

    plot_path = os.path.join(dest_root, "vasp_vs_mlip_interface_energy.png")
    xs: List[str] = []
    y_m: List[float] = []
    y_v: List[float] = []
    has_vasp_val = False
    for row in rows_out:
        if row.get("mlip_relaxed_min_it_E") is None:
            continue
        xs.append(f"{row['pair'][0]},{row['pair'][1]}")
        y_m.append(float(row["mlip_relaxed_min_it_E"]))
        gv = row.get("gamma_J_m2")
        if gv is not None:
            has_vasp_val = True
            y_v.append(float(gv))
        else:
            y_v.append(float("nan"))
    if xs:
        fig, ax = plt.subplots(figsize=(8, 4))
        xn = np.arange(len(xs))
        ax.bar(xn - 0.2, y_m, 0.4, label="MLIP γ (relaxed_min_it_E)")
        if has_vasp_val:
            ax.bar(xn + 0.2, y_v, 0.4, label="VASP γ (est.)")
        ax.set_xticks(xn)
        ax.set_xticklabels(xs, rotation=30, ha="right")
        ax.set_ylabel("J/m²")
        ax.legend()
        ax.set_title("Interface energy: MLIP vs VASP (estimated)")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    else:
        plot_path = ""

    out.update(
        {
            "success": True,
            "task_ref_used": task_ref,
            "progress": prog,
            "energies_by_tag": energies_by_tag,
            "rows": rows_out,
            "vasp_interface_energy_report_path": report_md,
            "vasp_interface_energy_plot_path": plot_path or None,
        }
    )
    if verbose:
        print("VASP report:", report_md)
        if plot_path:
            print("VASP plot:", plot_path)
    return out


def iomaker_fetch_results(
    dest_dir: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    write_report_md: Optional[str] = None,
    jf_bin: str = "jf",
    include_progress: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    One-line export API for notebooks / scripts.

    Copies report / stereographic figures / ``opt_results.pkl`` / ``pairs_best_it`` and related
    artifacts into *dest_dir*. If omitted, a sibling ``<task>_results`` directory is chosen.

    Also accepts the MLIP UUID directly as the first positional argument, e.g.
    ``iomaker_fetch_results("530a6c27-...")`` after a notebook restart.
    """
    if isinstance(dest_dir, dict) and result is None:
        result = dest_dir
        dest_dir = None
    elif (
        isinstance(dest_dir, str)
        and result is None
        and not ref.strip()
        and _FLOW_UUID_RE.match(dest_dir.strip())
    ):
        ref = dest_dir.strip()
        dest_dir = None
    task_ref = iomaker_resolve_ref(result, ref=ref, jf_job_id=jf_job_id, use_env=True)
    if not task_ref:
        out: Dict[str, Any] = {
            "success": False,
            "error": "missing_task_ref",
            "task_ref_used": "",
        }
        if verbose:
            print("job_id: (empty) — set result / ref / jf_job_id / INTEROPTIMUS_JOB_ID")
        return out
    dest_root = os.path.abspath(dest_dir or _default_iomaker_results_dir(result, task_ref))
    fr = iomaker_pull_artifacts(
        dest_root,
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        write_report_md=write_report_md,
        jf_bin=jf_bin,
        include_progress=include_progress,
        verbose=False,
    )
    fr = dict(fr)
    fr["export_dir"] = dest_root
    if include_progress:
        try:
            refreshed_prog = iomaker_status(
                result,
                ref=task_ref,
                jf_job_id=jf_job_id,
                verbose=False,
                jf_bin=jf_bin,
            )
            if isinstance(refreshed_prog, dict) and refreshed_prog.get("success"):
                fr["progress"] = refreshed_prog
        except Exception as e:
            fr["progress_refresh_error"] = str(e)
    if fr.get("success"):
        try:
            vr2 = iomaker_build_vasp_mlip_style_results(
                dest_root,
                result,
                ref=ref,
                jf_job_id=jf_job_id,
                jf_bin=jf_bin,
                verbose=False,
            )
            fr["vasp_results"] = vr2
            if vr2.get("success"):
                for key in (
                    "vasp_results_dir",
                    "vasp_io_report_path",
                    "vasp_pairs_summary_path",
                    "vasp_area_strain_path",
                    "vasp_opt_results_summary_path",
                    "vasp_pairs_best_it_dir",
                    "vasp_stereographic_path",
                    "vasp_stereographic_interactive_path",
                ):
                    fr[key] = vr2.get(key)
                # Overwrite MLIP-only artifacts at export root so opening stereographic.jpg / area_strain
                # shows DFT-merged plots (fetch copied run_dir files before this step).
                try:
                    vasp_dir = vr2.get("vasp_results_dir")
                    if vasp_dir and os.path.isdir(vasp_dir):
                        for src_name, dst_name in (
                            ("stereographic.jpg", "stereographic.jpg"),
                            ("stereographic_interactive.html", "stereographic_interactive.html"),
                            ("area_strain", "area_strain"),
                        ):
                            src = os.path.join(vasp_dir, src_name)
                            dst = os.path.join(dest_root, dst_name)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                        fr["merged_stereographic_to_export_root"] = True
                except Exception as e:
                    fr["merged_stereographic_to_export_root_error"] = str(e)
            elif vr2.get("error"):
                fr["vasp_results_error"] = vr2.get("error")
                if vr2.get("hint"):
                    fr["vasp_results_hint"] = vr2.get("hint")
        except Exception as e:
            fr["vasp_results"] = {"success": False, "error": str(e)}
            fr["vasp_results_error"] = str(e)
        try:
            vr = iomaker_build_vasp_interface_report(
                dest_root,
                result,
                ref=ref,
                jf_job_id=jf_job_id,
                jf_bin=jf_bin,
                verbose=False,
            )
            fr["vasp_interface_report"] = vr
            if vr.get("success"):
                fr["vasp_interface_energy_report_path"] = vr.get("vasp_interface_energy_report_path")
                fr["vasp_interface_energy_plot_path"] = vr.get("vasp_interface_energy_plot_path")
            elif vr.get("error"):
                fr["vasp_interface_report_error"] = vr.get("error")
                if vr.get("hint"):
                    fr["vasp_interface_report_hint"] = vr.get("hint")
        except Exception as e:
            fr["vasp_interface_report"] = {
                "success": False,
                "error": str(e),
            }
            fr["vasp_interface_report_error"] = str(e)
    if verbose:
        print("job_id:", task_ref)
        print(" export_dir:", dest_root)
        print(" success:", fr.get("success"), " run_dir:", fr.get("run_dir"))
        prog = fr.get("progress") if isinstance(fr.get("progress"), dict) else {}
        if prog.get("stage_summary"):
            print(" summary:", prog.get("stage_summary"))
        if fr.get("vasp_results_dir"):
            print(" vasp_results:", fr.get("vasp_results_dir"))
        if fr.get("vasp_stereographic_path"):
            print(" vasp stereographic:", fr.get("vasp_stereographic_path"))
        if fr.get("merged_stereographic_to_export_root"):
            print(
                " note: stereographic.jpg / stereographic_interactive.html / area_strain at export root "
                "were overwritten with VASP-merged versions (see vasp_results/ for copies)."
            )
        if fr.get("vasp_interface_energy_report_path"):
            print(" vasp_report:", fr.get("vasp_interface_energy_report_path"))
        if fr.get("vasp_interface_energy_plot_path"):
            print(" vasp_plot:", fr.get("vasp_interface_energy_plot_path"))
    return fr


def iomaker_pull_artifacts(
    dest_dir: str,
    result: Optional[Dict[str, Any]] = None,
    *,
    ref: str = "",
    jf_job_id: str = "",
    write_report_md: Optional[str] = None,
    jf_bin: str = "jf",
    include_progress: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Copy MLIP phase artifacts (images, ``opt_results.pkl``, etc.) into *dest_dir*.

    Uses the same **job ID** resolution as :func:`iomaker_progress`. If *write_report_md* is
    ``None``, writes ``fetched_summary.md`` under *dest_dir*.
    """
    task_ref = iomaker_resolve_ref(
        result,
        ref=ref,
        jf_job_id=jf_job_id,
        use_env=True,
    )
    if not task_ref:
        out: Dict[str, Any] = {
            "success": False,
            "error": "missing_task_ref",
            "task_ref_used": "",
        }
        if verbose:
            print("job_id: (empty) — set ref / jf_job_id / INTEROPTIMUS_JOB_ID")
        return out
    dest_root = os.path.abspath(dest_dir)
    os.makedirs(dest_root, exist_ok=True)
    rec_hint = _iomaker_task_record_hint_from_result(
        result,
        task_ref=task_ref,
        jf_job_id=jf_job_id,
    )
    if write_report_md is None:
        report_path: Optional[str] = str(Path(dest_root) / "fetched_summary.md")
    elif str(write_report_md).strip() == "":
        report_path = None
    else:
        report_path = str(write_report_md)
    fr = fetch_interoptimus_task_results(
        task_ref,
        jf_bin=jf_bin,
        copy_images_to=dest_root,
        write_report_to=os.path.abspath(report_path) if report_path else None,
        include_progress=include_progress,
        task_record_hint=rec_hint,
    )
    fr = dict(fr)
    fr["task_ref_used"] = task_ref
    if verbose:
        print("job_id:", task_ref)
        print(" success:", fr.get("success"), " run_dir:", fr.get("run_dir"))
        print(" dest:", dest_root)
    return fr


def _coerce_flow_doc_mapping(flow_doc: Any) -> Dict[str, Any]:
    if flow_doc is None:
        return {}
    if isinstance(flow_doc, dict):
        return flow_doc
    md = getattr(flow_doc, "model_dump", None)
    if callable(md):
        try:
            d = md()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    d2 = getattr(flow_doc, "__dict__", None)
    if isinstance(d2, dict):
        return {k: v for k, v in d2.items() if not str(k).startswith("_")}
    return {}


def _job_uuid_list_from_mongo_flow(flow_doc: Any) -> List[str]:
    """
    UUID strings from a jobflow-remote flow document: field ``jobs`` and/or ``ids``
    (``ids`` entries are ``(db_id, uuid, index)`` per FlowDoc).
    """
    d = _coerce_flow_doc_mapping(flow_doc)
    out: List[str] = []
    seen: set = set()

    def add(u: str) -> None:
        s = u.strip()
        if _FLOW_UUID_RE.match(s) and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)

    raw = d.get("jobs")
    if isinstance(raw, (list, tuple)):
        for x in raw:
            if isinstance(x, str):
                add(x)
            elif isinstance(x, dict):
                u = x.get("uuid")
                if isinstance(u, str):
                    add(u)
    raw_ids = d.get("ids")
    if isinstance(raw_ids, (list, tuple)):
        for item in raw_ids:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                u = item[1]
                if isinstance(u, str):
                    add(u)
            elif isinstance(item, dict):
                u = item.get("1")
                if u is None:
                    u = item.get(1)
                if isinstance(u, str):
                    add(u)
    return out


def _submit_dir_candidates(rec: Dict[str, Any]) -> List[str]:
    """Submit workdirs that may contain ``io_flow.json`` (from registry / task record only)."""
    dirs: List[str] = []
    seen: set = set()

    def add(p: str) -> None:
        t = os.path.abspath(os.path.expanduser((p or "").strip()))
        if t and t not in seen and os.path.isdir(t):
            seen.add(t)
            dirs.append(t)

    sw = (rec.get("submit_workdir") or "").strip()
    if sw:
        add(sw)
    return dirs


def _planned_vasp_from_io_flow_disk(rec: Dict[str, Any]) -> List[str]:
    """
    Planned VASP job UUIDs from ``io_flow.json`` on disk.

    IOMaker uses a **dynamic** second job for VASP; Mongo ``flows.jobs`` may only list the
    MLIP job until the first step finishes. The serialized ``io_flow.json`` still lists all
    top-level job UUIDs, so it is the reliable source for ``planned_vasp``.
    """
    best: List[str] = []
    for d in _submit_dir_candidates(rec):
        fp = Path(d) / "io_flow.json"
        if not fp.is_file():
            continue
        try:
            fd = json.loads(fp.read_text(encoding="utf-8"))
            plan = extract_interoptimus_flow_plan(fd)
            cand = list(plan.get("vasp_job_uuids") or [])
            if len(cand) > len(best):
                best = cand
        except Exception:
            continue
    return best


def _planned_vasp_from_flow_level_uuids(
    flow_level_uuids: List[str], mlip_ref: str
) -> List[str]:
    """Follow-up job UUIDs (VASP chain) given flow-level ``jobs`` and the MLIP job UUID."""
    if not flow_level_uuids:
        return []
    ml = (mlip_ref or "").strip().lower()
    if not ml:
        return list(flow_level_uuids[1:]) if len(flow_level_uuids) > 1 else []
    for i, u in enumerate(flow_level_uuids):
        if u.lower() == ml:
            return list(flow_level_uuids[i + 1 :])
    # *ml* not in list (stale Mongo vs regenerated io_flow): assume submit order [MLIP, …VASP]
    return list(flow_level_uuids[1:]) if len(flow_level_uuids) > 1 else []


def _registry_match_by_job_identifiers(reg: Dict[str, Any], needle: str) -> Optional[Dict[str, Any]]:
    """Match *needle* against ``jf_job_id`` / ``mlip_job_uuid`` / ``flow_uuid`` in registry values."""
    n = (needle or "").strip().lower()
    if not n:
        return None
    for _serial_key, rec in reg.items():
        if not isinstance(rec, dict):
            continue
        for fld in ("jf_job_id", "mlip_job_uuid", "flow_uuid"):
            v = rec.get(fld)
            if isinstance(v, str) and v.strip().lower() == n:
                return dict(rec)
    return None


def resolve_interoptimus_task_record(serial_id: str) -> Optional[Dict[str, Any]]:
    """
    Look up task metadata by **job / flow UUID**, or by registry key, or by path to
    ``io_interoptimus_task.json`` (loads that file or its parent directory when *serial_id*
    is a directory containing the file).

    Prefer passing a **UUID** from ``jf`` / ``result[\"mlip_job_uuid\"]``; paths are optional.

    After a successful server submit, the same metadata is also written under
    ``~/.interoptimus/iomaker_job_meta/`` with ``iomaker_job_id_aliases.json`` mapping
    MLIP / flow / submit / planned VASP UUIDs to that file, so VASP-inclusive flows stay
    recognizable even when the main registry row is not matched.
    """
    s = (serial_id or "").strip()
    if not s:
        return None
    reg = _load_task_registry()
    if s in reg:
        return dict(reg[s])
    reg_hit = _registry_match_by_job_identifiers(reg, s)
    if reg_hit:
        return reg_hit
    idx_hit = _resolve_interoptimus_task_record_from_index(s)
    if idx_hit:
        return idx_hit
    if _FLOW_UUID_RE.match(s):
        return {
            "jf_job_id": s,
            "do_vasp": None,
            "serial_id": None,
            "submit_workdir": None,
            "task_name": None,
        }
    disk = _load_task_record_from_disk(s)
    if disk:
        return disk
    try:
        p = Path(s).expanduser().resolve()
    except Exception:
        p = Path(s).expanduser()
    if p.is_file() and p.name == _TASK_JSON_NAME:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _resolve_job_uuid_from_ref(ref: str) -> Optional[str]:
    """
    ``submit_flow`` / stdout parsers may return a **numeric** id (Mongo ``db_id``, Slurm pid, etc.).
    Flow queries and ``get_job_doc(job_id=...)`` need the real **job UUID** string.
    """
    r = str(ref).strip()
    if not r:
        return None
    if _FLOW_UUID_RE.match(r):
        return r
    try:
        from jobflow_remote.cli.utils import get_job_controller

        jc = get_job_controller()
        ji = jc.get_job_info(db_id=r)
        if ji is None:
            ji = jc.get_job_info(job_id=r)
        if ji is None and r.isdigit():
            try:
                ji = jc.get_job_info_by_pid(int(r))
            except Exception:
                ji = None
        if ji is None and r.isdigit():
            try:
                ji = jc.get_job_info_by_pid(str(r))
            except Exception:
                ji = None
        if ji is None:
            return None
        u = getattr(ji, "uuid", None)
        return str(u) if u else None
    except Exception:
        return None


def _remote_queue_dict_from_job_doc(job_ref: str) -> Tuple[str, str]:
    """``(queue_out, queue_err)`` from Mongo ``remote`` on the job doc, if available."""
    uid = _resolve_job_uuid_from_ref(job_ref)
    if not uid:
        return "", ""
    try:
        from jobflow_remote.cli.utils import get_job_controller

        jc = get_job_controller()
        doc = jc.get_job_doc(job_id=uid)
        if doc is None:
            return "", ""
        remote = getattr(doc, "remote", None)
        if remote is None:
            return "", ""
        if hasattr(remote, "model_dump"):
            r = remote.model_dump()
        elif hasattr(remote, "dict"):
            r = remote.dict()
        elif isinstance(remote, dict):
            r = remote
        else:
            r = dict(remote) if remote else {}
        qo = r.get("queue_out") or ""
        qe = r.get("queue_err") or ""
        return str(qo), str(qe)
    except Exception:
        return "", ""


def _job_doc_queue_out_head(job_ref: str, max_lines: int = 100) -> str:
    """First *max_lines* of ``remote.queue_out`` from jobflow-remote MongoDB (if present)."""
    qo, _ = _remote_queue_dict_from_job_doc(job_ref)
    lines = str(qo).splitlines()
    return "\n".join(lines[:max_lines])


def _job_doc_queue_combined_head(job_ref: str, max_lines: int = 100) -> str:
    """
    ``remote.queue_out`` and ``remote.queue_err`` from Mongo (truncated).

    jobflow-remote only fills these after it downloads ``queue.out`` / ``queue.err`` into
    the controller's local data path; they may stay empty even when the job finished.
    """
    qo, qe = _remote_queue_dict_from_job_doc(job_ref)
    parts: List[str] = []
    if str(qo).strip():
        parts.append("\n".join(str(qo).splitlines()[:max_lines]))
    if str(qe).strip():
        err_cap = max(8, min(40, max_lines // 2))
        parts.append("\n".join(str(qe).splitlines()[:err_cap]))
    return "\n\n".join(parts).strip()


def _queue_files_tail_from_run_dir(
    run_dir: Optional[str],
    *,
    max_lines: int = 100,
    max_chars: int = 16000,
) -> Tuple[str, List[str]]:
    """
    Read ``queue.out`` / ``queue.err`` from *run_dir* if this host can see that path.

    Returns ``(text, filenames_read)``. Used when Mongo ``remote.queue_*`` is empty but
    the login node shares a filesystem with the worker (common on clusters).
    """
    if not run_dir:
        return "", []
    rd = os.path.abspath(run_dir)
    if not os.path.isdir(rd):
        return "", []
    chunks: List[str] = []
    read_names: List[str] = []
    for name in ("queue.out", "queue.err"):
        p = os.path.join(rd, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                body = f.read()
            if len(body) > max_chars:
                body = body[:max_chars] + "\n... [truncated]\n"
            lines = body.splitlines()
            tail = lines[-max_lines:] if len(lines) > max_lines else lines
            chunks.append(f"--- {name} ({rd}) ---\n" + "\n".join(tail))
            read_names.append(name)
        except OSError:
            continue
    return "\n\n".join(chunks).strip(), read_names


def _python_output_note_when_no_remote_log(
    job_state: Optional[str],
    run_dir: Optional[str],
) -> str:
    """Explain why ``python_output_head`` may be empty and what to do next."""
    st = _norm_job_state_str(job_state or "")
    terminal_fail = st in {"FAILED", "ERROR", "REMOTE_ERROR"}
    terminal_ok = st in {"COMPLETED", "COMPLETE", "DONE", "FINISHED"}
    pending = st in {
        "READY",
        "WAITING",
        "QUEUED",
        "LOCKED",
        "CHECKED_OUT",
        "UPLOADING",
        "DOWNLOADING",
    } or "QUEUE" in st or "PENDING" in st

    if run_dir:
        rd_hint = (
            f"本机若可访问 worker 的 run_dir，可直接打开：\n  {run_dir}/queue.out\n  {run_dir}/queue.err\n"
        )
    else:
        rd_hint = "当前尚无 run_dir；请稍后再查或运行 `jf job info <UUID>`。\n"

    if terminal_fail or terminal_ok:
        return (
            "Mongo 中 remote.queue_out / queue_err 仍为空（jobflow-remote 可能尚未把 worker 上的 "
            "queue.out 同步到数据库，或仅保存在控制端本地缓存目录）。"
            + rd_hint
            + "也可用终端：`jf job info <UUID>` 查看状态与路径。"
        )
    if pending or st in {"RUNNING", "REMOTE_RUNNING"} or "RUNNING" in st:
        return (
            "作业在队列或运行中时，remote.queue_out 常为空；worker 开始执行并回写后才会出现。"
            + rd_hint
            + "可隔一段时间再调用本函数，或增大轮询间隔。"
        )
    return (
        "尚无 queue_out（作业可能尚未开始、Mongo 未同步、或本机无法访问 worker 的 run_dir）。"
        + rd_hint
    )


def _job_info_to_dict(ji: Any) -> Dict[str, Any]:
    st = getattr(ji.state, "name", None) or str(getattr(ji, "state", ""))
    out = {
        "name": getattr(ji, "name", None),
        "state": st,
        "uuid": str(getattr(ji, "uuid", "") or ""),
        "index": getattr(ji, "index", None),
        "worker": str(getattr(ji, "worker", "") or ""),
        "run_dir": getattr(ji, "run_dir", None),
    }
    for attr in ("start_time", "end_time", "updated_on"):
        v = getattr(ji, attr, None)
        if v is not None:
            try:
                out[attr] = v.isoformat() if hasattr(v, "isoformat") else str(v)
            except Exception:
                out[attr] = str(v)
    return out


def _flow_jobs_for_root_job(
    job_uuid: str,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """
    Return ``(flow_uuid, job_infos_as_dicts, flow_doc_level_job_uuids)``.

    *flow_doc_level_job_uuids* comes from the flows collection ``jobs`` list. While the MLIP
    root job is still RUNNING, ``get_jobs_info`` often returns only that one row; the flow
    document still lists every job UUID submitted with the Flow (including planned VASP).
    """
    try:
        from jobflow_remote.cli.utils import initialize_config_manager, get_job_controller

        initialize_config_manager()
        jc = get_job_controller()
        ju = job_uuid.strip()
        flow_doc_raw = jc.get_flow_info_by_job_uuid(ju)
        if not flow_doc_raw and hasattr(jc, "flows"):
            try:
                coll = jc.flows
                flow_doc_raw = coll.find_one({"jobs": ju})
                if not flow_doc_raw:
                    flow_doc_raw = coll.find_one({"ids": {"$elemMatch": {"1": ju}}})
            except Exception:
                flow_doc_raw = None
        if not flow_doc_raw:
            return None, [], []
        d = _coerce_flow_doc_mapping(flow_doc_raw)
        flow_uuid = d.get("uuid")
        if not flow_uuid:
            flow_uuid = getattr(flow_doc_raw, "uuid", None) or getattr(
                flow_doc_raw, "flow_id", None
            )
        fu_str = str(flow_uuid).strip() if flow_uuid else ""
        flow_level = _job_uuid_list_from_mongo_flow(flow_doc_raw)
        if not fu_str:
            return None, [], flow_level
        jobs = jc.get_jobs_info(flow_ids=fu_str, limit=64) or []
        rows = [_job_info_to_dict(j) for j in jobs]
        rows.sort(key=lambda r: (r.get("index") is None, r.get("index") or 0))
        return fu_str, rows, flow_level
    except Exception:
        return None, [], []


def _flow_jobs_for_flow_uuid(
    flow_uuid: str,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """Return ``(flow_uuid, job_infos_as_dicts, flow_doc_level_job_uuids)`` for a known Flow UUID."""
    fu = str(flow_uuid or "").strip()
    if not fu:
        return None, [], []
    try:
        from jobflow_remote.cli.utils import initialize_config_manager, get_job_controller

        initialize_config_manager()
        jc = get_job_controller()
        flow_doc_raw = None
        if hasattr(jc, "flows"):
            try:
                flow_doc_raw = jc.flows.find_one({"uuid": fu})
            except Exception:
                flow_doc_raw = None
        flow_level = _job_uuid_list_from_mongo_flow(flow_doc_raw) if flow_doc_raw else []
        jobs = jc.get_jobs_info(flow_ids=fu, limit=256) or []
        rows = [_job_info_to_dict(j) for j in jobs]
        rows.sort(key=lambda r: (r.get("index") is None, r.get("index") or 0))
        return fu, rows, flow_level
    except Exception:
        return None, [], []


def _norm_job_state_str(st: Optional[str]) -> str:
    x = (st or "").strip().upper()
    if "." in x:
        x = x.split(".")[-1]
    return x


def _estimate_eta_note(
    flow_jobs: List[Dict[str, Any]],
    *,
    now: datetime,
) -> Tuple[Optional[float], str]:
    """Rough ETA (seconds) and a short note; best-effort only."""
    terminal = {
        "COMPLETED",
        "COMPLETE",
        "DONE",
        "FINISHED",
        "FAILED",
        "ERROR",
        "REMOTE_ERROR",
        "CANCELLED",
        "CANCELED",
        "REMOVED",
    }
    if not flow_jobs:
        return None, "尚无子作业信息（Flow 可能尚未写入数据库）。"
    states_norm = [_norm_job_state_str(str(j.get("state") or "")) for j in flow_jobs]
    if all(s in terminal for s in states_norm):
        return 0.0, "Flow 内全部步骤已结束。"
    running_idx = [i for i, s in enumerate(states_norm) if s not in terminal]
    if len(flow_jobs) >= 2 and len(running_idx) == 1:
        eta_sec: Optional[float] = None
        try:
            if states_norm[0] in terminal and states_norm[1] not in terminal:
                t_end0 = flow_jobs[0].get("end_time")
                t_start0 = flow_jobs[0].get("start_time")
                if t_end0 and t_start0:
                    e0 = datetime.fromisoformat(str(t_end0).replace("Z", "+00:00"))
                    s0 = datetime.fromisoformat(str(t_start0).replace("Z", "+00:00"))
                    if e0.tzinfo is None:
                        e0 = e0.replace(tzinfo=timezone.utc)
                    if s0.tzinfo is None:
                        s0 = s0.replace(tzinfo=timezone.utc)
                    mlip_sec = max(0.0, (e0 - s0).total_seconds())
                    eta_sec = mlip_sec * 1.5
        except Exception:
            eta_sec = None
        note = (
            "VASP 相关步骤进行中；下方为 Flow 内各作业状态与时间。"
            + (
                f" 粗略剩余时间估计：约 {int(eta_sec)} 秒（启发式：1.5× 已完成 MLIP 段 wall time）。"
                if eta_sec is not None
                else " 剩余时间受体系与队列影响，请结合各作业时间戳判断。"
            )
        )
        return eta_sec, note
    if running_idx:
        return (
            None,
            "作业进行中；剩余时间取决于队列与计算量，无法给出精确 ETA。",
        )
    return None, ""


def query_interoptimus_task_progress(
    serial_id: str,
    *,
    jf_bin: str = "jf",
    python_head_lines: int = 100,
    task_record_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query progress for a task registered by :func:`register_interoptimus_server_task`.

    **MLIP-only** (``do_vasp`` false): reports MLIP phase and the first *python_head_lines*
    of ``remote.queue_out`` (worker stdout captured by jobflow-remote), if available.

    **With VASP**: lists all jobs in the same Flow with states and a short ETA note.

    MLIP status uses the **MLIP root job UUID** stored at submit time (``mlip_job_uuid`` in
    ``io_interoptimus_task.json``). VASP follow-up jobs are listed in ``vasp_sub_jobs`` with
    per-job state and ``run_dir`` when the controller has registered them.

    Pass the **MLIP job UUID** (recommended), or any UUID that was registered at submit time
    (flow id, ``submit_flow`` return id, or a planned VASP job id): the local index under
    ``~/.interoptimus/iomaker_job_meta`` resolves them to the same record. Planned VASP UUIDs
    also come from ``io_flow.json`` on disk when ``submit_workdir`` is known, and from Mongo
    ``flows.ids`` / ``flows.jobs`` when available.

    *serial_id* should be a jobflow **UUID**; optional: absolute path to ``io_interoptimus_task.json``.
    """
    rec = _merge_interoptimus_task_record(
        resolve_interoptimus_task_record(serial_id),
        task_record_hint,
    )
    if not rec:
        return {
            "success": False,
            "error": "unknown_task_or_serial",
            "serial_id": serial_id,
            "hint": (
                "Could not resolve task: pass ``result['mlip_job_uuid']`` (or "
                "``server_submission['job_id']``), register ~/.interoptimus entry, or set "
                "INTEROPTIMUS_JOB_ID to the MLIP job UUID. Optional: path to "
                f"{_TASK_JSON_NAME}."
            ),
        }

    jf_id = rec.get("jf_job_id") or serial_id
    if not jf_id:
        return {"success": False, "error": "missing_or_invalid_jf_job_id", "record": rec}

    jf_ref = str(jf_id).strip()
    mlip_stored = (rec.get("mlip_job_uuid") or "").strip()
    mlip_track_ref = mlip_stored or jf_ref
    planned_vasp: List[str] = list(rec.get("vasp_job_uuids") or [])
    disk_pv = _planned_vasp_from_io_flow_disk(rec)
    if len(disk_pv) > len(planned_vasp):
        planned_vasp = disk_pv

    resolved_uuid = _resolve_job_uuid_from_ref(mlip_track_ref)

    do_vasp = rec.get("do_vasp")
    if isinstance(do_vasp, bool):
        pass
    else:
        do_vasp = None

    info = local_jf_job_info(mlip_track_ref, jf_bin=jf_bin)
    parsed = info.get("parsed") or {}
    state = parsed.get("state")
    run_dir = parsed.get("run_dir")
    if resolved_uuid is None:
        resolved_uuid = parsed.get("job_uuid") or _extract_first_uuid_from_text(
            _strip_rich_table_chars(
                _strip_ansi((info.get("stdout") or "") + "\n" + (info.get("stderr") or ""))
            )
        )

    mlip_queue_ref = str(resolved_uuid or mlip_track_ref).strip()

    flow_key = resolved_uuid or mlip_track_ref
    flow_uuid, flow_jobs, flow_level_job_uuids = _flow_jobs_for_root_job(flow_key)
    recorded_flow_uuid = str(rec.get("flow_uuid") or "").strip()
    if (not flow_uuid and not flow_jobs and not flow_level_job_uuids) and recorded_flow_uuid:
        flow_uuid, flow_jobs, flow_level_job_uuids = _flow_jobs_for_flow_uuid(recorded_flow_uuid)

    mongo_tail = _planned_vasp_from_flow_level_uuids(
        flow_level_job_uuids, mlip_queue_ref
    )
    # Only use flow-level UUID tail as a fallback before the Flow has materialized detailed jobs.
    # Once real flow jobs are available, keep submit-time planned roots separate from expanded jobs.
    if not flow_jobs and len(mongo_tail) > len(planned_vasp):
        planned_vasp = mongo_tail

    now = datetime.now(timezone.utc)
    out: Dict[str, Any] = {
        "success": True,
        "serial_id": rec.get("serial_id"),
        "submit_jf_job_id": jf_ref,
        "jf_job_id": jf_ref,
        "mlip_job_uuid": mlip_stored or resolved_uuid,
        "jf_job_uuid": resolved_uuid,
        "planned_vasp_job_uuids": planned_vasp,
        "flow_uuid_from_record": (rec.get("flow_uuid") or "").strip() or None,
        "task_name": rec.get("task_name"),
        "submit_workdir": rec.get("submit_workdir"),
        "parsed_source": info.get("parsed_source"),
        "job_state": state,
        "run_dir": run_dir,
    }
    if flow_level_job_uuids:
        out["flow_level_job_uuids"] = flow_level_job_uuids
    if mlip_stored and mlip_stored != jf_ref:
        out["tracking_note"] = (
            "job_state / run_dir / queue 日志对应 MLIP 根作业 (mlip_job_uuid)；"
            "submit_jf_job_id 为 submit_flow 返回值（可能与 Flow uuid 或首作业 uuid 不同）。"
        )
    if resolved_uuid is None:
        out["job_uuid_resolve_note"] = (
            "无法在数据库或 `jf job info` 输出中解析出 job UUID。"
            "请确认本机 `jf` 与 jobflow-remote 配置与提交时一致；或稍后重试。"
        )

    out["flow_uuid"] = flow_uuid
    vasp_tracking = _split_vasp_tracking(
        flow_jobs,
        mlip_uuid_key=mlip_queue_ref,
        planned_vasp_root_uuids=planned_vasp,
    )
    planned_root_rows = vasp_tracking["planned_roots"]
    expanded_vasp_rows = vasp_tracking["expanded_jobs"]
    planned_root_counts = _summarize_job_rows(planned_root_rows)
    expanded_vasp_counts = _summarize_job_rows(expanded_vasp_rows)
    out["planned_vasp_roots"] = planned_root_rows
    out["planned_vasp_root_counts"] = planned_root_counts
    out["vasp_sub_jobs"] = expanded_vasp_rows
    out["expanded_vasp_jobs"] = expanded_vasp_rows
    out["expanded_vasp_job_counts"] = expanded_vasp_counts
    out["vasp_job_counts"] = expanded_vasp_counts
    out["has_vasp"] = bool(planned_vasp or planned_root_rows or expanded_vasp_rows or len(flow_jobs) > 1)

    inferred_vasp = (
        bool(planned_vasp)
        or len(flow_jobs) > 1
        or len(flow_level_job_uuids) > 1
    )
    do_vasp_overridden = False
    if do_vasp is None:
        do_vasp = inferred_vasp
    elif do_vasp is False and inferred_vasp:
        do_vasp = True
        do_vasp_overridden = True

    if do_vasp_overridden:
        out["do_vasp_overridden"] = True
        out["do_vasp_note"] = (
            "任务记录里 do_vasp=False，但 io_flow.json 或 Flow 文档显示存在后续作业；"
            "已按 MLIP+VASP 展示（含 vasp_sub_jobs）。"
        )

    if not do_vasp:
        out["mode"] = "mlip_only"
        rec_do = rec.get("do_vasp")
        if isinstance(rec_do, bool) and rec_do is False:
            out["summary_line"] = "任务记录为 do_vasp=false，按仅 MLIP 展示；job_state 见上。"
            out["mode_note"] = (
                "若实际需要 VASP，请检查提交时的 settings.do_vasp 与注册是否一致。"
            )
        else:
            out["summary_line"] = (
                "未在元数据中解析到 VASP 后续作业；job_state 见上。"
                "（与 MLIP 是否已跑完无关，常见原因是查询 UUID 未命中含 vasp_job_uuids 的注册记录。）"
            )
            out["mode_note"] = (
                "mlip_only 仅表示「当前没有可用的 VASP 计划 UUID / 多作业信息」，不是断言你的 Flow 里一定没有 "
                "VASP。MLIP+VASP 时若只用裸 UUID且注册表对不上，或磁盘 io_flow 不可见，会出现本模式。"
                "优先在同一 session 用 iomaker_progress(result)；或确认 ~/.interoptimus 与 io_interoptimus_task.json 已写入。"
            )
        head = _job_doc_queue_combined_head(
            mlip_queue_ref, max_lines=python_head_lines
        )
        run_dir_files: List[str] = []
        if not head.strip() and run_dir:
            head, run_dir_files = _queue_files_tail_from_run_dir(
                run_dir, max_lines=python_head_lines
            )
        out["python_output_head"] = head
        if run_dir_files:
            out["python_output_source"] = "run_dir:" + ",".join(run_dir_files)
        elif str(head).strip():
            out["python_output_source"] = "mongo:remote.queue_out,queue_err"
        if not str(head).strip():
            out["python_output_note"] = _python_output_note_when_no_remote_log(state, run_dir)
        mlip_bucket = _job_state_bucket(state)
        out["current_phase"] = "done" if mlip_bucket in {"completed", "failed"} else "mlip"
        out["is_finished"] = bool(mlip_bucket in {"completed", "failed"})
        out["stage_summary"] = (
            "当前为 MLIP 阶段。"
            + (
                " 任务已结束。"
                if out["is_finished"]
                else f" MLIP 状态: {state or 'UNKNOWN'}。"
            )
        )
        return out

    out["mode"] = "mlip_and_vasp"
    out["vasp_flow_jobs"] = flow_jobs
    out["summary_line"] = (
        "MLIP 状态见 job_state；提交后先显示 planned_vasp_roots，展开后真实 VASP 子作业见 vasp_sub_jobs。"
    )
    if expanded_vasp_counts["total"] == 0 and planned_root_counts["total"] > 0:
        eta_sec, eta_note = (
            None,
            "VASP 动态入口已列入 Flow；真实 VASP 子任务会在 MLIP 完成后展开。",
        )
    else:
        eta_sec, eta_note = _estimate_eta_note(flow_jobs, now=now)
    out["eta_seconds_estimate"] = eta_sec
    out["eta_note"] = eta_note
    head = _job_doc_queue_combined_head(
        mlip_queue_ref, max_lines=python_head_lines
    )
    run_dir_files_v: List[str] = []
    if not head.strip() and run_dir:
        head, run_dir_files_v = _queue_files_tail_from_run_dir(
            run_dir, max_lines=python_head_lines
        )
    out["mlip_python_output_head"] = head
    if run_dir_files_v:
        out["mlip_python_output_source"] = "run_dir:" + ",".join(run_dir_files_v)
    elif str(head).strip():
        out["mlip_python_output_source"] = "mongo:remote.queue_out,queue_err"
    if not str(head).strip():
        out["mlip_python_output_note"] = _python_output_note_when_no_remote_log(state, run_dir)
    mlip_bucket = _job_state_bucket(state)
    if expanded_vasp_counts["total"] > 0 and expanded_vasp_counts["unfinished"] == 0:
        current_phase = "done"
    elif expanded_vasp_counts["total"] > 0:
        current_phase = "vasp"
    elif mlip_bucket not in {"completed", "failed"}:
        current_phase = "mlip"
    else:
        current_phase = "vasp"
    out["current_phase"] = current_phase
    out["is_finished"] = bool(current_phase == "done")
    phase_text = _phase_label_from_progress({"current_phase": current_phase})
    if current_phase == "done" and expanded_vasp_counts["failed"] > 0:
        phase_text = "全部结束（含失败）"
    if expanded_vasp_counts["total"] > 0:
        out["stage_summary"] = (
            f"当前为 {phase_text}。"
            f" 真实 VASP 子任务 {expanded_vasp_counts['finished']}/{expanded_vasp_counts['total']} 已结束；"
            f" completed={expanded_vasp_counts['completed']}, running={expanded_vasp_counts['running']},"
            f" pending={expanded_vasp_counts['pending']}, failed={expanded_vasp_counts['failed']}."
        )
    else:
        if current_phase == "mlip":
            out["stage_summary"] = (
                f"当前为 {phase_text}。"
                f" 已识别到 {planned_root_counts['total']} 个 VASP 动态入口，等待 MLIP 完成后展开真实子任务。"
            )
        else:
            out["stage_summary"] = (
                f"当前为 {phase_text}。"
                f" MLIP 已结束；已识别到 {planned_root_counts['total']} 个 VASP 动态入口，"
                "等待 jobflow-remote 展开真实 VASP 子任务。"
            )
    return out


def fetch_interoptimus_task_results(
    serial_id: str,
    *,
    jf_bin: str = "jf",
    copy_images_to: Optional[str] = None,
    write_report_to: Optional[str] = None,
    include_progress: bool = True,
    io_report_max_chars: Optional[int] = None,
    task_record_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    After :func:`query_interoptimus_task_progress`, load finished artifacts from ``run_dir``.

    Returns the structured energy list from ``io_remote_summary.json`` (all HKL / termination
    pairs), full ``io_report.txt`` and ``pairs_summary.txt`` text, and paths to:

    - ``project.jpg`` — lattice matching stereographic plot (HKL orientations);
    - ``stereographic.jpg`` — compact static stereographic summary plot;
    - ``stereographic_interactive.html`` — larger interactive stereographic plot with hover labels;
    - ``unique_matches.jpg`` — unique-match figure.

    *write_report_to* optionally saves :func:`collect_interoptimus_run_dir_results` ``report_markdown``
    (combined summary + tables + io_report) to a ``.md`` file.

    *copy_images_to*, if set, copies into that directory (created if missing):

    - the three JPEGs (``project.jpg``, ``stereographic.jpg``, ``unique_matches.jpg``);
    - ``opt_results.pkl`` (then materializes ``pairs_best_it/`` under the destination);
    - ``opt_results_summary.json``, ``area_strain``, and ``stereographic_interactive.html`` when present.

    Copied files are listed in ``copied_images`` (JPEGs) and ``copied_artifacts`` (files / materialized dir).
    """
    progress: Dict[str, Any] = {}
    if include_progress:
        progress = query_interoptimus_task_progress(
            serial_id,
            jf_bin=jf_bin,
            task_record_hint=task_record_hint,
        )

    run_dir = (progress.get("run_dir") or "").strip() or None
    if not run_dir:
        rec = _merge_interoptimus_task_record(
            resolve_interoptimus_task_record(serial_id),
            task_record_hint,
        )
        if rec:
            jf_ref = str(rec.get("jf_job_id") or serial_id).strip()
            info = local_jf_job_info(jf_ref, jf_bin=jf_bin)
            run_dir = (info.get("parsed") or {}).get("run_dir")
    if not run_dir:
        info2 = local_jf_job_info(str(serial_id).strip(), jf_bin=jf_bin)
        run_dir = (info2.get("parsed") or {}).get("run_dir")

    collected = collect_interoptimus_run_dir_results(run_dir, io_report_max_chars=io_report_max_chars)
    collected["query_progress_included"] = include_progress
    if include_progress:
        collected["progress"] = progress

    if write_report_to and collected.get("report_markdown"):
        wp = os.path.abspath(write_report_to)
        parent = os.path.dirname(wp)
        if parent:
            os.makedirs(parent, exist_ok=True)
        Path(wp).write_text(collected["report_markdown"], encoding="utf-8")
        collected["report_markdown_path"] = wp

    if copy_images_to and collected.get("success") and collected.get("run_dir"):
        dest_root = os.path.abspath(copy_images_to)
        os.makedirs(dest_root, exist_ok=True)
        rd = str(collected["run_dir"])
        copied_img: List[Dict[str, str]] = []
        for fname in ("project.jpg", "stereographic.jpg", "unique_matches.jpg"):
            src = os.path.join(rd, fname)
            if os.path.isfile(src):
                dst = os.path.join(dest_root, fname)
                shutil.copy2(src, dst)
                copied_img.append({"from": src, "to": dst, "kind": "image"})
        collected["copied_images"] = copied_img

        copied_art: List[Dict[str, str]] = []
        src_pkl = os.path.join(rd, "opt_results.pkl")
        if os.path.isfile(src_pkl):
            dst_pkl = os.path.join(dest_root, "opt_results.pkl")
            shutil.copy2(src_pkl, dst_pkl)
            copied_art.append({"from": src_pkl, "to": dst_pkl, "kind": "file"})
            try:
                from InterOptimus.jobflow import (
                    load_opt_results_pickle_payload,
                    materialize_pairs_best_it_dir,
                )

                payload = load_opt_results_pickle_payload(dst_pkl)
                dst_pbi = os.path.join(dest_root, "pairs_best_it")
                materialize_pairs_best_it_dir(
                    payload["opt_results"],
                    list(payload["materialize_pairs"]),
                    dst_pbi,
                    double_interface=bool(payload["double_interface"]),
                    strain_E_correction=bool(payload["strain_E_correction"]),
                )
                copied_art.append(
                    {"from": src_pkl, "to": dst_pbi, "kind": "materialized_pairs_best_it"}
                )
            except Exception as e:
                collected["pairs_materialize_error"] = str(e)
        else:
            pbi = os.path.join(rd, "pairs_best_it")
            if os.path.isdir(pbi):
                dst_pbi = os.path.join(dest_root, "pairs_best_it")
                try:
                    shutil.copytree(pbi, dst_pbi, dirs_exist_ok=True)
                except TypeError:
                    if os.path.isdir(dst_pbi):
                        shutil.rmtree(dst_pbi)
                    shutil.copytree(pbi, dst_pbi)
                copied_art.append({"from": pbi, "to": dst_pbi, "kind": "dir"})
        for fname in ("opt_results_summary.json", "area_strain", "stereographic_interactive.html"):
            src = os.path.join(rd, fname)
            if os.path.isfile(src):
                dst = os.path.join(dest_root, fname)
                shutil.copy2(src, dst)
                copied_art.append({"from": src, "to": dst, "kind": "file"})
        if copied_art:
            collected["copied_artifacts"] = copied_art

    return collected


if __name__ == "__main__":
    import sys

    print(
        "This module is a library. Use submit_io_flow_locally(...) from Python, "
        "or build a Flow JSON with InterOptimus.agents.simple_iomaker / interoptimus-simple.",
        file=sys.stderr,
    )
