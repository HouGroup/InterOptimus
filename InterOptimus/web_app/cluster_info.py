"""
Cluster (Slurm) introspection helpers for the web UI.

Pure stdlib: shells out to ``sinfo`` (and optionally ``squeue``) and parses the
text output. Cached briefly so the manage page can poll without hammering the
scheduler. Falls back gracefully when no Slurm client is on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_CACHE_TTL_SECONDS = 25.0


def _which_sinfo() -> Optional[str]:
    explicit = (os.environ.get("INTEROPTIMUS_SINFO_BIN") or "").strip()
    if explicit:
        return explicit
    return shutil.which("sinfo")


def _which_squeue() -> Optional[str]:
    explicit = (os.environ.get("INTEROPTIMUS_SQUEUE_BIN") or "").strip()
    if explicit:
        return explicit
    return shutil.which("squeue")


def _run(cmd: List[str], timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (FileNotFoundError, PermissionError) as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout: {e}"
    except OSError as e:
        return 1, "", str(e)


def _parse_partitions(out: str) -> List[Dict[str, Any]]:
    """
    Parse ``sinfo -h -o "%P|%a|%l|%D|%T|%C|%G"``:
        partition | avail | timelimit | nodes | state | cpus(A/I/O/T) | gres
    Aggregate per partition + state.
    """
    partitions: Dict[str, Dict[str, Any]] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        pname = parts[0].strip().rstrip("*") or "?"
        avail = parts[1].strip()
        timelimit = parts[2].strip()
        try:
            nodes = int(parts[3].strip())
        except ValueError:
            nodes = 0
        state = parts[4].strip()
        cpus_field = parts[5].strip()
        gres = parts[6].strip() or "(null)"

        cpu_alloc = cpu_idle = cpu_other = cpu_total = 0
        cpu_parts = cpus_field.split("/")
        if len(cpu_parts) == 4:
            try:
                cpu_alloc = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_other = int(cpu_parts[2])
                cpu_total = int(cpu_parts[3])
            except ValueError:
                pass

        is_default = parts[0].endswith("*")
        bucket = partitions.setdefault(
            pname,
            {
                "partition": pname,
                "is_default": is_default,
                "avail": avail,
                "timelimit": timelimit,
                "states": [],
                "nodes_total": 0,
                "cpus_alloc": 0,
                "cpus_idle": 0,
                "cpus_other": 0,
                "cpus_total": 0,
                "gres_set": set(),
            },
        )
        bucket["is_default"] = bucket["is_default"] or is_default
        bucket["states"].append({"state": state, "nodes": nodes, "cpus_idle": cpu_idle, "cpus_total": cpu_total, "gres": gres})
        bucket["nodes_total"] += nodes
        bucket["cpus_alloc"] += cpu_alloc
        bucket["cpus_idle"] += cpu_idle
        bucket["cpus_other"] += cpu_other
        bucket["cpus_total"] += cpu_total
        if gres and gres != "(null)":
            bucket["gres_set"].add(gres)
    out_list: List[Dict[str, Any]] = []
    for name in sorted(partitions):
        p = partitions[name]
        p["gres"] = sorted(p.pop("gres_set"))
        out_list.append(p)
    return out_list


def _parse_node_summary(out: str) -> List[Dict[str, Any]]:
    """
    Parse ``sinfo -N -h -o "%N|%P|%T|%C|%m|%G"``:
        nodelist | partition | state | cpus A/I/O/T | mem | gres
    """
    rows: List[Dict[str, Any]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        cpus_field = parts[3].strip()
        cpu_alloc = cpu_idle = cpu_other = cpu_total = 0
        cp = cpus_field.split("/")
        if len(cp) == 4:
            try:
                cpu_alloc, cpu_idle, cpu_other, cpu_total = (int(x) for x in cp)
            except ValueError:
                pass
        try:
            mem_mb = int(parts[4].strip())
        except ValueError:
            mem_mb = 0
        rows.append(
            {
                "node": parts[0].strip(),
                "partition": parts[1].strip().rstrip("*"),
                "state": parts[2].strip(),
                "cpu_alloc": cpu_alloc,
                "cpu_idle": cpu_idle,
                "cpu_other": cpu_other,
                "cpu_total": cpu_total,
                "mem_mb": mem_mb,
                "gres": parts[5].strip() or "(null)",
            }
        )
    return rows


def _parse_queue_summary(out: str) -> Dict[str, Any]:
    """
    Parse ``squeue -h -o "%T"``: counts by job state.
    """
    counts: Dict[str, int] = {}
    total = 0
    for raw in out.splitlines():
        s = raw.strip()
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
        total += 1
    return {"total": total, "by_state": counts}


def query_cluster_info(force: bool = False) -> Dict[str, Any]:
    """Cached ``sinfo`` snapshot used by the management UI."""
    now = time.time()
    if not force:
        with _CACHE_LOCK:
            cached = _CACHE.get("data")
            if cached and (now - float(_CACHE.get("ts") or 0)) < _CACHE_TTL_SECONDS:
                return {**cached, "cached": True, "age_seconds": round(now - _CACHE["ts"], 2)}

    sinfo = _which_sinfo()
    payload: Dict[str, Any] = {
        "available": False,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "sinfo_path": sinfo,
        "partitions": [],
        "nodes": [],
        "queue": None,
        "errors": [],
        "default_partition": (os.environ.get("INTEROPTIMUS_SLURM_PARTITION") or "").strip() or None,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cached": False,
        "age_seconds": 0.0,
    }
    if not sinfo:
        payload["errors"].append(
            "sinfo 不在 PATH（请确认服务运行在装有 Slurm 客户端的登录节点；可用 INTEROPTIMUS_SINFO_BIN 显式指定）。"
        )
        with _CACHE_LOCK:
            _CACHE["ts"] = now
            _CACHE["data"] = payload
        return payload

    rc1, out1, err1 = _run([sinfo, "-h", "-o", "%P|%a|%l|%D|%T|%C|%G"])
    if rc1 == 0 and out1:
        payload["partitions"] = _parse_partitions(out1)
        payload["available"] = True
        if not payload["default_partition"]:
            for p in payload["partitions"]:
                if p.get("is_default"):
                    payload["default_partition"] = p["partition"]
                    break
    else:
        payload["errors"].append(f"sinfo partition query failed (rc={rc1}): {err1.strip() or 'no output'}")

    rc2, out2, err2 = _run([sinfo, "-N", "-h", "-o", "%N|%P|%T|%C|%m|%G"])
    if rc2 == 0 and out2:
        payload["nodes"] = _parse_node_summary(out2)
    else:
        payload["errors"].append(f"sinfo node query failed (rc={rc2}): {err2.strip() or 'no output'}")

    squeue = _which_squeue()
    if squeue:
        rc3, out3, _err3 = _run([squeue, "-h", "-o", "%T"], timeout=6.0)
        if rc3 == 0:
            payload["queue"] = _parse_queue_summary(out3)

    with _CACHE_LOCK:
        _CACHE["ts"] = now
        _CACHE["data"] = payload
    return payload
