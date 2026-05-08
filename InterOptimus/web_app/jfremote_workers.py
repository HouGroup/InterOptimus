"""
jobflow-remote worker introspection helpers for the management UI.

Reads ``~/.jfremote/<project>.yaml`` to list workers and their settings
(``pre_run``, ``work_dir``, ``scheduler_type``, ``type``).  Provides a
``module avail`` parser so the form can populate VASP module-load options
from a dropdown, and an ``MlipCalc`` probe that runs a small static-energy
calculation inside a worker's ``pre_run`` environment so users only have
to specify the MLIP backend and the system finds (and validates) a
compatible worker for them.

All helpers are intentionally defensive / pure stdlib + ``yaml``: any
missing file, parse error or shell timeout returns an informative payload
so the web UI can render a useful message.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml as _yaml
except Exception:  # pragma: no cover - PyYAML is a hard runtime dep elsewhere
    _yaml = None


JFREMOTE_DIR_DEFAULT = Path.home() / ".jfremote"

# Mapping from MLIP backend name (mlip_calc) to:
#   (default_worker_name, default_conda_env, importable_module, pip_install_hint)
# Worker / env names follow the convention from deploy_jobflow_stack.py.
MLIP_TO_WORKER_HINTS: Dict[str, Dict[str, str]] = {
    "orb-models": {
        "worker": "orb",
        "env": "orb",
        "module": "orb_models",
        "pip": "orb-models",
    },
    "sevenn": {
        "worker": "sevenn",
        "env": "sevenn",
        "module": "sevenn",
        "pip": "sevenn",
    },
    "matris": {
        "worker": "matris",
        "env": "matris",
        "module": "matris",
        "pip": "matris (local zip / git)",
    },
    "dpa": {
        "worker": "dpa",
        "env": "dpa",
        "module": "deepmd",
        "pip": "deepmd-kit",
    },
}


def _normalize_probe_device(raw: Optional[str]) -> str:
    """Map form/API values to :class:`MlipCalc` device strings (``cpu`` | ``cuda``)."""
    d = (raw or "cpu").strip().lower()
    if d in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def _probe_slurm_walltime(seconds: int) -> str:
    """Format Slurm ``--time`` as HH:MM:SS (floor at 1 minute, cap at 24h)."""
    sec = max(60, min(int(seconds), 24 * 3600))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _probe_srun_cpus_per_task() -> int:
    """``srun --cpus-per-task`` for Si probe; override with ``INTEROPTIMUS_PROBE_SRUN_CPUS_PER_TASK``."""
    raw = (os.environ.get("INTEROPTIMUS_PROBE_SRUN_CPUS_PER_TASK") or "").strip()
    if raw.isdigit():
        return max(1, min(int(raw), 128))
    # PyTorch CPU + ORB often needs more than 2 logical CPUs worth of memory on shared Slurm sites.
    return 4


def _probe_thread_cap_shell() -> str:
    """Cap BLAS/OpenMP threads so a 1-task probe does not oversubscribe tiny allocations."""
    return (
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
    )


def _probe_remote_scratch_dir() -> Path:
    """Writable under home so batch nodes sharing NFS see probe scripts."""
    d = Path.home() / ".interoptimus_probe_scratch"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - fall back
        return Path.home()
    return d


def _jfremote_dir() -> Path:
    """Return the active ``~/.jfremote`` directory (env override wins)."""
    env = (os.environ.get("JFREMOTE_PROJECTS_FOLDER") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return JFREMOTE_DIR_DEFAULT.expanduser().resolve()


def _default_project_name() -> str:
    """Default jfremote project name (env override wins)."""
    return (os.environ.get("JFREMOTE_PROJECT") or "std").strip() or "std"


def _project_path(project: Optional[str] = None) -> Path:
    """Resolve ``~/.jfremote/<project>.yaml`` for the active project."""
    name = (project or _default_project_name()).strip() or "std"
    return _jfremote_dir() / f"{name}.yaml"


def list_jfremote_projects() -> List[str]:
    """Names of all ``*.yaml`` files in the active jfremote directory."""
    base = _jfremote_dir()
    if not base.is_dir():
        return []
    out: List[str] = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
            out.append(p.stem)
    return out


def _safe_load_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if _yaml is None:
        return None, "PyYAML not installed (pip install pyyaml)"
    if not path.is_file():
        return None, f"project file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
    except (OSError, _yaml.YAMLError) as e:
        return None, f"failed to read {path}: {e}"
    if not isinstance(data, dict):
        return None, f"unexpected top-level type in {path}: {type(data).__name__}"
    return data, None


def list_workers(project: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a payload describing the workers configured for ``project``.

    ``payload["workers"]`` is a list of dicts with at least ``name``,
    ``type``, ``scheduler_type``, ``work_dir``, ``pre_run`` (multiline
    string).  Empty ``workers`` + non-empty ``errors`` means the YAML
    could not be parsed; the UI should show ``errors``.
    """
    proj_name = (project or _default_project_name()).strip() or _default_project_name()
    path = _project_path(proj_name)
    payload: Dict[str, Any] = {
        "project": proj_name,
        "project_path": str(path),
        "available": False,
        "workers": [],
        "errors": [],
        "projects": list_jfremote_projects(),
    }
    data, err = _safe_load_yaml(path)
    if err is not None:
        payload["errors"].append(err)
        return payload

    workers = (data or {}).get("workers")
    if not isinstance(workers, dict):
        payload["errors"].append(
            f"'workers' missing or not a mapping in {path}; "
            "configure jobflow-remote first (see deploy_jobflow_stack.py)"
        )
        return payload

    items: List[Dict[str, Any]] = []
    for name in sorted(workers.keys()):
        w = workers.get(name) or {}
        if not isinstance(w, dict):
            continue
        pre_run = str(w.get("pre_run") or "")
        items.append(
            {
                "name": str(name),
                "type": str(w.get("type") or ""),
                "scheduler_type": str(w.get("scheduler_type") or ""),
                "work_dir": str(w.get("work_dir") or ""),
                "pre_run": pre_run,
                # convenience: what conda env (if any) the pre_run activates
                "conda_env": _detect_conda_env_in_pre_run(pre_run),
                # known MLIP backend hint (if worker name matches one of the standard MLIPs)
                "mlip_hint": _worker_mlip_hint(str(name), pre_run),
            }
        )
    payload["available"] = True
    payload["workers"] = items
    return payload


def _detect_conda_env_in_pre_run(pre_run: str) -> str:
    """Best-effort: extract the env name from a ``conda activate <env>`` line."""
    for raw in (pre_run or "").splitlines():
        line = raw.strip()
        # match "conda activate ENV" possibly preceded by `&&`
        if "conda activate" in line:
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            for i, tok in enumerate(tokens):
                if tok == "activate" and i + 1 < len(tokens):
                    return tokens[i + 1]
    return ""


def _worker_mlip_hint(name: str, pre_run: str) -> str:
    """
    Reverse-lookup: which MLIP backend (mlip_calc) does this worker most
    likely correspond to? Returns "" if not obviously an MLIP worker.
    """
    n = (name or "").strip().lower()
    env_raw = _detect_conda_env_in_pre_run(pre_run).strip()
    # accept both "orb" and "/home/.../envs/orb"
    env_basename = os.path.basename(env_raw).strip().lower() if env_raw else ""
    env = env_raw.lower()
    for calc, info in MLIP_TO_WORKER_HINTS.items():
        target = info["env"].lower()
        if n == info["worker"].lower() or env == target or env_basename == target:
            return calc
    return ""


# ---------------------------------------------------------------------------
# `module avail VASP` discovery
# ---------------------------------------------------------------------------


def _modulecmd_path() -> Optional[str]:
    """Locate the ``module`` shell function loader (``modulecmd``)."""
    explicit = (os.environ.get("INTEROPTIMUS_MODULECMD") or "").strip()
    if explicit:
        return explicit
    return shutil.which("modulecmd") or shutil.which("lmod")


def _run_shell(
    script: str, *, timeout: int = 30, env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """Run a shell snippet through ``bash -lc`` with a timeout."""
    try:
        proc = subprocess.run(
            ["bash", "-lc", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if hasattr(e, "stdout") else ""
        err = (e.stderr or b"").decode("utf-8", "replace") if hasattr(e, "stderr") else ""
        return 124, out, (err + f"\n[timeout after {timeout}s]")
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return proc.returncode, out, err


def _parse_module_avail(text: str, pattern: str) -> List[str]:
    """
    Extract module specifiers (``foo/bar``, ``VASP/6.4.2``) from ``module avail``
    output; both Lmod and TCL Modules write to stderr in two-column layout.
    """
    pat = (pattern or "").strip().lower()
    out: List[str] = []
    seen: set = set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip section headers like "------- /apps/modules -------"
        if line.startswith("--") or line.startswith("=="):
            continue
        # split into whitespace-delimited tokens; modules may span multiple columns
        for tok in line.split():
            t = tok.strip().rstrip("(L)").rstrip("(D)").rstrip("(S)").rstrip(":")
            if not t:
                continue
            # heuristic: a module spec contains at least one alpha char and may have / and version
            tl = t.lower()
            if pat and pat not in tl:
                continue
            if " " in t or t.startswith("/") or t.startswith("-"):
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def query_module_avail(pattern: str = "VASP", *, timeout: int = 20) -> Dict[str, Any]:
    """
    Run ``module avail <pattern>`` and parse the result.

    Returns ``{"available": bool, "modules": [...], "errors": [...], "pattern": ...}``.
    """
    payload: Dict[str, Any] = {
        "pattern": pattern,
        "available": False,
        "modules": [],
        "errors": [],
        "raw": "",
    }
    safe_pat = shlex.quote(pattern or "")
    script = (
        "if ! type module >/dev/null 2>&1; then "
        "if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; "
        "elif [ -f /usr/share/Modules/init/bash ]; then . /usr/share/Modules/init/bash; "
        "elif [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi; fi; "
        "if ! type module >/dev/null 2>&1; then echo '__no_module__' >&2; exit 127; fi; "
        f"module avail {safe_pat} 2>&1"
    )
    rc, out, err = _run_shell(script, timeout=timeout)
    combined = (out or "") + ("\n" + err if err else "")
    payload["raw"] = combined
    if "__no_module__" in combined:
        payload["errors"].append(
            "未检测到 environment-modules / Lmod；请在已加载 module 的登录节点上运行服务，"
            "或设置 INTEROPTIMUS_MODULECMD 显式指定。"
        )
        return payload
    if rc not in (0, 124):
        payload["errors"].append(f"`module avail {pattern}` 退出码 {rc}: {err.strip() or out.strip()}")
    modules = _parse_module_avail(combined, pattern)
    payload["modules"] = modules
    payload["available"] = bool(modules) or rc == 0
    return payload


# ---------------------------------------------------------------------------
# MLIP probe inside a worker's pre_run environment
# ---------------------------------------------------------------------------


_QUICK_PROBE_SCRIPT = """\
import json, os, sys, traceback
out = {"ok": False, "mode": "import_only"}
try:
    import InterOptimus  # noqa: F401
    out["interoptimus_version"] = getattr(InterOptimus, "__version__", "unknown")
    calc_name = sys.argv[1]
    if calc_name == "orb-models":
        importable = "orb_models"
    elif calc_name == "sevenn":
        importable = "sevenn"
    elif calc_name == "matris":
        importable = "matris"
    elif calc_name == "dpa":
        importable = "deepmd"
    else:
        importable = ""
    if importable:
        __import__(importable)
        out["mlip_module"] = importable
    out["ok"] = True
except BaseException as e:
    out["ok"] = False
    out["error"] = str(e)
    out["error_type"] = type(e).__name__
    out["traceback"] = traceback.format_exc()
finally:
    print("__INTEROPTIMUS_PROBE__" + json.dumps(out))
"""

_PROBE_SCRIPT = """\
import json, os, sys, traceback

def _probe_device():
    d = (sys.argv[2] if len(sys.argv) > 2 else "cpu").strip().lower()
    return "cuda" if d in ("cuda", "gpu") else "cpu"

device = _probe_device()
if device == "cpu":
    os.environ["INTEROPTIMUS_FORCE_MLIP_CPU"] = "1"
else:
    os.environ.pop("INTEROPTIMUS_FORCE_MLIP_CPU", None)
out = {"ok": False, "mode": "static_energy", "device": device}
try:
    import InterOptimus  # noqa: F401
    out["interoptimus_version"] = getattr(InterOptimus, "__version__", "unknown")
    from InterOptimus.mlip import MlipCalc
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor

    calc_name = sys.argv[1]
    if calc_name == "orb-models":
        importable = "orb_models"
    elif calc_name == "sevenn":
        importable = "sevenn"
    elif calc_name == "matris":
        importable = "matris"
    elif calc_name == "dpa":
        importable = "deepmd"
    else:
        importable = ""
    if importable:
        __import__(importable)
        out["mlip_module"] = importable

    s = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    atoms = AseAtomsAdaptor.get_atoms(s)
    mc = MlipCalc(calc_name, {"device": device})
    atoms.calc = mc.calc
    e = float(atoms.get_potential_energy())
    out["ok"] = True
    out["energy"] = e
    out["n_atoms"] = len(atoms)
except BaseException as e:
    out["ok"] = False
    out["error"] = str(e)
    out["error_type"] = type(e).__name__
    out["traceback"] = traceback.format_exc()
finally:
    print("__INTEROPTIMUS_PROBE__" + json.dumps(out))
"""


def _run_probe_script(
    script_body: str,
    mlip: str,
    pre_run: str,
    timeout: int,
    *,
    device: str = "cpu",
    probe_node: Optional[str] = None,
    probe_partition: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, str, str, float]:
    """
    Execute ``script_body`` via the worker's ``pre_run`` shell context.

    When ``probe_node`` and/or ``probe_partition`` is set, the probe runs
    under Slurm as ``srun [-p PART] [-w NODE] ... bash <runner.sh>`` so Si
    static energy is evaluated on a compute node (shared ``$HOME`` recommended
    for scratch scripts).

    If **only** ``probe_partition`` is set (no node), ``srun`` still allocates
    a node in that partition; otherwise selecting a partition in the UI would
    only affect the label while the job actually ran on the login/host shell.

    ``work_dir`` : if set, the probe runs after ``cd`` into that directory (same
    as jobflow-remote jobs for that worker), so relative paths and side effects
    match production.

    Returns ``(parsed_result_or_empty, rc, stdout, stderr, duration_s)``.
    """
    quoted_mlip = shlex.quote(mlip)
    quoted_dev = shlex.quote(_normalize_probe_device(device))
    node = (probe_node or "").strip()
    part = (probe_partition or "").strip()
    use_srun = bool(node) or bool(part)
    wd = (work_dir or "").strip()
    scratch = _probe_remote_scratch_dir() if use_srun else None
    probe_path = ""
    runner_path = ""
    try:
        probe_tf = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="interoptimus_probe_",
            delete=False,
            dir=str(scratch) if scratch is not None else None,
        )
        probe_tf.write(script_body)
        probe_tf.flush()
        probe_path = probe_tf.name
        probe_tf.close()

        quoted_probe = shlex.quote(probe_path)

        if use_srun:
            runner_tf = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".sh",
                prefix="interoptimus_probe_run_",
                delete=False,
                dir=str(scratch),
            )
            runner_lines = ["#!/usr/bin/env bash", "set +e", _probe_thread_cap_shell()]
            if wd:
                runner_lines.append(f"cd {shlex.quote(wd)} || exit 1")
            if pre_run.strip():
                runner_lines.append(pre_run.rstrip())
            runner_lines.append(f"python {quoted_probe} {quoted_mlip} {quoted_dev}\n")
            runner_tf.write("\n".join(runner_lines))
            runner_tf.flush()
            runner_path = runner_tf.name
            runner_tf.close()
            try:
                os.chmod(runner_path, 0o700)
            except OSError:
                pass

            wt = _probe_slurm_walltime(timeout)
            cpus = str(_probe_srun_cpus_per_task())
            cmd: List[str] = ["srun"]
            if part:
                cmd.extend(["-p", part])
            if node:
                cmd.extend(["-w", node])
            cmd.extend(
                [
                    "-N",
                    "1",
                    "-n",
                    "1",
                    f"--cpus-per-task={cpus}",
                    f"--time={wt}",
                    "bash",
                    runner_path,
                ]
            )
            script = "set +e\n" + shlex.join(cmd)
        else:
            script_lines: List[str] = ["set +e", _probe_thread_cap_shell()]
            if wd:
                script_lines.append(f"cd {shlex.quote(wd)} || exit 1")
            if pre_run.strip():
                script_lines.append(pre_run)
            script_lines.append(f"python {quoted_probe} {quoted_mlip} {quoted_dev}")
            script = "\n".join(script_lines)

        started = time.time()
        rc, out, err = _run_shell(script, timeout=timeout)
        duration = time.time() - started
    finally:
        for p in (runner_path, probe_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    marker = "__INTEROPTIMUS_PROBE__"
    parsed: Dict[str, Any] = {}
    for line in (out or "").splitlines()[::-1]:
        idx = line.find(marker)
        if idx >= 0:
            try:
                parsed = json.loads(line[idx + len(marker) :])
            except json.JSONDecodeError:
                parsed = {}
            break
    return parsed, rc, out, err, duration


def probe_mlip_in_worker(
    mlip: str,
    worker: Optional[Dict[str, Any]],
    *,
    timeout: int = 240,
    full: bool = True,
    device: str = "cpu",
    probe_node: Optional[str] = None,
    probe_partition: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a small static-energy calculation inside ``worker`` to verify that
    its environment can actually run ``mlip``.

    ``device`` is passed to :class:`InterOptimus.mlip.MlipCalc` (``cpu`` or
    ``cuda``; ``gpu`` is accepted and normalized to ``cuda``). For ``cpu``,
    ``INTEROPTIMUS_FORCE_MLIP_CPU=1`` is set inside the probe so it overrides
    any inherited ``0``/empty disabling from the parent shell.

    ``worker`` is the dict returned by :func:`list_workers`; its ``pre_run`` and
    (for the **full** Si step only) ``work_dir`` are applied so the probe matches
    jobflow-remote job startup. The quick import sweep ignores ``work_dir`` and
    runs on the host that serves the API. When ``full=False`` the probe only verifies
    that ``InterOptimus`` and the MLIP package import successfully (used
    as a fast pre-filter to avoid 30-90s torch/checkpoint loads on workers
    that obviously cannot satisfy the request).

    When ``probe_node`` is set (hostname for ``srun -w``), only the **full**
    Si probe uses ``srun`` on that node; the **quick** import sweep always
    runs in the current shell.

    When **only** ``probe_partition`` is set (typical manage-UI flow), the
    full probe uses ``srun -p <partition>`` without ``-w``, so Si static
    energy runs on a compute node in that partition — matching real MLIP
    jobs more closely than running ``pre_run`` on the login/host shell.

    Returns a payload with::

        {
            "ok": bool, "mode": "static_energy" | "import_only",
            "mlip": str, "worker": str,
            "energy": float | None, "n_atoms": int | None,
            "error": str | None, "error_type": str | None,
            "interoptimus_version": str, "duration_s": float,
            "device": str,
            "probe_node": str,
            "probe_partition": str,
            "stdout": str, "stderr": str, "work_dir": str, "traceback": str,
        }
    """
    dev = _normalize_probe_device(device)
    pn = (probe_node or "").strip() or None
    pp = (probe_partition or "").strip() or None
    name = str((worker or {}).get("name") or "")
    pre_run = str((worker or {}).get("pre_run") or "")
    work_dir_raw = str((worker or {}).get("work_dir") or "").strip()
    payload: Dict[str, Any] = {
        "ok": False,
        "mode": "static_energy" if full else "import_only",
        "mlip": mlip,
        "worker": name,
        "device": dev,
        "probe_node": pn or "",
        "probe_partition": pp or "",
        "energy": None,
        "n_atoms": None,
        "error": None,
        "error_type": None,
        "interoptimus_version": "",
        "duration_s": 0.0,
        "stdout": "",
        "stderr": "",
        "work_dir": work_dir_raw,
        "traceback": "",
    }
    if not name:
        payload["error"] = "missing worker"
        return payload
    if mlip not in MLIP_TO_WORKER_HINTS:
        payload["error"] = f"unsupported mlip: {mlip}"
        return payload

    script = _PROBE_SCRIPT if full else _QUICK_PROBE_SCRIPT
    parsed, rc, out, err, duration = _run_probe_script(
        script,
        mlip,
        pre_run,
        timeout,
        device=dev,
        probe_node=pn if full else None,
        probe_partition=pp if full else None,
        work_dir=(work_dir_raw or None) if full else None,
    )
    payload["duration_s"] = round(duration, 2)
    payload["stdout"] = out[-4000:]
    payload["stderr"] = err[-4000:]
    if not parsed:
        payload["error"] = (
            f"probe did not emit a result line (rc={rc}); "
            "the worker pre_run may have failed before reaching python — "
            "see stderr for details"
        )
        return payload

    payload["ok"] = bool(parsed.get("ok"))
    if parsed.get("device"):
        payload["device"] = str(parsed.get("device"))
    if "energy" in parsed:
        payload["energy"] = parsed.get("energy")
    if "n_atoms" in parsed:
        payload["n_atoms"] = parsed.get("n_atoms")
    if "interoptimus_version" in parsed:
        payload["interoptimus_version"] = str(parsed.get("interoptimus_version") or "")
    if not payload["ok"]:
        payload["error"] = parsed.get("error") or f"probe failed (rc={rc})"
        payload["error_type"] = parsed.get("error_type") or ""
        tb = parsed.get("traceback")
        if isinstance(tb, str) and tb.strip():
            payload["traceback"] = tb.strip()[-8000:]
    return payload


def auto_select_mlip_worker(
    mlip: str,
    project: Optional[str] = None,
    *,
    explicit_worker: Optional[str] = None,
    timeout: int = 240,
    quick_timeout: int = 45,
    stop_at_first: bool = False,
    probe_device: str = "cpu",
    probe_node: Optional[str] = None,
    probe_partition: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate every worker in the project that can run ``mlip`` and return
    the full set; the UI uses this to expose a dropdown of all passing
    workers (with an auto-pick suggestion).

    Strategy:
      1. Build a ranked candidate list:
         - ``explicit_worker`` first (if it exists in the project)
         - workers whose ``mlip_hint`` matches ``mlip``
         - remaining workers, alphabetical
      2. **Fast import-only sweep** (``full=False``, ``quick_timeout``):
         drops workers whose conda env obviously lacks ``InterOptimus`` or
         the MLIP package.
      3. **Full Si-static-energy probe** (``full=True``, ``timeout``) on
         every survivor (ordered, but always all of them unless
         ``stop_at_first`` is set; this lets the UI show every validated
         worker in a dropdown rather than auto-hide the alternatives).

    ``probe_device`` controls the Si static-energy step (``cpu`` or ``cuda``);
    the management form's MLIP ``device`` field should match what production
    jobs will use.

    ``probe_node`` / ``probe_partition`` (optional) run each **full** Si probe
    via ``srun`` when at least one is set: ``-p`` for partition, ``-w`` for a
    specific node, or partition-only for an arbitrary node in that partition.

    Returns::

        {
            "ok": bool,
            "mlip": str,
            "project": str,
            "selected": worker dict | None,
            "candidates_tried": [worker_name, ...],
            "quick_results": [...],
            "validated_workers": [{"worker": dict, "probe": full_probe_payload}, ...],
            "mlip_ready_workers": [{"worker": dict, "probe": ..., "si_ok": bool}, ...],
            "probe": full_probe_payload | None,
            "errors": [str, ...],
            "probe_device": str,
            "probe_node": str,
            "probe_partition": str,
        }

    ``mlip_ready_workers`` has one row per **import** survivor (same order as the
    full Si sweep).  Rows with ``si_ok`` false are still listed so the UI can
    offer e.g. ``std_worker`` when only ``gnu_worker`` passes the Si calculation.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "mlip": mlip,
        "project": project or _default_project_name(),
        "selected": None,
        "candidates_tried": [],
        "quick_results": [],
        "validated_workers": [],
        "mlip_ready_workers": [],
        "probe": None,
        "errors": [],
        "probe_device": _normalize_probe_device(probe_device),
        "probe_node": "",
        "probe_partition": "",
    }
    pd = str(out["probe_device"])
    pn = (probe_node or "").strip() or None
    pp = (probe_partition or "").strip() or None
    out["probe_node"] = pn or ""
    out["probe_partition"] = pp or ""
    info = list_workers(project)
    if info.get("errors"):
        out["errors"].extend(info["errors"])
    workers = info.get("workers") or []
    if not workers:
        out["errors"].append(
            f"未在 ~/.jfremote/{out['project']}.yaml 中发现任何 workers；请先运行 deploy_jobflow_stack.py"
        )
        return out

    by_name = {w["name"]: w for w in workers}
    order: List[Dict[str, Any]] = []
    if explicit_worker and explicit_worker in by_name:
        order.append(by_name[explicit_worker])
    for w in workers:
        if w in order:
            continue
        if w.get("mlip_hint") == mlip:
            order.append(w)
    for w in workers:
        if w in order:
            continue
        order.append(w)

    survivors: List[Dict[str, Any]] = []
    last_quick: Optional[Dict[str, Any]] = None
    for w in order:
        out["candidates_tried"].append(w["name"])
        q = probe_mlip_in_worker(mlip, w, timeout=quick_timeout, full=False)
        out["quick_results"].append(
            {
                "worker": w["name"],
                "ok": bool(q.get("ok")),
                "error": q.get("error") if not q.get("ok") else None,
                "duration_s": q.get("duration_s"),
            }
        )
        last_quick = q
        if q.get("ok"):
            survivors.append(w)

    if not survivors:
        out["probe"] = last_quick
        out["errors"].append(
            f"在已配置的 {len(order)} 个 worker 中没有 conda 环境同时含 InterOptimus 与 {mlip!r} 包；"
            "请用 `interoptimus deploy --with-mlip-workers` 或对应 conda env 手动 `pip install` 后重试。"
        )
        return out

    last_probe: Optional[Dict[str, Any]] = None
    mlip_ready: List[Dict[str, Any]] = []
    for w in survivors:
        full = probe_mlip_in_worker(
            mlip,
            w,
            timeout=timeout,
            full=True,
            device=pd,
            probe_node=pn,
            probe_partition=pp,
        )
        last_probe = full
        si_ok = bool(full.get("ok"))
        mlip_ready.append({"worker": w, "probe": full, "si_ok": si_ok})
        if si_ok:
            out["validated_workers"].append({"worker": w, "probe": full})
            if out["selected"] is None:
                out["selected"] = w
                out["probe"] = full
                out["ok"] = True
            if stop_at_first:
                out["mlip_ready_workers"] = mlip_ready
                return out
    out["mlip_ready_workers"] = mlip_ready
    if not out["ok"]:
        out["probe"] = last_probe
        out["errors"].append(
            f"已找到 {len(survivors)} 个可导入 {mlip!r} 的 worker，但 Si 静态能验证均失败；"
            "通常需要 checkpoint：将权重放到 ~/.cache/InterOptimus/checkpoints。"
        )
    return out
