#!/usr/bin/env python3
"""
Probe InterOptimus / jobflow-remote / HPC environment on the **current** machine.

Intended use: after ``ssh`` to the cluster login node, run::

    interoptimus-env

or::

    python -m InterOptimus.agents.server_env
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def _run(
    cmd: List[str],
    *,
    shell: bool = False,
    timeout: float = 120.0,
) -> Tuple[int, str, str]:
    try:
        if shell:
            proc = subprocess.run(
                cmd[0] if len(cmd) == 1 and isinstance(cmd[0], str) else cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]!r}"
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\n(timeout)"


def probe_python_interoptimus() -> Dict[str, Any]:
    """Current interpreter, version, and whether InterOptimus imports."""
    out: Dict[str, Any] = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
    }
    try:
        import InterOptimus  # noqa: F401

        out["interoptimus_import"] = "ok"
        out["interoptimus_file"] = getattr(InterOptimus, "__file__", None)
    except Exception as e:
        out["interoptimus_import"] = f"failed: {e}"
    return out


def probe_jf_cli(jf_bin: str = "jf") -> Dict[str, Any]:
    """``jf`` CLI presence and top-level help."""
    code, so, se = _run([jf_bin, "--help"], timeout=30)
    if code == 127:
        return {"jf_bin": jf_bin, "found": False, "error": se.strip()}
    return {
        "jf_bin": jf_bin,
        "found": True,
        "help_head": (so + se)[:4000],
    }


def probe_jf_tree(jf_bin: str = "jf") -> Dict[str, Any]:
    """``jf --tree`` (full command list)."""
    code, so, se = _run([jf_bin, "--tree"], timeout=60)
    return {
        "exit_code": code,
        "stdout": so,
        "stderr": se,
    }


def probe_jf_project_workers(jf_bin: str = "jf") -> Dict[str, Any]:
    """
    Try common jobflow-remote commands to list workers and execution configs.

    Different versions expose ``jf project worker list`` or similar.
    """
    attempts: List[Dict[str, Any]] = []
    for cmd in (
        [jf_bin, "project", "worker", "list"],
        [jf_bin, "worker", "list"],
        [jf_bin, "project", "exec_config", "list"],
    ):
        code, so, se = _run(cmd, timeout=90)
        attempts.append({
            "cmd": cmd,
            "exit_code": code,
            "stdout": so[:12000],
            "stderr": se[:4000],
        })
        if code == 0:
            return {"success": True, "primary": attempts[-1], "attempts": attempts}
    return {"success": False, "attempts": attempts}


def probe_jf_runner_status(jf_bin: str = "jf") -> Dict[str, Any]:
    code, so, se = _run([jf_bin, "runner", "status"], timeout=30)
    return {"exit_code": code, "stdout": so[:8000], "stderr": se[:2000]}


def probe_module_avail_vasp() -> Dict[str, Any]:
    """
    Lines from ``module avail`` that match ``vasp`` (case-insensitive), via bash.

    ``module`` is a shell function; must use ``bash -lc``.
    """
    script = "module avail 2>&1 | grep -i vasp | head -120 || true"
    code, so, se = _run(["bash", "-lc", script], timeout=90)
    lines = [ln for ln in (so + se).splitlines() if ln.strip()]
    return {
        "exit_code": code,
        "line_count": len(lines),
        "lines": lines[:120],
    }


def probe_which_binaries(names: Tuple[str, ...]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in names:
        out[name] = shutil.which(name)
    return out


def collect_server_env_report(*, jf_bin: str = "jf") -> Dict[str, Any]:
    """Single JSON-serializable dict for login-node diagnostics."""
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "cwd": os.getcwd(),
        "user": os.environ.get("USER"),
        "path_head": os.environ.get("PATH", "")[:500],
        "which": probe_which_binaries(("python", "python3", "conda", "module", jf_bin)),
        "python": probe_python_interoptimus(),
        "jf": probe_jf_cli(jf_bin),
        "jf_tree": probe_jf_tree(jf_bin),
        "jf_workers": probe_jf_project_workers(jf_bin),
        "jf_runner_status": probe_jf_runner_status(jf_bin),
        "module_avail_vasp_grep": probe_module_avail_vasp(),
    }


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="InterOptimus / jobflow-remote environment probe (run on the cluster login node).",
    )
    p.add_argument(
        "--jf-bin",
        default=os.environ.get("INTEROPTIMUS_JF_BIN", "jf"),
        help="jobflow-remote CLI (default: jf, or env INTEROPTIMUS_JF_BIN)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full report as JSON (for scripts)",
    )
    args = p.parse_args(argv)

    report = collect_server_env_report(jf_bin=args.jf_bin)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("InterOptimus — login-node environment")
    print(f"  hostname: {report.get('hostname')}")
    print(f"  user:     {report.get('user')}")
    print(f"  cwd:      {report.get('cwd')}")

    w = report.get("which") or {}
    print("\nWhich (PATH):")
    for k, v in w.items():
        print(f"  {k}: {v or '(not found)'}")

    py = report.get("python") or {}
    print("\nPython:")
    print(f"  executable: {py.get('executable')}")
    print(f"  version:    {py.get('version')}")
    print(f"  InterOptimus import: {py.get('interoptimus_import')}")

    jf = report.get("jf") or {}
    print("\njobflow-remote CLI (`jf`):")
    if jf.get("found"):
        print(f"  {args.jf_bin}: OK")
    else:
        print(f"  {args.jf_bin}: NOT FOUND — {jf.get('error', '')}")

    jw = report.get("jf_workers") or {}
    _print_section("jobflow-remote workers / exec (best attempt)")
    if jw.get("success") and jw.get("primary"):
        prim = jw["primary"]
        print("Command:", " ".join(prim.get("cmd", [])))
        out = (prim.get("stdout") or "") + (prim.get("stderr") or "")
        print(out.strip() or "(empty output)")
    else:
        print("Could not list workers automatically. Try manually:")
        print(f"  {args.jf_bin} project worker list")
        print(f"  {args.jf_bin} --tree")
        for att in (jw.get("attempts") or [])[:3]:
            print(f"\n--- tried: {' '.join(att.get('cmd', []))} (exit {att.get('exit_code')}) ---")
            print((att.get("stderr") or att.get("stdout") or "")[:1500])

    _print_section("module avail (lines matching vasp)")
    mod = report.get("module_avail_vasp_grep") or {}
    lines = mod.get("lines") or []
    if lines:
        for ln in lines:
            print(ln)
    else:
        print(
            "(no lines matched, or `module` unavailable). "
            "Try: module avail 2>&1 | grep -i vasp"
        )

    runner = report.get("jf_runner_status") or {}
    if runner.get("stdout") or runner.get("stderr"):
        _print_section("jf runner status")
        if runner.get("stdout"):
            print(runner["stdout"][:6000])
        if runner.get("stderr"):
            print(runner["stderr"][:2000])

    tree = report.get("jf_tree") or {}
    if tree.get("stdout"):
        _print_section("jf --tree (full)")
        print(tree["stdout"])


if __name__ == "__main__":
    main()
