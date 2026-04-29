"""Guided preflight checks for InterOptimus server setup."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from InterOptimus.checkpoints import CHECKPOINTS, checkpoint_status
from InterOptimus.mlip import default_mlip_checkpoint_dir

JOBFLOW_YAML = Path.home() / ".jobflow.yaml"
ATOMATE2_YAML = Path.home() / ".atomate2.yaml"
JFREMOTE_DIR = Path.home() / ".jfremote"


def _run(cmd: list[str], *, timeout: int = 20, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        return 127, f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None or not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _line(status: str, name: str, detail: str = "") -> None:
    print(f"{status:<7} {name}{': ' + detail if detail else ''}")


def _bashrc_has_conda_init() -> bool:
    bashrc = Path.home() / ".bashrc"
    if not bashrc.is_file():
        return False
    try:
        text = bashrc.read_text(errors="ignore").lower()
    except OSError:
        return False
    markers = (
        "conda initialize",
        "conda shell.bash hook",
        "conda.sh",
        "conda activate",
        "miniconda",
        "miniforge",
        "mambaforge",
        "anaconda",
    )
    return any(marker in text for marker in markers)


def _check_bash_login_conda() -> tuple[bool, str]:
    code, out = _run(
        [
            "bash",
            "-lc",
            "command -v conda >/dev/null && conda shell.bash hook >/dev/null",
        ],
        timeout=20,
    )
    if code == 0:
        code2, conda_path = _run(["bash", "-lc", "command -v conda"], timeout=20)
        return True, conda_path.splitlines()[0] if code2 == 0 and conda_path else "conda ok"
    return False, out or "bash -lc cannot initialize conda"


def run_doctor(
    *,
    project_name: str = "std",
    check_runner: bool = True,
    suggest_interactive: bool = True,
) -> bool:
    """Print a concise setup report. Returns True when core checks pass."""
    all_ok = True
    print("InterOptimus doctor")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    setup_hint = "运行 itom config --interactive" if suggest_interactive else "本向导将写入/补齐"

    if sys.version_info >= (3, 10):
        _line("OK", "Python >= 3.10")
    else:
        all_ok = False
        _line("MISSING", "Python >= 3.10", "请使用 Python 3.10 或更新版本")

    try:
        import InterOptimus  # noqa: F401

        _line("OK", "InterOptimus import", str(Path(InterOptimus.__file__ or "").parent))
    except Exception as exc:
        all_ok = False
        _line("MISSING", "InterOptimus import", f"pip install -e . 后重试 ({exc})")

    conda_path = shutil.which("conda")
    if conda_path:
        _line("OK", "conda (current shell)", conda_path)
    else:
        all_ok = False
        _line("MISSING", "conda (current shell)", "先 conda activate 对应环境")

    if _bashrc_has_conda_init():
        _line("OK", "~/.bashrc conda init", "found")
    else:
        all_ok = False
        _line(
            "MISSING",
            "~/.bashrc conda init",
            "运行 `conda init bash` 后重新登录，或确保 batch shell 能找到 conda",
        )

    bash_conda_ok, bash_conda_detail = _check_bash_login_conda()
    if bash_conda_ok:
        _line("OK", "bash -lc conda hook", bash_conda_detail)
    else:
        all_ok = False
        _line(
            "MISSING",
            "bash -lc conda hook",
            f"{bash_conda_detail[:300]}；worker pre_run 可能无法 activate conda 环境",
        )

    jf_path = shutil.which("jf")
    if jf_path:
        _line("OK", "jf", jf_path)
    else:
        all_ok = False
        _line("MISSING", "jf", "安装 jobflow-remote 或运行 itom config")

    if yaml is None:
        all_ok = False
        _line("MISSING", "PyYAML", "pip install pyyaml")
    else:
        _line("OK", "PyYAML")

    jf_cfg = _read_yaml(JOBFLOW_YAML)
    if jf_cfg and isinstance(jf_cfg.get("JOB_STORE"), dict):
        _line("OK", str(JOBFLOW_YAML), "JOB_STORE found")
    else:
        all_ok = False
        _line("MISSING", str(JOBFLOW_YAML), setup_hint)

    a2_cfg = _read_yaml(ATOMATE2_YAML)
    if a2_cfg and a2_cfg.get("VASP_CMD"):
        _line("OK", str(ATOMATE2_YAML), f"VASP_CMD={a2_cfg.get('VASP_CMD')}")
    else:
        all_ok = False
        _line("MISSING", str(ATOMATE2_YAML), f"{setup_hint}，或手动设置 VASP_CMD")

    project_file = JFREMOTE_DIR / f"{project_name}.yaml"
    if project_file.is_file():
        _line("OK", str(project_file))
    else:
        all_ok = False
        _line("MISSING", str(project_file), setup_hint)

    jf = shutil.which("jf")
    if jf and project_file.is_file():
        env = os.environ.copy()
        env.setdefault("JOBFLOW_CONFIG_FILE", str(JOBFLOW_YAML))
        env.setdefault("ATOMATE2_CONFIG_FILE", str(ATOMATE2_YAML))
        env.setdefault("JFREMOTE_PROJECT", project_name)
        env.setdefault("JFREMOTE_PROJECTS_FOLDER", str(JFREMOTE_DIR.resolve()))
        code, out = _run([jf, "project", "check", "--errors"], timeout=60, env=env)
        if code == 0:
            _line("OK", "jf project check")
        else:
            all_ok = False
            _line("MISSING", "jf project check", (out or "failed")[:500])

        if check_runner:
            code, out = _run([jf, "runner", "status"], timeout=20, env=env)
            if code == 0 and "shut_down" not in out:
                _line("OK", "jf runner status", out.splitlines()[0] if out else "running")
            else:
                _line("WARN", "jf runner", "配置后通常还需要执行 `jf runner start` 或 `jf runner restart`")

    print(f"\nCheckpoint 目录: {default_mlip_checkpoint_dir()}")
    for spec in CHECKPOINTS:
        ok, path = checkpoint_status(spec)
        if ok and path:
            _line("OK", f"checkpoint {spec.key}", str(path))
        else:
            all_ok = False
            detail = f"运行 `itom checkpoints download {spec.key}`"
            if path:
                detail += f"（发现替代文件: {path}）"
            _line("MISSING", f"checkpoint {spec.key}", detail)

    if all_ok:
        print("\n环境看起来已经就绪。")
    elif suggest_interactive:
        print("\n还有项目未就绪。推荐下一步: `itom config --interactive`")
    else:
        print("\n还有项目未就绪。下面的向导会继续询问并补齐这些配置。")
    return all_ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check InterOptimus environment readiness.")
    parser.add_argument("--project-name", default="std", help="jobflow-remote project name")
    parser.add_argument("--no-runner", action="store_true", help="skip jf runner status check")
    args = parser.parse_args(argv)
    if not run_doctor(project_name=args.project_name, check_runner=not args.no_runner):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
