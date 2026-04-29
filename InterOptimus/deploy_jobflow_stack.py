#!/usr/bin/env python3
"""
一键部署 jobflow / jobflow-remote（MongoDB）、atomate2（含 ~/.jobflow.yaml 与 ~/.atomate2.yaml）。
pymatgen / POTCAR（pmg、~/.pmgrc.yaml）请自行配置，本脚本不处理。

请在登录节点运行本脚本。MongoDB 的 host 默认 auto：在登录节点上用 ss 查找本机 mongod 在非 127.0.0.1 上的监听并解析主机名
（便于计算节点连 Mongo）；若只有 127.0.0.1 监听则退回 localhost，此时可显式传入 --mongo-host。

可选 --with-mlip-workers：在当前 conda 环境 pip install -e 安装 InterOptimus，并创建 conda 环境 orb / dpa / matris / sevenn，
在 jobflow-remote 当前项目中增加同名 local worker（pre_run 使用当前 conda 命令并 conda activate）。Runner 若已在运行需 jf runner restart。
InterOptimus / MatRIS 默认在 ~/software 下自动查找文件名分别含 InterOptimus、matris 的 .zip（不区分大小写，取修改时间最新）；也可用 --interoptimus-dir / --matris-local 等指定目录或压缩包。
MatRIS 优先顺序：--matris-local → 环境变量 INTEROPTIMUS_MATRIS_LOCAL / MATRIS_LOCAL_PACKAGE → ~/software 中含 matris 的 .zip（自动）→ InterOptimus 同级/子目录 MatRIS；否则再从 git 安装。

用法示例（与本机 Mongo 库 htp_test / 用户 htp_test 一致；默认 --mongo-host auto）:
  export INTEROPTIMUS_MONGO_PASSWORD=htp_test
  python deploy_jobflow_stack.py \\
    --mongo-db htp_test --mongo-user htp_test

脚本在写入任何配置前会先检测 MongoDB：连接、ping、在目标库列举集合并做一次探针写入/删除，以确认凭据可用且具备读写权限（缺 pymongo 时会自动 pip 安装后再检测）。

jobflow-remote 默认项目名为 std（写入 ~/.jfremote/std.yaml，可用 --project-name 覆盖），
默认仅一个 local worker，pre_run 会使用当前已激活 conda 环境中的 conda 命令并 activate 当前环境。

仅检测（不写入）:
  export INTEROPTIMUS_MONGO_PASSWORD=htp_test
  python deploy_jobflow_stack.py --verify-only --mongo-db htp_test \\
    --mongo-user htp_test
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

# When this script is executed by path (``python InterOptimus/deploy_jobflow_stack.py``),
# Python puts the package directory itself on sys.path. That directory contains
# ``jobflow.py``, which can shadow the third-party ``jobflow`` package during
# verification. The deploy helper is standalone, so remove that path early.
_SCRIPT_DIR = Path(__file__).resolve().parent
_clean_sys_path: list[str] = []
for _p in sys.path:
    try:
        if Path(_p or os.getcwd()).resolve() == _SCRIPT_DIR:
            continue
    except OSError:
        pass
    _clean_sys_path.append(_p)
sys.path[:] = _clean_sys_path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

BASHRC_MARKER_BEGIN = "# >>> jobflow-stack deploy (managed by deploy_jobflow_stack.py) >>>"
BASHRC_MARKER_END = "# <<< jobflow-stack deploy <<<"

DEFAULT_PROJECT = "std"
DEFAULT_WORKER_NAME = "default"
JOBFLOW_YAML = Path.home() / ".jobflow.yaml"
ATOMATE2_YAML = Path.home() / ".atomate2.yaml"
JFREMOTE_DIR = Path.home() / ".jfremote"

# InterOptimus：与 InterOptimus/mlip.py 中 calc 名称对应，各用独立 conda 环境避免 torch/依赖冲突
# 默认在 ~/software 下按文件名匹配 .zip（见 discover_software_zip）
SOFTWARE_DIR = Path.home() / "software"
INTEROPTIMUS_CORE_PIP = [
    "pymatgen",
    "interfacemaster",
    "dscribe",
    "scikit-optimize",
    "matplotlib",
    "numpy",
    "ase",
    "atomate2",
    "jobflow",
    "jobflow-remote",
    "qtoolkit",
    "adjustText",
    "ipywidgets",
    "tqdm",
    "mp-api",
    "openai",
]
# worker 名 = conda 环境名；pip 为在该环境中额外安装的计算器包（不含 InterOptimus/setup.py 里并列的其它 MLIP）
MLIP_CONDA_SPECS: list[dict[str, Any]] = [
    {"worker": "orb", "env": "orb", "pip": ["orb-models"]},
    # deepmd-kit PyPI 轮子在 CIBUILDWHEEL=1 时会按 pip 发行版 mpich 解析 libmpi；无则报 PackageNotFoundError: mpich
    {"worker": "dpa", "env": "dpa", "pip": ["deepmd-kit", "mpich"]},
    {"worker": "matris", "env": "matris", "pip": []},  # MatRIS 由 resolve_matris_install_source 决定本地或 git
    {"worker": "sevenn", "env": "sevenn", "pip": ["sevenn"]},
]

MATRIS_INSTALL_GIT_DEFAULT = "git+https://github.com/HPC-AI-Team/MatRIS.git"


def discover_software_zip(name_substring: str) -> Path | None:
    """在 ~/software 下查找文件名含 name_substring（不区分大小写）的 .zip，取修改时间最新。"""
    d = SOFTWARE_DIR
    if not d.is_dir():
        return None
    sub = name_substring.lower()
    candidates: list[Path] = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".zip":
            continue
        if sub in p.name.lower():
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_zip_find_pkg_root(zip_path: Path, cache_label: str) -> Path | None:
    """解压 zip 到 ~/.cache/deploy_jobflow_stack/<cache_label>，返回含 setup.py 或 pyproject.toml 的包根；否则 None。"""
    cache = Path.home() / ".cache" / "deploy_jobflow_stack" / cache_label
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache)
    for sub in sorted(cache.iterdir(), key=lambda x: x.name):
        if sub.is_dir() and ((sub / "setup.py").is_file() or (sub / "pyproject.toml").is_file()):
            return sub.resolve()
    if (cache / "setup.py").is_file() or (cache / "pyproject.toml").is_file():
        return cache.resolve()
    for root, _, files in os.walk(cache):
        if "setup.py" in files or "pyproject.toml" in files:
            return Path(root).resolve()
    return None


def _resolve_interoptimus_install_root(cli: Path | None) -> tuple[Path, str]:
    """返回 InterOptimus 源码根目录（含 setup.py|pyproject.toml）及说明。"""
    if cli is not None:
        p = cli.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".zip":
            root = _extract_zip_find_pkg_root(p, "interoptimus_extract")
            if root is None:
                _die(f"解压 {p} 后未找到 setup.py 或 pyproject.toml")
            return root, f"指定 zip: {p}"
        if p.is_dir():
            if not (p / "setup.py").is_file() and not (p / "pyproject.toml").is_file():
                _die(f"目录中未找到 setup.py / pyproject.toml: {p}")
            return p, f"指定目录: {p}"
        _die(f"--interoptimus-dir 必须是目录或 .zip 文件: {p}")
    source_root = _SCRIPT_DIR.parent
    if (source_root / "setup.py").is_file() or (source_root / "pyproject.toml").is_file():
        return source_root.resolve(), f"当前 InterOptimus 源码目录: {source_root.resolve()}"
    z = discover_software_zip("interoptimus")
    if z is None:
        _die(
            f"未在 {SOFTWARE_DIR} 中找到文件名含 InterOptimus 的 .zip（不区分大小写）；"
            "请放入压缩包或使用 --interoptimus-dir 指定目录或 zip"
        )
    root = _extract_zip_find_pkg_root(z, "interoptimus_extract")
    if root is None:
        _die(f"解压 {z} 后未找到 setup.py 或 pyproject.toml")
    return root, f"自动发现 {SOFTWARE_DIR} 下: {z.name}"


def _is_usable_matris_local_path(p: Path) -> bool:
    """本地 wheel/sdist 或含 setup.py / pyproject.toml 的源码目录。"""
    try:
        p = p.expanduser().resolve()
    except OSError:
        return False
    if not p.exists():
        return False
    if p.is_file():
        suf = p.suffix.lower()
        return suf == ".whl" or suf in (".gz", ".zip", ".bz2")
    if p.is_dir():
        return (p / "setup.py").is_file() or (p / "pyproject.toml").is_file()
    return False


def resolve_matris_install_source(
    interoptimus_src: Path,
    matris_local_cli: Path | None,
) -> tuple[str, str]:
    """
    返回 (pip install 的单个参数, 说明字符串)。
    优先顺序：--matris-local → INTEROPTIMUS_MATRIS_LOCAL / MATRIS_LOCAL_PACKAGE
    → ~/software 中含 matris 的 .zip（自动）→ 常见本地目录 → git。
    """
    ordered: list[tuple[str, Path]] = []
    if matris_local_cli is not None:
        p_cli = matris_local_cli.expanduser()
        if not _is_usable_matris_local_path(p_cli):
            print(f"警告: --matris-local 不是可用的 wheel/压缩包或源码目录，将忽略并尝试其它来源: {p_cli}")
        else:
            ordered.append(("--matris-local", p_cli))
    for key in ("INTEROPTIMUS_MATRIS_LOCAL", "MATRIS_LOCAL_PACKAGE"):
        v = (os.environ.get(key) or "").strip()
        if v:
            ordered.append((f"环境变量 {key}", Path(v).expanduser()))
    auto_matris = discover_software_zip("matris")
    if auto_matris is not None:
        ordered.append(("~/software 中含 matris 的 .zip（自动）", auto_matris))
    base = interoptimus_src.expanduser().resolve()
    ordered.extend(
        [
            ("InterOptimus 同级目录 MatRIS", (base.parent / "MatRIS")),
            ("InterOptimus 同级目录 matris", (base.parent / "matris")),
            ("InterOptimus 同级目录 MatRIS-main", (base.parent / "MatRIS-main")),
            ("InterOptimus 同级目录 matris-main", (base.parent / "matris-main")),
            ("InterOptimus/MatRIS 子目录", (base / "MatRIS")),
        ]
    )
    for label, path in ordered:
        if not _is_usable_matris_local_path(path):
            continue
        resolved = path.expanduser().resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".zip":
            root = _extract_zip_find_pkg_root(resolved, "matris_extract")
            if root is not None:
                return str(root), f"本地（{label}，已解压）: {resolved} -> {root}"
            return str(resolved), f"本地（{label}，pip 直装 zip）: {resolved}"
        return str(resolved), f"本地（{label}）: {resolved}"
    return MATRIS_INSTALL_GIT_DEFAULT, f"远程 git（未找到可用本地包）: {MATRIS_INSTALL_GIT_DEFAULT}"


def detect_mongo_client_host(port: int = 27017) -> str:
    """
    在登录节点上：用 ss 查找本机在非 127.0.0.1 上对 port 的监听，并尽量解析为主机名（如 master）；
    若仅有 127.0.0.1 或未检测到则退回 localhost。
    """
    try:
        r = subprocess.run(
            ["ss", "-tln", "-H"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "localhost"
    pat = re.compile(rf"(\d+\.\d+\.\d+\.\d+):{port}\b")
    for line in (r.stdout or "").splitlines():
        m = pat.search(line)
        if not m:
            continue
        ip = m.group(1)
        if ip.startswith("127."):
            continue
        try:
            return socket.gethostbyaddr(ip)[0]
        except (socket.herror, OSError):
            return ip
    return "localhost"


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _ensure_pymongo_for_mongo_check() -> None:
    """MongoDB 检测依赖 pymongo；若未安装则静默安装，便于在 _pip_install 之前先做连通性检查。"""
    try:
        import pymongo  # noqa: F401
    except ImportError:
        print("未检测到 pymongo，正在安装以进行 MongoDB 连通性检测…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pymongo", "-q"],
            check=True,
        )


def verify_mongodb_access(
    mongo: dict[str, Any],
    *,
    auth_source: str | None = None,
) -> list[str]:
    """
    验证能否用给定主机/端口/库名/用户名密码连接 MongoDB，并在目标库上具备 jobflow 所需的读写能力。

    依次尝试：server ping → 列出目标库集合 → 在临时集合中插入并删除一条探针文档。
    返回错误信息列表；空列表表示通过。
    """
    errors: list[str] = []
    try:
        from pymongo import MongoClient
        from pymongo.errors import OperationFailure, PyMongoError

        kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": 10_000, "connectTimeoutMS": 10_000}
        if mongo.get("username"):
            auth_db = auth_source or mongo["database"]
            client = MongoClient(
                host=mongo["host"],
                port=mongo["port"],
                username=mongo["username"],
                password=mongo["password"],
                authSource=auth_db,
                **kwargs,
            )
        else:
            client = MongoClient(host=mongo["host"], port=mongo["port"], **kwargs)

        client.admin.command("ping")

        db = client[mongo["database"]]
        try:
            db.list_collection_names()
        except OperationFailure as e:
            errors.append(
                f"已连接 MongoDB，但无法在数据库「{mongo['database']}」上列出集合（权限不足或库不可访问）: {e}"
            )
            client.close()
            return errors

        probe_coll = "_deploy_jobflow_stack_probe"
        coll = db[probe_coll]
        try:
            ins = coll.insert_one(
                {"_deploy_jobflow_stack_probe": True, "source": "deploy_jobflow_stack.py"}
            )
            coll.delete_one({"_id": ins.inserted_id})
        except OperationFailure as e:
            errors.append(
                f"已连接并可读库「{mongo['database']}」，但探针写入/删除失败（jobflow-remote 需要读写权限）: {e}"
            )
            client.close()
            return errors

        client.close()
    except PyMongoError as e:
        errors.append(f"MongoDB 连接或认证失败: {e}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"MongoDB 检测异常: {e}")
    return errors


def _ensure_yaml() -> None:
    if yaml is None:
        _die("需要 PyYAML：请 pip install pyyaml")


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def _pip_install(*, skip: bool, upgrade: bool) -> None:
    if skip:
        print("跳过 pip 安装（--skip-install）")
        return
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "jobflow",
        "jobflow-remote",
        "atomate2",
        "pyyaml",
        "maggma",
        "pymongo",
    ]
    if upgrade:
        cmd.append("--upgrade")
    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        _die(f"pip 安装失败: {e}。可改用 --skip-install 在离线/已装包环境继续。")


def _venv_pre_run() -> str:
    """非 conda 时回退：source 与当前解释器一致的 venv/bin/activate。"""
    bindir = Path(sys.executable).resolve().parent
    activate = bindir / "activate"
    if activate.is_file():
        return f"source {activate}"
    return ""


def _resolve_conda_env_name() -> str | None:
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return os.environ["CONDA_DEFAULT_ENV"]
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    p = Path(prefix).resolve()
    if p.name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", "micromamba"):
        return "base"
    return p.name


def _conda_hook_line() -> str:
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
    return f'eval "$({shlex.quote(conda_exe)} shell.bash hook)"'


def _conda_worker_pre_run() -> str:
    """与安装时相同的 conda 环境：使用当前可用 conda 命令再 activate。"""
    if not os.environ.get("CONDA_PREFIX"):
        return ""
    env_name = _resolve_conda_env_name()
    if not env_name:
        return ""
    lines = [
        _conda_hook_line(),
        f"conda activate {shlex.quote(env_name)}",
    ]
    return "\n".join(lines)


def _conda_activate_pre_run(env_name: str) -> str:
    """指定 conda 环境：使用当前可用 conda 命令再 activate。"""
    lines = [
        _conda_hook_line(),
        f"conda activate {shlex.quote(env_name)}",
    ]
    return "\n".join(lines)


def _conda_exe_or_none() -> str | None:
    return shutil.which("conda")


def _conda_exe() -> str:
    exe = _conda_exe_or_none()
    if not exe:
        _die("未找到 conda 可执行文件（PATH 中需有 conda）")
    return exe


def _conda_env_exists(env_name: str) -> bool:
    conda = _conda_exe_or_none()
    if not conda:
        return False
    r = subprocess.run(
        [conda, "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(re.search(rf"^\s*{re.escape(env_name)}\s+", r.stdout or "", re.MULTILINE))


def _conda_create_env(env_name: str, *, python_tag: str) -> None:
    if _conda_env_exists(env_name):
        print(f"conda 环境已存在，跳过创建: {env_name}")
        return
    _run(
        [
            _conda_exe(),
            "create",
            "-y",
            "-n",
            env_name,
            f"python={python_tag}",
        ]
    )


def _conda_run_pip(env_name: str, pip_args: list[str], **kwargs: Any) -> None:
    cmd = [_conda_exe(), "run", "-n", env_name, "--no-capture-output", "pip", *pip_args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True, **kwargs)


def _install_interoptimus_editable_in_env(env_name: str, src: Path) -> None:
    """在各 MLIP 环境中以 --no-deps 安装 InterOptimus，避免把其它 MLIP 一并装上。"""
    if not src.is_dir():
        _die(f"InterOptimus 源码目录不存在: {src}")
    setup = src / "setup.py"
    pyproject = src / "pyproject.toml"
    if not setup.is_file() and not pyproject.is_file():
        _die(f"未找到 {src} 下的 setup.py / pyproject.toml")
    _conda_run_pip(
        env_name,
        ["install", "-e", str(src.resolve()), "--no-deps"],
    )


def _setup_mlip_conda_envs(
    interoptimus_src: Path,
    *,
    python_tag: str,
    matris_local: Path | None,
) -> None:
    """为 orb / dpa / matris / sevenn 创建 conda 环境并安装对应 MLIP 与 InterOptimus（无其它 MLIP 依赖）。"""
    _conda_exe()
    matris_arg, matris_how = resolve_matris_install_source(interoptimus_src, matris_local)
    print(f"MatRIS: {matris_how}")
    for spec in MLIP_CONDA_SPECS:
        env = spec["env"]
        _conda_create_env(env, python_tag=python_tag)
        _conda_run_pip(env, ["install", "-U", "pip", "setuptools", "wheel"])
        _conda_run_pip(env, ["install", "torch"])
        _conda_run_pip(env, ["install", *INTEROPTIMUS_CORE_PIP])
        if spec["worker"] == "matris":
            _conda_run_pip(env, ["install", matris_arg])
        else:
            for pkg in spec["pip"]:
                _conda_run_pip(env, ["install", pkg])
        _install_interoptimus_editable_in_env(env, interoptimus_src)
        print(f"MLIP 环境就绪: {env} ({spec['worker']})")


def _pip_install_interoptimus_base_env(src: Path) -> None:
    """在当前 Python 环境安装 InterOptimus（可编辑），便于在登录节点提交/调试。"""
    if not src.is_dir():
        _die(f"InterOptimus 源码目录不存在: {src}")
    _run([sys.executable, "-m", "pip", "install", "-e", str(src.resolve())])


def _warn_runner_restart_if_needed(project_name: str) -> None:
    jf = shutil.which("jf")
    if not jf:
        return
    env = os.environ.copy()
    env.setdefault("JFREMOTE_PROJECT", project_name)
    env.setdefault("JFREMOTE_PROJECTS_FOLDER", str(JFREMOTE_DIR.expanduser().resolve()))
    try:
        r = subprocess.run(
            [jf, "runner", "status"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if "shut_down" in out:
            print("Runner 当前为 shut_down；若之后启动 runner，新 worker 会自动加载。")
        else:
            print(
                "提示: Runner 似乎未处于 shut_down。已修改 jobflow-remote 项目配置时，"
                "请执行 `jf runner restart`（或 stop 后再 start）以加载新的 worker。"
            )
    except OSError:
        pass


def _build_jfremote_workers(
    *,
    base_work_dir: Path,
    default_worker_name: str,
    default_pre_run: str,
    with_mlip_workers: bool,
) -> dict[str, Any]:
    workers: dict[str, Any] = {}
    workers[default_worker_name] = {
        "type": "local",
        "scheduler_type": "slurm",
        "work_dir": str(base_work_dir.resolve()),
    }
    if default_pre_run:
        workers[default_worker_name]["pre_run"] = default_pre_run
    if with_mlip_workers:
        for spec in MLIP_CONDA_SPECS:
            w = spec["worker"]
            env = spec["env"]
            subdir = base_work_dir / w
            subdir.mkdir(parents=True, exist_ok=True)
            workers[w] = {
                "type": "local",
                "scheduler_type": "slurm",
                "work_dir": str(subdir.resolve()),
                "pre_run": _conda_activate_pre_run(env),
            }
    return workers


def _update_bashrc(exports: dict[str, str]) -> None:
    bashrc = Path.home() / ".bashrc"
    block_lines = [BASHRC_MARKER_BEGIN]
    for k, v in exports.items():
        block_lines.append(f"export {k}={shlex.quote(v)}")
    block_lines.append(BASHRC_MARKER_END)
    new_block = "\n".join(block_lines) + "\n"

    old = bashrc.read_text() if bashrc.is_file() else ""
    if BASHRC_MARKER_BEGIN in old and BASHRC_MARKER_END in old:
        pattern = re.compile(
            re.escape(BASHRC_MARKER_BEGIN) + r".*?" + re.escape(BASHRC_MARKER_END) + r"\n?",
            re.DOTALL,
        )
        old = pattern.sub("", old)
    bashrc.write_text(old.rstrip() + "\n\n" + new_block)


def _backup_existing_file(path: Path) -> None:
    """Create a timestamped backup before overwriting a user config file."""
    if not path.is_file():
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    shutil.copy2(path, backup)
    print(f"已备份已有配置: {path} -> {backup}")


def _write_jobflow_yaml(cfg: dict[str, Any]) -> None:
    _ensure_yaml()
    _backup_existing_file(JOBFLOW_YAML)
    JOBFLOW_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"已写入 {JOBFLOW_YAML}")


def _write_jfremote_project(
    path: Path,
    *,
    name: str,
    mongo: dict[str, Any],
    workers: dict[str, Any],
) -> None:
    _ensure_yaml()
    m = mongo
    store_common: dict[str, Any] = {
        "type": "MongoStore",
        "host": m["host"],
        "port": m["port"],
        "database": m["database"],
        "collection_name": "PLACEHOLDER",
    }
    if m.get("username"):
        store_common["username"] = m["username"]
        store_common["password"] = m["password"]

    queue_store = dict(store_common)
    queue_store["collection_name"] = m.get("queue_jobs_collection", "jf_jobs")

    docs_store = dict(store_common)
    docs_store["collection_name"] = m.get("jobstore_docs_collection", "jf_outputs")

    gridfs_store: dict[str, Any] = {
        "type": "GridFSStore",
        "host": m["host"],
        "port": m["port"],
        "database": m["database"],
        "collection_name": m.get("jobstore_gridfs_collection", "jf_outputs_blobs"),
    }
    if m.get("username"):
        gridfs_store["username"] = m["username"]
        gridfs_store["password"] = m["password"]

    project = {
        "name": name,
        "workers": workers,
        "queue": {
            "store": queue_store,
            "flows_collection": m.get("flows_collection", "jf_flows"),
            "auxiliary_collection": m.get("auxiliary_collection", "jf_auxiliary"),
            "batches_collection": m.get("batches_collection", "jf_batches"),
        },
        "jobstore": {
            "docs_store": docs_store,
            "additional_stores": {"data": gridfs_store},
        },
        "exec_config": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing_file(path)
    path.write_text(yaml.safe_dump(project, sort_keys=False, allow_unicode=True))
    print(f"已写入 jobflow-remote 项目配置 {path}")


def _write_atomate2_yaml(cfg: dict[str, Any]) -> None:
    _ensure_yaml()
    _backup_existing_file(ATOMATE2_YAML)
    ATOMATE2_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"已写入 {ATOMATE2_YAML}")


def _build_jobflow_yaml_for_atomate2(
    mongo: dict[str, Any],
    *,
    docs_collection: str,
    gridfs_collection: str,
) -> dict[str, Any]:
    """与 atomate2 安装文档一致的 JOB_STORE（docs + GridFS data）。"""
    docs_store: dict[str, Any] = {
        "type": "MongoStore",
        "host": mongo["host"],
        "port": mongo["port"],
        "database": mongo["database"],
        "collection_name": docs_collection,
    }
    if mongo.get("username"):
        docs_store["username"] = mongo["username"]
        docs_store["password"] = mongo["password"]

    gridfs_store: dict[str, Any] = {
        "type": "GridFSStore",
        "host": mongo["host"],
        "port": mongo["port"],
        "database": mongo["database"],
        "collection_name": gridfs_collection,
    }
    if mongo.get("username"):
        gridfs_store["username"] = mongo["username"]
        gridfs_store["password"] = mongo["password"]

    return {
        "JOB_STORE": {
            "docs_store": docs_store,
            "additional_stores": {"data": gridfs_store},
        }
    }


def verify_configuration(
    *,
    jobflow_yaml: Path,
    jf_project_file: Path,
    project_name: str,
) -> list[str]:
    """返回错误列表；空表示通过。MongoDB 凭据已在 main() 中通过 verify_mongodb_access() 检测。"""
    errors: list[str] = []

    if yaml is None:
        errors.append("未安装 pyyaml，无法校验 YAML")
        return errors

    if not jobflow_yaml.is_file():
        errors.append(f"缺少 {jobflow_yaml}")
    else:
        try:
            jf_cfg = yaml.safe_load(jobflow_yaml.read_text()) or {}
            if "JOB_STORE" not in jf_cfg or "docs_store" not in jf_cfg["JOB_STORE"]:
                errors.append(f"{jobflow_yaml} 缺少 JOB_STORE.docs_store 结构")
            else:
                js = jf_cfg["JOB_STORE"]
                ads = js.get("additional_stores") or {}
                if "data" not in ads:
                    errors.append(
                        f"{jobflow_yaml} 缺少 JOB_STORE.additional_stores.data（atomate2 需要 GridFS data store）"
                    )
        except Exception as e:  # noqa: BLE001
            errors.append(f"解析 {jobflow_yaml} 失败: {e}")

    if not ATOMATE2_YAML.is_file():
        errors.append(f"缺少 atomate2 配置 {ATOMATE2_YAML}")
    else:
        try:
            a2 = yaml.safe_load(ATOMATE2_YAML.read_text()) or {}
            if "VASP_CMD" not in a2:
                errors.append(f"{ATOMATE2_YAML} 缺少 VASP_CMD")
        except Exception as e:  # noqa: BLE001
            errors.append(f"解析 {ATOMATE2_YAML} 失败: {e}")

    if not jf_project_file.is_file():
        errors.append(f"缺少 jobflow-remote 项目文件 {jf_project_file}")

    try:
        import jobflow  # noqa: F401
    except ImportError as e:
        errors.append(f"无法 import jobflow: {e}")

    try:
        import jobflow_remote  # noqa: F401
    except ImportError as e:
        errors.append(f"无法 import jobflow_remote: {e}")

    try:
        import atomate2  # noqa: F401
    except ImportError as e:
        errors.append(f"无法 import atomate2: {e}")

    # Mongo 已在 main() 开头通过 verify_mongodb_access() 检测；此处不再重复

    # jobflow-remote CLI
    jf = shutil.which("jf")
    if not jf:
        errors.append("未找到 jf 命令（jobflow-remote CLI）")
    else:
        env = os.environ.copy()
        env.setdefault("JOBFLOW_CONFIG_FILE", str(JOBFLOW_YAML.expanduser()))
        env.setdefault("ATOMATE2_CONFIG_FILE", str(ATOMATE2_YAML.expanduser()))
        env.setdefault("JFREMOTE_PROJECT", project_name)
        env.setdefault("JFREMOTE_PROJECTS_FOLDER", str(JFREMOTE_DIR.expanduser().resolve()))
        try:
            r = subprocess.run(
                [jf, "project", "check", "--errors"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            if r.returncode != 0:
                errors.append(
                    "jf project check 未通过:\n"
                    + (r.stdout or "")
                    + (r.stderr or "")
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"执行 jf project check 失败: {e}")

    return errors


def _prompt_text(
    prompt: str,
    *,
    default: str | None = None,
    required: bool = False,
    secret: bool = False,
) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        if secret:
            value = getpass.getpass(f"{prompt}{suffix}: ")
        else:
            value = input(f"{prompt}{suffix}: ")
        value = value.strip()
        if not value and default is not None:
            value = str(default)
        if value or not required:
            return value
        print("此项必填。")


def _prompt_bool(prompt: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "是", "true", "1"}:
            return True
        if value in {"n", "no", "否", "false", "0"}:
            return False
        print("请输入 y 或 n。")


def _prompt_int(prompt: str, *, default: int) -> int:
    while True:
        raw = _prompt_text(prompt, default=str(default), required=True)
        try:
            return int(raw)
        except ValueError:
            print("请输入整数。")


def _apply_interactive_config(args: argparse.Namespace) -> None:
    print("InterOptimus 交互式配置向导")
    print("此向导会先询问必要信息，再复用 itom config 的自动部署流程写入配置。")
    print()
    try:
        from InterOptimus.doctor import run_doctor

        run_doctor(project_name=args.project_name, check_runner=False, suggest_interactive=False)
    except Exception as exc:  # noqa: BLE001
        print(f"预检未能完整运行，继续进入配置向导: {exc}")
    print()

    print("Step 1/5: MongoDB")
    args.mongo_host = _prompt_text("MongoDB host（auto 会尝试检测本机 mongod）", default=args.mongo_host or "auto")
    args.mongo_port = _prompt_int("MongoDB port", default=args.mongo_port)
    args.mongo_db = _prompt_text("MongoDB database", default=args.mongo_db, required=True)
    args.mongo_user = _prompt_text("MongoDB user（无认证可留空）", default=args.mongo_user or "")
    if args.mongo_user:
        if args.mongo_password is None:
            env_pw = os.environ.get("INTEROPTIMUS_MONGO_PASSWORD")
            args.mongo_password = _prompt_text(
                "MongoDB password（留空使用 INTEROPTIMUS_MONGO_PASSWORD）",
                default=env_pw,
                secret=True,
            )
        args.mongo_auth_source = _prompt_text(
            "MongoDB authSource（用户建在 admin 时填 admin；默认同 database）",
            default=args.mongo_auth_source or "",
        ) or None

    print("\nStep 2/5: jobflow-remote")
    args.project_name = _prompt_text("jobflow-remote project name", default=args.project_name, required=True)
    args.jf_worker_name = _prompt_text("default worker name", default=args.jf_worker_name, required=True)
    args.work_dir = Path(_prompt_text("worker work dir", default=str(args.work_dir), required=True)).expanduser()

    print("\nStep 3/5: atomate2 / VASP")
    args.vasp_cmd = _prompt_text("VASP_CMD（暂不用 VASP 也可保留默认）", default=args.vasp_cmd, required=True)
    scratch = _prompt_text("CUSTODIAN_SCRATCH_DIR（可留空）", default=str(args.custodian_scratch_dir or ""))
    args.custodian_scratch_dir = Path(scratch).expanduser() if scratch else None

    print("\nStep 4/5: MLIP workers and checkpoints")
    args.with_mlip_workers = _prompt_bool("是否配置 orb/dpa/matris/sevenn MLIP workers", default=args.with_mlip_workers)
    if args.with_mlip_workers:
        install_mlip = _prompt_bool("是否创建/安装 MLIP conda 环境（会比较耗时）", default=not args.skip_mlip_conda)
        args.skip_mlip_conda = not install_mlip
        if not args.skip_mlip_conda:
            src_default = str(args.interoptimus_dir) if args.interoptimus_dir else str(_SCRIPT_DIR.parent)
            src = _prompt_text("InterOptimus 源码目录或 zip", default=src_default, required=True)
            args.interoptimus_dir = Path(src).expanduser()
            matris = _prompt_text("MatRIS 本地包/源码路径（可留空，找不到时从 git 安装）", default=str(args.matris_local or ""))
            args.matris_local = Path(matris).expanduser() if matris else None
        args.checkpoint_models = _prompt_text(
            "需要下载/校验的 checkpoint（all 或 orb,sevenn,dpa,matris）",
            default=args.checkpoint_models,
            required=True,
        )
        args.skip_checkpoint_download = not _prompt_bool(
            "是否自动下载/补齐 checkpoint",
            default=not args.skip_checkpoint_download,
        )

    print("\nStep 5/5: shell integration")
    args.skip_install = not _prompt_bool(
        "是否安装/补齐 jobflow/jobflow-remote/atomate2 依赖（会执行 pip install）",
        default=not args.skip_install,
    )
    args.skip_bashrc = not _prompt_bool("是否更新 ~/.bashrc 中的 JOBFLOW/JFREMOTE 环境变量", default=not args.skip_bashrc)

    print("\n配置摘要")
    print(f"  MongoDB: {args.mongo_host}:{args.mongo_port} / {args.mongo_db} user={args.mongo_user or '(none)'}")
    print(f"  project: {args.project_name}, worker: {args.jf_worker_name}, work_dir: {args.work_dir}")
    print(f"  VASP_CMD: {args.vasp_cmd}")
    print(f"  MLIP workers: {'yes' if args.with_mlip_workers else 'no'}")
    if args.with_mlip_workers:
        print(f"  MLIP conda install: {'no' if args.skip_mlip_conda else 'yes'}")
        print(f"  checkpoints: {args.checkpoint_models}, download={'no' if args.skip_checkpoint_download else 'yes'}")
    print(f"  install Python deps: {'no' if args.skip_install else 'yes'}")
    print(f"  update ~/.bashrc: {'no' if args.skip_bashrc else 'yes'}")
    if not _prompt_bool("确认开始写入配置并执行安装/校验", default=True):
        raise SystemExit("已取消。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在登录节点一键部署 jobflow + jobflow-remote + atomate2（不含 pymatgen/POTCAR）"
    )
    parser.add_argument(
        "--mongo-host",
        default="auto",
        help="MongoDB 主机名或 IP；默认 auto=在登录节点用 ss 检测本机 mongod 监听（非 127.0.0.1 优先）",
    )
    parser.add_argument("--mongo-port", type=int, default=27017)
    parser.add_argument("--mongo-db", default=None, help="MongoDB 数据库名，例如 htp_test")
    parser.add_argument("--mongo-user", default="", help="可为空（无认证）")
    parser.add_argument(
        "--mongo-password",
        default=None,
        help="MongoDB 密码；也可用环境变量 INTEROPTIMUS_MONGO_PASSWORD，未提供时会交互输入",
    )
    parser.add_argument(
        "--mongo-auth-source",
        default=None,
        help="认证数据库（默认与 --mongo-db 相同；用户建在 admin 时需指定）",
    )
    parser.add_argument(
        "--standalone-jobflow-collection",
        default="jobflow_outputs",
        help="~/.jobflow.yaml 中文档库 MongoStore 的 collection_name（与 jfremote jobstore 区分）",
    )
    parser.add_argument(
        "--standalone-jobflow-gridfs-collection",
        default=None,
        help="~/.jobflow.yaml 中 GridFSStore 的 collection_name，默认 <standalone-jobflow-collection>_blobs",
    )
    parser.add_argument(
        "--conda-module",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jf-worker-name",
        default=DEFAULT_WORKER_NAME,
        help="主 local worker 名称（atomate2/通用任务）；与 --with-mlip-workers 并存时为默认 worker 之一",
    )
    parser.add_argument(
        "--vasp-cmd",
        default="vasp_std",
        help="写入 ~/.atomate2.yaml 的 VASP_CMD（并行启动命令视集群自行修改）",
    )
    parser.add_argument(
        "--custodian-scratch-dir",
        type=Path,
        default=None,
        help="可选：写入 atomate2 的 CUSTODIAN_SCRATCH_DIR（绝对路径为佳）",
    )
    parser.add_argument("--project-name", default=DEFAULT_PROJECT, help="jobflow-remote 项目名（~/.jfremote/<name>.yaml）")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / "jfremote_work",
        help="local worker 执行目录",
    )
    parser.add_argument("--skip-install", action="store_true", help="跳过 pip install")
    parser.add_argument("--upgrade-packages", action="store_true", help="pip install --upgrade")
    parser.add_argument("--skip-bashrc", action="store_true", help="不修改 ~/.bashrc（仍写入配置文件）")
    parser.add_argument("--verify-only", action="store_true", help="仅检测，不写配置")
    parser.add_argument("--interactive", "-i", action="store_true", help="逐步检测并询问缺失配置")
    parser.add_argument(
        "--with-mlip-workers",
        action="store_true",
        help="安装 InterOptimus（可编辑）并配置 orb/dpa/matris/sevenn 四个 MLIP conda 环境、checkpoint 与 jfremote worker",
    )
    parser.add_argument(
        "--checkpoint-models",
        default="all",
        help="与 --with-mlip-workers 联用：下载/校验 checkpoint，all 或逗号分隔 orb,sevenn,dpa,matris",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=int,
        default=60,
        help="checkpoint 下载单次请求超时秒数",
    )
    parser.add_argument(
        "--skip-checkpoint-download",
        action="store_true",
        help="与 --with-mlip-workers 联用：不自动下载 checkpoint，只在最后提示校验状态",
    )
    parser.add_argument(
        "--interoptimus-dir",
        type=Path,
        default=None,
        help="InterOptimus：源码目录或 .zip；默认在 ~/software 中自动查找文件名含 InterOptimus 的 .zip（不区分大小写）",
    )
    parser.add_argument(
        "--skip-mlip-conda",
        action="store_true",
        help="与 --with-mlip-workers 联用：跳过本机 pip install -e 与四个 MLIP conda，仅把 worker 写入项目 YAML",
    )
    parser.add_argument(
        "--matris-local",
        type=Path,
        default=None,
        help="MatRIS 本地路径：wheel/tar.gz/zip 或源码目录；未设时查环境变量、~/software 中含 matris 的 .zip（自动）、InterOptimus 旁 MatRIS",
    )

    args = parser.parse_args()

    if args.interactive:
        _apply_interactive_config(args)

    if args.skip_mlip_conda and not args.with_mlip_workers:
        parser.error("--skip-mlip-conda 需与 --with-mlip-workers 同时使用")
    if args.skip_checkpoint_download and not args.with_mlip_workers:
        parser.error("--skip-checkpoint-download 需与 --with-mlip-workers 同时使用")
    if not args.mongo_db:
        parser.error("--mongo-db is required unless provided through --interactive")

    mongo_host = args.mongo_host
    if mongo_host == "auto":
        mongo_host = detect_mongo_client_host(args.mongo_port)
        print(f"Mongo host（auto）: {mongo_host}")

    mongo_password = args.mongo_password
    if args.mongo_user and mongo_password is None:
        mongo_password = os.environ.get("INTEROPTIMUS_MONGO_PASSWORD")
    if args.mongo_user and mongo_password is None:
        mongo_password = getpass.getpass(f"MongoDB password for {args.mongo_user}: ")

    mongo = {
        "host": mongo_host,
        "port": args.mongo_port,
        "database": args.mongo_db,
        "username": args.mongo_user,
        "password": mongo_password or "",
    }

    _ensure_pymongo_for_mongo_check()
    mongo_errs = verify_mongodb_access(mongo, auth_source=args.mongo_auth_source)
    if mongo_errs:
        print("MongoDB 凭据或权限检测失败：", file=sys.stderr)
        for e in mongo_errs:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"MongoDB 检测通过：{mongo['host']}:{mongo['port']} / 库「{mongo['database']}」"
        "（ping、列举集合、探针读写）"
    )

    _ensure_yaml()

    jf_path = JFREMOTE_DIR / f"{args.project_name}.yaml"

    if args.verify_only:
        errs = verify_configuration(
            jobflow_yaml=JOBFLOW_YAML,
            jf_project_file=jf_path,
            project_name=args.project_name,
        )
        if errs:
            print("校验失败：")
            for e in errs:
                print(" -", e)
            raise SystemExit(1)
        print("校验通过。")
        return

    _pip_install(skip=args.skip_install, upgrade=args.upgrade_packages)

    gridfs_coll = args.standalone_jobflow_gridfs_collection
    if not gridfs_coll:
        gridfs_coll = f"{args.standalone_jobflow_collection}_blobs"

    jobflow_cfg = _build_jobflow_yaml_for_atomate2(
        mongo,
        docs_collection=args.standalone_jobflow_collection,
        gridfs_collection=gridfs_coll,
    )
    _write_jobflow_yaml(jobflow_cfg)

    atomate2_cfg: dict[str, Any] = {"VASP_CMD": args.vasp_cmd}
    if args.custodian_scratch_dir is not None:
        atomate2_cfg["CUSTODIAN_SCRATCH_DIR"] = str(args.custodian_scratch_dir.expanduser().resolve())
    _write_atomate2_yaml(atomate2_cfg)

    pre_run = _conda_worker_pre_run()
    if not pre_run:
        print(
            "警告: 当前进程未检测到 conda 环境（无 CONDA_PREFIX）。"
            "worker pre_run 将回退为 venv 的 source …/activate；"
            "建议先 conda activate 目标环境后重新运行本脚本。"
        )
        pre_run = _venv_pre_run()

    if not pre_run.strip():
        print("警告: worker pre_run 为空；请在 ~/.jfremote 项目 YAML 里手动设置 workers.*.pre_run")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    with_mlip_workers = args.with_mlip_workers
    checkpoint_specs = []
    if with_mlip_workers:
        from InterOptimus.checkpoints import (
            download_checkpoints,
            parse_checkpoint_selection,
            verify_checkpoints,
        )

        try:
            checkpoint_specs = parse_checkpoint_selection(args.checkpoint_models)
        except ValueError as exc:
            parser.error(str(exc))

        if not _conda_exe_or_none():
            _die("--with-mlip-workers 需要 conda 在 PATH 中（登录节点先 conda activate）")
        if not args.skip_mlip_conda:
            inter_dir, inter_src_msg = _resolve_interoptimus_install_root(args.interoptimus_dir)
            print(f"InterOptimus 来源: {inter_src_msg}")
            _pip_install_interoptimus_base_env(inter_dir)
            py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
            _setup_mlip_conda_envs(
                inter_dir,
                python_tag=py_tag,
                matris_local=args.matris_local,
            )
        else:
            print(
                "已跳过当前环境的 InterOptimus pip 与 MLIP conda（--skip-mlip-conda），仅写入 worker 配置"
            )
        if args.skip_checkpoint_download:
            print("已跳过 checkpoint 自动下载（--skip-checkpoint-download）；稍后将只校验现有文件。")
        else:
            print("正在检查/下载 MLIP checkpoints（若失败会打印手动下载地址与目标路径）…")
            download_checkpoints(checkpoint_specs, timeout=args.checkpoint_timeout)
        print("MLIP checkpoint 校验：")
        verify_checkpoints(checkpoint_specs)

    workers = _build_jfremote_workers(
        base_work_dir=args.work_dir,
        default_worker_name=args.jf_worker_name,
        default_pre_run=pre_run,
        with_mlip_workers=with_mlip_workers,
    )
    _write_jfremote_project(
        jf_path,
        name=args.project_name,
        mongo=mongo,
        workers=workers,
    )
    if with_mlip_workers:
        _warn_runner_restart_if_needed(args.project_name)

    if not args.skip_bashrc:
        exports = {
            "JOBFLOW_CONFIG_FILE": str(JOBFLOW_YAML.expanduser()),
            "ATOMATE2_CONFIG_FILE": str(ATOMATE2_YAML.expanduser()),
            "JFREMOTE_PROJECT": args.project_name,
            "JFREMOTE_PROJECTS_FOLDER": str(JFREMOTE_DIR.expanduser().resolve()),
        }
        _update_bashrc(exports)
        print("已更新 ~/.bashrc（请执行 source ~/.bashrc 或重新登录）")
    else:
        print("已跳过 ~/.bashrc；请自行导出 JOBFLOW_CONFIG_FILE / JFREMOTE_*")

    errs = verify_configuration(
        jobflow_yaml=JOBFLOW_YAML,
        jf_project_file=jf_path,
        project_name=args.project_name,
    )
    if errs:
        print("部署后自检未完全通过：")
        for e in errs:
            print(" -", e)
        raise SystemExit(1)

    print("\n部署完成。")
    print(f"  jobflow 配置: {JOBFLOW_YAML}")
    print(f"  atomate2 配置: {ATOMATE2_YAML}")
    print(
        f"  jobflow-remote 项目: {jf_path} （worker={args.jf_worker_name!r}, JFREMOTE_PROJECT={args.project_name}）"
    )
    if with_mlip_workers:
        print("  MLIP workers: 已在 jfremote 中配置 orb, dpa, matris, sevenn（conda 环境同名）")
        if checkpoint_specs:
            print("  MLIP checkpoints: 已执行下载尝试与校验；可随时运行 `itom checkpoints verify` 复查")
    print("首次使用 jobflow-remote 时如需初始化数据库，请在确认无重要数据后执行: jf admin reset")


if __name__ == "__main__":
    main()
