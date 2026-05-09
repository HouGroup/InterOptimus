<img width="1536" height="1024" alt="interoptimus-logo-3d-interface-titled" src="https://github.com/user-attachments/assets/b0d7eb6f-458b-4dc7-b67c-6bbe9da5adf8" />

# InterOptimus

> **晶体界面高通量搜索与优化平台**（MLIP 加速 + 可选 VASP + Jobflow / jobflow-remote 调度）
>
> Crystal **interface** search and optimization with MLIP acceleration, optional VASP, and **Jobflow** / **jobflow-remote** execution.

InterOptimus 是面向第一性原理 / 机器学习势函数（MLIP）的固体界面（薄膜 / 衬底）建模与高通量优化软件包。一次运行即可完成「晶格匹配 → 终止面筛选 → 单 / 双界面构建 → MLIP 全局极小化 → 可选 VASP DFT 精算 → 报告与结构导出」全流程，并提供命令行（`itom` / `interoptimus-simple`）、Python API、和浏览器图形界面（`interoptimus-web`）三种使用方式。

---

## 主要特性 · Highlights

- **晶格匹配与终止面筛选**：基于 `interfacemaster` 与 `pymatgen` 的多 (h k l) 高通量扫描，自动给出最小应变、最小重复单元的候选界面，并生成立体投影图（`stereographic.jpg` / `stereographic_interactive.html`）。
- **单 / 双界面构建**：支持单侧 (film / substrate) slab + 单界面、对称双界面 (sandwich)、CNID 层间位移采样。
- **MLIP 全局极小化**：内置 ORB-Models / SevenNet / DeepMD / MatRIS 多种势函数，统一 `MLIPCalculator` 接口；自动跑应变 + 位移 + 终止面三层网格，输出 `opt_results.pkl` 与 `selected_interfaces.csv`。
- **VASP DFT 精算（可选）**：通过 `atomate2` + `jobflow-remote` 在远端 HPC 节点提交 ISMEAR=2 / SCF / relax 复合工作流，并用统一的能量提取路径计算 γ_VASP。
- **配置驱动 (one-shot config)**：`interoptimus-simple -c your.json|yaml` 或 `run_simple_iomaker(config_dict)` 一行触发整套流程。
- **远端任务全生命周期管理**：`iomaker_status` 实时查询 jobflow-remote 进度；`iomaker_fetch_results` 自动汇总 MLIP / VASP 阶段产物到本地 `mlip_results/` 与 `vasp_results/`。
- **浏览器 GUI**：`interoptimus-web` 提供任务提交、状态轮询、结果可视化（3D 界面结构、γ 曲线、CSV 表格）和会话管理。

---

## 安装 · Installation

需要 Python 3.10 / 3.11 / 3.12。从源码安装：

```bash
git clone https://github.com/HouGroup/InterOptimus.git
cd InterOptimus
pip install -e .            # 仅核心工作流（晶格匹配 / Jobflow / 界面流水线）
pip install -e '.[web]'     # 额外启用浏览器 GUI（FastAPI + Plotly）
```

或从 PyPI 安装（发布后）：

```bash
pip install InterOptimus              # 核心
pip install "InterOptimus[web]"       # 包含 interoptimus-web
```

**MLIP 后端** 不通过 `pip install` 自动装入。两种方式：
- 推荐：`itom config --with-mlip-workers` 自动配置 jobflow-remote worker（含 conda env）。
- 手动：在 worker / 本地 conda env 内安装 `torch`、`orb-models`、`sevenn`、`deepmd-kit`；MatRIS 不在 PyPI，按上游说明从源码构建。

首次部署 MongoDB / jobflow-remote / POTCAR / MLIP checkpoint 的完整指南见 [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)。

---

## 命令行入口 · CLI Entry Points

| 命令 | 模块 | 用途 |
|---|---|---|
| `itom config [...]` | `InterOptimus.deploy_jobflow_stack` | 一键配置 MongoDB、Jobflow、jobflow-remote、atomate2、MLIP worker |
| `interoptimus-env [...]` | `InterOptimus.agents.server_env` | 在登录节点 / 集群节点上做环境健康检查 |
| `interoptimus-simple -c <config>` | `InterOptimus.agents.simple_iomaker` | 由 JSON / YAML 配置一键提交 / 本地运行 IOMaker 流水线 |
| `interoptimus-web [--host ...]` | `InterOptimus.web_app.cli` | 启动浏览器 GUI（默认 `http://localhost:8000`） |

---

## 快速上手 · Quick Start

### 1. CLI 一键提交

```bash
# 复制示例配置并按需修改路径 / 集群 / worker 设置
cp InterOptimus/agents/simple_iomaker.example.json my_run.json
$EDITOR my_run.json

# 在登录节点（jobflow-remote / MongoDB / conda 已配置就绪）上：
interoptimus-simple -c my_run.json
```

### 2. Python API

```python
from pathlib import Path
import json

from InterOptimus.agents.simple_iomaker import run_simple_iomaker
from InterOptimus.agents.remote_submit import iomaker_status, iomaker_fetch_results

with open("my_run.json") as f:
    result = run_simple_iomaker(json.load(f))

# 用 result['mlip_job_uuid'] 跟踪远端任务
status = iomaker_status(result)
print(status["progress"])

# 完成后拉取结果到本地
iomaker_fetch_results(dest_dir=Path("./out"), result=result)
```

更底层的程序化入口（已规范化的 `settings` 字典）：

```python
from InterOptimus.agents.iomaker_job import (
    LocalBuildConfig,
    execute_iomaker_from_settings,
    normalize_iomaker_settings_from_full_dict,
)
```

完整参数手册见 [`docs/simple_iomaker_parameters.md`](docs/simple_iomaker_parameters.md)。

### 3. 浏览器 GUI

```bash
pip install -e '.[web]'
interoptimus-web --host 0.0.0.0 --port 8000
```

打开 `http://<host>:8000/manage` 即可可视化提交、查看状态、拉取结果与 3D 界面结构。会话目录默认为 `~/.interoptimus/web_sessions/`，可通过 `INTEROPTIMUS_WEB_SESSIONS` 环境变量覆盖。

### 4. 示例 notebook

`examples/` 目录提供端到端可运行示例：

- `examples/01_run_simple_iomaker_local.ipynb` — 从 CIF → 配置 → 提交 → 查询 → 拉取的完整中文注释流程。
- `examples/film.cif` / `examples/substrate.cif` — 配套的薄膜 / 衬底测试结构（Li / NiS）。

---

## 仓库结构 · Repository Layout

```
InterOptimus/
├── InterOptimus/                  # Python 包源码
│   ├── __init__.py
│   ├── itworker.py                # InterfaceWorker：物理 + MLIP + 优化主类
│   ├── matching.py                # 晶格匹配、终止面、立体投影图
│   ├── jobflow.py                 # IOMaker Jobflow makers + opt_results 持久化
│   ├── mlip.py                    # MLIPCalculator 工厂 + checkpoint 解析
│   ├── tool.py                    # 通用工具：位移分析、JSON 序列化、可视化
│   ├── checkpoints.py             # MLIP 权重资源管理
│   ├── CNID.py                    # CNID 层间位移格点
│   ├── equi_term.py               # 等效终止面化简
│   ├── iomaker_minimal_export.py  # selected_interfaces.csv / 最小化结果导出
│   ├── result_bundle.py           # 结果产物打包
│   ├── deploy_jobflow_stack.py    # `itom config` 配置脚本
│   ├── doctor.py                  # 服务器侧 preflight 检查
│   ├── verify_installation.py     # 安装健康自检
│   ├── session_workflow.py        # 浏览器会话 → run_simple_iomaker 桥接
│   ├── viz_ase_iface.py           # ASE 界面可视化
│   ├── viz_runtime.py             # MLIP relaxation telemetry
│   ├── agents/
│   │   ├── simple_iomaker.py      # `interoptimus-simple` CLI + run_simple_iomaker
│   │   ├── iomaker_job.py         # BuildConfig / execute_iomaker_from_settings
│   │   ├── iomaker_core.py        # 共享小工具
│   │   ├── remote_submit.py       # 远端提交、状态轮询、结果拉取
│   │   ├── server_env.py          # `interoptimus-env`
│   │   └── simple_iomaker.example.json
│   ├── web_app/                   # 浏览器 GUI（FastAPI + Plotly）
│   │   ├── app.py                 # 路由 + 业务逻辑
│   │   ├── job_worker.py          # 后台任务执行
│   │   ├── task_store.py          # 任务 / 结果元数据扫描
│   │   ├── session_artifacts.py   # 会话内产物定位
│   │   ├── cluster_info.py        # jfremote 集群信息
│   │   ├── jfremote_workers.py    # worker 状态汇总
│   │   ├── cif_view.py            # 结构 → 3Dmol CIF 文本
│   │   ├── cli.py / __main__.py   # `interoptimus-web` 入口
│   │   └── templates/             # Jinja2 模板
│   └── tests/                     # 单元测试
├── docs/
│   ├── GETTING_STARTED.md         # 服务器端首次部署指南
│   ├── simple_iomaker_parameters.md  # 配置参数手册
│   └── branding/                  # logo 资源
├── examples/                      # 端到端示例 + 测试 CIF
├── setup.py                       # 包定义 / 依赖 / 命令行入口
├── pyproject.toml                 # 构建系统声明
├── MANIFEST.in                    # 打包附带模板 / JSON
├── LICENSE                        # MIT
└── README.md
```

---

## 系统依赖 · Requirements

- **Python**：≥ 3.10、< 3.13
- **核心栈**：`pymatgen`、`interfacemaster`、`atomate2`、`jobflow`、`jobflow-remote`、`qtoolkit`、`ase`、`scikit-learn`、`scipy`、`pandas`、`matplotlib`、`numpy`、`adjustText`、`tqdm`、`mp-api`、`pyyaml`（精确版本约束见 `setup.py`）
- **GUI 可选**：`fastapi`、`uvicorn`、`jinja2`、`plotly`、`python-multipart`
- **MLIP 可选**：`torch` ≥ 2、`orb-models`、`sevenn`、`deepmd-kit`、`MatRIS`（按需）
- **运行依赖**：MongoDB（jobflow / jobflow-remote 元数据）；如使用 VASP DFT，需在 worker 节点配置 VASP 与 POTCAR

---

## 开源协议 · License

MIT License — Copyright (c) 2024 Yaoshu Xie. 详见 [`LICENSE`](LICENSE)。

---

## 引用 · Citation

如在学术论文中使用 InterOptimus，请引用本软件：

```bibtex
@software{InterOptimus,
  author = {Xie, Yaoshu and contributors},
  title  = {InterOptimus: high-throughput crystal interface search and optimization},
  year   = {2024},
  url    = {https://github.com/HouGroup/InterOptimus}
}
```

---

## 作者 · Author

- **Yaoshu Xie** — [jasonxie@sz.tsinghua.edu.cn](mailto:jasonxie@sz.tsinghua.edu.cn)
- HouGroup, Tsinghua University Shenzhen International Graduate School

欢迎通过 GitHub Issue 报告 bug / 提出特性请求。
