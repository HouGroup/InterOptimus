<img width="1536" height="1024" alt="InterOptimus crystal interface logo" src="https://github.com/user-attachments/assets/b0d7eb6f-458b-4dc7-b67c-6bbe9da5adf8" />

# InterOptimus

> 面向晶体异质界面的高通量搜索与优化：晶格匹配、终止面筛选、MLIP 全局优化，以及可选的 VASP 精算。

InterOptimus 将薄膜/基底界面建模组织为可重复的 Jobflow 工作流。它既可以在本地运行，也可以通过 jobflow-remote 提交到 Slurm 集群，并提供命令行、Python API、Jupyter notebook 和浏览器界面。

## 核心功能

- 基于 Zur–McGill（ZSL）方法搜索多组 Miller 面和面内超胞匹配。
- 构建带真空的单界面或 c 方向周期性的双界面。
- 枚举并化简薄膜/基底终止面，支持非极性终止面筛选。
- 使用 CNID 位移、Bayesian Optimization 和可选的 MLIP 梯度下降搜索界面配准。
- 支持 ORB、DPA、MatRIS 和 SevenNet 四类 MLIP 后端。
- 可选执行 VASP relax/static、应变能修正、偶极修正和有界 VASP-GD。
- 通过 jobflow-remote 管理 MLIP 与 VASP 子任务，并统一查询、下载和审计结果。
- 导出优化结构、能量汇总、匹配数据和静态/交互式立体投影图。

## 运行要求

- Python `>=3.11,<3.13`
- Linux/HPC 是远程 Slurm、MLIP GPU 和 VASP 工作流的主要运行环境
- MongoDB：保存 Jobflow/jobflow-remote 状态
- VASP 工作流：可用的 VASP 可执行文件和 POTCAR 库
- GPU MLIP：与集群驱动兼容的 PyTorch/CUDA 和对应模型包

核心依赖的精确上下限位于 [`setup.py`](setup.py)。当前代码显式适配：

- `pymatgen==2026.5.4`
- `pymatgen-core==2026.8.30`
- `atomate2>=0.1.5,<0.2`
- `emmet-core>=0.87.2,<0.88`
- `jobflow>=0.3.1,<0.4`
- `jobflow-remote>=1.0,<1.1`

## 安装

从源码安装核心工作流：

```bash
git clone https://github.com/HouGroup/InterOptimus.git
cd InterOptimus
python -m pip install -e .
```

安装浏览器界面的可选依赖：

```bash
python -m pip install -e '.[web]'
```

MLIP 后端不会随核心包自动安装。推荐在集群登录节点使用部署命令创建独立 worker 环境：

```bash
itom config --interactive --with-mlip-workers
```

也可以手动安装所需的 PyTorch、`orb-models`、`deepmd-kit`、SevenNet 或 MatRIS。不同后端最好使用独立 Conda 环境，避免 CUDA 和依赖冲突。

完整的服务器准备说明见 [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)。

## 环境配置与检查

查看可用管理命令：

```bash
itom --help
```

交互式配置 MongoDB、Jobflow、jobflow-remote 和 atomate2：

```bash
itom config --interactive
```

同时创建 `orb`、`dpa`、`matris` 和 `sevenn` worker：

```bash
itom config --interactive --with-mlip-workers
```

检查现有配置：

```bash
itom doctor --project-name std
interoptimus-env
jf project check --errors
```

列出、下载或验证 checkpoint：

```bash
itom checkpoints list all
itom checkpoints download orb,sevenn,dpa,matris
itom checkpoints verify all
```

默认 checkpoint 目录是 `~/.cache/InterOptimus/checkpoints/`，可通过 `INTEROPTIMUS_CHECKPOINT_DIR` 修改。运行时解析器可识别同一模型的多种兼容文件名。MatRIS 还会按其上游约定使用 `~/.cache/matris/`。

## 快速开始

推荐使用下面四个顶层字段：

- `workflow_name`：任务名
- `IO_workflow_config`：结构、匹配、优化和 VASP 参数
- `execution`：`local` 或 `server`
- `cluster`：MLIP/VASP worker 与 Slurm 资源

```python
config = {
    "workflow_name": "si_ge_111",
    "execution": "server",
    "IO_workflow_config": {
        "cost_preset": "low",
        "bulk_cifs": {
            "film_cif": "/path/to/si.cif",
            "substrate_cif": "/path/to/ge.cif",
        },
        "lattice_matching_settings": {
            "max_area": 35,
            "max_length_tol": 0.05,
            "max_angle_tol": 0.05,
            "film_max_miller": 1,
            "substrate_max_miller": 1,
            "film_millers": [[1, 1, 1]],
            "substrate_millers": [[1, 1, 1]],
        },
        "structure_settings": {
            "termination_ftol": 0.15,
            "film_thickness": 6,
            "substrate_thickness": 6,
            "double_interface": False,
            "vacuum_over_film": 8,
        },
        "optimization_settings": {
            "calc": "orb-models",
            "device": "cuda",
            "fmax": 0.2,
            "steps": 100,
            "discut": 0.6,
            "n_calls_density": 1,
            "z_range": [1.5, 2.5],
            "strain_E_correction": False,
        },
        "vasp_settings": {
            "do_vasp": False,
            "vasp_pair_selection": "each_match_lowest",
        },
    },
    "cluster": {
        "mlip": {
            "mlip_worker": "orb",
            "mlip_project": "std",
            "slurm_partition": "gpu",
            "cpus_per_gpu": 5,
        },
        "vasp": {},
    },
}
```

`cost_preset` 可取 `low`、`medium` 或 `high`。显式写入的参数会覆盖对应预设。完整字段和兼容的旧格式见 [`docs/simple_iomaker_parameters.md`](docs/simple_iomaker_parameters.md)。

### 命令行运行

从示例配置开始：

```bash
cp InterOptimus/agents/simple_iomaker.example.json my_run.json
interoptimus-simple -c my_run.json
```

配置文件支持 JSON 和 YAML。

### Python API

```python
from InterOptimus.agents.simple_iomaker import run_simple_iomaker
from InterOptimus.agents.remote_submit import iomaker_fetch_results, iomaker_status

result = run_simple_iomaker(config)
job_uuid = result["mlip_job_uuid"]

# 可传提交结果，也可在重启 notebook 后直接传 UUID。
status = iomaker_status(job_uuid)

if status["is_finished"]:
    bundle = iomaker_fetch_results(
        dest_dir="./si_ge_results",
        ref=job_uuid,
    )
```

`execution="local"` 会在当前 Python 环境执行 MLIP 工作流；该环境必须已经安装所选 MLIP 后端。当前本地执行路径会强制关闭 VASP，因此 VASP 任务应使用 `execution="server"`，由 jobflow-remote 提交。

## 界面与优化选项

常用建模分支：

- `double_interface=False`：带真空层的单界面。
- `double_interface=True`：c 方向周期性的双界面。
- `non_polar_film_termination` / `non_polar_substrate_termination`：使用 pymatgen 的 slab 极性判据筛选终止面。
- `strain_E_correction=True`：计算并扣除应变能贡献。
- `do_mlip_gd=True`：在 MLIP 优化阶段增加梯度下降。
- `gd_max_steps` 和 `gd_max_displacement`：限制 MLIP-GD 的步数和单步位移。

### Match 与 termination 对称性去重

Match 去重以原始晶胞中的晶格方向为依据：

- 先把 pymatgen 的 ZSL slab display vectors 还原到 crystal frame。
- 将向量转换为整数晶格方向并约去公因数，因此向量长度、界面面积和应变不参与等价类定义。
- Film 和 substrate 各自必须存在一个点群旋转，同时映射完整的两根面内基矢。
- 两侧必须采用相同的向量交换/符号重标记，以保持外延对应关系。
- 所有候选先按 `von_mises_strain` 升序排列，所以每个等价类保留最小应变 match。

Termination 去重会区分边界条件：带真空的单界面保留有向表面法线，周期性双界面才允许法线反向等价。代表 termination 通过实际 film/substrate slab shift 映射到 pymatgen 标签，不依赖标签列表顺序。

Film 和 substrate 可使用不同的非极性筛选参数：

```python
config["IO_workflow_config"]["structure_settings"].update({
    "non_polar_film_termination": {
        "oxidation_states": {"Zn": 2, "O": -2},
        "tol_dipole_per_unit_area": 1e-4,
    },
    "non_polar_substrate_termination": {
        "oxidation_states": {"Ga": 3, "N": -3},
        "tol_dipole_per_unit_area": 2e-3,
    },
})
```

Bayesian Optimization 会把原子碰撞结构视为高能惩罚项；若所有采样点碰撞、能量非有限或所有弛豫失败，工作流会给出明确错误，而不是继续提交无效 VASP 任务。

## VASP 工作流

启用 VASP：

```python
config["IO_workflow_config"]["vasp_settings"].update({
    "do_vasp": True,
    "vasp_pair_selection": "each_match_lowest",
    "relax_user_incar_settings": {"ALGO": "Fast"},
    "static_user_incar_settings": {"ALGO": "Fast"},
    "relax_user_potcar_functional": "PBE_54",
    "static_user_potcar_functional": "PBE_54",
})
```

VASP 资源写在 `cluster.vasp`：

```python
config["cluster"]["vasp"] = {
    "vasp_worker": "default",
    "vasp_slurm_partition": "compute",
    "vasp_nodes": 1,
    "vasp_processes_per_node": 48,
    "vasp_pre_run": "module load VASP/6.5.1",
}
```

特殊分支：

- `vasp_dipole_correction=True`：为单界面启用偶极修正。
- `do_vasp_gd=True`：执行 VASP 刚体配准梯度下降。
- `vasp_gd_kwargs={"max_steps": ..., "max_displacement": ...}`：限制 VASP-GD，建议保留有限步数和位移上限。
- `vasp_pair_selection` 支持 `each_match_lowest`、`each_plane_lowest` 和 `all_optimized`。

使用 VASP 前必须设置 `PMG_VASP_PSP_DIR`，并确保 worker 能加载 VASP、atomate2 和同一版本的 InterOptimus。

## 结果与监控

远程任务可以通过提交返回值或 MLIP 根 UUID 查询：

```python
from InterOptimus.agents.remote_submit import (
    iomaker_fetch_results,
    iomaker_status,
    query_interoptimus_task_progress,
)

status = iomaker_status(job_uuid)
progress = query_interoptimus_task_progress(job_uuid)
results = iomaker_fetch_results(dest_dir="./results", ref=job_uuid)
```

根据启用的分支，结果包可包含：

- `opt_results.pkl`
- `selected_interfaces.csv`
- `pairs_best_it/`
- `mlip_results/` 和 `vasp_results/`
- `stereographic.jpg` 与 `stereographic_interactive.html`
- 面积/应变数据和界面能报告
- VASP relax/static 结构及汇总报告

仓库还提供 [`scripts/audit_interface_workflow_results.py`](scripts/audit_interface_workflow_results.py)，用于检查原子分区、晶格有限性、正体积、最小原子距离、有限能量及 VASP 输出完整性，并支持 `.gz` 压缩的 `vasprun.xml`、`OUTCAR` 和 `INCAR`。

## 浏览器界面

```bash
python -m pip install -e '.[web]'
interoptimus-web --host 0.0.0.0 --port 8765
```

默认端口为 `8765`。浏览器界面提供配置、任务提交、状态查看、结果文件和结构可视化。Web 会话默认保存在 `~/.interoptimus/web_sessions/`，可通过 `INTEROPTIMUS_WEB_SESSIONS` 覆盖。

## 示例 notebook

[`examples/`](examples/) 中包含：

- `01_run_simple_iomaker_local.ipynb`：基础配置、提交、状态查询和结果拉取。
- `02_remote_mlip_workflows.ipynb`：ORB、DPA、MatRIS 和 SevenNet 远程工作流。
- `03_remote_vasp_standard_and_double.ipynb`：标准单界面和双界面 VASP。
- `04_remote_vasp_dipole_and_gd.ipynb`：偶极修正和有界 VASP-GD。
- `05_monitor_and_audit_remote_workflows.ipynb`：统一监控、下载和结果审计。

远程示例默认 `SUBMIT=False`，检查 worker、checkpoint、分区、POTCAR 和 VASP 模块后再显式开启提交。

## 命令行入口

- `itom config`：部署或检查 Jobflow、jobflow-remote、atomate2 和 MongoDB。
- `itom checkpoints`：列出、下载和验证 MLIP checkpoint。
- `itom doctor`：检查项目、MongoDB、worker、runner 和 checkpoint 状态。
- `interoptimus-env`：输出服务器环境与 worker 提示。
- `interoptimus-simple -c CONFIG`：由 JSON/YAML 配置运行或提交工作流。
- `interoptimus-web`：启动 FastAPI 浏览器界面。

各命令均可使用 `--help` 查看当前代码支持的参数。

## 仓库结构

```text
InterOptimus/
├── InterOptimus/
│   ├── itworker.py              # 界面构建与 MLIP 优化
│   ├── matching.py              # 晶格匹配和匹配化简
│   ├── pymatgen_compat.py       # 新版 pymatgen 兼容层
│   ├── jobflow.py               # IOMaker 与 VASP 工作流
│   ├── mlip.py                  # MLIP 统一入口和 checkpoint 解析
│   ├── checkpoints.py           # checkpoint CLI
│   ├── deploy_jobflow_stack.py  # 集群配置
│   ├── doctor.py                # 环境诊断
│   ├── agents/                  # 配置、提交、状态和结果 API
│   ├── web_app/                 # FastAPI Web UI
│   └── tests/
├── docs/
├── examples/
├── scripts/
├── setup.py
└── README.md
```

## 测试

在安装完整核心依赖的环境中运行：

```bash
python -m unittest discover -s InterOptimus/tests -p 'test_*.py'
```

测试覆盖配置兼容、checkpoint 解析、pymatgen 晶体学兼容、三方与 nonsymmorphic 晶系、match/termination 对称性去重、工作流失败边界和远程状态汇总。真实 MLIP/VASP 运行仍取决于模型、GPU、MongoDB、Slurm、POTCAR 和 VASP 环境。

## 引用

如果 InterOptimus 用于学术工作，请引用：

> Yaoshu Xie, Lu Jiang, Tingzheng Hou, *et al.*
> **InterOptimus: An AI-assisted robust workflow for screening ground-state heterogeneous interface structures in lithium batteries.**
> *Journal of Energy Chemistry* **106**, 631–641 (2025).
> https://doi.org/10.1016/j.jechem.2025.03.007

```bibtex
@article{InterOptimus2025,
  author  = {Xie, Yaoshu and Jiang, Lu and Hou, Tingzheng and others},
  title   = {{InterOptimus}: An {AI}-assisted robust workflow for screening
             ground-state heterogeneous interface structures in lithium batteries},
  journal = {Journal of Energy Chemistry},
  volume  = {106},
  pages   = {631--641},
  year    = {2025},
  doi     = {10.1016/j.jechem.2025.03.007}
}
```

## License

MIT License. Copyright (c) 2024 Yaoshu Xie.

问题与功能建议请通过 GitHub Issues 提交。
