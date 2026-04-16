# `simple_iomaker` 配置参数说明

适用于命令 **`interoptimus-simple -c <config.json|yaml>`** 与 Python **`InterOptimus.agents.simple_iomaker.run_simple_iomaker`**。  
配置为 JSON 或 YAML（YAML 需安装 `pyyaml`）。

---

## 1. 总体结构（以代码为准）

根对象**必须**包含一个完整的 **`settings`**：与完整 IOMaker 参数字典同形（见 **`InterOptimus.agents.iomaker_job.normalize_iomaker_settings_from_full_dict`**），至少包含：

`name`, `mode`, `inputs`, `lattice_matching_settings`, `structure_settings`, `optimization_settings`, `global_minimization_settings`

以及你需要的其它键，例如 **`bulk`**（本地 CIF 路径）、**`do_vasp`**、**`do_vasp_gd`**、**`relax_user_*`** / **`static_user_*`**（与 `InterfaceWorker.patch_jobflow_jobs` / pymatgen input set 一致）、**`vasp_relax_settings`** / **`vasp_static_settings`**（旧版桶，仅当未设 `user_*` 时生效）等。

**计算相关参数只能写在 `settings` 里**，不要在根上再写一份（见下文「禁止的顶层键」）。

### `settings.bulk`（本地 CIF 路径）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `film_cif` | string | `film.cif` | 薄膜 CIF 路径（相对或绝对）。 |
| `substrate_cif` | string | `substrate.cif` | 基底 CIF 路径。 |

归一化由 **`normalize_iomaker_settings_from_full_dict`** 填充默认值；加载结构时覆盖 **`BuildConfig`** 中的 `film_cif` / `substrate_cif`。

可选分组仅两种：

| 分组 | 作用 |
|------|------|
| **`execution`** | 运行模式、输出 Flow 文件名、是否打印参数 |
| **`cluster`** | 推荐 **`mlip`**（InterOptimus / submit_flow / MLIP Slurm）、**`vasp`**（VASP 子 Flow）；仍支持旧版**单层** `cluster`。 |

加载时由 **`bundled_config_to_flat`** 把 `execution` / `cluster` 展开为扁平键，再与根上同名字段合并；**同一键不能既在分组里又在根上重复出现**。

仓库示例：**`InterOptimus/agents/simple_iomaker.example.json`**。

---

## 2. 已移除的旧写法（会报错）

若根对象出现以下**任一分组**，将直接报错，需改为「只保留 `settings` + `execution` + `cluster`」：

- `structure`, `task`, `physics`, `vasp`, `overrides`

不再支持：旧版「无 `settings`、用 `physics` + `interface` 合并 tutorial 预设」、**`overrides` 深度合并**、以及把 `max_area` / `do_vasp` 等放在根上的扁平捷径。

---

## 3. `execution`

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `run` | string | `server` | `local`：本机跑 Flow；`server`：在当前机器上 `submit_flow`（登录节点场景）。 |
| `output_flow_json` | string | `io_flow.json` | 输出 Flow JSON 文件名。 |
| `print_settings` | bool | `true` | 是否在终端打印参数。 |

---

## 4. `cluster`

推荐使用**嵌套**结构（三者都可省略；`vasp` 在 `do_vasp: false` 时常为空对象）：

```json
"cluster": {
  "mlip": { ... },
  "vasp": { ... }
}
```

以下为各小节**允许的键**（与源码 `_CLUSTER_SCHEMA_*` 一致）。也支持**旧版**：所有键平铺在同一个 `cluster` 对象里。

### 4.1 `shared`（可选、可省略）

历史分组名；**当前不允许任何子键**（可写空对象 `{}` 或整段省略）。CIF 路径一律在 **`settings.bulk`**；**`settings.inputs.type`** 固定为 **`local_cif`**。

### 4.2 `mlip`：InterOptimus 主流程 / submit_flow / MLIP Slurm（`run` 为 `server` 时常用）

MLIP 作业的 **`QResources.nodes` 固定为 1**，**`processes_per_node` 固定为 1**。资源模板由 **`settings.optimization_settings.device`** 决定：

| `device` | 行为 |
|----------|------|
| **`cuda` / `gpu`** | 使用 GPU 模板：`gpus_per_job`、`partition`、`#SBATCH --cpus-per-gpu=…`（与 `mlip_scheduler_kwargs` 合并）。 |
| **`cpu`**（默认） | 使用 CPU 模板：`partition` 与 **`mlip_cpus_per_task`**（单 Slurm task、多核，常用于内存），与 `mlip_scheduler_kwargs` 合并。 |

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `slurm_partition` | string | `interactive` 或 **`INTEROPTIMUS_SLURM_PARTITION`** | MLIP 作业 Slurm 分区。 |
| `server_run_parent` | string，可选 | 无 | 运行目录父路径；不设则在当前工作目录下建任务子目录。 |
| `server_pre_cmd` | string | **`INTEROPTIMUS_SERVER_PRE_CMD`** 或空 | 提交前 shell（如 `conda activate …`）。 |
| `server_python` | string | `python` | 执行 `submit_flow` 的 Python。 |
| `server_jf_bin` | string | `jf` | jobflow CLI 可执行文件。 |
| `cpus_per_gpu` | int | `1` | 仅 GPU 路径：`--cpus-per-gpu`。 |
| `mlip_cpus_per_task` | int | `1` | **仅 MLIP / submit_flow 根作业**（CPU 路径）：Slurm 单 task 的核数。 |
| `cpus_per_task` | int | — | **已弃用别名**，等同于 `mlip_cpus_per_task`（仅为兼容旧 JSON）。 |
| `gpus_per_job` | int | `1` | 仅 GPU 路径。 |
| `mlip_scheduler_kwargs` | object | 无 | 与 MLIP 的 `scheduler_kwargs` 合并。 |
| `mlip_worker` | string | `std_worker` | jobflow-remote worker（InterOptimus / MLIP 段）。 |
| `mlip_project` | string | `std` | `submit_flow` 的 project。 |

### 4.3 `vasp`：VASP 子 Flow 资源（仅 **`settings.do_vasp: true`** 时常填）

下列键只描述 **集群上 VASP 怎么跑**（分区、核数、加载模块等），**不是** pymatgen 的 INCAR/KPOINTS；后者放在 **`settings`** 的 **`relax_user_*`** / **`static_user_*`** 等。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `vasp_worker` | string | `std_worker` | VASP 子 Flow 的 jobflow-remote worker。 |
| `vasp_slurm_partition` | string | 同 `slurm_partition` | VASP 作业分区。 |
| `vasp_nodes` | int | `1` | 节点数。 |
| `vasp_processes_per_node` | int | `4` | 每节点 MPI 进程数；**默认不在 Slurm 脚本里写 `cpus_per_task`**，并行度由 `nodes` × `processes_per_node` 表达。若集群仍需 `--cpus-per-task` 等，请放进 `vasp_scheduler_kwargs`。 |
| `vasp_scheduler_kwargs` | object | 无 | VASP 作业 Slurm 附加参数（与上面 `partition` 合并）。 |
| `vasp_pre_run` | string | 无 | 若未设 **`vasp_exec_config`**，可用来生成 `exec_config.pre_run`（如 `module load VASP/…`）。 |
| `vasp_exec_config` | object | 无 | 传给 jobflow-remote 的 `exec_config`（如 `{"pre_run": "…"}`）；与 `vasp_pre_run` 二选一或按源码合并逻辑。 |

---

## 5. `settings` 中与 VASP 相关的提醒

- **`user_kpoints_settings`** 类字段应使用 pymatgen **`DictSet`** 支持的键（如 **`reciprocal_density`**、**`grid_density`**），不要误用不支持的 **`kpoints`** 网格列表；详见 pymatgen `pymatgen.io.vasp.sets` 中 `DictSet.kpoints` 的实现说明。
- **`do_vasp_gd`**：写在 **`settings`** 中；`IOMaker` 映射为 `patch_jobflow_jobs(..., do_dft_gd=...)`。

---

## 6. 禁止出现在根上的键（与 `settings` 重复时）

若根对象里已有 **`settings`**，则下列键**不得**再出现在根上（非 `null`），应全部放进 `settings` 或对应子字典：

`name`, `interface`, `mode`, `inputs`, `max_area`, `film_thickness`, `substrate_thickness`, `vacuum_over_film`, `fmax`, `steps`, `device`, `z_range`, `n_calls_density`, `do_vasp`, `do_vasp_gd`, `film_mp_id`, `substrate_mp_id`, `mlip_calc`, `ckpt_path`, `do_mlip_gd`, `vasp_relax_settings`, `vasp_static_settings`, **`bulk`**（应写在 **`settings.bulk`**），以及全部 **`relax_user_*` / `static_user_*` / `vasp_gd_kwargs` / `vasp_dipole_correction`**。

根上允许的**基础设施**键名集合见源码 **`InterOptimus/agents.simple_iomaker._ALLOWED_INFRA_KEYS`**（与展开后的 `execution`/`cluster` 字段一致）。

---

## 7. 环境变量

| 变量 | 作用 |
|------|------|
| `INTEROPTIMUS_SLURM_PARTITION` | 未指定 `slurm_partition` 时的默认分区。 |
| `INTEROPTIMUS_SERVER_PRE_CMD` | 未指定 `server_pre_cmd` 时的默认提交前命令。 |
| `INTEROPTIMUS_TASK_REGISTRY_DIR` | 可选；任务 serial 注册表目录（默认 `~/.interoptimus`）。 |
| `INTEROPTIMUS_CHECKPOINT_DIR` | 可选；MLIP checkpoint 默认目录（未设置时见 InterOptimus 内解析逻辑）。 |

---

## 8. 相关命令

| 命令 | 说明 |
|------|------|
| `interoptimus-simple -c <config>` | 从 JSON/YAML 构建 Flow 并按 `run` 执行。 |
| `interoptimus-env` | 检查本机 Python、`jf`、worker 等。 |

---

## 9. 源码位置（与本文不一致时以代码为准）

- 分组、白名单、校验：`InterOptimus/agents/simple_iomaker.py`（**`bundled_config_to_flat`**、**`_BUNDLE_SCHEMA`**、**`_validate_simple_config`**、**`_ALLOWED_INFRA_KEYS`**、**`_FORBID_FLAT_WITH_SETTINGS`**）
- 归一化 `settings`：`InterOptimus/agents/iomaker_job.py`（**`normalize_iomaker_settings_from_full_dict`**）
- 构建 `BuildConfig` 与执行：`build_config_from_simple_dict`、`execute_iomaker_from_settings`（同模块）
