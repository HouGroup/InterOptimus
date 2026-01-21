# Remote Submission Guide

本指南介绍如何使用 `BuildConfig` 自动将生成的 jobflow 任务提交到远程服务器。

## 功能概述

`BuildConfig` 类现在支持以下功能：

1. **远程服务器提交**：自动将生成的 `io_flow.json` 提交到远程服务器
2. **Jobflow 资源配置**：配置 MLIP 和 VASP 任务的资源和 worker
3. **自动验证**：如果 `do_vasp=True`，自动验证必需的 VASP 参数

## 基本用法

### 1. 仅生成本地 JSON（不提交）

```python
from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt

cfg = BuildConfig(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/",
    # submit_to_remote=False (默认)
)

result = build_iomaker_flow_from_prompt("Create Si on SiC interface", cfg)
print(f"Flow JSON: {result['flow_json_path']}")
```

### 2. 生成并自动提交到远程服务器

```python
from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt
from qtoolkit.core.data_objects import QResources

def mlip_resources():
    return QResources(
        nodes=1,
        processes_per_node=1,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 10},
    )

cfg = BuildConfig(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/",
    
    # 启用远程提交
    submit_to_remote=True,
    remote_host="xys@10.103.65.21",
    remote_identity_file="~/.ssh/id_ed25519",
    remote_workdir="~/io_runs/si_sic_run1",
    remote_python="python",
    remote_pre_cmd="source ~/.bashrc && conda activate atomate2",
    remote_passphrase="your_passphrase",  # 可选，如果使用 ssh-agent 可以省略
    
    # Jobflow 配置
    mlip_resources=mlip_resources,
    mlip_worker="std_worker",
)

result = build_iomaker_flow_from_prompt("Create Si on SiC interface", cfg)

# 检查提交结果
if "remote_submission" in result:
    submit_result = result["remote_submission"]
    if submit_result["success"]:
        print(f"Job ID: {submit_result.get('job_id')}")
    else:
        print(f"Submission failed: {submit_result.get('error')}")
```

### 3. 使用 VASP 计算（必须提供 VASP 资源）

```python
from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt
from qtoolkit.core.data_objects import QResources

def mlip_resources():
    return QResources(
        nodes=1,
        processes_per_node=1,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 10},
    )

def vasp_resources():
    return QResources(
        nodes=1,
        processes_per_node=4,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 40},
    )

cfg = BuildConfig(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/",
    
    submit_to_remote=True,
    remote_host="xys@10.103.65.21",
    remote_identity_file="~/.ssh/id_ed25519",
    remote_workdir="~/io_runs/si_sic_run1",
    remote_passphrase="your_passphrase",
    
    # MLIP 配置
    mlip_resources=mlip_resources,
    mlip_worker="std_worker",
    
    # VASP 配置（必需，如果 LLM 输出中 do_vasp=True）
    vasp_resources=vasp_resources,  # 必需！
    vasp_worker="std_worker",  # 必需！
)

# LLM 会自动检测是否需要 VASP，如果 do_vasp=True，会验证 vasp_resources 和 vasp_worker
result = build_iomaker_flow_from_prompt(
    "Create Si on SiC interface with vacuum, do VASP calculation",
    cfg
)
```

## BuildConfig 参数说明

### LLM 配置（必需）

- `api_key`: OpenAI API 密钥
- `base_url`: API 基础 URL
- `model`: 模型名称（默认: "gpt-3.5-turbo"）

### 远程提交配置（可选）

- `submit_to_remote`: 是否自动提交到远程服务器（默认: `False`）
- `remote_host`: 远程服务器地址，格式 `"user@hostname"`（必需如果 `submit_to_remote=True`）
- `remote_identity_file`: SSH 私钥文件路径（必需如果 `submit_to_remote=True`）
- `remote_workdir`: 远程工作目录（必需如果 `submit_to_remote=True`）
- `remote_python`: 远程服务器上的 Python 命令（默认: `"python"`）
- `remote_pre_cmd`: 执行 Python 脚本前的命令，如激活 conda 环境（可选）
- `remote_passphrase`: SSH 密钥的 passphrase（可选，如果使用 ssh-agent 可以省略）
- `remote_use_paramiko`: 是否使用 paramiko 而不是 pexpect（默认: `False`）
- `remote_debug`: 是否启用调试输出（默认: `False`）

### Jobflow 资源配置（可选）

- `mlip_resources`: 返回 `QResources` 的函数，用于 MLIP 任务（可选）
- `vasp_resources`: 返回 `QResources` 的函数，用于 VASP 任务（**必需**如果 `do_vasp=True`）
- `mlip_worker`: MLIP 任务的 worker 名称（默认: `"std_worker"`）
- `vasp_worker`: VASP 任务的 worker 名称（默认: `"std_worker"`，**必需**如果 `do_vasp=True`）

## 验证规则

1. **如果 `submit_to_remote=True`**：
   - 必须提供 `remote_host`
   - 必须提供 `remote_identity_file`
   - 必须提供 `remote_workdir`

2. **如果 `do_vasp=True`（由 LLM 自动检测）**：
   - 必须提供 `vasp_resources`（不能为 `None`）
   - 必须提供 `vasp_worker`（不能为空字符串）

## 返回值

`build_iomaker_flow_from_prompt` 返回一个字典，包含：

- `flow_json_path`: 生成的 Flow JSON 文件路径
- `flow_dict`: Flow 字典对象
- `settings`: LLM 输出的设置
- `structures_meta`: 结构元数据
- `remote_submission`: 如果 `submit_to_remote=True`，包含提交结果
  - `success`: 是否成功
  - `job_id`: 任务 ID（如果成功）
  - `stdout`: 标准输出
  - `stderr`: 标准错误
  - `error`: 错误类型

## 示例脚本

参考 `InterOptimus/agents/demos/demo_llm_iomaker_with_remote.py` 查看完整示例。

## 故障排除

### 1. SSH 连接失败

- 检查 `remote_host` 格式是否正确（`user@hostname`）
- 检查 `remote_identity_file` 路径是否正确
- 如果使用 passphrase，确保 `remote_passphrase` 正确
- 或者使用 `ssh-agent` 管理密钥

### 2. VASP 验证失败

如果看到错误 "vasp_resources is required when do_vasp=True"：

- 确保在 `BuildConfig` 中提供了 `vasp_resources`（函数，不是 `None`）
- 确保提供了 `vasp_worker`（非空字符串）

### 3. 远程目录不存在

- 确保 `remote_workdir` 路径正确
- 代码会自动创建目录，但如果权限不足可能会失败

## 安全建议

⚠️ **不要在代码中硬编码 passphrase！**

推荐做法：

1. **使用环境变量**：
```python
import os
cfg = BuildConfig(
    # ...
    remote_passphrase=os.getenv("SSH_PASSPHRASE"),
)
```

2. **使用 ssh-agent**（最安全）：
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
# 然后省略 remote_passphrase 参数
```

3. **使用配置文件**（不要提交到 git）：
```python
import json
with open("~/.ssh_config.json") as f:
    config = json.load(f)
    cfg = BuildConfig(
        # ...
        remote_passphrase=config["passphrase"],
    )
```
