# Remote Jobflow Submission with Passphrase Support

这个模块提供了通过SSH提交jobflow任务到远程服务器的功能，支持自动输入SSH密钥的passphrase。

## 安装依赖

```bash
# 方式1: 使用pexpect (推荐)
pip install pexpect

# 方式2: 使用paramiko (替代方案)
pip install paramiko
```

或者安装所有依赖：
```bash
pip install -r InterOptimus/agents/requirements_remote.txt
```

## 使用方法

### 基本用法

```python
from InterOptimus.agents.remote_submit import submit_io_flow_via_ssh

# 提交任务，自动输入passphrase
result = submit_io_flow_via_ssh(
    host="xys@10.103.65.21",
    identity_file="~/.ssh/id_ed25519",
    local_io_flow_json="io_flow.json",
    remote_workdir="~/io_runs/si_sic_run1",
    remote_python="python",
    pre_cmd="source ~/.bashrc && conda activate atomate2",
    passphrase="your_passphrase_here",  # 直接传入passphrase
)

print(f"提交成功: {result['success']}")
print(f"任务ID: {result.get('job_id', 'N/A')}")
print(f"输出:\n{result['stdout']}")
```

### 使用paramiko方式

```python
result = submit_io_flow_via_ssh(
    host="xys@10.103.65.21",
    identity_file="~/.ssh/id_ed25519",
    local_io_flow_json="io_flow.json",
    remote_workdir="~/io_runs/si_sic_run2",
    passphrase="your_passphrase_here",
    use_paramiko=True,  # 使用paramiko而不是pexpect
)
```

### 检查任务状态

```python
from InterOptimus.agents.remote_submit import check_job_status_via_ssh

status = check_job_status_via_ssh(
    host="xys@10.103.65.21",
    identity_file="~/.ssh/id_ed25519",
    job_id="12345",  # 从submit结果中获取
    passphrase="your_passphrase_here",
)

print(f"状态: {status['stdout']}")
```

## 参数说明

### `submit_io_flow_via_ssh`

- `host`: SSH主机地址，格式 `"user@hostname"` 或 `"hostname"`
- `identity_file`: SSH私钥文件路径（默认: `~/.ssh/id_ed25519`）
- `local_io_flow_json`: 本地jobflow JSON文件路径（默认: `"io_flow.json"`）
- `remote_workdir`: 远程工作目录（会自动创建）
- `remote_python`: 远程服务器上的Python命令（默认: `"python"`）
- `pre_cmd`: 执行Python脚本前的命令（如激活conda环境）
- `passphrase`: SSH密钥的passphrase（如果为None，会尝试使用ssh-agent或提示输入）
- `use_paramiko`: 是否使用paramiko库（默认False，使用pexpect）

### `check_job_status_via_ssh`

- `host`: SSH主机地址
- `identity_file`: SSH私钥文件路径
- `job_id`: 任务ID（可选，如果提供则查询特定任务）
- `remote_workdir`: 远程工作目录（可选，如果提供则查询该目录下的任务）
- `passphrase`: SSH密钥的passphrase
- `use_paramiko`: 是否使用paramiko库

## 两种方式对比

### pexpect方式（默认）
- ✅ 兼容性好，模拟真实的交互式SSH会话
- ✅ 支持所有SSH选项和配置
- ✅ 可以处理复杂的SSH配置
- ❌ 需要安装pexpect库

### paramiko方式
- ✅ 纯Python实现，更Pythonic
- ✅ 更好的错误处理和连接管理
- ✅ 支持SFTP直接上传文件
- ❌ 需要安装paramiko库
- ❌ 某些SSH配置可能不支持

## 安全建议

⚠️ **重要**: 不要在代码中硬编码passphrase！

推荐做法：

1. **使用环境变量**:
```python
import os
passphrase = os.getenv("SSH_PASSPHRASE")
```

2. **使用配置文件**（不要提交到git）:
```python
import json
with open("~/.ssh_config.json") as f:
    config = json.load(f)
    passphrase = config["passphrase"]
```

3. **使用ssh-agent**（最安全）:
```bash
# 启动ssh-agent
eval "$(ssh-agent -s)"

# 添加密钥（会提示输入passphrase一次）
ssh-add ~/.ssh/id_ed25519

# 之后就不需要passphrase了
```

如果使用ssh-agent，可以省略`passphrase`参数：
```python
result = submit_io_flow_via_ssh(
    # ... 其他参数
    # passphrase=None,  # 会尝试使用ssh-agent
)
```

## 完整示例

参考 `InterOptimus/agents/demos/demo_remote_submit.py`
