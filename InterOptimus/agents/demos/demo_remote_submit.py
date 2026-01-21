#!/usr/bin/env python3
"""
Demo: Remote jobflow submission with passphrase support.

This script demonstrates how to submit a jobflow JSON file to a remote server
with automatic passphrase handling.
"""

from pathlib import Path
from InterOptimus.agents.remote_submit import submit_io_flow_via_ssh, check_job_status_via_ssh


def main():
    # 方式1: 使用pexpect (推荐，兼容性好)
    print("=== 使用 pexpect 方式提交 ===")
    result = submit_io_flow_via_ssh(
        host="xys@10.103.65.21",
        identity_file="~/.ssh/id_ed25519",
        local_io_flow_json="io_flow.json",  # 确保这个文件存在
        remote_workdir="~/io_runs/si_sic_run1",
        remote_python="python",
        pre_cmd="source ~/.bashrc && conda activate atomate2",
        passphrase="jinlhr542",  # 直接传入passphrase
        use_paramiko=False,  # 使用pexpect
        debug=True,  # 开启调试模式，显示详细输出
    )
    
    print(f"提交成功: {result['success']}")
    if result.get('job_id'):
        print(f"任务ID: {result['job_id']}")
    if result.get('error'):
        print(f"错误类型: {result['error']}")
    print(f"标准输出:\n{result['stdout']}")
    if result['stderr']:
        print(f"标准错误:\n{result['stderr']}")
    
    # 如果失败，打印调试信息
    if not result['success']:
        print("\n=== 调试信息 ===")
        print(f"完整输出: {result.get('stdout', '')}")
        print(f"错误信息: {result.get('stderr', '')}")
        print(f"错误类型: {result.get('error', 'unknown')}")
    
    # 方式2: 使用paramiko (更Pythonic，但需要安装paramiko)
    # print("\n=== 使用 paramiko 方式提交 ===")
    # result2 = submit_io_flow_via_ssh(
    #     host="xys@10.103.65.21",
    #     identity_file="~/.ssh/id_ed25519",
    #     local_io_flow_json="io_flow.json",
    #     remote_workdir="~/io_runs/si_sic_run2",
    #     passphrase="your_passphrase_here",
    #     use_paramiko=True,  # 使用paramiko
    # )
    
    # 检查任务状态
    if result.get('job_id'):
        print("\n=== 检查任务状态 ===")
        status = check_job_status_via_ssh(
            host="xys@10.103.65.21",
            identity_file="~/.ssh/id_ed25519",
            job_id=result['job_id'],
            passphrase="your_passphrase_here",
        )
        print(f"状态查询成功: {status['success']}")
        print(f"状态信息:\n{status['stdout']}")


if __name__ == "__main__":
    main()
