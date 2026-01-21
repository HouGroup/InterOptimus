#!/usr/bin/env python3
"""
Remote jobflow submission via SSH with passphrase support.

This module provides functions to submit jobflow JSON files to a remote server
via SSH, with support for SSH key passphrases.
"""

import subprocess
import shlex
from pathlib import Path
from typing import Optional

try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    PEXPECT_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


def submit_io_flow_via_ssh(
    host="xys@10.103.65.21",
    identity_file="~/.ssh/id_ed25519",
    local_io_flow_json="io_flow.json",
    remote_workdir="~/io_runs/si_sic_run1",
    remote_python="python",
    pre_cmd="source ~/.bashrc && conda activate atomate2",
    passphrase: Optional[str] = None,
    use_paramiko: bool = False,
    debug: bool = False,
):
    """
    Submit a jobflow JSON file to a remote server via SSH.
    
    Args:
        host: SSH host in format "user@hostname"
        identity_file: Path to SSH private key file
        local_io_flow_json: Local path to the jobflow JSON file
        remote_workdir: Remote working directory (will be created if not exists)
        remote_python: Python command to use on remote server
        pre_cmd: Command to run before Python script (e.g., conda activate)
        passphrase: SSH key passphrase (if None, will try ssh-agent or prompt)
        use_paramiko: If True, use paramiko library instead of pexpect
    
    Returns:
        dict with 'success', 'stdout', 'stderr', 'job_id' (if available)
    
    Raises:
        FileNotFoundError: If local_io_flow_json doesn't exist
        ImportError: If required library (pexpect or paramiko) not available
        subprocess.CalledProcessError: If SSH/SCP commands fail
    """
    local_io_flow_json = Path(local_io_flow_json).resolve()
    if not local_io_flow_json.exists():
        raise FileNotFoundError(f"Local file not found: {local_io_flow_json}")

    identity_file = str(Path(identity_file).expanduser())
    
    if use_paramiko:
        if not PARAMIKO_AVAILABLE:
            raise ImportError(
                "paramiko not available. Install with: pip install paramiko"
            )
        return _submit_via_paramiko(
            host=host,
            identity_file=identity_file,
            local_io_flow_json=local_io_flow_json,
            remote_workdir=remote_workdir,
            remote_python=remote_python,
            pre_cmd=pre_cmd,
            passphrase=passphrase,
        )
    else:
        if not PEXPECT_AVAILABLE:
            raise ImportError(
                "pexpect not available. Install with: pip install pexpect"
            )
        return _submit_via_pexpect(
            host=host,
            identity_file=identity_file,
            local_io_flow_json=local_io_flow_json,
            remote_workdir=remote_workdir,
            remote_python=remote_python,
            pre_cmd=pre_cmd,
            passphrase=passphrase,
            debug=debug,
        )


def _submit_via_pexpect(
    host: str,
    identity_file: str,
    local_io_flow_json: Path,
    remote_workdir: str,
    remote_python: str,
    pre_cmd: str,
    passphrase: Optional[str],
    debug: bool = False,
):
    """Submit using pexpect (interactive passphrase input)."""
    import re
    ssh_opts = [
        "-i", identity_file,
        "-o", "IdentitiesOnly=yes",
        "-o", "PreferredAuthentications=publickey",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    
    # Build SSH command
    ssh_cmd = ["ssh"] + ssh_opts + [host]
    scp_cmd = ["scp"] + ssh_opts + [str(local_io_flow_json), f"{host}:{remote_workdir.rstrip('/')}/io_flow.json"]
    
    def run_with_passphrase(cmd, expect_patterns=None, cmd_debug=None):
        """Run command with passphrase handling."""
        import sys
        import io
        
        if cmd_debug is None:
            cmd_debug = debug
        
        if expect_patterns is None:
            expect_patterns = [
                (r"Enter passphrase for key.*:", passphrase),
                (r"passphrase for key.*:", passphrase),
                (r"password.*:", passphrase),
                (r"\(yes/no\)", "yes"),
                (r"Are you sure.*", "yes"),
            ]
        
        # Capture output reliably (even in debug mode). In debug mode, tee to stdout.
        output_buffer = io.StringIO()

        class _Tee:
            def __init__(self, *files):
                self._files = files

            def write(self, s):
                for f in self._files:
                    try:
                        f.write(s)
                        f.flush()
                    except Exception:
                        pass

            def flush(self):
                for f in self._files:
                    try:
                        f.flush()
                    except Exception:
                        pass

        # Use list form for spawn to avoid shell issues
        child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8", timeout=60)
        child.logfile_read = _Tee(output_buffer, sys.stdout) if cmd_debug else output_buffer
        
        all_output = []
        max_iterations = 20  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            try:
                # Wait for any of the patterns or EOF
                patterns_to_match = [p[0] for p in expect_patterns] + [pexpect.EOF, pexpect.TIMEOUT]
                idx = child.expect(patterns_to_match, timeout=60)
                
                if idx < len(expect_patterns):  # Matched a passphrase pattern
                    pattern, response = expect_patterns[idx]
                    current_output = child.before if child.before else ""
                    all_output.append(current_output)
                    if cmd_debug:
                        print(f"[DEBUG] Matched pattern: {pattern}")
                        print(f"[DEBUG] Current output: {current_output}")
                    if response:
                        if cmd_debug:
                            print(f"[DEBUG] Sending passphrase...")
                        child.sendline(response)
                    else:
                        # No passphrase provided, try to get from user
                        import getpass
                        pwd = getpass.getpass(f"Enter passphrase for {identity_file}: ")
                        child.sendline(pwd)
                elif idx == len(expect_patterns):  # EOF - command finished
                    all_output.append(child.before if child.before else "")
                    if cmd_debug:
                        print(f"[DEBUG] Command finished (EOF)")
                    break
                elif idx == len(expect_patterns) + 1:  # Timeout
                    current_output = child.before if child.before else ""
                    all_output.append(current_output)
                    if cmd_debug:
                        print(f"[DEBUG] Timeout occurred")
                        print(f"[DEBUG] Output so far: {current_output}")
                    # Don't raise immediately, try to continue
                    if iteration >= max_iterations:
                        child.close()
                        raise TimeoutError(f"Timeout waiting for prompt after {max_iterations} iterations. Output: {''.join(all_output)}")
            except pexpect.EOF:
                all_output.append(child.before if child.before else "")
                if cmd_debug:
                    print(f"[DEBUG] EOF exception")
                break
            except pexpect.TIMEOUT:
                current_output = child.before if child.before else ""
                all_output.append(current_output)
                if cmd_debug:
                    print(f"[DEBUG] Timeout exception")
                if iteration >= max_iterations:
                    break
            except Exception as e:
                current_output = child.before if child.before else ""
                all_output.append(current_output)
                if cmd_debug:
                    print(f"[DEBUG] Exception: {e}")
                child.close()
                raise RuntimeError(f"Error during SSH command execution: {e}. Output: {''.join(all_output)}")
        
        # Wait for process to finish
        try:
            child.expect(pexpect.EOF, timeout=5)
            all_output.append(child.before if child.before else "")
        except:
            pass
        
        exit_status = child.exitstatus if child.exitstatus is not None else -1
        full_output = ''.join(all_output) + output_buffer.getvalue()
        
        if cmd_debug:
            print(f"[DEBUG] Exit status: {exit_status}")
            print(f"[DEBUG] Full output length: {len(full_output)}")
        
        child.close()
        
        return exit_status, full_output

    def _parse_remote_home(output: str) -> str:
        """
        Extract remote HOME from ssh output.
        We expect the remote command to print markers: __HOME__<path>__END__.
        This avoids passphrase prompts / debug output corrupting the parsed path.
        """
        if not output:
            raise ValueError("Empty output while detecting remote HOME")
        # Normalize CRLF and stray carriage returns
        norm = output.replace("\r", "\n")
        m = re.findall(r"__HOME__(.*?)__END__", norm, flags=re.DOTALL)
        if not m:
            # fallback: last absolute /home/... token
            m2 = re.findall(r"(/home/[^\s'\"\\]+)", norm)
            if not m2:
                raise ValueError(f"Could not parse remote HOME from output: {output!r}")
            return m2[-1].strip()
        return m[-1].strip()
    
    error_messages = []
    
    # Step 0: Get HOME directory and expand ~ in remote_workdir
    try:
        # IMPORTANT: ssh joins remote args into a single command line.
        # We must ensure bash receives the *entire* -c script as ONE argument,
        # otherwise it will run just "printf" and print usage.
        get_home_cmd = ssh_cmd + ["bash -lc 'printf \"__HOME__%s__END__\" \"$HOME\"'"]
        exit_status, home_output = run_with_passphrase(get_home_cmd, cmd_debug=debug)
        if exit_status != 0:
            return {
                'success': False,
                'stdout': home_output,
                'stderr': f"Failed to get HOME directory: {home_output}",
                'job_id': None,
                'error': 'get_home_failed',
            }
        remote_home = _parse_remote_home(home_output)
        # Expand ~ in remote_workdir
        if remote_workdir.startswith('~/'):
            remote_workdir_abs = remote_workdir.replace('~', remote_home, 1)
        elif remote_workdir.startswith('~'):
            remote_workdir_abs = remote_home
        else:
            remote_workdir_abs = remote_workdir
        if debug:
            print(f"[DEBUG] Remote HOME: {remote_home}")
            print(f"[DEBUG] Remote workdir (expanded): {remote_workdir_abs}")
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f"Error getting HOME directory: {e}",
            'job_id': None,
            'error': 'get_home_exception',
        }
    
    # Step 1: Create remote directory (use absolute path)
    try:
        mkdir_cmd = ssh_cmd + [f"mkdir -p {shlex.quote(remote_workdir_abs)}"]
        exit_status, output = run_with_passphrase(mkdir_cmd, cmd_debug=debug)
        if exit_status != 0:
            error_messages.append(f"Failed to create remote directory. Exit code: {exit_status}, Output: {output}")
            return {
                'success': False,
                'stdout': output,
                'stderr': f"Failed to create remote directory: {output}",
                'job_id': None,
                'error': 'mkdir_failed',
            }
    except Exception as e:
        error_messages.append(f"Error creating remote directory: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'job_id': None,
            'error': 'mkdir_exception',
        }
    
    # Step 2: Upload file (use absolute path)
    try:
        scp_cmd_abs = ["scp"] + ssh_opts + [str(local_io_flow_json), f"{host}:{remote_workdir_abs.rstrip('/')}/io_flow.json"]
        exit_status, output = run_with_passphrase(scp_cmd_abs, cmd_debug=debug)
        if exit_status != 0:
            error_messages.append(f"Failed to upload file. Exit code: {exit_status}, Output: {output}")
            return {
                'success': False,
                'stdout': output,
                'stderr': f"Failed to upload file: {output}",
                'job_id': None,
                'error': 'scp_failed',
            }
    except Exception as e:
        error_messages.append(f"Error uploading file: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'job_id': None,
            'error': 'scp_exception',
        }
    
    # Step 3: Submit job
    remote_script = r"""
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow
from qtoolkit.core.data_objects import QResources
import json

with open('io_flow.json','r') as f:
    flow_json = json.load(f)

flow = Flow.from_dict(flow_json)

resources = QResources(
    nodes=1,
    processes_per_node=1,
    scheduler_kwargs={"partition": "standard", "cpus-per-task": 10},
)

print(submit_flow(flow, worker="std_worker", resources=resources, project="std"))
""".strip()
    
    cmd = f"""
set -e
cd {shlex.quote(remote_workdir_abs)}
{pre_cmd}
{shlex.quote(remote_python)} - <<'PY'
{remote_script}
PY
""".strip()
    
    try:
        # Same quoting rule as above: pass ONE remote argument so bash -lc gets
        # the full multi-line script as a single -c string.
        full_cmd = ssh_cmd + [f"bash -lc {shlex.quote(cmd)}"]
        exit_status, output = run_with_passphrase(full_cmd, cmd_debug=debug)
        
        # Try to extract job ID from output
        job_id = None
        if output:
            import re
            job_match = re.search(r'job[_-]?id["\']?\s*[:=]\s*["\']?(\d+)', output, re.I)
            if job_match:
                job_id = job_match.group(1)
        
        return {
            'success': exit_status == 0,
            'stdout': output,
            'stderr': '' if exit_status == 0 else f"Exit code: {exit_status}",
            'job_id': job_id,
            'error': None if exit_status == 0 else 'submit_failed',
        }
    except Exception as e:
        error_messages.append(f"Error submitting job: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'job_id': None,
            'error': 'submit_exception',
        }


def _submit_via_paramiko(
    host: str,
    identity_file: str,
    local_io_flow_json: Path,
    remote_workdir: str,
    remote_python: str,
    pre_cmd: str,
    passphrase: Optional[str],
):
    """Submit using paramiko (direct key handling)."""
    # Parse host
    if "@" in host:
        username, hostname = host.split("@", 1)
    else:
        username = None
        hostname = host
        # Try to get username from current user
        import os
        username = os.getenv("USER") or os.getenv("USERNAME")
    
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Load private key
    try:
        if passphrase:
            pkey = paramiko.Ed25519Key.from_private_key_file(identity_file, password=passphrase)
        else:
            # Try without passphrase first
            try:
                pkey = paramiko.Ed25519Key.from_private_key_file(identity_file)
            except paramiko.ssh_exception.SSHException:
                # If that fails, try with passphrase prompt
                import getpass
                pwd = getpass.getpass(f"Enter passphrase for {identity_file}: ")
                pkey = paramiko.Ed25519Key.from_private_key_file(identity_file, password=pwd)
    except Exception as e:
        raise ValueError(f"Failed to load SSH key: {e}")
    
    try:
        # Connect
        ssh.connect(hostname, username=username, pkey=pkey)
        
        # Create remote directory
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_workdir}")
        stdout.channel.recv_exit_status()  # Wait for completion
        
        # Upload file using SFTP
        sftp = ssh.open_sftp()
        remote_file_path = f"{remote_workdir.rstrip('/')}/io_flow.json"
        sftp.put(str(local_io_flow_json), remote_file_path)
        sftp.close()
        
        # Submit job
        remote_script = r"""
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow
from qtoolkit.core.data_objects import QResources
import json

with open('io_flow.json','r') as f:
    flow_json = json.load(f)

flow = Flow.from_dict(flow_json)

resources = QResources(
    nodes=1,
    processes_per_node=1,
    scheduler_kwargs={"partition": "standard", "cpus-per-task": 10},
)

print(submit_flow(flow, worker="std_worker", resources=resources, project="std"))
""".strip()
        
        cmd = f"""
set -e
cd {shlex.quote(remote_workdir)}
{pre_cmd}
{shlex.quote(remote_python)} - <<'PY'
{remote_script}
PY
""".strip()
        
        stdin, stdout, stderr = ssh.exec_command(f"bash -lc {shlex.quote(cmd)}")
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode('utf-8')
        stderr_text = stderr.read().decode('utf-8')
        
        # Try to extract job ID
        job_id = None
        if stdout_text:
            import re
            job_match = re.search(r'job[_-]?id["\']?\s*[:=]\s*["\']?(\d+)', stdout_text, re.I)
            if job_match:
                job_id = job_match.group(1)
        
        return {
            'success': exit_status == 0,
            'stdout': stdout_text,
            'stderr': stderr_text,
            'job_id': job_id,
        }
    
    finally:
        ssh.close()


def check_job_status_via_ssh(
    host="xys@10.103.65.21",
    identity_file="~/.ssh/id_ed25519",
    job_id: Optional[str] = None,
    remote_workdir: Optional[str] = None,
    passphrase: Optional[str] = None,
    use_paramiko: bool = False,
):
    """
    Check the status of a submitted jobflow job.
    
    Args:
        host: SSH host
        identity_file: Path to SSH private key
        job_id: Job ID to check (if None, will try to find from remote_workdir)
        remote_workdir: Remote working directory
        passphrase: SSH key passphrase
        use_paramiko: Use paramiko instead of pexpect
    
    Returns:
        dict with job status information
    """
    identity_file = str(Path(identity_file).expanduser())
    
    if use_paramiko and PARAMIKO_AVAILABLE:
        # Use paramiko for status check
        if "@" in host:
            username, hostname = host.split("@", 1)
        else:
            import os
            username = os.getenv("USER") or os.getenv("USERNAME")
            hostname = host
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            if passphrase:
                pkey = paramiko.Ed25519Key.from_private_key_file(identity_file, password=passphrase)
            else:
                pkey = paramiko.Ed25519Key.from_private_key_file(identity_file)
        except paramiko.ssh_exception.SSHException:
            import getpass
            pwd = getpass.getpass(f"Enter passphrase for {identity_file}: ")
            pkey = paramiko.Ed25519Key.from_private_key_file(identity_file, password=pwd)
        
        ssh.connect(hostname, username=username, pkey=pkey)
        
        try:
            # Check job status (example command - adjust based on your jobflow_remote setup)
            if job_id:
                cmd = f"jobflow_remote status {job_id}"
            elif remote_workdir:
                cmd = f"cd {remote_workdir} && jobflow_remote status"
            else:
                raise ValueError("Either job_id or remote_workdir must be provided")
            
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            return {
                'success': exit_status == 0,
                'stdout': stdout_text,
                'stderr': stderr_text,
            }
        finally:
            ssh.close()
    else:
        # Use pexpect for status check
        if not PEXPECT_AVAILABLE:
            raise ImportError("pexpect not available")
        
        ssh_opts = [
            "-i", identity_file,
            "-o", "IdentitiesOnly=yes",
            "-o", "PreferredAuthentications=publickey",
        ]
        
        if job_id:
            cmd = ["ssh"] + ssh_opts + [host, f"jobflow_remote status {job_id}"]
        elif remote_workdir:
            cmd = ["ssh"] + ssh_opts + [host, f"cd {remote_workdir} && jobflow_remote status"]
        else:
            raise ValueError("Either job_id or remote_workdir must be provided")
        
        # Similar pexpect handling as in _submit_via_pexpect
        import pexpect
        child = pexpect.spawn(" ".join(shlex.quote(str(c)) for c in cmd), encoding='utf-8')
        
        if passphrase:
            patterns = [
                (r"Enter passphrase for key.*:", passphrase),
                (r"password:", passphrase),
            ]
            for pattern, response in patterns:
                try:
                    idx = child.expect([pattern, pexpect.EOF], timeout=10)
                    if idx == 0:
                        child.sendline(response)
                except pexpect.EOF:
                    break
        
        child.expect(pexpect.EOF)
        child.close()
        
        return {
            'success': child.exitstatus == 0,
            'stdout': child.before,
            'stderr': '',
        }


if __name__ == "__main__":
    # Example usage
    result = submit_io_flow_via_ssh(
        passphrase="your_passphrase_here",  # 直接传入passphrase
        # use_paramiko=True,  # 或者使用paramiko方式
    )
    
    print("Submission result:")
    print(f"Success: {result['success']}")
    print(f"Job ID: {result.get('job_id', 'N/A')}")
    print(f"Output:\n{result['stdout']}")
    if result['stderr']:
        print(f"Errors:\n{result['stderr']}")
