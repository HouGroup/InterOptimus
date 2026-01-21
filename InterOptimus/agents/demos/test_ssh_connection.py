#!/usr/bin/env python3
"""
Test SSH connection with passphrase to diagnose issues.
"""

import pexpect
import sys

def test_ssh_connection(host, identity_file, passphrase):
    """Test basic SSH connection."""
    print(f"Testing SSH connection to {host}...")
    print(f"Using key: {identity_file}")
    
    ssh_cmd = [
        "ssh",
        "-i", identity_file,
        "-o", "IdentitiesOnly=yes",
        "-o", "PreferredAuthentications=publickey",
        "-o", "StrictHostKeyChecking=accept-new",
        host,
        "echo 'SSH connection successful!'"
    ]
    
    print(f"\nCommand: {' '.join(ssh_cmd)}")
    print("\n=== Starting SSH connection (you'll see all output) ===\n")
    
    child = pexpect.spawn(ssh_cmd[0], ssh_cmd[1:], encoding='utf-8', timeout=30)
    child.logfile_read = sys.stdout  # Show all output
    
    patterns = [
        (r"Enter passphrase for key.*:", passphrase),
        (r"passphrase for key.*:", passphrase),
        (r"password.*:", passphrase),
        (r"\(yes/no\)", "yes"),
        (r"Are you sure.*", "yes"),
        (pexpect.EOF, None),
        (pexpect.TIMEOUT, None),
    ]
    
    try:
        while True:
            idx = child.expect([p[0] for p in patterns], timeout=30)
            pattern, response = patterns[idx]
            
            if pattern == pexpect.EOF:
                print("\n=== Connection closed ===")
                break
            elif pattern == pexpect.TIMEOUT:
                print("\n=== Timeout ===")
                print(f"Current output: {child.before}")
                break
            elif response:
                print(f"\n[Auto-responding to: {pattern}]")
                child.sendline(response)
            else:
                print(f"\n[Matched pattern but no response: {pattern}]")
                break
    except Exception as e:
        print(f"\n=== Exception: {e} ===")
        print(f"Current output: {child.before}")
    
    child.expect(pexpect.EOF, timeout=5)
    exit_status = child.exitstatus if child.exitstatus is not None else -1
    child.close()
    
    print(f"\n=== Exit status: {exit_status} ===")
    return exit_status == 0


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    host = "xys@10.103.65.21"
    identity_file = str(Path("~/.ssh/id_ed25519").expanduser())
    passphrase = "jinlhr542"  # 替换为你的passphrase
    
    success = test_ssh_connection(host, identity_file, passphrase)
    
    if success:
        print("\n✅ SSH connection test PASSED!")
    else:
        print("\n❌ SSH connection test FAILED!")
        print("\n请检查:")
        print("1. passphrase是否正确")
        print("2. SSH密钥路径是否正确")
        print("3. 服务器地址和用户名是否正确")
        print("4. 网络连接是否正常")
