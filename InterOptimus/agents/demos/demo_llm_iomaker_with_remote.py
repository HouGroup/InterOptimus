#!/usr/bin/env python3
"""
Demo: LLM IOMaker with remote submission.

This script demonstrates how to use BuildConfig to automatically submit
generated jobflow jobs to a remote server with VASP calculation.

Key features:
- Uses default MPRelaxSet/MPStaticSet when vasp_relax_settings/vasp_static_settings are not provided
- Automatic remote submission via SSH
- Passphrase support for SSH keys
"""

from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt
from qtoolkit.core.data_objects import QResources
import os


def mlip_resources():
    """Return QResources for MLIP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=1,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 10},
    )


def vasp_resources():
    """Return QResources for VASP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=4,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 40},
    )


def main():
    # Configure LLM API
    API_SECRET_KEY = "sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35"
    BASE_URL = "https://api.zhizengzeng.com/v1/"

    # Create BuildConfig with remote submission enabled
    cfg = BuildConfig(
        # LLM configuration
        api_key=API_SECRET_KEY,
        base_url=BASE_URL,
        model="gpt-3.5-turbo",
        
        # Materials Project (optional, only needed if using MP IDs)
        mp_api_key="LBPWGcTye2eanNBb7dtmTp2deXR4aF9E",
        
        # Input files (defaults to film.cif and substrate.cif in current directory)
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow.json",
        print_settings=True,
        
        # Jobflow resources and workers
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # VASP configuration (required if do_vasp=True in LLM output)
        vasp_resources=vasp_resources,  # Required if do_vasp=True
        vasp_worker="std_worker",  # Required if do_vasp=True
        
        # VASP settings: Not provided - will use default MPRelaxSet/MPStaticSet
        # vasp_relax_settings=None,  # Default: uses MPRelaxSet defaults
        # vasp_static_settings=None,  # Default: uses MPStaticSet defaults
        
        # Remote submission configuration
        submit_to_remote=True,  # Enable automatic remote submission
        remote_host="xys@10.103.65.21",
        remote_identity_file="~/.ssh/id_ed25519",
        remote_workdir="~/io_runs/si_sic_run1",
        remote_python="python",
        remote_pre_cmd="source ~/.bashrc && conda activate atomate2",
        # Use environment variable for passphrase in production!
        remote_passphrase=os.getenv("SSH_PASSPHRASE", "jinlhr542"),
        remote_use_paramiko=False,  # Use pexpect (recommended)
        remote_debug=False,  # Set to True for verbose output
    )

    # Build jobflow from natural language prompt
    user_prompt = "建立双界面模型，进行VASP模拟"
    
    print("=" * 80)
    print("Building IOMaker jobflow from prompt...")
    print(f"Prompt: {user_prompt}")
    print("=" * 80)
    
    try:
        result = build_iomaker_flow_from_prompt(user_prompt, cfg)
        
        print("\n✅ Jobflow generation completed!")
        print(f"📄 Flow JSON: {result['flow_json_path']}")
        
        # Check remote submission result
        if "remote_submission" in result:
            submit_result = result["remote_submission"]
            if submit_result["success"]:
                print(f"\n✅ Remote submission successful!")
                if submit_result.get("job_id"):
                    print(f"📋 Job ID: {submit_result['job_id']}")
            else:
                print(f"\n❌ Remote submission failed:")
                print(f"   Error: {submit_result.get('error', 'unknown')}")
                print(f"   Details: {submit_result.get('stderr', '')}")
        
        if "remote_submission_error" in result:
            print(f"\n⚠️  Remote submission error: {result['remote_submission_error']}")
            
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
