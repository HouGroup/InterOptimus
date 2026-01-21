#!/usr/bin/env python3
"""
Complete Demo: LLM IOMaker with remote submission and VASP settings.

This script demonstrates various usage scenarios:
1. Basic usage without VASP (MLIP only)
2. VASP with default MPRelaxSet/MPStaticSet
3. VASP with custom settings
4. Remote submission
"""

from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt
from qtoolkit.core.data_objects import QResources


def mlip_resources():
    """Return QResources for MLIP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=1,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 1},
    )


def vasp_resources():
    """Return QResources for VASP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=4,
        scheduler_kwargs={"partition": "standard", "cpus-per-task": 48},
    )


def demo_1_mlip_only():
    """Demo 1: MLIP only, no VASP calculation."""
    print("=" * 80)
    print("Demo 1: MLIP Only (No VASP)")
    print("=" * 80)
    
    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        
        # Input files
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow_mlip_only.json",
        print_settings=True,
        
        # MLIP resources
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # No remote submission
        submit_to_remote=False,
    )
    
    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，不进行VASP计算",
        cfg
    )
    
    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def demo_2_vasp_with_defaults():
    """Demo 2: VASP with default MPRelaxSet/MPStaticSet."""
    print("=" * 80)
    print("Demo 2: VASP with Default MPRelaxSet/MPStaticSet")
    print("=" * 80)
    
    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        
        # Input files
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow_vasp_defaults.json",
        print_settings=True,
        
        # MLIP resources
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # VASP resources (required when do_vasp=True)
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        
        # No VASP settings provided - will use default MPRelaxSet/MPStaticSet
        # vasp_relax_settings=None,  # Default: uses MPRelaxSet defaults
        # vasp_static_settings=None,  # Default: uses MPStaticSet defaults
        
        # No remote submission
        submit_to_remote=False,
    )
    
    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg
    )
    
    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def demo_3_vasp_with_custom_settings():
    """Demo 3: VASP with custom settings."""
    print("=" * 80)
    print("Demo 3: VASP with Custom Settings")
    print("=" * 80)
    
    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        
        # Input files
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow_vasp_custom.json",
        print_settings=True,
        
        # MLIP resources
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # VASP resources
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        
        # Custom VASP settings
        vasp_relax_settings={
            'INCAR': {
                'EDIFF': 1e-5,
                'EDIFFG': -0.01,
                'NSW': 100,
            },
            'KPOINTS': {
                'kpoints': [[4, 4, 1]],  # Custom k-points
            },
            'POTCAR_FUNCTIONAL': 'PBE',
            'POTCAR': {},  # Use default POTCAR settings
            'GD': False,  # No gradient descent
            'GDTOL': 5e-4,
        },
        vasp_static_settings={
            'INCAR': {
                'EDIFF': 1e-6,
                'NSW': 0,  # Static calculation
            },
            'KPOINTS': {
                'kpoints': [[6, 6, 1]],  # Finer k-points for static
            },
            'POTCAR_FUNCTIONAL': 'PBE',
            'POTCAR': {},
        },
        
        # No remote submission
        submit_to_remote=False,
    )
    
    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg
    )
    
    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def demo_4_remote_submission():
    """Demo 4: Remote submission with VASP."""
    print("=" * 80)
    print("Demo 4: Remote Submission with VASP")
    print("=" * 80)
    
    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        
        # Materials Project (optional, only needed if using MP IDs)
        mp_api_key="LBPWGcTye2eanNBb7dtmTp2deXR4aF9E",
        
        # Input files
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow_remote.json",
        print_settings=True,
        
        # MLIP resources
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # VASP resources (required when do_vasp=True)
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        
        # Use default MPRelaxSet/MPStaticSet
        # vasp_relax_settings=None,
        # vasp_static_settings=None,
        
        # Remote submission configuration
        submit_to_remote=True,
        remote_host="xys@10.103.65.21",
        remote_identity_file="~/.ssh/id_ed25519",
        remote_workdir="~/io_runs/si_sic_run1",
        remote_python="python",
        remote_pre_cmd="source ~/.bashrc && conda activate atomate2",
        remote_passphrase="jinlhr542",  # Use environment variable in production!
        remote_use_paramiko=False,  # Use pexpect
        remote_debug=False,  # Set to True for verbose output
    )
    
    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg
    )
    
    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    
    # Check remote submission result
    if "remote_submission" in result:
        submit_result = result["remote_submission"]
        if submit_result["success"]:
            print(f"✅ Remote submission successful!")
            if submit_result.get("job_id"):
                print(f"📋 Job ID: {submit_result['job_id']}")
        else:
            print(f"❌ Remote submission failed:")
            print(f"   Error: {submit_result.get('error', 'unknown')}")
            print(f"   Details: {submit_result.get('stderr', '')}")
    
    print()


def demo_5_partial_vasp_settings():
    """Demo 5: VASP with partial settings (some defaults, some custom)."""
    print("=" * 80)
    print("Demo 5: VASP with Partial Settings")
    print("=" * 80)
    
    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        
        # Input files
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        
        # Output
        output_flow_json="io_flow_vasp_partial.json",
        print_settings=True,
        
        # MLIP resources
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        
        # VASP resources
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        
        # Partial VASP settings - only override INCAR, use defaults for others
        vasp_relax_settings={
            'INCAR': {
                'EDIFF': 1e-5,  # Only override EDIFF
                # Other INCAR settings use MPRelaxSet defaults
            },
            # KPOINTS, POTCAR_FUNCTIONAL, POTCAR use defaults
        },
        # vasp_static_settings=None,  # Use MPStaticSet defaults
        
        # No remote submission
        submit_to_remote=False,
    )
    
    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg
    )
    
    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("InterOptimus LLM IOMaker Complete Demo")
    print("=" * 80 + "\n")
    
    # Uncomment the demo you want to run:
    
    # Demo 1: MLIP only
    # demo_1_mlip_only()
    
    # Demo 2: VASP with defaults
    # demo_2_vasp_with_defaults()
    
    # Demo 3: VASP with custom settings
    # demo_3_vasp_with_custom_settings()
    
    # Demo 4: Remote submission
    # demo_4_remote_submission()
    
    # Demo 5: Partial VASP settings
    # demo_5_partial_vasp_settings()
    
    print("\n" + "=" * 80)
    print("Note: Uncomment the demo you want to run in main()")
    print("=" * 80)


if __name__ == "__main__":
    # Example: Run a specific demo
    import sys
    
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        if demo_num == "1":
            demo_1_mlip_only()
        elif demo_num == "2":
            demo_2_vasp_with_defaults()
        elif demo_num == "3":
            demo_3_vasp_with_custom_settings()
        elif demo_num == "4":
            demo_4_remote_submission()
        elif demo_num == "5":
            demo_5_partial_vasp_settings()
        else:
            print(f"Unknown demo number: {demo_num}")
            print("Usage: python demo_llm_iomaker_complete.py [1|2|3|4|5]")
    else:
        main()
