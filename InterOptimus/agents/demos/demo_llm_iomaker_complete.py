#!/usr/bin/env python3
"""
Complete Demo: LLM IOMaker with VASP settings and optional login-node (server) submit.

Scenarios:
1. Basic usage without VASP (MLIP only)
2. VASP with default MPRelaxSet/MPStaticSet
3. VASP with custom settings
4. Login-node submission (submit_target="server", run on cluster after ssh)
5. VASP with partial settings
"""

from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt
from qtoolkit.core.data_objects import QResources


def mlip_resources():
    """Return QResources for MLIP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=1,
        scheduler_kwargs={"partition": "standard", "cpus_per_task": 1},
    )


def vasp_resources():
    """Return QResources for VASP jobs."""
    return QResources(
        nodes=1,
        processes_per_node=4,
        scheduler_kwargs={"partition": "standard", "cpus_per_task": 48},
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
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        output_flow_json="io_flow_mlip_only.json",
        print_settings=True,
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        submit_target="local",
    )

    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，不进行VASP计算",
        cfg,
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
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        output_flow_json="io_flow_vasp_defaults.json",
        print_settings=True,
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        submit_target="local",
    )

    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg,
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
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        output_flow_json="io_flow_vasp_custom.json",
        print_settings=True,
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        relax_user_incar_settings={
            "EDIFF": 1e-5,
            "EDIFFG": -0.01,
            "NSW": 100,
        },
        # pymatgen DictSet: use reciprocal_density / grid_density / length / … — not a raw "kpoints" mesh.
        relax_user_kpoints_settings={"reciprocal_density": 64},
        relax_user_potcar_functional="PBE",
        relax_user_potcar_settings={},
        static_user_incar_settings={
            "EDIFF": 1e-6,
            "NSW": 0,
        },
        static_user_kpoints_settings={"reciprocal_density": 100},
        static_user_potcar_functional="PBE",
        static_user_potcar_settings={},
        vasp_gd_kwargs={"tol": 5e-4},
        submit_target="local",
    )

    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg,
    )

    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def demo_4_server_submission():
    """Demo 4: Login-node jobflow-remote submit (run this script on the cluster head node)."""
    print("=" * 80)
    print("Demo 4: Server (login-node) submission")
    print("=" * 80)

    cfg = BuildConfig(
        api_key="sk-zk29a8909a9badfb2973536dfd5a1bf41c7696742c764c35",
        base_url="https://api.zhizengzeng.com/v1/",
        model="gpt-3.5-turbo",
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        output_flow_json="io_flow_server.json",
        print_settings=True,
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        submit_target="server",
        server_pre_cmd="source ~/.bashrc && conda activate atomate2",
    )

    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg,
    )

    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    if result.get("server_submission"):
        print("server_submission:", result["server_submission"])
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
        film_cif="film.cif",
        substrate_cif="substrate.cif",
        output_flow_json="io_flow_vasp_partial.json",
        print_settings=True,
        mlip_resources=mlip_resources,
        mlip_worker="std_worker",
        vasp_resources=vasp_resources,
        vasp_worker="std_worker",
        relax_user_incar_settings={"EDIFF": 1e-5},
        submit_target="local",
    )

    result = build_iomaker_flow_from_prompt(
        "建立双界面模型，进行VASP模拟",
        cfg,
    )

    print(f"✅ Flow JSON generated: {result['flow_json_path']}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("InterOptimus LLM IOMaker Complete Demo")
    print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("Note: Uncomment the demo you want to run in main()")
    print("=" * 80)


if __name__ == "__main__":
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
            demo_4_server_submission()
        elif demo_num == "5":
            demo_5_partial_vasp_settings()
        else:
            print(f"Unknown demo number: {demo_num}")
            print("Usage: python demo_llm_iomaker_complete.py [1|2|3|4|5]")
    else:
        main()
