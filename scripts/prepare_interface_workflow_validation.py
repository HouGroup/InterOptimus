#!/usr/bin/env python3
"""Prepare a bounded multi-material InterOptimus remote validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.core import Lattice, Structure


def structures() -> dict[str, Structure]:
    u = 0.375
    return {
        "si": Structure.from_spacegroup(
            "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
        ),
        "ge": Structure.from_spacegroup(
            "Fd-3m", Lattice.cubic(5.55), ["Ge"], [[0, 0, 0]]
        ),
        "mgo": Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(4.21),
            ["Mg", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        ),
        "sto": Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(3.905),
            ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
        ),
        "zno": Structure(
            Lattice.hexagonal(3.25, 5.21),
            ["Zn", "Zn", "O", "O"],
            [[0, 0, 0], [2 / 3, 1 / 3, 0.5], [0, 0, u], [2 / 3, 1 / 3, 0.5 + u]],
        ),
        "gan": Structure(
            Lattice.hexagonal(3.19, 5.19),
            ["Ga", "Ga", "N", "N"],
            [[0, 0, 0], [2 / 3, 1 / 3, 0.5], [0, 0, u], [2 / 3, 1 / 3, 0.5 + u]],
        ),
        "lco_a": Structure.from_spacegroup(
            "R-3m",
            Lattice.hexagonal(2.815, 14.05),
            ["Li", "Co", "O"],
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.241]],
        ),
        "lco_b": Structure.from_spacegroup(
            "R-3m",
            Lattice.hexagonal(2.84, 14.15),
            ["Li", "Co", "O"],
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.241]],
        ),
        "alumina_a": Structure.from_spacegroup(
            "R-3c",
            Lattice.hexagonal(4.76, 12.99),
            ["Al", "O"],
            [[0, 0, 0.3522], [0.306, 0, 0.25]],
        ),
        "alumina_b": Structure.from_spacegroup(
            "R-3c",
            Lattice.hexagonal(4.80, 13.08),
            ["Al", "O"],
            [[0, 0, 0.3522], [0.306, 0, 0.25]],
        ),
    }


def workflow(
    *,
    name: str,
    film: str,
    substrate: str,
    calc: str,
    worker: str,
    miller: tuple[int, int, int],
    max_area: float,
    double_interface: bool = False,
    strain: bool = False,
    mlip_gd: bool = False,
    polarity: bool = False,
    do_vasp: bool = False,
    do_vasp_gd: bool = False,
    dipole: bool = False,
    pair_selection: str = "each_match_lowest",
) -> dict:
    structure_settings: dict = {
        "termination_ftol": 0.15,
        "film_thickness": 6,
        "substrate_thickness": 6,
        "double_interface": double_interface,
        "vacuum_over_film": 8,
    }
    if polarity:
        structure_settings["non_polar_film_termination"] = {
            "oxidation_states": {"Al": 3, "O": -2}
        }
        structure_settings["non_polar_substrate_termination"] = {
            "oxidation_states": {"Al": 3, "O": -2}
        }

    vasp_settings = {
        "do_vasp": do_vasp,
        "do_vasp_gd": do_vasp_gd,
        "vasp_dipole_correction": dipole,
        "vasp_pair_selection": pair_selection,
        "vasp_gd_kwargs": {
            "min_steps": 0,
            "max_steps": 1,
            "max_displacement": 0.2,
        },
        "relax_user_incar_settings": {
            "ALGO": "Fast",
            "ENCUT": 300,
            "EDIFF": 1e-4,
            "EDIFFG": 1.0,
            "NSW": 5,
            "LREAL": "Auto",
        },
        "static_user_incar_settings": {
            "ALGO": "Fast",
            "ENCUT": 300,
            "EDIFF": 1e-4,
            "NELM": 40,
            "LREAL": "Auto",
        },
        "relax_user_kpoints_settings": {"reciprocal_density": 20},
        "static_user_kpoints_settings": {"reciprocal_density": 20},
        "relax_user_potcar_functional": "PBE_54",
        "static_user_potcar_functional": "PBE_54",
    }
    return {
        "workflow_name": name,
        "execution": "server",
        "IO_workflow_config": {
            "cost_preset": "low",
            "bulk_cifs": {
                "film_cif": f"materials/{film}.cif",
                "substrate_cif": f"materials/{substrate}.cif",
            },
            "lattice_matching_settings": {
                "max_area": max_area,
                "max_length_tol": 0.1,
                "max_angle_tol": 0.05,
                "film_max_miller": 1,
                "substrate_max_miller": 1,
                "film_millers": [list(miller)],
                "substrate_millers": [list(miller)],
            },
            "structure_settings": structure_settings,
            "optimization_settings": {
                "fmax": 0.2,
                "steps": 10,
                "device": "cuda",
                "discut": 0.6,
                "n_calls_density": 0.1,
                "z_range": [1.5, 2.5],
                "calc": calc,
                "strain_E_correction": strain,
                "do_mlip_gd": mlip_gd,
                "gd_max_steps": 3,
                "gd_max_displacement": 0.2,
            },
            "vasp_settings": vasp_settings,
        },
        "cluster": {
            "mlip": {
                "mlip_worker": worker,
                "slurm_partition": "gpu2",
                "mlip_project": "std",
                "cpus_per_gpu": 5,
            },
            "vasp": {
                "vasp_worker": "default",
                "vasp_slurm_partition": "spr",
                "vasp_nodes": 1,
                "vasp_processes_per_node": 12,
                "vasp_pre_run": "module load VASP/6.5.1",
            },
        },
    }


def matrix() -> list[dict]:
    return [
        workflow(name="validate_mlip_orb_si_ge", film="si", substrate="ge", calc="orb-models", worker="orb", miller=(1, 1, 1), max_area=35),
        workflow(name="validate_mlip_dpa_lco", film="lco_a", substrate="lco_b", calc="dpa", worker="dpa", miller=(0, 0, 1), max_area=12, double_interface=True),
        workflow(name="validate_mlip_matris_mgo_sto", film="mgo", substrate="sto", calc="matris", worker="matris", miller=(0, 0, 1), max_area=100, strain=True),
        workflow(name="validate_mlip_sevenn_zno_gan", film="zno", substrate="gan", calc="sevenn", worker="sevenn", miller=(0, 0, 1), max_area=15, mlip_gd=True),
        workflow(name="validate_polar_orb_alumina", film="alumina_a", substrate="alumina_b", calc="orb-models", worker="orb", miller=(0, 0, 1), max_area=25, polarity=True),
        workflow(name="validate_double_matris_si_ge", film="si", substrate="ge", calc="matris", worker="matris", miller=(1, 1, 1), max_area=35, double_interface=True),
        workflow(name="validate_vasp_standard_si_ge", film="si", substrate="ge", calc="orb-models", worker="orb", miller=(1, 1, 1), max_area=35, strain=True, do_vasp=True),
        workflow(name="validate_vasp_double_mgo_sto", film="mgo", substrate="sto", calc="matris", worker="matris", miller=(0, 0, 1), max_area=100, double_interface=True, do_vasp=True),
        workflow(name="validate_vasp_dipole_zno_gan", film="zno", substrate="gan", calc="sevenn", worker="sevenn", miller=(0, 0, 1), max_area=15, do_vasp=True, dipole=True, pair_selection="each_plane_lowest"),
        workflow(name="validate_vasp_gd_lco", film="lco_a", substrate="lco_b", calc="dpa", worker="dpa", miller=(0, 0, 1), max_area=12, double_interface=True, do_vasp=True, do_vasp_gd=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    material_dir = root / "materials"
    config_dir = root / "configs"
    material_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    for name, structure in structures().items():
        structure.to(filename=material_dir / f"{name}.cif")
    manifest = []
    for config in matrix():
        path = config_dir / f"{config['workflow_name']}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        manifest.append(
            {
                "name": config["workflow_name"],
                "config": str(path),
                "do_vasp": config["IO_workflow_config"]["vasp_settings"]["do_vasp"],
                "calc": config["IO_workflow_config"]["optimization_settings"]["calc"],
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
