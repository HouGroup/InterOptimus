#!/usr/bin/env python3
"""
IOMaker CLI: build ``io_flow.json`` from JSON/YAML.

Preferred input shape::

    {
      "workflow_name": "...",
      "IO_workflow_config": {
        "cost_preset": "low | medium | high",
        "bulk_cifs": {...},
        "lattice_matching_settings": {...},
        "structure_settings": {...},
        "optimization_settings": {...},  # also absorbs former global_minimization_settings keys
        "vasp_settings": {...}
      },
      "execution": "local" | "server",
      "cluster": {...}
    }

Internally this is normalized back to the full ``settings`` dict expected by
``execute_iomaker_from_settings``. The older ``{"settings": ..., "execution": ..., "cluster": ...}``
shape is still accepted for compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from qtoolkit.core.data_objects import QResources

from .iomaker_job import (
    BaseBuildConfig,
    DEFAULT_FILM_CIF,
    DEFAULT_SUBSTRATE_CIF,
    LocalBuildConfig,
    ServerBuildConfig,
    TUTORIAL_WITHOUT_VACUUM,
    TUTORIAL_WITH_VACUUM,
    execute_iomaker_from_settings,
    normalize_iomaker_settings_from_full_dict,
    uses_legacy_full_settings_dict,
)


_CLUSTER_SCHEMA_SHARED = frozenset()
_CLUSTER_SCHEMA_MLIP = frozenset(
    {
        "slurm_partition",
        "server_run_parent",
        "server_pre_cmd",
        "server_python",
        "server_jf_bin",
        "cpus_per_gpu",
        "mlip_cpus_per_task",
        "cpus_per_task",
        "gpus_per_job",
        "mlip_scheduler_kwargs",
        "mlip_worker",
        "mlip_project",
    }
)
_CLUSTER_SCHEMA_VASP = frozenset(
    {
        "vasp_worker",
        "vasp_slurm_partition",
        "vasp_nodes",
        "vasp_processes_per_node",
        "vasp_scheduler_kwargs",
        "vasp_pre_run",
        "vasp_exec_config",
    }
)
_CLUSTER_SCHEMA_SECTIONS: Dict[str, frozenset] = {
    "shared": _CLUSTER_SCHEMA_SHARED,
    "mlip": _CLUSTER_SCHEMA_MLIP,
    "vasp": _CLUSTER_SCHEMA_VASP,
}
_CLUSTER_LEGACY_KEYS: frozenset = _CLUSTER_SCHEMA_SHARED | _CLUSTER_SCHEMA_MLIP | _CLUSTER_SCHEMA_VASP

# Only ``execution`` and ``cluster`` bundles (besides top-level ``settings``).
_BUNDLE_SCHEMA: Dict[str, frozenset] = {
    "execution": frozenset({"run", "output_flow_json", "print_settings"}),
    # Legacy: flat ``cluster`` keys; nested ``cluster`` is expanded by :func:`_flatten_cluster_bundle`.
    "cluster": _CLUSTER_LEGACY_KEYS,
}
_BUNDLE_NAMES = frozenset(_BUNDLE_SCHEMA)

_LEGACY_BUNDLE_KEYS = frozenset(
    {"structure", "task", "physics", "vasp", "overrides"}
)

_NEW_STYLE_TOP_LEVEL_KEYS = frozenset(
    {"workflow_name", "IO_workflow_config", "execution", "cluster"}
)
_NEW_IO_WORKFLOW_KEYS = frozenset(
    {
        "cost_preset",
        "bulk_cifs",
        "lattice_matching_settings",
        "structure_settings",
        "optimization_settings",
        "vasp_settings",
    }
)
_GLOBAL_MINIMIZATION_KEYS = frozenset(
    {
        "n_calls_density",
        "z_range",
        "calc",
        "strain_E_correction",
        "term_screen_tol",
        "name",
    }
)
_VASP_SETTINGS_KEYS = frozenset(
    {
        "do_vasp",
        "do_vasp_gd",
        "relax_user_incar_settings",
        "relax_user_potcar_settings",
        "relax_user_kpoints_settings",
        "relax_user_potcar_functional",
        "static_user_incar_settings",
        "static_user_potcar_settings",
        "static_user_kpoints_settings",
        "static_user_potcar_functional",
        "vasp_gd_kwargs",
        "vasp_dipole_correction",
        "vasp_relax_settings",
        "vasp_static_settings",
        "lowest_energy_pairs_settings",
        "vasp_pair_selection",
    }
)
_COST_PRESET_NAMES = frozenset({"low", "medium", "high"})
_COST_PRESET_OVERRIDES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "with_vacuum": {
        "low": {
            "lattice_matching_settings": {
                "max_area": 30,
                "film_max_miller": 2,
                "substrate_max_miller": 2,
            },
            "structure_settings": {
                "film_thickness": 8,
                "substrate_thickness": 8,
            },
            "optimization_settings": {
                "steps": 500,
                "do_mlip_gd": False,
            },
            "global_minimization_settings": {
                "n_calls_density": 1,
            },
        },
        "medium": {
            "lattice_matching_settings": {
                "max_area": 60,
            },
            "structure_settings": {
                "film_thickness": 10,
                "substrate_thickness": 10,
            },
            "optimization_settings": {
                "steps": 500,
            },
        },
        "high": {
            "lattice_matching_settings": {
                "max_area": 100,
                "film_max_miller": 4,
                "substrate_max_miller": 4,
            },
            "structure_settings": {
                "film_thickness": 15,
                "substrate_thickness": 15,
            },
            "optimization_settings": {
                "steps": 500,
                "do_mlip_gd": True,
            },
            "global_minimization_settings": {
                "n_calls_density": 6,
            },
        },
    },
    "without_vacuum": {
        "low": {
            "lattice_matching_settings": {
                "max_area": 30,
                "film_max_miller": 2,
                "substrate_max_miller": 2,
            },
            "structure_settings": {
                "film_thickness": 8,
                "substrate_thickness": 8,
            },
            "optimization_settings": {
                "steps": 500,
            },
            "global_minimization_settings": {
                "n_calls_density": 1,
            },
        },
        "medium": {
            "lattice_matching_settings": {
                "max_area": 60,
            },
            "structure_settings": {
                "film_thickness": 10,
                "substrate_thickness": 10,
            },
            "optimization_settings": {
                "steps": 500,
            },
        },
        "high": {
            "lattice_matching_settings": {
                "max_area": 100,
                "film_max_miller": 4,
                "substrate_max_miller": 4,
            },
            "structure_settings": {
                "film_thickness": 15,
                "substrate_thickness": 15,
            },
            "optimization_settings": {
                "steps": 500,
            },
            "global_minimization_settings": {
                "n_calls_density": 6,
            },
        },
    },
}

# Flat keys allowed besides ``settings`` (after bundle expansion).
_ALLOWED_INFRA_KEYS = frozenset(
    {
        "settings",
        "run",
        "film_cif",
        "substrate_cif",
        "output_flow_json",
        "print_settings",
        "server_run_parent",
        "server_pre_cmd",
        "server_python",
        "server_jf_bin",
        "mlip_worker",
        "vasp_worker",
        "mlip_project",
        "slurm_partition",
        "cpus_per_gpu",
        "mlip_cpus_per_task",
        "cpus_per_task",
        "gpus_per_job",
        "mlip_scheduler_kwargs",
        "vasp_slurm_partition",
        "vasp_nodes",
        "vasp_processes_per_node",
        "vasp_scheduler_kwargs",
        "vasp_pre_run",
        "vasp_exec_config",
    }
)

# Must not duplicate computation fields at top level when ``settings`` is present.
_FORBID_FLAT_WITH_SETTINGS = frozenset(
    {
        "name",
        "interface",
        "mode",
        "inputs",
        "max_area",
        "film_thickness",
        "substrate_thickness",
        "vacuum_over_film",
        "fmax",
        "steps",
        "device",
        "z_range",
        "n_calls_density",
        "do_vasp",
        "do_vasp_gd",
        "film_mp_id",
        "substrate_mp_id",
        "mlip_calc",
        "ckpt_path",
        "do_mlip_gd",
        "vasp_relax_settings",
        "vasp_static_settings",
        "relax_user_incar_settings",
        "relax_user_potcar_settings",
        "relax_user_kpoints_settings",
        "relax_user_potcar_functional",
        "static_user_incar_settings",
        "static_user_potcar_settings",
        "static_user_kpoints_settings",
        "static_user_potcar_functional",
        "vasp_gd_kwargs",
        "vasp_dipole_correction",
        "bulk",
    }
)


def _flatten_cluster_bundle(sub: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand ``cluster`` into the flat dict expected by :func:`build_config_from_simple_dict`.

    **Nested** (recommended): keys are ``shared`` (optional, currently no allowed keys), ``mlip``, ``vasp`` — each an object with
    keys allowed for that section.

    **Legacy**: a single flat object whose keys are any of ``_CLUSTER_LEGACY_KEYS``.
    """
    if not isinstance(sub, dict):
        raise TypeError("config['cluster'] must be an object")
    section_names = frozenset(_CLUSTER_SCHEMA_SECTIONS)
    top = set(sub.keys())
    nested = not top or top.issubset(section_names)
    if not nested:
        for k in sub:
            if k not in _CLUSTER_LEGACY_KEYS:
                raise ValueError(
                    f"Unknown key cluster.{k!r}. Use nested cluster with shared/mlip/vasp sections, "
                    f"or legacy flat keys allowed: {sorted(_CLUSTER_LEGACY_KEYS)}"
                )
        return dict(sub)
    out: Dict[str, Any] = {}
    for sec in ("shared", "mlip", "vasp"):
        if sec not in sub:
            continue
        block = sub[sec]
        if block is None:
            continue
        if not isinstance(block, dict):
            raise TypeError(f"config['cluster'][{sec!r}] must be an object")
        allowed = _CLUSTER_SCHEMA_SECTIONS[sec]
        for k, v in block.items():
            if k not in allowed:
                raise ValueError(
                    f"Unknown key cluster.{sec}.{k!r}. Allowed: {sorted(allowed)}"
                )
            if k in out:
                raise ValueError(f"Duplicate config key {k!r} in cluster")
            out[k] = v
    return out


def load_config_file(path: str) -> Dict[str, Any]:
    """Load JSON or YAML (YAML requires PyYAML: ``pip install pyyaml``)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    path_lower = path.lower()
    if path_lower.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return data
    return json.loads(raw)


def _ensure_mapping(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return deepcopy(value)


def _split_merged_optimization_settings(value: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = _ensure_mapping(value, "config['IO_workflow_config']['optimization_settings']")
    opt: Dict[str, Any] = {}
    gm: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in _GLOBAL_MINIMIZATION_KEYS:
            gm[k] = v
        else:
            opt[k] = v
    return opt, gm


def _cost_preset_overrides(double_interface: bool, preset_name: str) -> Dict[str, Dict[str, Any]]:
    mode = "without_vacuum" if double_interface else "with_vacuum"
    name = str(preset_name or "").strip().lower()
    if name not in _COST_PRESET_NAMES:
        raise ValueError(
            f"Unknown cost_preset={preset_name!r}. Allowed: {sorted(_COST_PRESET_NAMES)}"
        )
    return deepcopy(_COST_PRESET_OVERRIDES[mode].get(name, {}))


def _normalize_new_style_simple_config(config: Dict[str, Any]) -> Dict[str, Any]:
    unknown = [k for k in config if k not in _NEW_STYLE_TOP_LEVEL_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown top-level key(s) for simple_iomaker config: {unknown}. "
            f"Allowed: {sorted(_NEW_STYLE_TOP_LEVEL_KEYS)}"
        )

    workflow_name = str(config.get("workflow_name") or "").strip()
    if not workflow_name:
        raise ValueError("config['workflow_name'] is required and must be a non-empty string.")

    io_cfg = config.get("IO_workflow_config")
    if not isinstance(io_cfg, dict):
        raise TypeError("config['IO_workflow_config'] must be an object")

    unknown_io = [k for k in io_cfg if k not in _NEW_IO_WORKFLOW_KEYS]
    if unknown_io:
        raise ValueError(
            f"Unknown key(s) in config['IO_workflow_config']: {unknown_io}. "
            f"Allowed: {sorted(_NEW_IO_WORKFLOW_KEYS)}"
        )

    bulk = _ensure_mapping(io_cfg.get("bulk_cifs"), "config['IO_workflow_config']['bulk_cifs']")
    cost_preset = str(io_cfg.get("cost_preset") or "medium").strip().lower()
    lattice = _ensure_mapping(
        io_cfg.get("lattice_matching_settings"),
        "config['IO_workflow_config']['lattice_matching_settings']",
    )
    structure = _ensure_mapping(
        io_cfg.get("structure_settings"),
        "config['IO_workflow_config']['structure_settings']",
    )
    opt_patch, gm_patch = _split_merged_optimization_settings(
        io_cfg.get("optimization_settings")
    )
    vasp = _ensure_mapping(io_cfg.get("vasp_settings"), "config['IO_workflow_config']['vasp_settings']")

    unknown_vasp = [k for k in vasp if k not in _VASP_SETTINGS_KEYS]
    if unknown_vasp:
        raise ValueError(
            f"Unknown key(s) in config['IO_workflow_config']['vasp_settings']: {unknown_vasp}. "
            f"Allowed: {sorted(_VASP_SETTINGS_KEYS)}"
        )

    double_interface = bool(structure.get("double_interface", False))
    cost_overrides = _cost_preset_overrides(double_interface, cost_preset)
    preset = deepcopy(TUTORIAL_WITHOUT_VACUUM if double_interface else TUTORIAL_WITH_VACUUM)

    settings: Dict[str, Any] = {
        "name": workflow_name,
        "mode": "without_vacuum" if double_interface else "with_vacuum",
        "inputs": {"type": "local_cif"},
        "lattice_matching_settings": deepcopy(preset["lattice_matching_settings"]),
        "structure_settings": deepcopy(preset["structure_settings"]),
        "optimization_settings": deepcopy(preset["optimization_settings"]),
        "global_minimization_settings": deepcopy(preset["global_minimization_settings"]),
        "do_vasp": False,
        "bulk": {
            "film_cif": str(bulk.get("film_cif") or DEFAULT_FILM_CIF),
            "substrate_cif": str(bulk.get("substrate_cif") or DEFAULT_SUBSTRATE_CIF),
        },
        "cost_preset": cost_preset,
    }

    settings["lattice_matching_settings"].update(cost_overrides.get("lattice_matching_settings", {}))
    settings["structure_settings"].update(cost_overrides.get("structure_settings", {}))
    settings["optimization_settings"].update(cost_overrides.get("optimization_settings", {}))
    settings["global_minimization_settings"].update(
        cost_overrides.get("global_minimization_settings", {})
    )
    settings["lattice_matching_settings"].update(lattice)
    settings["structure_settings"].update(structure)
    settings["optimization_settings"].update(opt_patch)
    settings["global_minimization_settings"].update(gm_patch)

    settings["do_vasp"] = bool(vasp.pop("do_vasp", False))
    if "do_vasp_gd" in vasp:
        settings["do_vasp_gd"] = bool(vasp.pop("do_vasp_gd"))
    for k, v in vasp.items():
        settings[k] = v

    execution = config.get("execution", "server")
    if not isinstance(execution, str):
        raise TypeError("config['execution'] must be a string: 'local' or 'server'")
    run = execution.strip().lower().replace("-", "_")
    if run not in ("local", "server"):
        raise ValueError(f"Unknown execution={execution!r}. Use 'local' or 'server'.")

    normalized: Dict[str, Any] = {
        "settings": settings,
        "execution": {"run": run},
    }
    if "cluster" in config:
        normalized["cluster"] = config["cluster"]
    return normalized


def normalize_simple_iomaker_root_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either the new simple_iomaker schema or the legacy ``settings``-based one.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if "settings" in config:
        return config
    if "IO_workflow_config" in config or "workflow_name" in config:
        return _normalize_new_style_simple_config(config)
    raise ValueError(
        "Invalid simple_iomaker config shape. Use either the new "
        "{workflow_name, IO_workflow_config, execution, cluster} layout or the legacy "
        "{settings, execution, cluster} layout."
    )


def bundled_config_to_flat(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand ``execution`` / ``cluster`` bundles into a flat dict.

    ``cluster`` may be **nested** (``shared`` / ``mlip`` / ``vasp``) or **legacy flat**; see
    :func:`_flatten_cluster_bundle`.

    Root must include ``settings``. Legacy bundles (``physics``, ``task``, ``structure``, ``vasp``, ``overrides``) are rejected.
    """
    if not isinstance(d, dict):
        raise TypeError("config must be a dict")

    legacy = [k for k in _LEGACY_BUNDLE_KEYS if k in d and d[k] is not None]
    if legacy:
        raise ValueError(
            f"Remove legacy bundle keys {legacy!r}. Put all computation parameters in `settings` only; "
            f"use `execution` and `cluster` for run/output/Slurm/paths."
        )

    flat: Dict[str, Any] = {}
    used_bundles = [bn for bn in _BUNDLE_SCHEMA if bn in d and d[bn] is not None]

    for bname in used_bundles:
        sub = d[bname]
        if not isinstance(sub, dict):
            raise TypeError(f"config[{bname!r}] must be an object, got {type(sub).__name__}")
        if bname == "cluster":
            merged = _flatten_cluster_bundle(sub)
        else:
            allowed = _BUNDLE_SCHEMA[bname]
            merged = {}
            for k, v in sub.items():
                if k not in allowed:
                    raise ValueError(
                        f"Unknown key {bname!r}.{k!r}. Allowed: {sorted(allowed)}"
                    )
                merged[k] = v
        for k, v in merged.items():
            if k in flat:
                raise ValueError(
                    f"Duplicate config key {k!r} (appears more than once across bundles or top level)."
                )
            flat[k] = v

    for k, v in d.items():
        if k in _BUNDLE_SCHEMA:
            continue
        if k in flat:
            raise ValueError(
                f"Duplicate config key {k!r}: use either grouped bundles or flat keys, not both."
            )
        flat[k] = v

    return flat


def _validate_simple_config(flat: Dict[str, Any]) -> None:
    if "only_generate_flow" in flat:
        raise ValueError(
            "Invalid key 'only_generate_flow': use run: 'local' or 'server'."
        )
    st = flat.get("settings")
    if not isinstance(st, dict):
        raise TypeError("config must include a dict 'settings' with full IOMaker parameters.")

    if not uses_legacy_full_settings_dict(st):
        raise ValueError(
            "simple_iomaker requires a **complete** `settings` object "
            "(name, mode, inputs, lattice_matching_settings, structure_settings, "
            "optimization_settings, global_minimization_settings, do_vasp, …). "
            "Legacy `physics` / `task` / flat shortcuts are no longer supported."
        )

    for k in _FORBID_FLAT_WITH_SETTINGS:
        if k in flat and flat[k] is not None:
            raise ValueError(
                f"Do not set top-level {k!r}; put it inside `settings` (or under optimization_settings / …)."
            )

    for k in flat:
        if k not in _ALLOWED_INFRA_KEYS:
            raise ValueError(
                f"Unknown top-level key {k!r}. Allowed: {sorted(_ALLOWED_INFRA_KEYS)}"
            )


def _infer_mlip_use_cuda(settings: Dict[str, Any]) -> bool:
    opt = settings.get("optimization_settings") or {}
    dev = str(opt.get("device", "cpu")).lower()
    return dev in ("cuda", "gpu")


def _make_mlip_resources(
    *,
    use_cuda: bool,
    partition: str,
    cpus_per_gpu: int,
    cpus_per_task: int,
    gpus_per_job: int,
    extra_scheduler_kwargs: Optional[Dict[str, Any]],
) -> Callable[[], QResources]:
    def _fn() -> QResources:
        if use_cuda:
            cpg = max(1, int(cpus_per_gpu))
            g = max(1, int(gpus_per_job))
            sk: Dict[str, Any] = {
                "partition": partition,
                "qverbatim": f"#SBATCH --cpus-per-gpu={cpg}",
            }
            if extra_scheduler_kwargs:
                sk = {**sk, **extra_scheduler_kwargs}
            return QResources(
                nodes=1,
                processes_per_node=1,
                gpus_per_job=g,
                scheduler_kwargs=sk,
            )
        cpt = max(1, int(cpus_per_task))
        sk_cpu: Dict[str, Any] = {
            "partition": partition,
            "cpus_per_task": cpt,
        }
        if extra_scheduler_kwargs:
            sk_cpu = {**sk_cpu, **extra_scheduler_kwargs}
        return QResources(
            nodes=1,
            processes_per_node=1,
            scheduler_kwargs=sk_cpu,
        )

    return _fn


def build_config_from_simple_dict(d: Dict[str, Any], settings: Dict[str, Any]) -> BaseBuildConfig:
    """Build LocalBuildConfig or ServerBuildConfig from infrastructure keys only."""
    run = str(d.get("run", "server")).strip().lower().replace("-", "_")
    if run in ("flow_only", "json_only", "generate_only"):
        raise ValueError(
            f"run={d.get('run')!r} is no longer supported. Use 'local' or 'server'."
        )

    _bk = settings.get("bulk") if isinstance(settings.get("bulk"), dict) else {}
    film_cif = str(_bk.get("film_cif") or d.get("film_cif") or "film.cif")
    substrate_cif = str(_bk.get("substrate_cif") or d.get("substrate_cif") or "substrate.cif")
    out_json = d.get("output_flow_json", "io_flow.json")
    print_settings = bool(d.get("print_settings", True))

    server_run_parent = d.get("server_run_parent")
    # 默认用当前进程的解释器（如 Jupyter / CLI 所在 conda 环境），避免子进程 `python` 找不到 jobflow-remote
    server_pre_cmd = str(
        d.get("server_pre_cmd", os.getenv("INTEROPTIMUS_SERVER_PRE_CMD", ""))
    )
    server_python = str(
        d.get("server_python", os.getenv("INTEROPTIMUS_SERVER_PYTHON", sys.executable))
    )
    server_jf_bin = str(d.get("server_jf_bin", "jf"))

    mlip_worker = str(d.get("mlip_worker", "std_worker"))
    vasp_worker = str(d.get("vasp_worker", "std_worker"))
    mlip_project = str(d.get("mlip_project", "std"))

    partition = str(
        d.get("slurm_partition")
        or os.getenv("INTEROPTIMUS_SLURM_PARTITION")
        or "interactive"
    )
    cpus_per_gpu = int(d.get("cpus_per_gpu", 1))
    # MLIP only: one Slurm task, multiple CPUs (e.g. memory). Not used for VASP resources.
    mlip_cpus_per_task = int(
        d.get(
            "mlip_cpus_per_task",
            d.get("cpus_per_task", d.get("mlip_processes_per_node", d.get("mlip_ppn", 1))),
        )
    )
    gpus_per_job = int(d.get("gpus_per_job", 1))
    extra_sk = d.get("mlip_scheduler_kwargs") if isinstance(d.get("mlip_scheduler_kwargs"), dict) else None
    use_cuda = _infer_mlip_use_cuda(settings)
    mlip_resources = _make_mlip_resources(
        use_cuda=use_cuda,
        partition=partition,
        cpus_per_gpu=cpus_per_gpu,
        cpus_per_task=mlip_cpus_per_task,
        gpus_per_job=gpus_per_job,
        extra_scheduler_kwargs=extra_sk,
    )

    do_vasp = bool(settings.get("do_vasp", False))
    vasp_resources = None
    if do_vasp:
        vpart = str(d.get("vasp_slurm_partition", partition))
        v_nodes = int(d.get("vasp_nodes", 1))
        v_ppn = int(d.get("vasp_processes_per_node", 4))
        v_extra = d.get("vasp_scheduler_kwargs") if isinstance(d.get("vasp_scheduler_kwargs"), dict) else None

        def _vasp_res() -> QResources:
            # Parallelism from nodes × processes_per_node; do not inject cpus_per_task unless
            # the user adds it via vasp_scheduler_kwargs (some sites derive Slurm CPUs from ntasks).
            sk: Dict[str, Any] = {"partition": vpart}
            if v_extra:
                sk = {**sk, **v_extra}
            return QResources(nodes=v_nodes, processes_per_node=v_ppn, scheduler_kwargs=sk)

        vasp_resources = _vasp_res

    vasp_exec_cfg = d.get("vasp_exec_config")
    if isinstance(vasp_exec_cfg, dict):
        vasp_exec_config = vasp_exec_cfg
    elif d.get("vasp_pre_run"):
        vasp_exec_config = {"pre_run": str(d["vasp_pre_run"])}
    else:
        vasp_exec_config = None

    common: Dict[str, Any] = {
        "film_cif": film_cif,
        "substrate_cif": substrate_cif,
        "output_flow_json": out_json,
        "print_settings": print_settings,
        "server_run_parent": server_run_parent,
        "server_pre_cmd": server_pre_cmd,
        "server_python": server_python,
        "server_jf_bin": server_jf_bin,
        "mlip_resources": mlip_resources,
        "vasp_resources": vasp_resources,
        "mlip_worker": mlip_worker,
        "vasp_worker": vasp_worker,
        "mlip_project": mlip_project,
        "do_vasp_gd": bool(settings.get("do_vasp_gd", False)),
        "vasp_exec_config": vasp_exec_config,
    }

    if run == "local":
        return LocalBuildConfig(**common)

    if run == "server":
        return ServerBuildConfig(**common)

    raise ValueError(f"Unknown run={d.get('run')!r}. Use: local | server")


def run_simple_iomaker(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load structures, build Flow JSON, optionally run or submit (see ``run`` in config)."""
    normalized = normalize_simple_iomaker_root_config(config)
    flat = bundled_config_to_flat(normalized)
    _validate_simple_config(flat)
    settings = normalize_iomaker_settings_from_full_dict(flat["settings"])
    cfg = build_config_from_simple_dict(flat, settings)
    return execute_iomaker_from_settings(settings, cfg, user_prompt="")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build IOMaker io_flow.json from JSON/YAML config."
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to JSON or YAML config (see InterOptimus/agents/simple_iomaker.example.json)",
    )
    args = p.parse_args()
    cfg_dict = load_config_file(args.config)
    result = run_simple_iomaker(cfg_dict)
    print("✅ Flow JSON:", result["flow_json_path"])
    print("📦 Structures:", result["structures_meta"].get("source"))


if __name__ == "__main__":
    main()
