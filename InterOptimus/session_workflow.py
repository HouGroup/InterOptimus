"""
Web (and headless) session driver: form / CIF bytes → ``run_simple_iomaker`` on disk.

Sessions live under :func:`sessions_root` (same layout as the browser UI worker).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from InterOptimus.agents.iomaker_core import _normalize_mlip_calc
from InterOptimus.agents.simple_iomaker import run_simple_iomaker
from InterOptimus.result_bundle import finalize_session_result_bundle
from InterOptimus.jobflow import (
    load_opt_results_pickle_payload,
    materialize_pairs_best_it_dir,
    resolve_opt_results_pickle_path,
)


def sessions_root() -> Path:
    env = (
        (os.environ.get("INTEROPTIMUS_DESKTOP_SESSIONS") or "").strip()
        or (os.environ.get("INTEROPTIMUS_WEB_SESSIONS") or "").strip()
    )
    root = Path(env) if env else (Path.home() / ".interoptimus" / "web_sessions")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")


def _parse_float(s: str, default: float) -> float:
    t = (s or "").strip()
    if not t:
        return default
    return float(t)


def _clamp01(x: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, x))


def _relax_ratios_from_form(form: Dict[str, Any]) -> tuple[float, float]:
    """
    Film / substrate **relax** fractions in ``relax_in_ratio`` mode: fraction of each slab
    thickness that participates in relaxation (stored as ``set_relax_thicknesses``; see
    ``itworker.parse_optimization_params``).

    Preferred keys: ``relax_film_ratio``, ``relax_substrate_ratio`` (0–1).

    Legacy keys ``fix_film_fraction`` / ``fix_substrate_fraction`` denoted **fixed**
    (non-relaxing) fraction; they are converted as ``relax_ratio = 1 - fix_fraction``.
    """
    has_new = "relax_film_ratio" in form or "relax_substrate_ratio" in form
    if has_new:
        return (
            _clamp01(_parse_float(_form_get(form, "relax_film_ratio", "0.5"), 0.5)),
            _clamp01(_parse_float(_form_get(form, "relax_substrate_ratio", "0.5"), 0.5)),
        )
    ff = _form_get(form, "fix_film_fraction", "0.5")
    fs = _form_get(form, "fix_substrate_fraction", "0.5")
    return (
        _clamp01(1.0 - _parse_float(ff, 0.5)),
        _clamp01(1.0 - _parse_float(fs, 0.5)),
    )


def _materialize_pair_poscars(workdir: Path) -> Tuple[Optional[Path], Optional[str]]:
    pkl = workdir / "opt_results.pkl"
    if not pkl.is_file():
        alt = resolve_opt_results_pickle_path(workdir)
        if alt and os.path.isfile(alt):
            pkl = Path(alt)
    if not pkl.is_file():
        return None, "opt_results.pkl not found"
    try:
        payload = load_opt_results_pickle_payload(str(pkl))
    except Exception as e:
        return None, f"Failed to load opt_results.pkl: {e}"
    opt = payload.get("opt_results")
    keys_raw = payload.get("materialize_pairs")
    if not isinstance(opt, dict):
        return None, "Invalid opt_results in pickle"
    pair_keys: List[Tuple[int, int]] = []
    if isinstance(keys_raw, list) and keys_raw:
        for item in keys_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pair_keys.append((int(item[0]), int(item[1])))
    if not pair_keys:
        for k, v in opt.items():
            if isinstance(k, tuple) and len(k) == 2:
                rb = (v or {}).get("relaxed_best_interface") or {}
                if rb.get("structure") is not None:
                    pair_keys.append((int(k[0]), int(k[1])))
        pair_keys.sort()
    if not pair_keys:
        return None, "No pairs to materialize"
    pairs_dir = workdir / "pairs_best_it"
    try:
        materialize_pairs_best_it_dir(
            opt,
            pair_keys,
            str(pairs_dir),
            double_interface=bool(payload.get("double_interface")),
            strain_E_correction=bool(payload.get("strain_E_correction")),
        )
    except Exception as e:
        return None, str(e)
    return pairs_dir, None


def _build_config(
    *,
    workflow_name: str,
    cost_preset: str,
    double_interface: bool,
    execution: str,
    cluster: Optional[Dict[str, Any]],
    lattice_matching_settings: Dict[str, Any],
    structure_settings: Dict[str, Any],
    optimization_settings: Dict[str, Any],
    vasp_settings: Optional[Dict[str, Any]] = None,
    do_vasp: bool = False,
) -> Dict[str, Any]:
    vsettings: Dict[str, Any] = {"do_vasp": bool(do_vasp)}
    if vasp_settings:
        for k, v in vasp_settings.items():
            if v is None or v == "" or v == {} or v == []:
                continue
            vsettings[k] = v
        vsettings["do_vasp"] = bool(do_vasp)
    cfg: Dict[str, Any] = {
        "workflow_name": workflow_name.strip() or "IO_web",
        "IO_workflow_config": {
            "cost_preset": cost_preset,
            "bulk_cifs": {
                "film_cif": "film.cif",
                "substrate_cif": "substrate.cif",
            },
            "lattice_matching_settings": lattice_matching_settings,
            "structure_settings": structure_settings,
            "optimization_settings": optimization_settings,
            "vasp_settings": vsettings,
        },
        "execution": execution,
    }
    if cluster:
        cfg["cluster"] = cluster
    return cfg


def _run_in_workdir(workdir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    old = os.getcwd()
    try:
        os.chdir(workdir)
        return run_simple_iomaker(config)
    finally:
        os.chdir(old)


def _json_safe_result(result: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"flow_dict", "settings"}
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = f"<{type(v).__name__}>"
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = str(v)[:500]
    return out


def _run_dir_from_result(result: Dict[str, Any], session_workdir: Path) -> str:
    """
    Directory that actually contains flow outputs (nested slug dir under the session).

    ``result['local_workdir']`` is usually correct; if it is missing or points at the session
    root while artifacts live in a subdirectory, fall back to ``opt_results_pkl``'s parent or
    a shallow search under the session folder.
    """
    lv = result.get("local_workdir")
    if isinstance(lv, str) and lv.strip():
        root = str(Path(lv).expanduser().resolve(strict=False))
        if os.path.isfile(os.path.join(root, "stereographic.jpg")) or os.path.isfile(
            os.path.join(root, "opt_results.pkl")
        ) or os.path.isfile(os.path.join(root, "opt_results_.pkl")):
            return root
    op = result.get("opt_results_pkl")
    if isinstance(op, str) and op.strip():
        p = Path(op).expanduser()
        try:
            if p.is_file():
                return str(p.parent.resolve())
        except OSError:
            pass
    try:
        for sub in sorted(session_workdir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                if (
                    (sub / "stereographic.jpg").is_file()
                    or (sub / "opt_results.pkl").is_file()
                    or (sub / "opt_results_.pkl").is_file()
                ):
                    return str(sub.resolve())
    except OSError:
        pass
    if isinstance(lv, str) and lv.strip():
        return str(Path(lv).expanduser().resolve(strict=False))
    return str(session_workdir.resolve())


def _form_get(form: Dict[str, Any], key: str, default: str = "") -> str:
    v = form.get(key, default)
    return default if v is None else str(v)


def _parse_int_opt(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _coerce_scalar(v: str) -> Any:
    s = v.strip()
    if not s:
        return ""
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_json_or_keyvals(text: str) -> Dict[str, Any]:
    """
    Accept either a JSON object or a textarea with one ``KEY = VAL`` per line.

    Used for ``relax_user_incar_settings`` / ``vasp_scheduler_kwargs`` etc.
    Returns ``{}`` if the input is empty / blank. Raises ``ValueError`` on a
    malformed JSON object.
    """
    s = (text or "").strip()
    if not s:
        return {}
    if s[0] in "{[":
        out = json.loads(s)
        if not isinstance(out, dict):
            raise ValueError("Expected a JSON object")
        return out
    out: Dict[str, Any] = {}
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse line as KEY=VALUE: {raw!r}")
            k, v = parts
        out[k.strip()] = _coerce_scalar(v)
    return out


def _build_cluster_from_form(form: Dict[str, Any], *, want_vasp: bool) -> Dict[str, Any]:
    """Build a ``cluster`` dict (``mlip`` + ``vasp`` sections) from web-form keys."""
    mlip: Dict[str, Any] = {}
    partition = _form_get(form, "slurm_partition", "").strip()
    if not partition:
        partition = (os.environ.get("INTEROPTIMUS_SLURM_PARTITION") or "interactive").strip()
    mlip["slurm_partition"] = partition

    cpg = _parse_int_opt(_form_get(form, "cpus_per_gpu", ""))
    if cpg is not None:
        mlip["cpus_per_gpu"] = max(1, cpg)
    mcpu = _parse_int_opt(_form_get(form, "mlip_cpus_per_task", ""))
    if mcpu is not None:
        mlip["mlip_cpus_per_task"] = max(1, mcpu)
    gpj = _parse_int_opt(_form_get(form, "gpus_per_job", ""))
    if gpj is not None:
        mlip["gpus_per_job"] = max(1, gpj)

    for k_form, k_cfg in (
        ("server_pre_cmd", "server_pre_cmd"),
        ("server_run_parent", "server_run_parent"),
        ("server_python", "server_python"),
        ("server_jf_bin", "server_jf_bin"),
        ("mlip_worker", "mlip_worker"),
        ("mlip_project", "mlip_project"),
    ):
        v = _form_get(form, k_form, "").strip()
        if v:
            mlip[k_cfg] = v

    extra_sk_text = _form_get(form, "mlip_scheduler_kwargs", "").strip()
    if extra_sk_text:
        try:
            extra_sk = _parse_json_or_keyvals(extra_sk_text)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"mlip_scheduler_kwargs 解析失败: {e}") from e
        if extra_sk:
            mlip["mlip_scheduler_kwargs"] = extra_sk

    cluster: Dict[str, Any] = {"mlip": mlip}

    if want_vasp:
        vasp: Dict[str, Any] = {}
        vpart = _form_get(form, "vasp_slurm_partition", "").strip()
        if vpart:
            vasp["vasp_slurm_partition"] = vpart
        vn = _parse_int_opt(_form_get(form, "vasp_nodes", ""))
        if vn is not None:
            vasp["vasp_nodes"] = max(1, vn)
        vppn = _parse_int_opt(_form_get(form, "vasp_processes_per_node", ""))
        if vppn is not None:
            vasp["vasp_processes_per_node"] = max(1, vppn)
        vw = _form_get(form, "vasp_worker", "").strip()
        if vw:
            vasp["vasp_worker"] = vw
        vpre = _form_get(form, "vasp_pre_run", "").strip()
        if vpre:
            vasp["vasp_pre_run"] = vpre
        cluster["vasp"] = vasp

    return cluster


def _build_vasp_settings_from_form(form: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the contents of ``IO_workflow_config.vasp_settings`` from the web form.
    INCAR / KPOINTS / POTCAR fields accept JSON or ``KEY=VAL`` lines; pair selection
    and dipole correction are simple strings.
    """
    out: Dict[str, Any] = {}

    sel = _form_get(form, "vasp_pair_selection", "").strip()
    if sel:
        out["vasp_pair_selection"] = sel

    dgd = _form_get(form, "do_vasp_gd", "").strip().lower()
    if dgd in ("1", "true", "yes", "on"):
        out["do_vasp_gd"] = True
    elif dgd in ("0", "false", "no", "off"):
        out["do_vasp_gd"] = False

    dipole = _form_get(form, "vasp_dipole_correction", "").strip().lower()
    if dipole in ("true", "1", "yes", "on"):
        out["vasp_dipole_correction"] = True
    elif dipole in ("false", "0", "no", "off"):
        out["vasp_dipole_correction"] = False

    funct = _form_get(form, "relax_user_potcar_functional", "").strip()
    if funct:
        out["relax_user_potcar_functional"] = funct
    funct_s = _form_get(form, "static_user_potcar_functional", "").strip()
    if funct_s:
        out["static_user_potcar_functional"] = funct_s

    object_keys = (
        "relax_user_incar_settings",
        "relax_user_potcar_settings",
        "relax_user_kpoints_settings",
        "static_user_incar_settings",
        "static_user_potcar_settings",
        "static_user_kpoints_settings",
        "vasp_gd_kwargs",
        "vasp_relax_settings",
        "vasp_static_settings",
    )
    for key in object_keys:
        text = _form_get(form, key, "").strip()
        if not text:
            continue
        try:
            parsed = _parse_json_or_keyvals(text)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{key} 解析失败: {e}") from e
        if parsed:
            out[key] = parsed

    return out


def vasp_runtime_effective_dict(vasp_settings: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Effective VASP inputs matching what :class:`~InterOptimus.jobflow.IOMaker` passes to
    ``MPRelaxSet`` / ``MPStaticSet``: InterOptimus default INCAR merged with user
    ``relax_user_incar_settings`` / ``static_user_incar_settings``. POTCAR fields are
    user overrides only (empty dict ⇒ pymatgen chooses per species and functional).

    Pass ``None`` when VASP is not used.
    """
    if vasp_settings is None:
        return None
    from InterOptimus.itworker import default_relax_incar_settings, default_static_incar_settings

    def _as_dict(v: Any) -> Optional[Dict[str, Any]]:
        if isinstance(v, dict):
            return dict(v)
        return None

    vs = vasp_settings
    r_i = _as_dict(vs.get("relax_user_incar_settings"))
    s_i = _as_dict(vs.get("static_user_incar_settings"))
    r_p = _as_dict(vs.get("relax_user_potcar_settings")) or {}
    s_p = _as_dict(vs.get("static_user_potcar_settings")) or {}
    relax_fn = vs.get("relax_user_potcar_functional") or "PBE_54"
    static_fn = vs.get("static_user_potcar_functional") or relax_fn
    relax_incar = default_relax_incar_settings(r_i, num_atoms=None)
    static_incar = default_static_incar_settings(s_i, num_atoms=None)
    return {
        "relax_incar_merged": relax_incar,
        "static_incar_merged": static_incar,
        "relax_potcar_overrides": r_p,
        "static_potcar_overrides": s_p,
        "relax_potcar_functional": str(relax_fn),
        "static_potcar_functional": str(static_fn),
        "potcar_note": (
            "POTCAR 此处仅显示表单中的元素级覆盖；空字典表示不覆盖，由 "
            "MPRelaxSet/MPStaticSet 按结构元素与上述 functional 自动选择赝势。"
        ),
    }


def vasp_runtime_effective_from_form(form: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Preview merged VASP settings from raw form (when ``web_result.json`` is not written yet)."""
    if not _parse_bool(_form_get(form, "do_vasp", "false")):
        return None
    try:
        vs = _build_vasp_settings_from_form(form)
    except ValueError:
        return None
    return vasp_runtime_effective_dict(vs)


def _merge_advanced_mlip_form(optimization_settings: Dict[str, Any], form: Dict[str, Any]) -> None:
    """
    Optional MLIP / global-minimization overrides merged into ``optimization_settings``.
    :func:`InterOptimus.agents.simple_iomaker._split_merged_optimization_settings` sends
    ``n_calls_density``, ``z_range``, ``strain_E_correction``, ``term_screen_tol``, ``calc``
    into ``global_minimization_settings``; other keys stay in ``optimization_settings``.
    """
    ck = _form_get(form, "adv_ckpt_path", "").strip()
    if ck:
        optimization_settings["ckpt_path"] = ck

    for fk, ok in (
        ("adv_fmax", "fmax"),
        ("adv_discut", "discut"),
        ("adv_gd_tol", "gd_tol"),
        ("adv_bo_coord_bin", "BO_coord_bin_size"),
        ("adv_bo_energy_bin", "BO_energy_bin_size"),
        ("adv_bo_rms_bin", "BO_rms_bin_size"),
    ):
        s = _form_get(form, fk, "").strip()
        if not s:
            continue
        try:
            optimization_settings[ok] = float(s)
        except ValueError:
            pass

    ncd = _form_get(form, "adv_n_calls_density", "").strip()
    if ncd:
        try:
            optimization_settings["n_calls_density"] = float(ncd)
        except ValueError:
            pass

    se = _form_get(form, "adv_strain_E_correction", "").strip().lower()
    if se in ("true", "false"):
        optimization_settings["strain_E_correction"] = se == "true"

    tst = _form_get(form, "adv_term_screen_tol", "").strip()
    if tst:
        try:
            optimization_settings["term_screen_tol"] = float(tst)
        except ValueError:
            pass

    zlo = _form_get(form, "adv_z_range_lo", "").strip()
    zhi = _form_get(form, "adv_z_range_hi", "").strip()
    if zlo and zhi:
        try:
            optimization_settings["z_range"] = [float(zlo), float(zhi)]
        except ValueError:
            pass


def run_iomaker_session(
    *,
    film_cif_path: Path | None = None,
    substrate_cif_path: Path | None = None,
    film_bytes: bytes | None = None,
    substrate_bytes: bytes | None = None,
    form: Dict[str, Any],
    session_id: str | None = None,
) -> Dict[str, Any]:
    """
    Copy CIFs into a session workdir, build config, call :func:`run_simple_iomaker`,
    optionally materialize POSCAR zip for local execution.

    Pass either ``film_cif_path`` + ``substrate_cif_path``, or ``film_bytes`` + ``substrate_bytes``.

    ``form`` keys match the web form payload (string values).
    """
    sid = session_id or str(uuid.uuid4())
    workdir = sessions_root() / sid
    workdir.mkdir(parents=True, exist_ok=True)

    film_path = workdir / "film.cif"
    sub_path = workdir / "substrate.cif"
    try:
        if film_bytes is not None and substrate_bytes is not None:
            film_path.write_bytes(film_bytes)
            sub_path.write_bytes(substrate_bytes)
        elif film_cif_path is not None and substrate_cif_path is not None:
            shutil.copy2(film_cif_path, film_path)
            shutil.copy2(substrate_cif_path, sub_path)
        else:
            return {
                "ok": False,
                "session_id": sid,
                "error": "Provide film/substrate paths or bytes.",
                "workdir": str(workdir.resolve()),
            }
    except OSError as e:
        return {
            "ok": False,
            "session_id": sid,
            "error": f"Failed to copy CIFs: {e}",
            "workdir": str(workdir.resolve()),
        }

    workflow_name = _form_get(form, "workflow_name", "IO_web")
    # Web 表单不再暴露 low/medium/high（易引起误解）；固定按 medium 档位合并默认。
    cost_preset = "medium"
    double_interface = _form_get(form, "double_interface", "true")
    execution = _form_get(form, "execution", "local")
    lm_max_area = _form_get(form, "lm_max_area", "20")
    lm_max_length_tol = _form_get(form, "lm_max_length_tol", "0.03")
    lm_max_angle_tol = _form_get(form, "lm_max_angle_tol", "0.03")
    lm_film_max_miller = _form_get(form, "lm_film_max_miller", "3")
    lm_substrate_max_miller = _form_get(form, "lm_substrate_max_miller", "3")
    st_film_thickness = _form_get(form, "st_film_thickness", "10")
    st_substrate_thickness = _form_get(form, "st_substrate_thickness", "10")
    st_termination_ftol = _form_get(form, "st_termination_ftol", "0.15")
    st_vacuum_over_film = _form_get(form, "st_vacuum_over_film", "5")
    opt_device = _form_get(form, "opt_device", "cpu")
    do_vasp = _parse_bool(_form_get(form, "do_vasp", "false"))
    adv_steps = _form_get(form, "adv_steps", "").strip()
    if not adv_steps:
        adv_steps = _form_get(form, "opt_steps", "500")
    do_mlip_gd = _form_get(form, "do_mlip_gd", "false")
    relax_in_ratio = _form_get(form, "relax_in_ratio", "true")
    relax_in_layers = _form_get(form, "relax_in_layers", "false")
    set_relax_film_ang = _form_get(form, "set_relax_film_ang", "0")
    set_relax_substrate_ang = _form_get(form, "set_relax_substrate_ang", "0")

    ex = execution.strip().lower().replace("-", "_")
    if ex not in ("local", "server"):
        return {"ok": False, "session_id": sid, "error": "execution must be local or server"}
    di = _parse_bool(double_interface)

    lattice_matching_settings: Dict[str, Any] = {
        "max_area": _parse_float(lm_max_area, 20.0),
        "max_length_tol": _parse_float(lm_max_length_tol, 0.03),
        "max_angle_tol": _parse_float(lm_max_angle_tol, 0.03),
        "film_max_miller": int(_parse_float(lm_film_max_miller, 3)),
        "substrate_max_miller": int(_parse_float(lm_substrate_max_miller, 3)),
        "film_millers": None,
        "substrate_millers": None,
    }

    structure_settings: Dict[str, Any] = {
        "film_thickness": _parse_float(st_film_thickness, 10.0),
        "substrate_thickness": _parse_float(st_substrate_thickness, 10.0),
        "termination_ftol": _parse_float(st_termination_ftol, 0.15),
        "vacuum_over_film": _parse_float(st_vacuum_over_film, 5.0),
        "double_interface": di,
    }

    dev = (opt_device or "cpu").strip().lower()
    if dev not in ("cpu", "cuda", "gpu"):
        return {"ok": False, "session_id": sid, "error": "opt_device must be cpu, cuda, or gpu"}
    if dev == "gpu":
        dev = "cuda"

    opt_steps_i = int(_parse_float(adv_steps, 500))
    gd = _parse_bool(do_mlip_gd)
    rir = _parse_bool(relax_in_ratio)
    ril = _parse_bool(relax_in_layers)

    optimization_settings: Dict[str, Any] = {
        "calc": "orb-models",
        "device": dev,
        "steps": max(1, opt_steps_i),
        "do_mlip_gd": gd,
        "relax_in_ratio": rir,
        "relax_in_layers": ril,
    }

    if rir:
        rf, rs = _relax_ratios_from_form(form)
        optimization_settings["set_relax_thicknesses"] = (rf, rs)
    else:
        optimization_settings["set_relax_thicknesses"] = (
            max(0.0, _parse_float(set_relax_film_ang, 0.0)),
            max(0.0, _parse_float(set_relax_substrate_ang, 0.0)),
        )

    _merge_advanced_mlip_form(optimization_settings, form)

    # Web form: explicit MLIP backend (default orb-models, same as HTML select).
    mc_raw = _form_get(form, "mlip_calc", "orb-models").strip()
    mc = _normalize_mlip_calc(mc_raw) or "orb-models"
    optimization_settings["calc"] = mc
    optimization_settings.pop("model_name", None)
    optimization_settings.pop("model_variant", None)

    print(f"[InterOptimus] MLIP backend: {mc}", flush=True)

    if mc == "matris":
        mm = _form_get(form, "matris_model", "").strip()
        mt = _form_get(form, "matris_task", "").strip()
        if mm:
            optimization_settings["model"] = mm
        if mt:
            optimization_settings["task"] = mt

    cluster = None
    if ex == "server":
        try:
            cluster = _build_cluster_from_form(form, want_vasp=do_vasp)
        except ValueError as e:
            return {"ok": False, "session_id": sid, "error": str(e), "workdir": str(workdir.resolve())}

    try:
        vasp_settings = _build_vasp_settings_from_form(form)
    except ValueError as e:
        return {"ok": False, "session_id": sid, "error": str(e), "workdir": str(workdir.resolve())}

    config = _build_config(
        workflow_name=workflow_name,
        cost_preset=cost_preset,
        double_interface=di,
        execution=ex,
        cluster=cluster,
        lattice_matching_settings=lattice_matching_settings,
        structure_settings=structure_settings,
        optimization_settings=optimization_settings,
        vasp_settings=vasp_settings,
        do_vasp=do_vasp,
    )

    try:
        result = _run_in_workdir(workdir, config)
    except Exception as e:
        tb = traceback.format_exc()
        hint = (
            "各 MLIP 权重可放在 ~/.cache/InterOptimus/checkpoints/（如 orb-v3-*.ckpt、*sevennet*.pth、"
            "dpa*.pb）。请确认已安装所选后端（orb_models / sevennet / matris / deepmd-kit）及 PyTorch。"
        )
        return {
            "ok": False,
            "session_id": sid,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb,
            "python_executable": sys.executable,
            "hint": hint,
            "workdir": str(workdir.resolve()),
        }

    wd = _run_dir_from_result(result, workdir)
    report_path = result.get("report_path")
    report_text = ""
    if report_path and os.path.isfile(report_path):
        try:
            report_text = Path(report_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            report_text = ""

    poscars_error: Optional[str] = None
    n_pairs: Optional[int] = None
    wd_path = Path(wd)
    if ex == "local":
        _, err = _materialize_pair_poscars(wd_path)
        if err:
            poscars_error = err
        pb = wd_path / "pairs_best_it"
        if pb.is_dir():
            try:
                n_pairs = len([p for p in pb.iterdir() if p.is_dir() and p.name.startswith("match_")])
            except OSError:
                pass

    wd_display = wd
    if ex == "local":
        try:
            wd_display = str(finalize_session_result_bundle(wd_path))
            merged_report = Path(wd_display) / "io_report.txt"
            if merged_report.is_file():
                report_text = merged_report.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[InterOptimus] finalize_session_result_bundle failed: {e}", file=sys.stderr)
            wd_display = wd

    try:
        wd_resolved = str(Path(wd_display).expanduser().resolve(strict=False))
    except OSError:
        wd_resolved = str(wd_display)
    try:
        job_root = str(Path(wd).expanduser().resolve(strict=False))
    except OSError:
        job_root = str(wd)
    stereo = os.path.join(wd_resolved, "stereographic.jpg")
    stereo_html = os.path.join(wd_resolved, "stereographic_interactive.html")
    project_jpg = os.path.join(job_root, "project.jpg")

    payload = {
        "ok": True,
        "session_id": sid,
        "result": _json_safe_result(result),
        "report_text": report_text,
        "config_effective": {
            "lattice_matching_settings": lattice_matching_settings,
            "structure_settings": structure_settings,
            "optimization_settings": {
                **optimization_settings,
                "set_relax_thicknesses": list(optimization_settings.get("set_relax_thicknesses", ()))
                if isinstance(optimization_settings.get("set_relax_thicknesses"), tuple)
                else optimization_settings.get("set_relax_thicknesses"),
            },
            "vasp_settings": vasp_settings,
            "vasp_runtime_effective": vasp_runtime_effective_dict(vasp_settings if do_vasp else None),
            "cluster": cluster,
        },
        "artifacts": {
            "stereographic_jpg": stereo if os.path.isfile(stereo) else None,
            "stereographic_interactive_html": stereo_html if os.path.isfile(stereo_html) else None,
            "project_jpg": project_jpg if os.path.isfile(project_jpg) else None,
            "local_workdir": wd_resolved,
            "pairs_count": n_pairs,
            "poscars_error": poscars_error,
        },
    }
    return payload
