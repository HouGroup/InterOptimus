"""
Eqnorm ``simple_iomaker`` workflow for the desktop GUI (and any local caller; no HTTP server in this build).
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
from InterOptimus.desktop_app.result_bundle import finalize_desktop_result_bundle
from InterOptimus.jobflow import (
    load_opt_results_pickle_payload,
    materialize_pairs_best_it_dir,
    resolve_opt_results_pickle_path,
)


def sessions_root() -> Path:
    env = os.environ.get("INTEROPTIMUS_WEB_SESSIONS", "").strip()
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
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "workflow_name": workflow_name.strip() or "IO_web_eqnorm",
        "IO_workflow_config": {
            "cost_preset": cost_preset,
            "bulk_cifs": {
                "film_cif": "film.cif",
                "substrate_cif": "substrate.cif",
            },
            "lattice_matching_settings": lattice_matching_settings,
            "structure_settings": structure_settings,
            "optimization_settings": optimization_settings,
            "vasp_settings": {"do_vasp": False},
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

    ev = _form_get(form, "adv_eqnorm_variant", "").strip()
    if ev:
        optimization_settings["model_variant"] = ev

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


def run_eqnorm_session(
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

    ``form`` keys match the desktop GUI payload (string values).
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

    workflow_name = _form_get(form, "workflow_name", "IO_web_eqnorm")
    cost_preset = _form_get(form, "cost_preset", "medium")
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
    opt_steps = _form_get(form, "opt_steps", "500")
    do_mlip_gd = _form_get(form, "do_mlip_gd", "false")
    relax_in_ratio = _form_get(form, "relax_in_ratio", "true")
    relax_in_layers = _form_get(form, "relax_in_layers", "false")
    fix_film_fraction = _form_get(form, "fix_film_fraction", "0.5")
    fix_substrate_fraction = _form_get(form, "fix_substrate_fraction", "0.5")
    set_relax_film_ang = _form_get(form, "set_relax_film_ang", "0")
    set_relax_substrate_ang = _form_get(form, "set_relax_substrate_ang", "0")

    if cost_preset not in ("low", "medium", "high"):
        return {"ok": False, "session_id": sid, "error": "cost_preset must be low, medium, or high"}
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

    opt_steps_i = int(_parse_float(opt_steps, 500))
    gd = _parse_bool(do_mlip_gd)
    rir = _parse_bool(relax_in_ratio)
    ril = _parse_bool(relax_in_layers)

    optimization_settings: Dict[str, Any] = {
        "calc": "eqnorm",
        "device": dev,
        "steps": max(1, opt_steps_i),
        "do_mlip_gd": gd,
        "relax_in_ratio": rir,
        "relax_in_layers": ril,
        "model_name": "eqnorm",
        "model_variant": "eqnorm-mptrj",
    }

    if rir:
        ff = _clamp01(_parse_float(fix_film_fraction, 0.5))
        fs = _clamp01(_parse_float(fix_substrate_fraction, 0.5))
        optimization_settings["set_relax_thicknesses"] = (1.0 - ff, 1.0 - fs)
    else:
        optimization_settings["set_relax_thicknesses"] = (
            max(0.0, _parse_float(set_relax_film_ang, 0.0)),
            max(0.0, _parse_float(set_relax_substrate_ang, 0.0)),
        )

    _merge_advanced_mlip_form(optimization_settings, form)

    # Desktop / API: explicit MLIP backend (default Eqnorm). Keeps runs aligned with the GUI selection.
    mc_raw = _form_get(form, "mlip_calc", "eqnorm").strip()
    mc = _normalize_mlip_calc(mc_raw) or "eqnorm"
    optimization_settings["calc"] = mc
    if mc != "eqnorm":
        optimization_settings.pop("model_name", None)
        optimization_settings.pop("model_variant", None)
    else:
        optimization_settings.setdefault("model_name", "eqnorm")
        optimization_settings.setdefault("model_variant", "eqnorm-mptrj")

    print(f"[InterOptimus] MLIP backend: {mc}", flush=True)

    cluster = None
    if ex == "server":
        cluster = {
            "mlip": {
                "slurm_partition": os.environ.get("INTEROPTIMUS_SLURM_PARTITION", "interactive"),
            },
            "vasp": {},
        }

    config = _build_config(
        workflow_name=workflow_name,
        cost_preset=cost_preset,
        double_interface=di,
        execution=ex,
        cluster=cluster,
        lattice_matching_settings=lattice_matching_settings,
        structure_settings=structure_settings,
        optimization_settings=optimization_settings,
    )

    try:
        result = _run_in_workdir(workdir, config)
    except Exception as e:
        tb = traceback.format_exc()
        hint = (
            "Eqnorm / 其它 MLIP 权重通常放在 ~/.cache/InterOptimus/checkpoints/（eqnorm*.pt）。"
            "请确认已安装 eqnorm（见 https://github.com/yzchen08/eqnorm ）、torch、torch_geometric 与权重可用。"
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
            wd_display = str(finalize_desktop_result_bundle(wd_path))
            merged_report = Path(wd_display) / "io_report.txt"
            if merged_report.is_file():
                report_text = merged_report.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[InterOptimus] finalize_desktop_result_bundle failed: {e}", file=sys.stderr)
            wd_display = wd

    stereo = os.path.join(wd_display, "stereographic.jpg")
    stereo_html = os.path.join(wd_display, "stereographic_interactive.html")
    project_jpg = os.path.join(wd, "project.jpg")

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
        },
        "artifacts": {
            "stereographic_jpg": stereo if os.path.isfile(stereo) else None,
            "stereographic_interactive_html": stereo_html if os.path.isfile(stereo_html) else None,
            "project_jpg": project_jpg if os.path.isfile(project_jpg) else None,
            "local_workdir": wd_display,
            "pairs_count": n_pairs,
            "poscars_error": poscars_error,
        },
    }
    return payload
