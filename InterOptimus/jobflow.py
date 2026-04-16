"""
InterOptimus Jobflow Module

This module defines jobflow-based workflows for interface optimization
calculations, including gradient descent optimization and VASP calculations
with automatic resource management.
"""

from __future__ import annotations

from pymatgen.core.structure import Structure
from jobflow import Flow, Response, job, Maker
from qtoolkit.core.data_objects import QResources
import os
from InterOptimus.itworker import InterfaceWorker
from InterOptimus.mlip import resolve_mlip_checkpoint
from dataclasses import dataclass
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.interface import Interface
from typing import Callable, Dict, Any, Optional, List, Tuple
import re
import numpy as np
import pickle
from jobflow_remote import set_run_config
import base64
import json
import datetime


def _hpc_job_ids() -> Dict[str, str]:
    """Best-effort scheduler job id for remote runs (SLURM/PBS/LSF)."""
    out: Dict[str, str] = {}
    for key in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID"):
        v = os.environ.get(key)
        if v:
            out[key] = v
    return out


def _write_opt_results_summary_json(cwd: str, opt_results: Dict) -> None:
    """
    Write JSON-serializable energy / coordinate summaries (no pymatgen structures).

    Full structures also go to ``opt_results.pkl`` on disk; this JSON is for quick inspection
    without unpickling.
    """
    out: Dict[str, Any] = {}
    for (i, j), d in opt_results.items():
        key = f"match_{i}_term_{j}"
        od: Dict[str, Any] = {"match_id": int(i), "term_id": int(j)}
        for name in (
            "relaxed_min_it_E",
            "relaxed_min_bd_E",
            "film_atom_count",
            "substrate_atom_count",
            "match_area",
            "strain",
        ):
            if name not in d or d[name] is None:
                continue
            v = d[name]
            try:
                od[name] = float(v) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                od[name] = v
        if "supcl_E" in d:
            try:
                od["supcl_E"] = [float(x) for x in d["supcl_E"]]
            except Exception:
                pass
        for arr_name in ("xyzs_frac", "xyzs_cart"):
            if arr_name not in d:
                continue
            try:
                od[arr_name] = np.asarray(d[arr_name]).tolist()
            except Exception:
                pass
        if "relaxed_interface_sup_Es" in d:
            try:
                od["relaxed_interface_sup_Es"] = [float(x) for x in d["relaxed_interface_sup_Es"]]
            except Exception:
                pass
        rb = d.get("relaxed_best_interface")
        if isinstance(rb, dict) and "e" in rb:
            try:
                od["relaxed_best_interface_E"] = float(rb["e"])
            except (TypeError, ValueError):
                od["relaxed_best_interface_E"] = rb["e"]
        out[key] = od
    path = os.path.join(cwd, "opt_results_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _ensure_structure_for_export(obj: Any) -> Optional[Structure]:
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        return Structure.from_dict(obj)
    if isinstance(obj, str):
        try:
            return Structure.from_dict(json.loads(obj))
        except Exception:
            return None
    return None


def materialize_pairs_best_it_dir(
    opt_results: Dict[Any, Any],
    pair_keys: List[Tuple[int, int]],
    pairs_dir: str,
    *,
    double_interface: bool,
    strain_E_correction: bool,
) -> None:
    """
    Write ``match_<i>_term_<j>/`` POSCAR trees under *pairs_dir* (legacy layout).

    Used after :func:`fetch_interoptimus_task_results` loads ``opt_results.pkl`` so the
    compute job does not create ``pairs_best_it/`` on the cluster filesystem.
    """
    os.makedirs(pairs_dir, exist_ok=True)
    for (i, j) in pair_keys:
        pair_dir = os.path.join(pairs_dir, f"match_{i}_term_{j}")
        os.makedirs(pair_dir, exist_ok=True)

        best_it_obj = opt_results.get((i, j), {}).get("relaxed_best_interface", {}).get("structure")
        best_it = _ensure_structure_for_export(best_it_obj)
        if best_it is not None:
            best_it.to(fmt="poscar", filename=os.path.join(pair_dir, "best_it_POSCAR"))

        if not double_interface:
            slabs = opt_results.get((i, j), {}).get("slabs", {})
            film_slab = _ensure_structure_for_export(slabs.get("film", {}).get("structure"))
            substrate_slab = _ensure_structure_for_export(slabs.get("substrate", {}).get("structure"))
            if film_slab is not None:
                film_slab.to(fmt="poscar", filename=os.path.join(pair_dir, "film_slab_POSCAR"))
            if substrate_slab is not None:
                substrate_slab.to(fmt="poscar", filename=os.path.join(pair_dir, "substrate_slab_POSCAR"))

        if strain_E_correction:
            sfilm_obj = opt_results.get((i, j), {}).get("strain_film")
            sfilm = _ensure_structure_for_export(sfilm_obj)
            if sfilm is not None:
                sfilm.to(fmt="poscar", filename=os.path.join(pair_dir, "sfilm_POSCAR"))


def _normalize_opt_results_pickle_payload(raw: Any) -> Dict[str, Any]:
    """Accept new wrapper dict or legacy raw ``opt_results`` mapping."""
    if isinstance(raw, dict) and "opt_results" in raw:
        return raw
    if not isinstance(raw, dict):
        raise TypeError("opt_results.pkl: expected dict or wrapped payload")
    opt_results = raw
    pair_keys: List[Tuple[int, int]] = []
    for k, v in opt_results.items():
        if not isinstance(k, tuple) or len(k) != 2:
            continue
        rb = (v or {}).get("relaxed_best_interface") or {}
        if rb.get("structure") is not None:
            pair_keys.append((int(k[0]), int(k[1])))
    pair_keys.sort()
    has_it = any((opt_results.get(k) or {}).get("relaxed_min_it_E") is not None for k in pair_keys)
    has_bd = any((opt_results.get(k) or {}).get("relaxed_min_bd_E") is not None for k in pair_keys)
    if has_it and not has_bd:
        double_interface = True
    elif has_bd and not has_it:
        double_interface = False
    else:
        double_interface = True
    strain_E_correction = any(
        (opt_results.get(k) or {}).get("strain_film") is not None for k in pair_keys
    )
    return {
        "version": 0,
        "opt_results": opt_results,
        "materialize_pairs": pair_keys,
        "double_interface": double_interface,
        "strain_E_correction": strain_E_correction,
    }


def load_opt_results_pickle_payload(path: str) -> Dict[str, Any]:
    """Load ``opt_results.pkl`` from a completed run (wrapper or legacy format)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _normalize_opt_results_pickle_payload(raw)


def _write_opt_results_pickle(
    cwd: str,
    iw: InterfaceWorker,
    pair_keys: List[Tuple[int, int]],
) -> None:
    path = os.path.join(cwd, "opt_results.pkl")
    payload = {
        "version": 1,
        "opt_results": iw.opt_results,
        "materialize_pairs": list(pair_keys),
        "double_interface": bool(iw.double_interface),
        "strain_E_correction": bool(iw.strain_E_correction),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _write_remote_io_summaries(
    *,
    cwd: str,
    pairs_dir: str,
    summary_path: str,
    pairs_summary: List[dict],
    film_conv,
    substrate_conv,
    double_interface: bool,
    lattice_matching_settings: Optional[Dict[str, Any]],
    optimization_settings: Optional[Dict[str, Any]],
    global_minimization_settings: Optional[Dict[str, Any]],
    do_vasp: bool,
) -> None:
    """
    Write io_remote_summary.md + io_remote_summary.json for remote / batch users:
    one place to open after rsync or in the job directory.
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    lm = lattice_matching_settings or {}
    opt = optimization_settings or {}
    gm = global_minimization_settings or {}

    def _rel(p: str) -> str:
        try:
            return os.path.relpath(p, cwd)
        except ValueError:
            return p

    ckpt = opt.get("ckpt_path")
    ckpt_base = os.path.basename(str(ckpt)) if ckpt else None

    payload: Dict[str, Any] = {
        "generated_at_utc": utc_now.isoformat(),
        "working_directory": os.path.abspath(cwd),
        "pairs_directory": os.path.abspath(pairs_dir),
        "film_reduced_formula": getattr(film_conv.composition, "reduced_formula", None),
        "substrate_reduced_formula": getattr(substrate_conv.composition, "reduced_formula", None),
        "double_interface": bool(double_interface),
        "scheduler": _hpc_job_ids(),
        "settings": {
            "max_area": lm.get("max_area"),
            "calc": gm.get("calc"),
            "ckpt_path": ckpt,
            "ckpt_basename": ckpt_base,
            "do_mlip_gd": opt.get("do_mlip_gd"),
            "do_vasp": do_vasp,
        },
        "pairs": pairs_summary,
        "artifacts": {
            "io_report": _rel(os.path.join(cwd, "io_report.txt")),
            "pairs_summary": _rel(summary_path) if os.path.isfile(summary_path) else None,
            "pairs_best_it_dir": _rel(os.path.join(cwd, "pairs_best_it"))
            if os.path.isdir(os.path.join(cwd, "pairs_best_it"))
            else None,
            "opt_results_pickle": "opt_results.pkl" if os.path.isfile(os.path.join(cwd, "opt_results.pkl")) else None,
            "opt_results_summary": "opt_results_summary.json"
            if os.path.isfile(os.path.join(cwd, "opt_results_summary.json"))
            else None,
            "area_strain": "area_strain" if os.path.isfile(os.path.join(cwd, "area_strain")) else None,
            "project_image": "project.jpg" if os.path.isfile(os.path.join(cwd, "project.jpg")) else None,
            "stereographic_image": "stereographic.jpg"
            if os.path.isfile(os.path.join(cwd, "stereographic.jpg"))
            else None,
            "stereographic_interactive": "stereographic_interactive.html"
            if os.path.isfile(os.path.join(cwd, "stereographic_interactive.html"))
            else None,
            "unique_matches_plot": "unique_matches.jpg" if os.path.isfile(os.path.join(cwd, "unique_matches.jpg")) else None,
        },
    }

    json_path = os.path.join(cwd, "io_remote_summary.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

    # Markdown: readable without tooling
    lines_md: List[str] = []
    lines_md.append("# InterOptimus remote summary")
    lines_md.append("")
    lines_md.append(f"- **UTC time:** {payload['generated_at_utc']}")
    lines_md.append(f"- **Working directory:** `{payload['working_directory']}`")
    if payload["scheduler"]:
        sch_s = ", ".join(f"{k}={v}" for k, v in payload["scheduler"].items())
        lines_md.append(f"- **Scheduler:** {sch_s}")
    lines_md.append(f"- **Film:** {payload['film_reduced_formula']}")
    lines_md.append(f"- **Substrate:** {payload['substrate_reduced_formula']}")
    lines_md.append(f"- **Double interface:** {payload['double_interface']}")
    lines_md.append("")
    lines_md.append("## Key settings")
    lines_md.append("")
    lines_md.append(f"| Key | Value |")
    lines_md.append(f"| --- | --- |")
    lines_md.append(f"| max_area | {lm.get('max_area')} |")
    lines_md.append(f"| calc | {gm.get('calc')} |")
    lines_md.append(f"| ckpt | {ckpt_base or ckpt or '—'} |")
    lines_md.append(f"| do_mlip_gd | {opt.get('do_mlip_gd')} |")
    lines_md.append(f"| do_vasp | {do_vasp} |")
    lines_md.append("")
    lines_md.append("## Best pairs (lowest-energy set)")
    lines_md.append("")
    if not pairs_summary:
        lines_md.append("_No rows._")
    else:
        lines_md.append(
            "| match | term | film Miller | sub Miller | energy type | energy | film atoms | sub atoms | match_area | strain |"
        )
        lines_md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in pairs_summary:
            if "interface_energy" in item:
                et, ev = "interface_energy", item.get("interface_energy")
            else:
                et, ev = "cohesive_energy", item.get("cohesive_energy")
            row = [
                item.get("match_id"),
                item.get("term_id"),
                item.get("film_conventional_miller"),
                item.get("substrate_conventional_miller"),
                et,
                ev,
                item.get("film_atom_count"),
                item.get("substrate_atom_count"),
                item.get("match_area"),
                item.get("strain"),
            ]
            esc = [str(x).replace("|", "\\|") for x in row]
            lines_md.append("| " + " | ".join(esc) + " |")
    lines_md.append("")
    lines_md.append("## Files to open next")
    lines_md.append("")
    art = payload["artifacts"]
    lines_md.append(f"- Full report: `{art['io_report']}`")
    if art["pairs_summary"]:
        lines_md.append(f"- Tab-separated table: `{art['pairs_summary']}`")
    if art.get("opt_results_pickle"):
        lines_md.append(
            f"- Full optimization data (POSCARs via fetch): `{art['opt_results_pickle']}`"
        )
    if art.get("pairs_best_it_dir"):
        lines_md.append(f"- Best-interface structures (legacy on-disk): `{art['pairs_best_it_dir']}/`")
    if art.get("opt_results_summary"):
        lines_md.append(f"- Optimization summary (JSON): `{art['opt_results_summary']}`")
    if art.get("area_strain"):
        lines_md.append(f"- Stereographic data: `{art['area_strain']}`")
    if art.get("stereographic_interactive"):
        lines_md.append(f"- Interactive stereographic plot: `{art['stereographic_interactive']}`")
    if art.get("stereographic_image"):
        lines_md.append(f"- Stereographic plot: `{art['stereographic_image']}`")
    if art["project_image"]:
        lines_md.append(f"- Matching plot: `{art['project_image']}`")
    if art["unique_matches_plot"]:
        lines_md.append(f"- Unique matches: `{art['unique_matches_plot']}`")
    lines_md.append(f"- Machine-readable: `{_rel(json_path)}`")
    lines_md.append("")

    md_path = os.path.join(cwd, "io_remote_summary.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_md))
    except OSError:
        pass

@dataclass
class GDVaspMaker(Maker):
    name: str = 'it_gradient_descend'
    dx: float = 0.05
    tol: float = 5e-4
    initial_r: float = 0.1
    initial_it: Interface = None
    film_indices: List = None
    min_steps: int = 3
    # atomate2 RelaxMaker — lazy-only at runtime for VASP GD paths; avoid importing
    # atomate2 at module import (pymatgen/atomate2 version skew breaks MLIP-only remote jobs).
    relax_maker: Any = None
    relax_maker_dp: Any = None
    metadata: Dict[str, Any] = None
    
    @job
    def update_n_p_1(self, y_dns, saved_data):
        #calculate g_n
        g_n = (np.array(y_dns) - np.array(saved_data['ys'][-1])) / self.dx
        if saved_data['n'] == 0:
            r = 0.01
        else:
            #x_n & x_n_1 have been updated in saved_data
            x_n, x_n_1 = np.array(saved_data['xs'][-1]), np.array(saved_data['xs'][-2])
            #g_n_1 has not been updated
            g_n_1 = np.array(saved_data['gs'][-1])
            r = abs(np.dot((x_n - x_n_1), (g_n - g_n_1))) / np.linalg.norm(g_n - g_n_1) ** 2
        
        #translate it_n
        tt = TranslateSitesTransformation(self.film_indices, - r * g_n, False)
        it = tt.apply_transformation(saved_data['its'][-1])
        #vasp_job = self.relax_maker.make(structure = it, prev_dir = saved_data['vasp_dir'])
        vasp_job = self.relax_maker.make(structure = it)
        saved_data['n'] += 1
        ask_gradient_job = self.ask_gradient(
                               saved_data,
                                np.array(saved_data['xs'][-1]) - r * g_n,
                                vasp_job.output.output.energy,
                                g_n,
                                vasp_job.output.output.structure,
                                vasp_job.output.dir_name
                                )
        return Response(replace = Flow([vasp_job, ask_gradient_job]))
        
    @job
    def save_final_data(self, saved_data):
        if self.relax_maker_dp is None:
            return saved_data
        else:
            #stct = Structure.from_dict(saved_data['its'][-1])
            job = self.relax_maker_dp.make(saved_data['its'][-1], prev_dir = saved_data['vasp_dir'])
            metadata = self.metadata.copy()
            metadata['job'] += '_dp'
            job.update_metadata(metadata)
            return Response(output = saved_data, replace = Flow([job]))
        
    @job
    #def ask_gradient(self, saved_data, x_n, y_n, g_n_1, it_n):
    def ask_gradient(self, saved_data, x_n, y_n, g_n_1, it_n, vasp_dir):
        # if first round, save vasp running dictionary
        saved_data['vasp_dir'] = vasp_dir
        # update n_p_1 data
        saved_data['xs'].append(x_n)
        saved_data['ys'].append(y_n)
        saved_data['gs'].append(g_n_1)
        saved_data['its'].append(it_n)
        
        print(type(it_n))
        #it_n_structure = Structure.from_dict(it_n)
        it_n_structure = it_n
        
        if saved_data['n'] > self.min_steps and abs(saved_data['ys'][-1] - saved_data['ys'][-2]) < self.tol * len(it_n_structure):
            save_job = self.save_final_data(saved_data)
            save_job.update_metadata(self.metadata)
            return Response(replace = Flow([save_job]))
            
        # jobs to calculate g_n
        jobs = []
        for i in range(3):
            pdx = np.zeros(3)
            pdx[i] = self.dx
            tt = TranslateSitesTransformation(self.film_indices, pdx, False)
            it = tt.apply_transformation(it_n_structure)
            #jobs.append(self.relax_maker.make(structure = it, prev_dir = vasp_dir))
            jobs.append(self.relax_maker.make(structure = it))
        
        # job to calculate r, x_n_p_1 & y_n_p_1
        n_p_1_job = self.update_n_p_1([jobs[0].output.output.energy,
                                        jobs[1].output.output.energy,
                                        jobs[2].output.output.energy],
                                        saved_data)
        return Response(replace = Flow(jobs + [n_p_1_job]))
    
    def make(self):
        saved_data = {}
        saved_data['xs'] = []
        saved_data['ys'] = []
        saved_data['gs'] = []
        saved_data['its'] = []
        saved_data['n'] = 0
        #first vasp job
        vasp_job = self.relax_maker.make(structure = self.initial_it)
        #update saved_data with the first-vasp-job output
        ask_gradient_job = self.ask_gradient(
                            saved_data = saved_data,
                            x_n = np.array([0,0,0]),
                            y_n = vasp_job.output.output.energy,
                            g_n_1 = 0,
                            it_n = vasp_job.output.output.structure,
                            vasp_dir = vasp_job.output.dir_name
                            )
        return [vasp_job, ask_gradient_job]


def legacy_vasp_bucket_dicts_to_user_kwargs(
    vasp_relax_settings: Optional[Dict[str, Any]],
    vasp_static_settings: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Any,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Any,
    Dict[str, Any],
]:
    """
    Map deprecated ``vasp_relax_settings`` / ``vasp_static_settings`` buckets
    (``INCAR``, ``KPOINTS``, ``POTCAR``, ``POTCAR_FUNCTIONAL``, ``GDTOL``)
    to kwargs of :meth:`InterfaceWorker.patch_jobflow_jobs` (``relax_user_*``, ``static_user_*``, ``gd_kwargs``).
    """
    rr = vasp_relax_settings or {}
    rs = vasp_static_settings or {}
    gd = {"tol": float(rr.get("GDTOL", 5e-4))}
    return (
        rr.get("INCAR"),
        rr.get("POTCAR"),
        rr.get("KPOINTS"),
        rr.get("POTCAR_FUNCTIONAL"),
        rs.get("INCAR"),
        rs.get("POTCAR"),
        rs.get("KPOINTS"),
        rs.get("POTCAR_FUNCTIONAL"),
        gd,
    )


def iomaker_resolve_patch_jobflow_vasp_kwargs(maker: "IOMaker") -> Dict[str, Any]:
    """
    Resolve VASP arguments for :meth:`InterfaceWorker.patch_jobflow_jobs` from an :class:`IOMaker` instance.

    Priority: explicit ``relax_user_*`` / ``static_user_*`` (same names as ``itworker``); else legacy
    ``vasp_relax_settings`` / ``vasp_static_settings`` buckets; else defaults (all ``None``, ``gd_kwargs`` tol 5e-4).
    """
    default_potcar_functional = "PBE_54"

    has_explicit = any(
        getattr(maker, k, None) is not None
        for k in (
            "relax_user_incar_settings",
            "relax_user_potcar_settings",
            "relax_user_kpoints_settings",
            "relax_user_potcar_functional",
            "static_user_incar_settings",
            "static_user_potcar_settings",
            "static_user_kpoints_settings",
            "static_user_potcar_functional",
        )
    )
    if has_explicit:
        r_i = maker.relax_user_incar_settings
        r_p = maker.relax_user_potcar_settings
        r_k = maker.relax_user_kpoints_settings
        r_f = maker.relax_user_potcar_functional or default_potcar_functional
        s_i = maker.static_user_incar_settings
        s_p = maker.static_user_potcar_settings
        s_k = maker.static_user_kpoints_settings
        s_f = maker.static_user_potcar_functional or default_potcar_functional
    elif maker.vasp_relax_settings or maker.vasp_static_settings:
        r_i, r_p, r_k, r_f, s_i, s_p, s_k, s_f, gd0 = legacy_vasp_bucket_dicts_to_user_kwargs(
            maker.vasp_relax_settings,
            maker.vasp_static_settings,
        )
        gd = dict(maker.vasp_gd_kwargs) if maker.vasp_gd_kwargs is not None else dict(gd0)
        gd.setdefault("tol", float(gd0.get("tol", 5e-4)))
        return {
            "relax_user_incar_settings": r_i,
            "relax_user_potcar_settings": r_p,
            "relax_user_kpoints_settings": r_k,
            "relax_user_potcar_functional": r_f or default_potcar_functional,
            "static_user_incar_settings": s_i,
            "static_user_potcar_settings": s_p,
            "static_user_kpoints_settings": s_k,
            "static_user_potcar_functional": s_f or default_potcar_functional,
            "gd_kwargs": gd,
            "dipole_correction": bool(maker.vasp_dipole_correction),
        }
    else:
        r_i = r_p = r_k = s_i = s_p = s_k = None
        r_f = s_f = default_potcar_functional

    gd = dict(maker.vasp_gd_kwargs) if maker.vasp_gd_kwargs is not None else {}
    gd.setdefault("tol", 5e-4)
    if (
        maker.vasp_gd_kwargs is None
        and maker.vasp_relax_settings
        and isinstance(maker.vasp_relax_settings, dict)
        and "GDTOL" in maker.vasp_relax_settings
    ):
        gd["tol"] = float(maker.vasp_relax_settings["GDTOL"])
    return {
        "relax_user_incar_settings": r_i,
        "relax_user_potcar_settings": r_p,
        "relax_user_kpoints_settings": r_k,
        "relax_user_potcar_functional": r_f,
        "static_user_incar_settings": s_i,
        "static_user_potcar_settings": s_p,
        "static_user_kpoints_settings": s_k,
        "static_user_potcar_functional": s_f,
        "gd_kwargs": gd,
        "dipole_correction": bool(maker.vasp_dipole_correction),
    }


@dataclass
class IOMaker(Maker):
    name: str = 'IO_std'
    lattice_matching_settings: Dict[str, Any] = None
    structure_settings: Dict[str, Any] = None
    optimization_settings: Dict[str, Any] = None
    global_minimization_settings: Dict[str, Any] = None
    do_vasp: bool = False
    do_vasp_gd: bool = False
    #: Same kwargs as :meth:`InterfaceWorker.patch_jobflow_jobs` (pymatgen ``MPRelaxSet`` / ``MPStaticSet``).
    relax_user_incar_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_settings: Optional[Dict[str, Any]] = None
    relax_user_kpoints_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_functional: Any = "PBE_54"
    static_user_incar_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_settings: Optional[Dict[str, Any]] = None
    static_user_kpoints_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_functional: Any = "PBE_54"
    #: Passed to ``patch_jobflow_jobs(..., gd_kwargs=...)`` when ``do_vasp_gd`` is True.
    vasp_gd_kwargs: Optional[Dict[str, Any]] = None
    #: Same as ``patch_jobflow_jobs(..., dipole_correction=...)``.
    vasp_dipole_correction: bool = False
    #: Deprecated: ``{"INCAR": ..., "KPOINTS": ..., "POTCAR": ..., "POTCAR_FUNCTIONAL": ..., "GDTOL": ...}`` buckets.
    #: Used only when all ``relax_user_*`` / ``static_user_*`` above are unset.
    vasp_relax_settings: Optional[Dict[str, Any]] = None
    vasp_static_settings: Optional[Dict[str, Any]] = None
    lowest_energy_pairs_settings: Dict[str, Any] = None
    pairs_output_dir: Optional[str] = None  # unused; POSCAR trees are produced locally from opt_results.pkl
    mlip_resources: Callable = None
    vasp_resources: Callable = None
    mlip_worker: str = 'std_worker'
    vasp_worker: str = 'std_worker'
    #: Passed to jobflow-remote ``set_run_config(..., exec_config=...)`` for the VASP sub-flow only.
    #: Use ``{"pre_run": "module load VASP/6.x"}`` (or set env ``INTEROPTIMUS_VASP_PRE_RUN`` when this is ``None``).
    vasp_exec_config: Optional[Dict[str, Any]] = None

    @job(data="IO_results")
    def IO_HT_job(self, film_conv, substrate_conv):
        iw = InterfaceWorker(film_conv, substrate_conv)
        iw.lattice_matching(**self.lattice_matching_settings)
        iw.ems.plot_unique_matches()
        iw.ems.plot_matching_data(['film', 'substrate'],'project.jpg', show_millers = True, show_legend = False)
        iw.parse_interface_structure_params(**self.structure_settings)
        iw.parse_optimization_params(**self.optimization_settings)
        iw.global_minimization(**self.global_minimization_settings)

        # Visualize minimization results with formatted reduced formulas
        def _formula_to_subscript(formula: str) -> str:
            return re.sub(r"(\d+)", r"$_\1$", formula or "")

        film_formula = getattr(film_conv.composition, "reduced_formula", "film")
        substrate_formula = getattr(substrate_conv.composition, "reduced_formula", "substrate")
        iw.visualize_minimization_results(
            film_name=_formula_to_subscript(film_formula),
            substrate_name=_formula_to_subscript(substrate_formula),
        )

        # Selected pairs table + opt_results.pkl (no pairs_best_it/ on compute filesystem;
        # POSCAR trees are materialized locally by fetch_interoptimus_task_results).
        pairs_kwargs = self.lowest_energy_pairs_settings or {}
        pairs = iw.get_lowest_energy_pairs_each_match(**pairs_kwargs)
        pairs_summary = []

        cwd_here = os.getcwd()
        pairs_dir = cwd_here
        summary_path = os.path.join(cwd_here, "pairs_summary.txt")

        for (i, j) in pairs:
            best_it_obj = iw.opt_results.get((i, j), {}).get("relaxed_best_interface", {}).get("structure")

            # Collect summary info for report
            try:
                idx_data = iw.unique_matches_indices_data[i]
                film_hkl = idx_data.get("film_conventional_miller")
                sub_hkl = idx_data.get("substrate_conventional_miller")
            except Exception:
                film_hkl = None
                sub_hkl = None
            if iw.double_interface:
                energy = iw.opt_results.get((i, j), {}).get("relaxed_min_it_E")
                energy_label = "interface_energy"
            else:
                energy = iw.opt_results.get((i, j), {}).get("relaxed_min_bd_E")
                energy_label = "cohesive_energy"
            film_atoms = iw.opt_results.get((i, j), {}).get("film_atom_count")
            substrate_atoms = iw.opt_results.get((i, j), {}).get("substrate_atom_count")
            try:
                # Prefer counts from the original interface object if available
                if hasattr(best_it_obj, "film_indices") and hasattr(best_it_obj, "substrate_indices"):
                    film_atoms = len(best_it_obj.film_indices)
                    substrate_atoms = len(best_it_obj.substrate_indices)
                elif hasattr(best_it_obj, "film") and hasattr(best_it_obj, "substrate"):
                    film_atoms = len(best_it_obj.film)
                    substrate_atoms = len(best_it_obj.substrate)
                elif isinstance(best_it_obj, dict):
                    if "film_indices" in best_it_obj:
                        film_atoms = len(best_it_obj.get("film_indices") or [])
                    if "substrate_indices" in best_it_obj:
                        substrate_atoms = len(best_it_obj.get("substrate_indices") or [])
            except Exception:
                pass
            # Fallback: use cached indices in opt_results (set in post_bayesian_process)
            if film_atoms is None:
                fi = iw.opt_results.get((i, j), {}).get("film_indices")
                if fi is not None:
                    film_atoms = len(fi)
            if substrate_atoms is None:
                si = iw.opt_results.get((i, j), {}).get("substrate_indices")
                if si is not None:
                    substrate_atoms = len(si)
            # Fallback: use sampled interface indices
            if film_atoms is None or substrate_atoms is None:
                try:
                    sample_it = iw.opt_results.get((i, j), {}).get("sampled_interfaces", [None])[0]
                    if film_atoms is None and hasattr(sample_it, "film_indices"):
                        film_atoms = len(sample_it.film_indices)
                    if substrate_atoms is None and hasattr(sample_it, "substrate_indices"):
                        substrate_atoms = len(sample_it.substrate_indices)
                    if film_atoms is None and hasattr(sample_it, "film"):
                        film_atoms = len(sample_it.film)
                    if substrate_atoms is None and hasattr(sample_it, "substrate"):
                        substrate_atoms = len(sample_it.substrate)
                except Exception:
                    pass
            _od = iw.opt_results.get((i, j), {}) or {}
            _ma = _od.get("match_area")
            _st = _od.get("strain")
            pairs_summary.append(
                {
                    "match_id": i,
                    "term_id": j,
                    "film_conventional_miller": film_hkl,
                    "substrate_conventional_miller": sub_hkl,
                    energy_label: energy,
                    "film_atom_count": film_atoms,
                    "substrate_atom_count": substrate_atoms,
                    "match_area": _ma,
                    "strain": _st,
                }
            )

        # Write pairs summary for downstream report (text table)
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                headers = [
                    "match_id",
                    "term_id",
                    "film_conventional_miller",
                    "substrate_conventional_miller",
                    "energy_type",
                    "energy_value",
                    "film_atom_count",
                    "substrate_atom_count",
                    "match_area",
                    "strain",
                ]
                f.write("\t".join(headers) + "\n")
                for item in pairs_summary:
                    # energy_type is either interface_energy or cohesive_energy
                    if "interface_energy" in item:
                        energy_type = "interface_energy"
                        energy_value = item.get("interface_energy")
                    else:
                        energy_type = "cohesive_energy"
                        energy_value = item.get("cohesive_energy")
                    row = [
                        str(item.get("match_id")),
                        str(item.get("term_id")),
                        str(item.get("film_conventional_miller")),
                        str(item.get("substrate_conventional_miller")),
                        str(energy_type),
                        str(energy_value),
                        str(item.get("film_atom_count")),
                        str(item.get("substrate_atom_count")),
                        str(item.get("match_area")),
                        str(item.get("strain")),
                    ]
                    f.write("\t".join(row) + "\n")
        except Exception:
            pass

        try:
            _write_opt_results_pickle(cwd_here, iw, pairs)
        except Exception:
            pass

        # Write report locally (server-safe, uses current working dir)
        try:
            report_path = os.path.join(os.getcwd(), "io_report.txt")
            lines = []
            lines.append("=" * 80)
            lines.append("InterOptimus IO Report")
            lines.append("Remote quick view: io_remote_summary.md | io_remote_summary.json")
            lines.append("=" * 80)
            lines.append("")
            lines.append("Structures")
            lines.append("-" * 80)
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                film_sg = SpacegroupAnalyzer(film_conv).get_space_group_symbol()
                film_sg_num = SpacegroupAnalyzer(film_conv).get_space_group_number()
                sub_sg = SpacegroupAnalyzer(substrate_conv).get_space_group_symbol()
                sub_sg_num = SpacegroupAnalyzer(substrate_conv).get_space_group_number()
                lines.append(
                    f"Film: {film_conv.composition.reduced_formula} | "
                    f"{film_conv.composition.formula} | SG {film_sg} ({film_sg_num})"
                )
                lines.append(
                    f"Substrate: {substrate_conv.composition.reduced_formula} | "
                    f"{substrate_conv.composition.formula} | SG {sub_sg} ({sub_sg_num})"
                )
            except Exception:
                lines.append("Film: (info unavailable)")
                lines.append("Substrate: (info unavailable)")
            lines.append("")
            lines.append("Key Settings")
            lines.append("-" * 80)
            lm = self.lattice_matching_settings or {}
            st = self.structure_settings or {}
            opt = self.optimization_settings or {}
            gm = self.global_minimization_settings or {}
            lines.append(f"max_area: {lm.get('max_area')}")
            lines.append(f"max_length_tol: {lm.get('max_length_tol')}")
            lines.append(f"max_angle_tol: {lm.get('max_angle_tol')}")
            lines.append(f"film_thickness: {st.get('film_thickness')}")
            lines.append(f"substrate_thickness: {st.get('substrate_thickness')}")
            lines.append(f"vacuum_over_film: {st.get('vacuum_over_film')}")
            lines.append(f"calc: {gm.get('calc')}")
            ckpt = opt.get("ckpt_path")
            lines.append(f"ckpt_path: {ckpt}")
            if ckpt:
                lines.append(f"ckpt_model_name: {os.path.basename(str(ckpt))}")
            lines.append(f"do_mlip_gd: {opt.get('do_mlip_gd')}")
            lines.append(f"do_vasp: {self.do_vasp}")
            lines.append(f"do_vasp_gd: {self.do_vasp_gd}")
            lines.append("")
            lines.append("IOMaker Parameters (full)")
            lines.append("-" * 80)
            lines.append("lattice_matching_settings:")
            lines.append(json.dumps(self.lattice_matching_settings or {}, indent=2, ensure_ascii=False))
            lines.append("structure_settings:")
            lines.append(json.dumps(self.structure_settings or {}, indent=2, ensure_ascii=False))
            lines.append("optimization_settings:")
            lines.append(json.dumps(self.optimization_settings or {}, indent=2, ensure_ascii=False))
            lines.append("global_minimization_settings:")
            lines.append(json.dumps(self.global_minimization_settings or {}, indent=2, ensure_ascii=False))
            if self.do_vasp:
                vk = iomaker_resolve_patch_jobflow_vasp_kwargs(self)
                lines.append("VASP (InterfaceWorker.patch_jobflow_jobs, resolved):")
                lines.append(json.dumps(vk, indent=2, ensure_ascii=False))
                if self.vasp_relax_settings is not None:
                    lines.append("vasp_relax_settings (deprecated bucket, if any):")
                    lines.append(json.dumps(self.vasp_relax_settings, indent=2, ensure_ascii=False))
                if self.vasp_static_settings is not None:
                    lines.append("vasp_static_settings (deprecated bucket, if any):")
                    lines.append(json.dumps(self.vasp_static_settings, indent=2, ensure_ascii=False))
            if self.lowest_energy_pairs_settings is not None:
                lines.append("lowest_energy_pairs_settings:")
                lines.append(json.dumps(self.lowest_energy_pairs_settings, indent=2, ensure_ascii=False))
            lines.append("")
            lines.append("Outputs")
            lines.append("-" * 80)
            lines.append(f"Run directory (pairs_summary.txt, opt_results.pkl): {pairs_dir}")
            lines.append(
                "Best-interface POSCAR trees are not written on the compute node; "
                "use fetch_interoptimus_task_results(..., copy_images_to=...) locally to materialize pairs_best_it/."
            )
            if os.path.exists(summary_path):
                lines.append("")
                lines.append("Pairs Summary")
                lines.append("-" * 80)
                with open(summary_path, "r", encoding="utf-8") as f:
                    lines.extend([line.rstrip("\n") for line in f])
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

        try:
            _write_opt_results_summary_json(os.getcwd(), iw.opt_results)
        except Exception:
            pass

        try:
            _write_remote_io_summaries(
                cwd=os.getcwd(),
                pairs_dir=pairs_dir,
                summary_path=summary_path,
                pairs_summary=pairs_summary,
                film_conv=film_conv,
                substrate_conv=substrate_conv,
                double_interface=iw.double_interface,
                lattice_matching_settings=self.lattice_matching_settings,
                optimization_settings=self.optimization_settings,
                global_minimization_settings=self.global_minimization_settings,
                do_vasp=self.do_vasp,
            )
        except Exception:
            pass

        results = {}
        results['unique_matches'] = iw.unique_matches
        results['all_match_data'] = iw.ems.all_matche_data
        results['match_data'] = iw.ems.matching_data
        results['opt_results'] = iw.opt_results
        io_results_bytes = pickle.dumps(results)
        io_results_b64 = base64.b64encode(io_results_bytes).decode('utf-8')
        if self.do_vasp:
            vk = iomaker_resolve_patch_jobflow_vasp_kwargs(self)
            flow = iw.patch_jobflow_jobs(
                filter_name=self.name,
                only_lowest_energy_each_plane=True,
                relax_user_incar_settings=vk["relax_user_incar_settings"],
                relax_user_potcar_settings=vk["relax_user_potcar_settings"],
                relax_user_kpoints_settings=vk["relax_user_kpoints_settings"],
                relax_user_potcar_functional=vk["relax_user_potcar_functional"],
                static_user_incar_settings=vk["static_user_incar_settings"],
                static_user_potcar_settings=vk["static_user_potcar_settings"],
                static_user_kpoints_settings=vk["static_user_kpoints_settings"],
                static_user_potcar_functional=vk["static_user_potcar_functional"],
                do_dft_gd=self.do_vasp_gd,
                gd_kwargs=vk["gd_kwargs"],
                dipole_correction=vk["dipole_correction"],
            )

            return {'IO_results': io_results_b64, 'flow': Flow(flow).to_json()}
        else:
            return {'IO_results':io_results_b64}
    
    @job
    def vasp_job(self, flow_json):
        return Response(replace = Flow.from_dict(json.loads(flow_json)))
        
    def make(self, film_conv, substrate_conv):
        if self.do_vasp:
            IO_job = self.IO_HT_job(film_conv, substrate_conv)
            
            # Call resources function to get QResources object (not function)
            mlip_res = self.mlip_resources() if callable(self.mlip_resources) else self.mlip_resources
            
            IO_job = set_run_config(IO_job, worker = self.mlip_worker,
                              resources = mlip_res,
                              priority=10,
                              dynamic = False)
            
            vasp_job = self.vasp_job(IO_job.output['flow'])
            
            # Call resources function to get QResources object (not function)
            vasp_res = self.vasp_resources() if callable(self.vasp_resources) else self.vasp_resources

            default_pre = "module load VASP/6.4.3"
            if self.vasp_exec_config is None:
                exec_cfg = {"pre_run": os.environ.get("INTEROPTIMUS_VASP_PRE_RUN", default_pre)}
            else:
                exec_cfg = dict(self.vasp_exec_config)
                if "pre_run" not in exec_cfg:
                    exec_cfg["pre_run"] = os.environ.get("INTEROPTIMUS_VASP_PRE_RUN", default_pre)

            vasp_job = set_run_config(vasp_job, worker = self.vasp_worker,
                              resources = vasp_res,
                              priority=0,
                              exec_config=exec_cfg,
                              dynamic = True)
            
            return Flow([IO_job, vasp_job])
        else:
            # NOTE: use the passed-in conventional structures
            IO_job = self.IO_HT_job(film_conv, substrate_conv)
            # Apply MLIP resources if provided
            if self.mlip_resources is not None:
                mlip_res = self.mlip_resources() if callable(self.mlip_resources) else self.mlip_resources
                IO_job = set_run_config(IO_job, worker = self.mlip_worker,
                                  resources = mlip_res,
                                  priority=10,
                                  dynamic = False)
            return Flow([IO_job])
        
@job
def check_it_phase_stability(film_conv, substrate_conv, device='cpu',
                                                        fmax=0.5,
                                                        steps=500,
                                                        n_calls=30,
                                                        calc='sevenn'):
    """
    Evaluate interface phase stability using MLIP optimization.

    Performs lattice matching, interface structure generation, and phase
    stability evaluation using machine learning interatomic potentials.
    Compares static and relaxed interface structures to assess stability.

    Args:
        film_conv: Conventional unit cell of the film material
        substrate_conv: Conventional unit cell of the substrate material
        device (str): Device for MLIP calculations ('cpu' or 'cuda')
        fmax (float): Maximum force threshold for relaxation
        steps (int): Maximum steps for geometry optimization
        n_calls (int): Number of optimization calls for interface registration
        calc (str): MLIP calculator to use ('sevenn', 'orb-models', 'matris', 'dpa', etc.)

    Returns:
        dict: Results containing:
            - static_it: JSON representation of static interface
            - relaxed_it: JSON representation of relaxed interface
            - rms_frac: RMS displacement in fractional coordinates
            - rms_cart: RMS displacement in Cartesian coordinates
    """
    
    iw = InterfaceWorker(film_conv, substrate_conv)
    _ckpt = resolve_mlip_checkpoint(calc)

    iw.lattice_matching(max_area = 100,
                        max_length_tol = 0.03,
                        max_angle_tol = 0.03,
                        film_max_miller = 4,
                        substrate_max_miller = 4)
                        
    iw.parse_interface_structure_params(termination_ftol = 2, c_periodic = True, \
                                    vacuum_over_film = 10, film_thickness = 15, \
                                    substrate_thickness = 15, shift_to_bottom = True)
                                    
    iw.parse_optimization_params(do = True,
                             set_fix_thicknesses = (0,0),
                             fix_in_layers = True,
                             whole_slab_fixed = False,
                             fmax = fmax,
                             steps = steps,
                             device = device,
                             ckpt_path = _ckpt)
    
    static_it, relaxed_it, dx_frac, dx_cart = iw.phase_stability_evaluation(
                                                                            n_calls = n_calls,
                                                                            z_range = (0, 3),
                                                                            calc = calc,
                                                                            discut = 0.5,
                                                                            )
    
    return {'static_it': static_it.to_json(),
            'relaxed_it': relaxed_it.to_json(),
            'rms_frac': dx_frac,
            'rms_cart': dx_cart}
