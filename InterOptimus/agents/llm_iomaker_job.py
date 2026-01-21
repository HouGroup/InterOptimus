#!/usr/bin/env python3
"""
LLM -> InterOptimus.jobflow.IOMaker job builder.

Step 1 (per user request):
- Use an OpenAI-compatible LLM (api_key + base_url) to generate the complex
  parameter dicts needed by InterOptimus.jobflow.IOMaker.
- Load input structures from ./film.cif and ./substrate.cif by default.
- If those files do not exist, use Materials Project API to fetch conventional
  structures for user-specified formulas (e.g., film=Si, substrate=SiC).
- Build a jobflow Flow using IOMaker.make(...) and write the Flow to JSON.

This module intentionally does NOT yet do jobflow-remote submission/status/report.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Literal, Callable

from pymatgen.core.structure import Structure

from ..jobflow import IOMaker


DEFAULT_FILM_CIF = "film.cif"
DEFAULT_SUBSTRATE_CIF = "substrate.cif"

# ---------------------------------------------------------------------
# Tutorial presets (match Tutorial/bicrystal_with_vaccum.ipynb and
# Tutorial/bicrystal_without_vaccum.ipynb)
# ---------------------------------------------------------------------

TUTORIAL_WITH_VACUUM = {
    "lattice_matching_settings": {
        "max_area": 20,
        "max_length_tol": 0.03,
        "max_angle_tol": 0.03,
        "film_max_miller": 3,
        "substrate_max_miller": 3,
        "film_millers": None,
        "substrate_millers": None,
    },
    "structure_settings": {
        "termination_ftol": 0.15,
        "film_thickness": 8,
        "substrate_thickness": 8,
        "double_interface": False,
        "vacuum_over_film": 5,
    },
    "optimization_settings": {
        "set_relax_thicknesses": (1 / 2, 1 / 2),
        "relax_in_layers": False,
        "relax_in_ratio": True,
        "fmax": 0.05,
        "steps": 200,
        "device": "cpu",
        "discut": 0.8,
        "ckpt_path": None,
        "BO_coord_bin_size": 0.25,
        "BO_energy_bin_size": 0.05,
        "BO_rms_bin_size": 0.3,
        "do_gd": True,
    },
    "global_minimization_settings": {
        "n_calls_density": 1,
        "z_range": [0.5, 3.0],
        "calc": "orb-models",
        "strain_E_correction": True,
    },
}

TUTORIAL_WITHOUT_VACUUM = {
    "lattice_matching_settings": {
        "max_area": 25,
        "max_length_tol": 0.03,
        "max_angle_tol": 0.03,
        "film_max_miller": 3,
        "substrate_max_miller": 3,
        "film_millers": None,
        "substrate_millers": None,
    },
    "structure_settings": {
        "termination_ftol": 0.15,
        "film_thickness": 8,
        "substrate_thickness": 8,
        "double_interface": True,
        "vacuum_over_film": 0,
    },
    "optimization_settings": {
        "fmax": 0.05,
        "steps": 200,
        "device": "cpu",
        "discut": 0.8,
        "ckpt_path": "",
    },
    "global_minimization_settings": {
        "n_calls_density": 1,
        "z_range": [0.5, 3.0],
        "calc": "orb-models",
        "strain_E_correction": True,
    },
}


def _tutorial_mode_from_prompt(user_prompt: str) -> str:
    """
    Decide which Tutorial parameter set to use.
    Default: with vacuum.
    """
    t = (user_prompt or "").lower()
    if "without vacuum" in t or "no vacuum" in t or "without_vacuum" in t:
        return "without_vacuum"
    if "double_interface" in t or "double interface" in t:
        return "without_vacuum"
    # Chinese: double interface implies no vacuum layer
    if "双界面" in (user_prompt or ""):
        return "without_vacuum"
    # Chinese hints
    if "无真空" in t or "不加真空" in t or "不要真空" in t:
        return "without_vacuum"
    return "with_vacuum"


def _force_without_vacuum_from_prompt(user_prompt: str) -> bool:
    """
    Return True if user explicitly requests a double-interface model.
    Per requirement: double-interface == without vacuum (no vacuum layer).
    """
    raw = user_prompt or ""
    t = raw.lower()
    if "double_interface" in t or "double interface" in t:
        return True
    if "双界面" in raw:
        return True
    return False


def _disable_gd_from_prompt(user_prompt: str) -> bool:
    """
    Return True if the user explicitly asks to disable gradient descent (GD).

    Examples:
    - 不要/不做/不进行 梯度下降
    - 不要 GD / 不用 GD
    - no gradient descent / disable GD
    """
    raw = user_prompt or ""
    t = raw.lower()

    # Chinese negations around "梯度下降" / "GD"
    if re.search(r"(不要|不做|不进行|不用|禁止).{0,8}(梯度下降|gd)", raw):
        return True

    # English phrases
    if re.search(r"\b(no|without|disable|dont|don't|do not)\b.{0,16}\b(gradient\s*descent|gd)\b", t):
        return True

    # Short explicit tokens
    if re.search(r"\bno\s*gd\b", t) or re.search(r"\bdisable\s*gd\b", t):
        return True

    return False


def _slug(text: str, maxlen: int = 32) -> str:
    """
    Make a filesystem-safe, compact token for naming.
    """
    s = (text or "NA").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return (s[:maxlen] or "NA")


def _summarize_key_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select a stable subset of settings for name hashing.
    """
    lm = settings.get("lattice_matching_settings", {}) or {}
    opt = settings.get("optimization_settings", {}) or {}
    gm = settings.get("global_minimization_settings", {}) or {}
    return {
        "mode": settings.get("mode"),
        "do_vasp": settings.get("do_vasp"),
        "calc": gm.get("calc"),
        "z_range": gm.get("z_range"),
        "n_calls_density": gm.get("n_calls_density"),
        "strain_E_correction": gm.get("strain_E_correction"),
        "max_area": lm.get("max_area"),
        "max_length_tol": lm.get("max_length_tol"),
        "max_angle_tol": lm.get("max_angle_tol"),
        "do_gd": opt.get("do_gd"),
        "ckpt_path": opt.get("ckpt_path"),
    }


def _auto_iomaker_name(
    settings: Dict[str, Any],
    film_label: str,
    substrate_label: str,
) -> str:
    """
    Build a deterministic, human-readable name for IOMaker jobs.
    """
    mode = settings.get("mode", "with_vacuum")
    do_vasp = bool(settings.get("do_vasp", False))
    mode_tag = "double" if mode == "without_vacuum" else "vac"

    gm = settings.get("global_minimization_settings", {}) or {}
    calc = gm.get("calc", "mlip")
    calc_tag = {"orb-models": "orb", "sevenn": "sevenn", "dpa": "dpa"}.get(calc, _slug(str(calc), 12))

    key_settings = _summarize_key_settings(settings)
    blob = json.dumps(key_settings, sort_keys=True, ensure_ascii=False)
    h8 = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    return f"IO_{_slug(film_label)}-{_slug(substrate_label)}_m-{mode_tag}_v{1 if do_vasp else 0}_mlip-{calc_tag}_{ts}_{h8}"


@dataclass
class BuildConfig:
    """Configuration for building an IOMaker Flow from text."""

    # LLM
    api_key: str
    base_url: str
    model: str = "gpt-3.5-turbo"

    # Materials Project (optional, only needed if CIF files missing)
    mp_api_key: Optional[str] = None

    # Inputs
    # By default these are written/read in the CURRENT WORKING DIRECTORY.
    # (i.e., wherever the user runs the Python command)
    film_cif: str = DEFAULT_FILM_CIF
    substrate_cif: str = DEFAULT_SUBSTRATE_CIF

    # Output
    output_flow_json: str = "io_flow.json"

    # Print settings to stdout after building
    print_settings: bool = True

    # (Deprecated) previously used for constraint-relaxation MP search.
    # We now only support local CIFs or explicit MP IDs.
    strict_mp_constraints: bool = False

    # Remote submission (optional)
    # If provided, the generated job will be automatically submitted to the remote server
    submit_to_remote: bool = False
    remote_host: Optional[str] = None  # e.g., "xys@10.103.65.21"
    remote_identity_file: Optional[str] = None  # e.g., "~/.ssh/id_ed25519"
    remote_workdir: Optional[str] = None  # e.g., "~/io_runs/si_sic_run1"
    remote_python: str = "python"  # Python command on remote server
    remote_pre_cmd: Optional[str] = None  # e.g., "source ~/.bashrc && conda activate atomate2"
    remote_passphrase: Optional[str] = None  # SSH key passphrase
    remote_use_paramiko: bool = False  # Use paramiko instead of pexpect
    remote_debug: bool = False  # Enable debug output for remote submission

    # Jobflow resources and workers
    # These are passed to IOMaker for jobflow_remote configuration
    mlip_resources: Optional[Callable] = None  # Function returning QResources for MLIP jobs
    vasp_resources: Optional[Callable] = None  # Function returning QResources for VASP jobs (required if do_vasp=True)
    mlip_worker: str = "std_worker"  # Worker name for MLIP jobs
    vasp_worker: str = "std_worker"  # Worker name for VASP jobs (required if do_vasp=True)
    
    # VASP settings (optional, defaults to MPRelaxSet/MPStaticSet if None)
    vasp_relax_settings: Optional[Dict[str, Any]] = None  # VASP relaxation settings dict
    vasp_static_settings: Optional[Dict[str, Any]] = None  # VASP static settings dict
    # Whether to run gradient descent for VASP jobs (IO -> VASP GD)
    do_vasp_gd: bool = False

    # MLIP checkpoint path (optional).
    # If provided, this will be written into optimization_settings["ckpt_path"].
    # If None, the workflow will rely on env fallbacks inside InterfaceWorker.set_energy_calculator.
    ckpt_path: Optional[str] = None


def _openai_client(api_key: str, base_url: str):
    """Create OpenAI-compatible client (lazy import)."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency 'openai'. Please install it (pip install openai)."
        ) from e
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_material_formulas_fallback(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very small fallback: try to extract 2 formulas from text.
    Example: "film is Si, substrate is SiC" -> ("Si", "SiC")
    """
    # prioritize explicit "film ... substrate ..." patterns
    film = None
    sub = None

    m = re.search(r"film\s*(?:=|is|:)\s*([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
    if m:
        film = m.group(1)
    m = re.search(r"substrate\s*(?:=|is|:)\s*([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
    if m:
        sub = m.group(1)

    if film and sub:
        return film, sub

    # generic chemical formula-ish tokens
    tokens = re.findall(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b", text)
    # remove common english tokens
    bad = {"Create", "Generate", "Make", "With", "Without", "Vacuum", "Interface", "Jobflow", "MLIP"}
    tokens = [t for t in tokens if t not in bad]
    if len(tokens) >= 2:
        return film or tokens[0], sub or tokens[1]
    return film, sub


def _mp_rester(mp_api_key: str):
    """
    Create a Materials Project rester.
    Prefer mp_api.client.MPRester (new API). Fallback to pymatgen.ext.matproj.MPRester.
    """
    # New MP API
    try:
        from mp_api.client import MPRester  # type: ignore
        return MPRester(mp_api_key)
    except Exception:
        pass

    # Old MP API (pymatgen)
    try:
        from pymatgen.ext.matproj import MPRester  # type: ignore
        return MPRester(mp_api_key)
    except Exception as e:
        raise ImportError(
            "Cannot import Materials Project client. Install mp-api (recommended) or pymatgen with MP support."
        ) from e


def _fetch_mp_summary_info_by_id(mpr: Any, mp_id: str) -> Dict[str, Any]:
    """Fetch a small, printable summary for a given mp-id (best effort)."""
    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "spacegroup",
        "band_gap",
        "density",
        "nsites",
        "nelements",
        "is_metal",
    ]
    # mp-api client
    if mpr.__class__.__module__.startswith("mp_api"):
        try:
            doc = mpr.summary.get_data_by_id(mp_id, fields=fields)
            # get_data_by_id may return a list in some versions
            if isinstance(doc, list) and doc:
                doc = doc[0]
            info: Dict[str, Any] = {}
            for k in fields:
                if hasattr(doc, k):
                    info[k] = getattr(doc, k)
            sg = getattr(doc, "spacegroup", None)
            if sg is not None:
                if hasattr(sg, "symbol"):
                    info["spacegroup_symbol"] = sg.symbol
                if hasattr(sg, "number"):
                    info["spacegroup_number"] = sg.number
                if hasattr(sg, "crystal_system"):
                    info["crystal_system"] = sg.crystal_system
            return info
        except Exception:
            return {"material_id": mp_id}

    # pymatgen older client
    try:
        doc = mpr.summary.get_data_by_id(mp_id)
        if isinstance(doc, list) and doc:
            doc = doc[0]
        if isinstance(doc, dict):
            return {k: doc.get(k) for k in fields if k in doc} | {"material_id": mp_id}
    except Exception:
        pass
    return {"material_id": mp_id}


def fetch_mp_conventional_structure_by_id(mp_id: str, mp_api_key: str) -> Tuple[Structure, Dict[str, Any]]:
    """
    Fetch the conventional Structure for a given Materials Project ID (mp-id).
    """
    mpr = _mp_rester(mp_api_key)
    info = _fetch_mp_summary_info_by_id(mpr, mp_id)
    try:
        st = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    except TypeError:
        st = mpr.get_structure_by_material_id(mp_id)
    return st, info


def load_structures(
    film_cif: str,
    substrate_cif: str,
    *,
    mp_api_key: Optional[str],
    film_mp_id: Optional[str],
    substrate_mp_id: Optional[str],
) -> Tuple[Structure, Structure, Dict[str, Any]]:
    """
    Load structures from local CIFs if present; otherwise fetch from MP.
    Returns: (film_conv, substrate_conv, meta)
    """
    meta: Dict[str, Any] = {"source": None}

    # Normalize paths: relative paths are relative to current working directory
    # (this is what users expect when they say "write to current folder").
    film_cif_path = film_cif if os.path.isabs(film_cif) else os.path.join(os.getcwd(), film_cif)
    substrate_cif_path = (
        substrate_cif if os.path.isabs(substrate_cif) else os.path.join(os.getcwd(), substrate_cif)
    )

    # Mode A: local CIFs (must exist)
    if os.path.exists(film_cif_path) and os.path.exists(substrate_cif_path) and not (film_mp_id or substrate_mp_id):
        film_conv = Structure.from_file(film_cif_path)
        substrate_conv = Structure.from_file(substrate_cif_path)
        meta["source"] = "local_cif"
        meta["film_cif"] = os.path.abspath(film_cif_path)
        meta["substrate_cif"] = os.path.abspath(substrate_cif_path)
        return film_conv, substrate_conv, meta

    # Mode B: MP IDs (must be provided explicitly)
    if not (film_mp_id and substrate_mp_id):
        raise FileNotFoundError(
            f"Local CIF inputs not found (need '{film_cif_path}' and '{substrate_cif_path}'), "
            "and MP IDs were not provided. Please provide film_mp_id/substrate_mp_id."
        )
    if not mp_api_key:
        raise ValueError("MP IDs provided but mp_api_key is missing (set MP_API_KEY or pass BuildConfig.mp_api_key).")

    film_conv, film_info = fetch_mp_conventional_structure_by_id(film_mp_id, mp_api_key)
    substrate_conv, substrate_info = fetch_mp_conventional_structure_by_id(substrate_mp_id, mp_api_key)
    meta["source"] = "materials_project_id"
    meta["film_mp_id"] = film_mp_id
    meta["substrate_mp_id"] = substrate_mp_id
    meta["mp_film"] = film_info
    meta["mp_substrate"] = substrate_info

    # Write downloaded conventional structures to CIF files for reproducibility.
    # This will overwrite existing local CIFs if present.
    try:
        # Ensure parent directories exist if user provided a path.
        os.makedirs(os.path.dirname(film_cif_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(substrate_cif_path) or ".", exist_ok=True)

        film_conv.to(filename=film_cif_path)
        substrate_conv.to(filename=substrate_cif_path)
        meta["film_cif"] = os.path.abspath(film_cif_path)
        meta["substrate_cif"] = os.path.abspath(substrate_cif_path)
        meta["cif_written"] = True
    except Exception as e:
        # CIF writing failure should be visible but not necessarily fatal for building the Flow.
        # Still, per user request "download cif file", we surface the error explicitly.
        raise RuntimeError(f"Downloaded MP structures but failed to write CIF files: {e}") from e

    return film_conv, substrate_conv, meta


def llm_generate_iomaker_settings(
    *,
    user_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    """
    Call LLM to generate settings for IOMaker.

    Returns a dict containing:
    - materials: {film, substrate}
    - lattice_matching_settings
    - structure_settings
    - optimization_settings
    - global_minimization_settings
    - do_vasp (optional)
    - vasp_relax_settings / vasp_static_settings (optional)
    """
    schema_hint = {
        # We follow Tutorial presets. LLM mainly decides input source + vacuum mode.
        "inputs": {
            "type": "local_cif  # or mp_id",
            "film_cif": "film.cif",
            "substrate_cif": "substrate.cif",
            "film_mp_id": "mp-149",
            "substrate_mp_id": "mp-8062",
        },
        "mode": "with_vacuum  # or without_vacuum",
        "do_vasp": False,
        "notes": "Parameter dicts will be filled using the exact Tutorial preset for the chosen mode.",
    }

    sys_msg = (
        "You are an expert in crystal interface simulation using InterOptimus. "
        "Your job is to convert user requirements into parameter dictionaries that can be "
        "passed into InterOptimus.jobflow.IOMaker.\n"
        "Return ONLY valid JSON. Do not include code fences."
    )

    user_msg = (
        "User requirement:\n"
        f"{user_prompt}\n\n"
        "Output JSON schema (example values):\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Rules:\n"
        "- Always output key 'inputs'.\n"
        "- inputs.type must be 'local_cif' or 'mp_id'.\n"
        "- If user provides mp-IDs, set inputs.type='mp_id' and fill film_mp_id/substrate_mp_id.\n"
        "- Otherwise set inputs.type='local_cif' and use film.cif/substrate.cif in current folder.\n"
        "- Always output key 'mode' as 'with_vacuum' or 'without_vacuum'.\n"
        "- If user does not specify vacuum, choose 'with_vacuum'.\n"
        "- If user asks no vacuum / double_interface, choose 'without_vacuum'.\n"
        "- Set do_vasp=false unless user explicitly requests VASP.\n"
    )

    client = _openai_client(api_key, base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    content = resp.choices[0].message.content or ""

    # best-effort JSON parse
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise ValueError(f"LLM did not return JSON. Raw:\n{content}")
        data = json.loads(m.group(0))

    # Make sure we at least have inputs/mode
    if "mode" not in data:
        data["mode"] = _tutorial_mode_from_prompt(user_prompt)
    if "inputs" not in data:
        # best-effort: if prompt contains mp- ids, use them
        mpids = re.findall(r"\bmp-\d+\b", user_prompt or "")
        if len(mpids) >= 2:
            data["inputs"] = {"type": "mp_id", "film_mp_id": mpids[0], "substrate_mp_id": mpids[1]}
        else:
            data["inputs"] = {"type": "local_cif", "film_cif": DEFAULT_FILM_CIF, "substrate_cif": DEFAULT_SUBSTRATE_CIF}
    return data


def _clean_for_json_serialization(obj: Any, visited: Optional[set] = None) -> Any:
    """
    Recursively remove non-serializable objects (like functions) from a dictionary.
    This is needed because jobflow Flow objects may contain function references
    that cannot be serialized to JSON.
    """
    if visited is None:
        visited = set()
    
    # Handle circular references
    if isinstance(obj, (dict, list, tuple)):
        obj_id = id(obj)
        if obj_id in visited:
            return None  # Circular reference detected
        visited.add(obj_id)
    
    try:
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # Skip function objects and other callables (but not types/classes)
                if callable(value) and not isinstance(value, type):
                    continue  # Skip functions
                try:
                    cleaned[key] = _clean_for_json_serialization(value, visited)
                except (TypeError, ValueError, RecursionError):
                    # If we can't serialize it, skip it
                    continue
            if isinstance(obj, dict):
                visited.discard(id(obj))
            return cleaned
        elif isinstance(obj, (list, tuple)):
            cleaned = []
            for item in obj:
                if callable(item) and not isinstance(item, type):
                    continue  # Skip functions
                try:
                    cleaned.append(_clean_for_json_serialization(item, visited))
                except (TypeError, ValueError, RecursionError):
                    # If we can't serialize it, skip it
                    continue
            if isinstance(obj, (list, tuple)):
                visited.discard(id(obj))
            return cleaned if isinstance(obj, list) else tuple(cleaned)
        else:
            # For other types, check if they're callable (but not types/classes)
            if callable(obj) and not isinstance(obj, type):
                return None  # Replace functions with None
            return obj
    except (TypeError, ValueError, RecursionError):
        return None


def _print_final_settings(
    settings: Dict[str, Any], 
    structures_meta: Dict[str, Any], 
    flow_json_path: str,
    cfg: Optional[BuildConfig] = None
) -> None:
    """
    Print all settings for user visibility after building a job.
    """
    print("\n" + "=" * 80)
    print("✅ InterOptimus IOMaker job generated")
    print(f"📄 Flow JSON: {flow_json_path}")
    print(f"📦 Structures source: {structures_meta.get('source')}")
    if structures_meta.get("film_cif") and structures_meta.get("substrate_cif"):
        print(f"🧱 film.cif: {structures_meta.get('film_cif')}")
        print(f"🧱 substrate.cif: {structures_meta.get('substrate_cif')}")
    # Print selected Materials Project info when available
    if structures_meta.get("mp_film"):
        print("\n🧾 Materials Project (film) selection:")
        print(json.dumps(structures_meta["mp_film"], indent=2, ensure_ascii=False))
    if structures_meta.get("mp_substrate"):
        print("\n🧾 Materials Project (substrate) selection:")
        print(json.dumps(structures_meta["mp_substrate"], indent=2, ensure_ascii=False))
    
    # Print jobflow configuration
    if cfg:
        print("\n⚙️  Jobflow configuration:")
        print(f"   MLIP worker: {cfg.mlip_worker}")
        print(f"   MLIP resources: {'✅ Set' if cfg.mlip_resources else '❌ Not set (using defaults)'}")
        if settings.get("do_vasp"):
            print(f"   VASP worker: {cfg.vasp_worker}")
            print(f"   VASP resources: {'✅ Set' if cfg.vasp_resources else '❌ Not set'}")
            print(f"   VASP GD: {'✅ On' if settings.get('do_vasp_gd') else '❌ Off'}")

            print("\n🧪 VASP input settings:")
            vrel = settings.get("vasp_relax_settings", None)
            vsta = settings.get("vasp_static_settings", None)
            if vrel is None:
                print("   - vasp_relax_settings: (default) pymatgen.io.vasp.sets.MPRelaxSet")
            else:
                print("   - vasp_relax_settings:")
                print(json.dumps(vrel, indent=2, ensure_ascii=False))
            if vsta is None:
                print("   - vasp_static_settings: (default) pymatgen.io.vasp.sets.MPStaticSet")
            else:
                print("   - vasp_static_settings:")
                print(json.dumps(vsta, indent=2, ensure_ascii=False))
        
        if cfg.submit_to_remote:
            print("\n🌐 Remote submission configuration:")
            print(f"   Host: {cfg.remote_host}")
            print(f"   Remote workdir: {cfg.remote_workdir}")
            print(f"   Python: {cfg.remote_python}")
            if cfg.remote_pre_cmd:
                print(f"   Pre-command: {cfg.remote_pre_cmd}")
    
    print("\n🔧 Parameter settings (LLM output normalized):")
    print(json.dumps(settings, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")


def build_iomaker_flow_from_prompt(user_prompt: str, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Main entry: build an IOMaker Flow JSON from natural language.

    Returns:
      dict with keys:
        - flow_json_path
        - flow_dict
        - settings (llm output)
        - structures_meta
    """
    llm_out = llm_generate_iomaker_settings(
        user_prompt=user_prompt,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
    )

    # Decide tutorial mode (LLM output preferred, otherwise prompt heuristic)
    mode = (llm_out.get("mode") or _tutorial_mode_from_prompt(user_prompt)).strip()
    if mode not in ("with_vacuum", "without_vacuum"):
        mode = _tutorial_mode_from_prompt(user_prompt)

    # Hard rule: "double interface" / "双界面" => without_vacuum
    if _force_without_vacuum_from_prompt(user_prompt):
        mode = "without_vacuum"

    preset = TUTORIAL_WITH_VACUUM if mode == "with_vacuum" else TUTORIAL_WITHOUT_VACUUM

    # Final settings used by IOMaker: EXACTLY match tutorial presets
    settings: Dict[str, Any] = {
        "name": llm_out.get("name", "IO_llm"),
        "mode": mode,
        "inputs": llm_out.get("inputs", {}) or {},
        # deepcopy so per-run overrides do not mutate module-level presets
        "lattice_matching_settings": deepcopy(preset["lattice_matching_settings"]),
        "structure_settings": deepcopy(preset["structure_settings"]),
        "optimization_settings": deepcopy(preset["optimization_settings"]),
        "global_minimization_settings": deepcopy(preset["global_minimization_settings"]),
        "do_vasp": bool(llm_out.get("do_vasp", False)),
    }

    # Allow user to override MLIP checkpoint path via BuildConfig
    # (goes into InterfaceWorker.opt_kwargs via parse_optimization_params)
    if cfg.ckpt_path is not None:
        settings["optimization_settings"]["ckpt_path"] = cfg.ckpt_path

    # If user explicitly asks to NOT do gradient descent, force-disable it.
    # This controls MLIP-side gradient descent in InterfaceWorker.parse_optimization_params().
    if _disable_gd_from_prompt(user_prompt):
        settings["optimization_settings"]["do_gd"] = False

    inputs = settings.get("inputs", {}) or {}
    in_type = inputs.get("type", "local_cif")
    film_mp_id = inputs.get("film_mp_id")
    substrate_mp_id = inputs.get("substrate_mp_id")
    # (No further normalization needed: preset is authoritative)


    # Load structures:
    effective_mp_key = cfg.mp_api_key or os.getenv("MP_API_KEY")
    film_conv, substrate_conv, meta = load_structures(
        cfg.film_cif,
        cfg.substrate_cif,
        mp_api_key=effective_mp_key,
        film_mp_id=film_mp_id if in_type == "mp_id" else None,
        substrate_mp_id=substrate_mp_id if in_type == "mp_id" else None,
    )

    # Validate VASP requirements if do_vasp is True
    do_vasp = bool(settings.get("do_vasp", False))
    if do_vasp:
        if cfg.vasp_resources is None:
            raise ValueError(
                "vasp_resources is required when do_vasp=True. "
                "Please provide a Callable that returns QResources for VASP jobs."
            )
        if not cfg.vasp_worker:
            raise ValueError(
                "vasp_worker is required when do_vasp=True. "
                "Please provide the worker name for VASP jobs."
            )

    # VASP settings priority: BuildConfig > LLM output > None (default MPRelaxSet/MPStaticSet)
    vasp_relax_settings = cfg.vasp_relax_settings
    if vasp_relax_settings is None and "vasp_relax_settings" in settings:
        vasp_relax_settings = settings.get("vasp_relax_settings")

    vasp_static_settings = cfg.vasp_static_settings
    if vasp_static_settings is None and "vasp_static_settings" in settings:
        vasp_static_settings = settings.get("vasp_static_settings")

    # If user explicitly asks to disable gradient descent, also disable VASP GD.
    do_vasp_gd = bool(cfg.do_vasp_gd)
    if _disable_gd_from_prompt(user_prompt):
        do_vasp_gd = False

    # Record the effective VASP settings for printing/debugging (only matters when do_vasp=True)
    if do_vasp:
        settings["vasp_relax_settings"] = vasp_relax_settings
        settings["vasp_static_settings"] = vasp_static_settings
        settings["do_vasp_gd"] = do_vasp_gd

    # Auto-generate IOMaker name if user didn't specify one
    if not settings.get("name") or settings.get("name") == "IO_llm":
        if in_type == "mp_id" and film_mp_id and substrate_mp_id:
            film_label = film_mp_id
            substrate_label = substrate_mp_id
        else:
            film_label = os.path.splitext(os.path.basename(cfg.film_cif))[0]
            substrate_label = os.path.splitext(os.path.basename(cfg.substrate_cif))[0]
        settings["name"] = _auto_iomaker_name(
            settings=settings,
            film_label=film_label,
            substrate_label=substrate_label,
        )
    
    maker = IOMaker(
        name=settings.get("name", "IO_llm"),
        lattice_matching_settings=settings["lattice_matching_settings"],
        structure_settings=settings["structure_settings"],
        optimization_settings=settings["optimization_settings"],
        global_minimization_settings=settings["global_minimization_settings"],
        do_vasp=do_vasp,
        do_vasp_gd=do_vasp_gd,
        vasp_relax_settings=vasp_relax_settings,
        vasp_static_settings=vasp_static_settings,
        mlip_resources=cfg.mlip_resources,
        vasp_resources=cfg.vasp_resources,
        mlip_worker=cfg.mlip_worker,
        vasp_worker=cfg.vasp_worker,
    )

    flow = maker.make(film_conv, substrate_conv)

    # Write Flow to JSON file
    # Try to use jobflow's to_json() method first, which handles serialization properly
    out_path = os.path.abspath(cfg.output_flow_json)
    flow_dict = None
    flow_json_str = None
    
    try:
        if hasattr(flow, "to_json"):
            # jobflow Flow has to_json() method that returns a JSON string
            flow_json_str = flow.to_json()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(flow_json_str)
            # Parse back to dict for return value
            flow_dict = json.loads(flow_json_str)
    except (TypeError, AttributeError) as e:
        # If to_json() fails (e.g., due to function objects), try cleaning the dict
        print(f"Warning: to_json() failed ({e}), attempting to clean and serialize manually...")
        try:
            flow_dict = flow.as_dict() if hasattr(flow, "as_dict") else flow.to_dict()
            # Remove non-serializable objects (like functions) from flow_dict
            flow_dict = _clean_for_json_serialization(flow_dict)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(flow_dict, f, indent=2)
            flow_json_str = json.dumps(flow_dict, indent=2)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to serialize Flow to JSON. "
                f"to_json() error: {e}, "
                f"manual serialization error: {e2}. "
                f"This may be due to non-serializable objects (like functions) in the Flow. "
                f"Please check that mlip_resources and vasp_resources are not stored in the Flow."
            ) from e2
    
    if flow_dict is None:
        raise RuntimeError("Failed to serialize Flow to JSON")

    if cfg.print_settings:
        _print_final_settings(settings=settings, structures_meta=meta, flow_json_path=out_path, cfg=cfg)

    result = {
        "flow_json_path": out_path,
        "flow_dict": flow_dict,
        "settings": settings,
        "structures_meta": meta,
    }

    # Submit to remote server if configured
    if cfg.submit_to_remote:
        if not cfg.remote_host:
            raise ValueError("submit_to_remote=True requires remote_host to be set")
        if not cfg.remote_identity_file:
            raise ValueError("submit_to_remote=True requires remote_identity_file to be set")
        if not cfg.remote_workdir:
            raise ValueError("submit_to_remote=True requires remote_workdir to be set")

        try:
            from .remote_submit import submit_io_flow_via_ssh

            submit_result = submit_io_flow_via_ssh(
                host=cfg.remote_host,
                identity_file=cfg.remote_identity_file,
                local_io_flow_json=out_path,
                remote_workdir=cfg.remote_workdir,
                remote_python=cfg.remote_python,
                pre_cmd=cfg.remote_pre_cmd or "",
                passphrase=cfg.remote_passphrase,
                use_paramiko=cfg.remote_use_paramiko,
                debug=cfg.remote_debug,
            )

            result["remote_submission"] = submit_result

            if submit_result["success"]:
                print(f"\n✅ Job submitted successfully to {cfg.remote_host}")
                if submit_result.get("job_id"):
                    print(f"📋 Job ID: {submit_result['job_id']}")
            else:
                print(f"\n❌ Remote submission failed:")
                print(f"   Error: {submit_result.get('error', 'unknown')}")
                print(f"   Stderr: {submit_result.get('stderr', '')}")
        except ImportError as e:
            raise ImportError(
                f"Remote submission requires 'pexpect' or 'paramiko'. "
                f"Install with: pip install pexpect (or pip install paramiko). "
                f"Original error: {e}"
            )
        except Exception as e:
            print(f"\n⚠️  Remote submission failed with error: {e}")
            result["remote_submission_error"] = str(e)

    return result


def main():
    import argparse

    p = argparse.ArgumentParser(description="LLM -> IOMaker Flow builder")
    p.add_argument("--prompt", required=True, help="User natural language requirement")
    p.add_argument("--api-key", required=False, default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--base-url", required=False, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--model", required=False, default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    p.add_argument("--mp-api-key", required=False, default=os.getenv("MP_API_KEY"))
    p.add_argument("--film-cif", required=False, default=DEFAULT_FILM_CIF)
    p.add_argument("--substrate-cif", required=False, default=DEFAULT_SUBSTRATE_CIF)
    p.add_argument("--out", required=False, default="io_flow.json")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key (or env OPENAI_API_KEY).")

    cfg = BuildConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        mp_api_key=args.mp_api_key,
        film_cif=args.film_cif,
        substrate_cif=args.substrate_cif,
        output_flow_json=args.out,
    )

    result = build_iomaker_flow_from_prompt(args.prompt, cfg)
    print("✅ Flow JSON written:", result["flow_json_path"])
    print("📦 Structures source:", result["structures_meta"].get("source"))


if __name__ == "__main__":
    main()

