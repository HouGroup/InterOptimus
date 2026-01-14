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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Literal

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
        "ckpt_path": "",
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
    # Chinese hints
    if "无真空" in t or "不加真空" in t or "不要真空" in t:
        return "without_vacuum"
    return "with_vacuum"


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


def _print_final_settings(settings: Dict[str, Any], structures_meta: Dict[str, Any], flow_json_path: str) -> None:
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

    preset = TUTORIAL_WITH_VACUUM if mode == "with_vacuum" else TUTORIAL_WITHOUT_VACUUM

    # Final settings used by IOMaker: EXACTLY match tutorial presets
    settings: Dict[str, Any] = {
        "name": llm_out.get("name", "IO_llm"),
        "mode": mode,
        "inputs": llm_out.get("inputs", {}) or {},
        "lattice_matching_settings": preset["lattice_matching_settings"],
        "structure_settings": preset["structure_settings"],
        "optimization_settings": preset["optimization_settings"],
        "global_minimization_settings": preset["global_minimization_settings"],
        "do_vasp": bool(llm_out.get("do_vasp", False)),
    }

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

    maker = IOMaker(
        name=settings.get("name", "IO_llm"),
        lattice_matching_settings=settings["lattice_matching_settings"],
        structure_settings=settings["structure_settings"],
        optimization_settings=settings["optimization_settings"],
        global_minimization_settings=settings["global_minimization_settings"],
        do_vasp=bool(settings.get("do_vasp", False)),
        vasp_relax_settings=settings.get("vasp_relax_settings"),
        vasp_static_settings=settings.get("vasp_static_settings"),
        mlip_worker=settings.get("mlip_worker", "std_worker"),
        vasp_worker=settings.get("vasp_worker", "std_worker"),
    )

    flow = maker.make(film_conv, substrate_conv)

    # Write Flow to JSON file
    flow_dict = flow.as_dict() if hasattr(flow, "as_dict") else flow.to_dict()  # jobflow compat
    out_path = os.path.abspath(cfg.output_flow_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flow_dict, f, indent=2)

    if cfg.print_settings:
        _print_final_settings(settings=settings, structures_meta=meta, flow_json_path=out_path)

    return {
        "flow_json_path": out_path,
        "flow_dict": flow_dict,
        "settings": settings,
        "structures_meta": meta,
    }


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

