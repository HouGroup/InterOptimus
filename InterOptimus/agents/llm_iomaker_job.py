#!/usr/bin/env python3
"""
LLM -> InterOptimus.jobflow.IOMaker job builder.

Step 1 (per user request):
- One OpenAI-compatible LLM call (api_key + base_url) embeds a code/preset summary and
  parameter glossary, then outputs the parameter dicts for InterOptimus.jobflow.IOMaker.
- Load input structures from user-provided CIF paths (defaults: ./film.cif and ./substrate.cif).
- Build a jobflow Flow using IOMaker.make(...) and write the Flow to JSON.

On a cluster **login node**, use ``submit_target="server"``: build the flow, then
:func:`InterOptimus.agents.remote_submit.submit_io_flow_locally`; track progress with
``mlip_job_uuid`` / :func:`InterOptimus.agents.remote_submit.iomaker_progress`.
See :mod:`InterOptimus.agents.server_env` (``interoptimus-env`` CLI).
"""

from __future__ import annotations

from types import SimpleNamespace

import json
import os
import re
import sys
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, replace
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Literal, Callable

from pymatgen.core.structure import Structure

from ..jobflow import IOMaker

from .remote_submit import (
    persist_iomaker_task_index_fallback,
    register_interoptimus_server_task,
    submit_io_flow_locally,
)


DEFAULT_FILM_CIF = "film.cif"
DEFAULT_SUBSTRATE_CIF = "substrate.cif"


def _default_server_python() -> str:
    """Python for login-node ``submit_flow``; same env as the running process unless overridden."""
    return (os.getenv("INTEROPTIMUS_SERVER_PYTHON") or "").strip() or sys.executable


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
        "do_mlip_gd": True,
    },
    "global_minimization_settings": {
        "n_calls_density": 4,
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
        "vacuum_over_film": 5,
    },
    "optimization_settings": {
        "fmax": 0.05,
        "steps": 200,
        "device": "cpu",
        "discut": 0.8,
        "ckpt_path": None,
    },
    "global_minimization_settings": {
        "n_calls_density": 4,
        "z_range": [0.5, 3.0],
        "calc": "orb-models",
        "strain_E_correction": True,
    },
}


def _json_safe_for_llm_doc(obj: Any) -> Any:
    """Make preset dicts JSON-serializable for the LLM reference block (tuples → lists)."""
    if isinstance(obj, dict):
        return {k: _json_safe_for_llm_doc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_for_llm_doc(x) for x in obj]
    return obj


IOMAKER_PIPELINE_SUMMARY = """
InterOptimus.jobflow.IOMaker builds a Jobflow workflow that:
(1) Lattice-matches film and substrate supercells (lattice_matching_settings).
(2) Builds interface / bilayer structures (structure_settings).
(3) Relaxes interfaces with an MLIP (optimization_settings: fmax, steps, optional GD, etc.).
(4) Runs global minimization over interface degrees of freedom (global_minimization_settings: z_range, n_calls_density, calc, …).
(5) Optionally runs VASP if do_vasp is true.

mode=with_vacuum: single-interface slab with a vacuum region; typically double_interface=false, vacuum_over_film>0.
mode=without_vacuum: e.g. bicrystal / double-interface; double_interface=true; preset uses vacuum_over_film=5 (Å gap; override in structure_settings / physics if needed).

The JSON below lists the exact default parameter trees for each mode. Your job is to output a JSON object that
follows the output schema: only include nested keys that the user explicitly requests or that you must set to
satisfy the user (e.g. mode, inputs). For any key you omit, the runtime will keep the tutorial default for the
chosen mode after merge.
"""


IOMAKER_PARAMETER_GLOSSARY = """
Parameter glossary (names match IOMaker / InterfaceWorker):

inputs:
  type: must be "local_cif". Structures are always loaded from CIF files (paths in settings.bulk or defaults).

lattice_matching_settings:
  max_area: maximum supercell area (Å²) for lattice matching.
  max_length_tol, max_angle_tol: fractional tolerance on lengths / angles for matching.
  film_max_miller, substrate_max_miller: maximum |h|+|k|+|l| for Miller indices to search.
  film_millers, substrate_millers: optional explicit lists of [h,k,l]; omit or null for auto search.

structure_settings:
  termination_ftol: tolerance for terminating surface planes.
  film_thickness, substrate_thickness: slab thickness in Å (approximate layer count context).
  double_interface: true for bicrystal / two bulk interfaces; false for single slab + vacuum.
  vacuum_over_film: vacuum / gap thickness (Å); with_vacuum uses it as slab vacuum; without_vacuum preset defaults to 5.

optimization_settings (MLIP relaxation):
  set_relax_thicknesses: [fraction_film, fraction_substrate] in (0,1) — relax region as fraction of thicknesses (with_vacuum preset only).
  relax_in_layers, relax_in_ratio: how movable atoms are chosen (layer-wise vs ratio).
  fmax: force convergence criterion (small float, e.g. 0.05 eV/Å). NOT the same as steps.
  steps: integer count of MLIP relaxation steps (must be >= 1 for real runs; never confuse with fmax decimals like 0.05).
  device: "cpu" or "cuda".
  discut: distance cutoff for neighbor lists (Å).
  ckpt_path: path to MLIP checkpoint file, or null for env default.
  BO_coord_bin_size, BO_energy_bin_size, BO_rms_bin_size: Bayesian optimization bin sizes (with_vacuum preset).
  do_mlip_gd: whether to run MLIP gradient descent stage.

global_minimization_settings:
  n_calls_density: sampling density for the global search (higher = more evaluations).
  z_range: [z_min, z_max] interface separation or distance range (Å) for the search.
  calc: MLIP backend id string: orb-models | sevenn | dpa | matris (and similar).
  strain_E_correction: whether to apply strain energy correction for the film.

do_vasp: if true, add VASP jobs (user must supply vasp_resources in code config).
vasp_pre_run: optional shell snippet for the VASP worker (mapped to jobflow-remote exec_config pre_run), e.g. module load VASP/6.4.3.
vasp_exec_config: optional full exec_config dict for the VASP sub-flow; overrides vasp_pre_run when both are set in LLM output.
mlip_calc: optional hint for the same as global_minimization_settings.calc (SevenNet → sevenn).
"""


def _iomaker_llm_reference_block() -> str:
    """Single text block embedded in the one-shot LLM prompt (defaults + meanings)."""
    w = json.dumps(_json_safe_for_llm_doc(TUTORIAL_WITH_VACUUM), indent=2, ensure_ascii=False)
    wo = json.dumps(_json_safe_for_llm_doc(TUTORIAL_WITHOUT_VACUUM), indent=2, ensure_ascii=False)
    return (
        IOMAKER_PIPELINE_SUMMARY.strip()
        + "\n\n--- preset: with_vacuum (defaults) ---\n"
        + w
        + "\n\n--- preset: without_vacuum (defaults) ---\n"
        + wo
        + "\n\n"
        + IOMAKER_PARAMETER_GLOSSARY.strip()
    )


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


_VALID_MLIP_CALCS = frozenset({"orb-models", "sevenn", "dpa", "matris"})


def _normalize_mlip_calc(name: Optional[str]) -> Optional[str]:
    """Map user/LLM strings to InterfaceWorker global_minimization calc names."""
    if not name or not isinstance(name, str):
        return None
    s = name.strip().lower().replace("_", "-")
    if s in _VALID_MLIP_CALCS:
        return s
    aliases = {
        "orb": "orb-models",
        "orbmodels": "orb-models",
        "sevennet": "sevenn",
        "seven-net": "sevenn",
        "7net": "sevenn",
        "7-net": "sevenn",
    }
    if s in aliases:
        return aliases[s]
    return None


def _infer_mlip_calc_from_prompt(user_prompt: str) -> Optional[str]:
    """
    Infer MLIP backend from natural language (tutorial presets default to orb-models).
    Handles e.g. SevenNet / sevennet / sevenn / ORB / DPA.
    """
    raw = user_prompt or ""
    low = raw.lower()
    # SevenNet — check before generic "orb" to avoid mis-ordering
    if re.search(r"(?i)sevennet|seven-net|seven_net|\bsevenn\b|7-net|\b7net\b", low):
        return "sevenn"
    if re.search(r"(?i)orb-models|orb_models", low):
        return "orb-models"
    if re.search(r"(?i)\bmatris\b", low):
        return "matris"
    if re.search(r"(?i)\bdpa\b", low) or "deep potential" in low:
        return "dpa"
    if re.search(r"(?i)\bmlip\b.{0,24}\borb\b", low) or re.search(
        r"(?i)\borb\b.{0,16}(mlip|势|calculator|计算)", low
    ):
        return "orb-models"
    return None


def _resolve_mlip_calc(
    *,
    cfg_mlip: Optional[str],
    llm_mlip: Any,
    user_prompt: str,
) -> Optional[str]:
    """
    Priority: BuildConfig > explicit prompt keywords > LLM (mlip_calc / merged calc).

    Prompt inference runs before LLM so a user saying \"sevennet\" is not overridden
    by the tutorial default orb-models the LLM may echo.
    """
    n = _normalize_mlip_calc(cfg_mlip)
    if n is not None:
        return n
    infer = _infer_mlip_calc_from_prompt(user_prompt)
    if infer is not None:
        return infer
    if isinstance(llm_mlip, str):
        n2 = _normalize_mlip_calc(llm_mlip)
        if n2 is not None:
            return n2
    return None


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


def _extract_max_strain_tolerances(user_prompt: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract max length/angle strain tolerances from prompt.
    Examples:
      - "最大长度、角度应变不超过5%" -> (0.05, 0.05)
      - "长度5%，角度3%" -> (0.05, 0.03)
    """
    raw = user_prompt or ""
    t = raw.lower()

    # Specific split: "长度5%，角度3%" (order-insensitive)
    m_len = re.search(r"(长度|length).{0,4}(\d+(?:\.\d+)?)\s*%+", raw, flags=re.IGNORECASE)
    m_ang = re.search(r"(角度|angle).{0,4}(\d+(?:\.\d+)?)\s*%+", raw, flags=re.IGNORECASE)
    len_tol = float(m_len.group(2)) / 100.0 if m_len else None
    ang_tol = float(m_ang.group(2)) / 100.0 if m_ang else None
    if len_tol is not None or ang_tol is not None:
        return len_tol, ang_tol

    # Common Chinese phrasing with shared percent
    m = re.search(
        r"(最大|最长|长度|角度).{0,8}(应变).{0,8}(不超过|<=|≤|小于等于|低于|小于)\s*(\d+(?:\.\d+)?)\s*%?",
        raw,
    )
    if m:
        val = float(m.group(4)) / 100.0
        return val, val

    # Generic percent in prompt when mentioning length/angle strain
    if ("长度" in raw or "angle" in t or "角度" in raw) and ("应变" in raw or "strain" in t):
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*%+", raw)
        if m2:
            val = float(m2.group(1)) / 100.0
            return val, val

    return None, None


def _extract_thicknesses(user_prompt: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract film/substrate thickness from prompt (Angstrom).

    Examples:
      - "界面厚度为10" -> (10, 10)
      - "薄膜厚度8 基底厚度12" -> (8, 12)
      - "film thickness 6, substrate thickness 9" -> (6, 9)
    """
    raw = user_prompt or ""
    t = raw.lower()

    # Film/ substrate specific patterns
    film_re = re.search(r"(薄膜|film).{0,6}(厚度|thickness).{0,6}(\d+(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    sub_re = re.search(r"(基底|衬底|substrate).{0,6}(厚度|thickness).{0,6}(\d+(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    film_thk = float(film_re.group(3)) if film_re else None
    sub_thk = float(sub_re.group(3)) if sub_re else None
    if film_thk is not None or sub_thk is not None:
        return film_thk, sub_thk

    # Generic "thickness=10" applies to both
    gen = re.search(r"(界面|interface|厚度|thickness).{0,6}(为|=|is)?\s*(\d+(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if gen:
        val = float(gen.group(3))
        return val, val

    return None, None


def _extract_ckpt_path_from_prompt(user_prompt: str) -> Optional[str]:
    """
    Extract ckpt_path from prompt.
    Examples:
      - "ckpt_path=/path/to/model.ckpt"
      - "checkpoint: /path/to/model.pth"
      - "模型路径=/path/to/model"
    """
    raw = user_prompt or ""
    # Prefer quoted path if provided
    m = re.search(r"(ckpt_path|checkpoint|ckpt|模型路径|权重路径)\s*(=|:|为|是)\s*['\"]([^'\"]+)['\"]", raw, flags=re.IGNORECASE)
    if m:
        return m.group(3).strip()
    # Unquoted path
    m2 = re.search(r"(ckpt_path|checkpoint|ckpt|模型路径|权重路径)\s*(=|:|为|是)\s*([^\s,;，]+)", raw, flags=re.IGNORECASE)
    if m2:
        return m2.group(3).strip()
    return None


def _extract_max_area(user_prompt: str) -> Optional[float]:
    """
    Extract max matching area from prompt (Å²).
    Examples:
      - "最大匹配面积不超过25" -> 25
      - "匹配面积不超过 80 平方埃" -> 80
      - "max area <= 30" -> 30
    """
    raw = user_prompt or ""
    t = raw.lower()
    m = re.search(
        r"(最大|最大匹配|匹配)?(面积|area).{0,12}(不超过|<=|≤|小于等于|低于|小于|max)\s*(\d+(?:\.\d+)?)\s*(?:平方埃|Å\s*\^?\s*2|a\s*\^?\s*2)?",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(4))
    # "面积 80 平方埃" / "面积80"
    m2 = re.search(r"(匹配|最大)?(面积|area)\s*(为|是|=|:)?\s*(\d+(?:\.\d+)?)\s*(平方埃|Å)", raw, flags=re.IGNORECASE)
    if m2:
        return float(m2.group(4))
    # English pattern
    m3 = re.search(r"\b(max\s*area|max_area)\b.{0,6}(<=|≤|=|is)?\s*(\d+(?:\.\d+)?)", t)
    if m3:
        return float(m3.group(3))
    return None


_MERGE_SKIP = object()


def _coerce_merge_value(cur: Any, v: Any, key: str) -> Any:
    """Merge one LLM value into a preset; return _MERGE_SKIP to keep cur."""
    if v is None:
        if key in ("film_millers", "substrate_millers", "ckpt_path"):
            return None
        return _MERGE_SKIP
    if isinstance(cur, bool):
        return v if isinstance(v, bool) else cur
    if isinstance(cur, (int, float)) and type(cur) is not bool:
        if isinstance(v, (int, float)) and type(v) is not bool:
            return type(cur)(v)
        return cur
    if isinstance(cur, list):
        return v if isinstance(v, list) else cur
    if isinstance(cur, tuple):
        if isinstance(v, (list, tuple)) and len(v) == len(cur):
            return tuple(v)
        return cur
    if cur is None:
        if key in ("film_millers", "substrate_millers") and isinstance(v, list):
            return v
        return cur
    if isinstance(cur, str) or cur is None:
        if key in ("device", "ckpt_path", "calc") and isinstance(v, str):
            return v.strip()
    return cur


def _merge_llm_partial_settings(settings: Dict[str, Any], llm_out: Dict[str, Any]) -> None:
    """
    Merge partial dicts from llm_out into settings (in-place).
    Only keys that already exist in the preset section are updated; types are coerced safely.
    """
    for section in (
        "lattice_matching_settings",
        "structure_settings",
        "optimization_settings",
        "global_minimization_settings",
    ):
        patch = llm_out.get(section)
        if not isinstance(patch, dict):
            continue
        base = settings.get(section)
        if not isinstance(base, dict):
            continue
        for k, v in patch.items():
            if k not in base:
                continue
            merged = _coerce_merge_value(base[k], v, k)
            if merged is not _MERGE_SKIP:
                settings[section][k] = merged


def _infer_do_vasp_from_prompt(user_prompt: str) -> Optional[bool]:
    """
    If the user clearly requests MLIP-only or VASP, override LLM do_vasp.
    Returns None if ambiguous.
    """
    raw = user_prompt or ""
    low = raw.lower()
    # MLIP-only / no VASP
    if re.search(r"(不要|不做|不进行|不用|仅|只|只要|only).{0,12}(vasp|dft)", raw, flags=re.IGNORECASE):
        return False
    if re.search(r"(?i)(仅|只).{0,6}(mlip|机器学习势|势函数)", raw):
        return False
    if re.search(r"(?i)\bmlip\s+only\b|\bno\s+vasp\b", low):
        return False
    # Explicit VASP / DFT
    if re.search(r"(?i)(需要|加上|运行|用|做|计算).{0,8}(vasp|dft)", raw):
        return True
    if re.search(r"(?i)vasp\s+(弛豫|静态|计算|scf)|dft\s+", raw):
        return True
    return None


def _extract_vacuum_over_film(user_prompt: str) -> Optional[float]:
    """Vacuum layer thickness (Å) when using with_vacuum mode."""
    raw = user_prompt or ""
    m = re.search(
        r"(真空层|真空|vacuum).{0,8}(?:厚度|为|是|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:埃|Å|angstrom|a)?",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(2))
    m2 = re.search(
        r"(vacuum_over_film|vacuum\s+layer).{0,4}(?:=|:)?\s*(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if m2:
        return float(m2.group(2))
    return None


def _extract_z_range_from_prompt(user_prompt: str) -> Optional[Tuple[float, float]]:
    """
    Global minimization z-range (interface distance), e.g. [0.5, 3.0].
    """
    raw = user_prompt or ""
    m = re.search(
        r"(z_range|z\s*区间|界面\s*距离|interface\s+distance).{0,6}[\[\(]?\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*[\]\)]?",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(2)), float(m.group(3))
    m2 = re.search(
        r"(z_range|z\s*范围).{0,4}(?:=|:)\s*[\[\(]?\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if m2:
        return float(m2.group(2)), float(m2.group(3))
    return None


def _extract_n_calls_density_from_prompt(user_prompt: str) -> Optional[float]:
    raw = user_prompt or ""
    m = re.search(
        r"(n_calls_density|采样密度|调用密度).{0,6}(?:=|:|为|是)?\s*(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(2))
    return None


def _extract_fmax_steps_from_prompt(user_prompt: str) -> Tuple[Optional[float], Optional[int]]:
    """MLIP relaxation fmax and steps."""
    raw = user_prompt or ""
    fmax = None
    steps = None
    mf = re.search(
        r"(fmax|力收敛|force\s*tolerance).{0,6}(?:=|:|为|是)?\s*(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if mf:
        fmax = float(mf.group(2))
    st = re.search(
        r"(?:steps|步数|最大步|优化步).{0,8}(?:=|:|为|是)?\s*(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    if st is None:
        st = re.search(
            r"(?:mlip|MLIP)[^。；;\n]{0,24}?(?:steps|步数)\s*为\s*(\d+)",
            raw,
            flags=re.IGNORECASE,
        )
    if st:
        steps = int(st.group(1))
    return fmax, steps


def _extract_miller_max_from_prompt(user_prompt: str) -> Tuple[Optional[int], Optional[int]]:
    """film_max_miller / substrate_max_miller."""
    raw = user_prompt or ""
    fm = re.search(
        r"(film|薄膜).{0,6}(?:miller|Miller|米勒).{0,4}(?:=|:|不超过|≤|<=)?\s*(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    sm = re.search(
        r"(substrate|基底|衬底).{0,6}(?:miller|Miller|米勒).{0,4}(?:=|:|不超过|≤|<=)?\s*(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    fmv = int(fm.group(2)) if fm else None
    smv = int(sm.group(2)) if sm else None
    if fmv is None and smv is None:
        m = re.search(
            r"(?:miller|米勒).{0,4}(?:指数|index).{0,4}(?:不超过|≤|max)?\s*(\d+)",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            v = int(m.group(1))
            return v, v
    return fmv, smv


def _extract_strain_E_correction_from_prompt(user_prompt: str) -> Optional[bool]:
    raw = user_prompt or ""
    if re.search(r"(应变能修正|strain[\s_-]E[\s_-]correction).{0,4}(=|:|为)?\s*(true|1|yes|开|是)", raw, re.I):
        return True
    if re.search(r"(应变能修正|strain[\s_-]E[\s_-]correction).{0,4}(=|:|为)?\s*(false|0|no|关|否)", raw, re.I):
        return False
    if re.search(r"(不要|关闭|禁用).{0,4}(应变能修正)", raw):
        return False
    if re.search(r"(开启|启用).{0,4}(应变能修正)", raw):
        return True
    return None


def _extract_relax_mode_from_prompt(user_prompt: str) -> Tuple[Optional[bool], Optional[bool]]:
    """
    relax_in_layers: True if user asks for layer-wise relaxation.
    relax_in_ratio: False if user explicitly asks not to use ratio mode.
    """
    raw = user_prompt or ""
    low = raw.lower()
    in_layers = None
    in_ratio = None
    if re.search(r"(按层|逐层).{0,4}(放松|弛豫|relax)", raw) or re.search(
        r"(?i)relax\s+in\s+layers", low
    ):
        in_layers = True
        in_ratio = False
    if re.search(r"(按比例|比例放松)", raw) or re.search(r"(?i)relax\s+in\s+ratio", low):
        in_ratio = True
    return in_layers, in_ratio


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
        "do_mlip_gd": opt.get("do_mlip_gd"),
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
    calc_tag = {"orb-models": "orb", "sevenn": "sevenn", "matris": "matris", "dpa": "dpa"}.get(calc, _slug(str(calc), 12))

    key_settings = _summarize_key_settings(settings)
    blob = json.dumps(key_settings, sort_keys=True, ensure_ascii=False)
    h8 = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    return f"IO_{_slug(film_label)}-{_slug(substrate_label)}_m-{mode_tag}_v{1 if do_vasp else 0}_mlip-{calc_tag}_{ts}_{h8}"


def _formula_to_subscript(formula: str) -> str:
    """
    Convert a reduced formula like 'Li2S' to 'Li$_2$S' for plotting.
    """
    return re.sub(r"(\d+)", r"$_\1$", formula or "")


def _local_run_workdir(name: str) -> str:
    """
    Local run directory name, based on IOMaker name.
    """
    return _slug(name, maxlen=128)


def _run_flow_locally(flow, workdir: str) -> None:
    """
    Run jobflow locally inside a specific working directory.
    """
    os.makedirs(workdir, exist_ok=True)
    old_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        try:
            from jobflow import run_locally  # type: ignore
            run_locally(flow, create_folders=True, ensure_success=True)
        except Exception:
            from jobflow.managers.local import run_locally  # type: ignore
            run_locally(flow, create_folders=True, ensure_success=True)
    finally:
        os.chdir(old_cwd)


def _write_run_report(report_path: str, settings: Dict[str, Any], structures_meta: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Write a concise run report to a text file.
    """
    def _dump_json(value: Any) -> str:
        return json.dumps(_clean_for_json_serialization(value), indent=2, ensure_ascii=False)

    lines = []
    lines.append("InterOptimus LLM IOMaker Run Report")
    lines.append("=" * 80)
    lines.append(f"Flow JSON: {result.get('flow_json_path')}")
    lines.append(f"IOMaker name: {settings.get('name')}")
    lines.append(f"Mode: {settings.get('mode')}")
    lines.append(f"Do VASP: {settings.get('do_vasp')}")
    if settings.get("do_vasp"):
        lines.append(f"Do VASP GD: {settings.get('do_vasp_gd')}")
    lines.append("")
    lines.append("Structures")
    lines.append("-" * 80)
    lines.append(f"Source: {structures_meta.get('source')}")
    struct_info = result.get("structures_info", {})
    if struct_info:
        film_info = struct_info.get("film", {})
        sub_info = struct_info.get("substrate", {})
        if film_info:
            lines.append(f"Film: {film_info.get('reduced_formula')} | {film_info.get('pretty_formula')} | SG {film_info.get('spacegroup_symbol')} ({film_info.get('spacegroup_number')})")
        if sub_info:
            lines.append(f"Substrate: {sub_info.get('reduced_formula')} | {sub_info.get('pretty_formula')} | SG {sub_info.get('spacegroup_symbol')} ({sub_info.get('spacegroup_number')})")
    if structures_meta.get("film_cif"):
        lines.append(f"Film CIF: {structures_meta.get('film_cif')}")
    if structures_meta.get("substrate_cif"):
        lines.append(f"Substrate CIF: {structures_meta.get('substrate_cif')}")
    lines.append("")
    lines.append("Key Settings")
    lines.append("-" * 80)
    lm = settings.get("lattice_matching_settings", {})
    st = settings.get("structure_settings", {})
    opt = settings.get("optimization_settings", {})
    gm = settings.get("global_minimization_settings", {})
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
    lines.append("")
    lines.append("IOMaker Parameters (full)")
    lines.append("-" * 80)
    lines.append("lattice_matching_settings:")
    lines.append(_dump_json(settings.get("lattice_matching_settings", {})))
    lines.append("structure_settings:")
    lines.append(_dump_json(settings.get("structure_settings", {})))
    lines.append("optimization_settings:")
    lines.append(_dump_json(settings.get("optimization_settings", {})))
    lines.append("global_minimization_settings:")
    lines.append(_dump_json(settings.get("global_minimization_settings", {})))
    lines.append(f"do_vasp: {settings.get('do_vasp')}")
    lines.append(f"do_vasp_gd: {settings.get('do_vasp_gd')}")
    if settings.get("do_vasp"):
        try:
            from InterOptimus.jobflow import iomaker_resolve_patch_jobflow_vasp_kwargs

            _vk = iomaker_resolve_patch_jobflow_vasp_kwargs(
                SimpleNamespace(
                    relax_user_incar_settings=settings.get("relax_user_incar_settings"),
                    relax_user_potcar_settings=settings.get("relax_user_potcar_settings"),
                    relax_user_kpoints_settings=settings.get("relax_user_kpoints_settings"),
                    relax_user_potcar_functional=settings.get("relax_user_potcar_functional"),
                    static_user_incar_settings=settings.get("static_user_incar_settings"),
                    static_user_potcar_settings=settings.get("static_user_potcar_settings"),
                    static_user_kpoints_settings=settings.get("static_user_kpoints_settings"),
                    static_user_potcar_functional=settings.get("static_user_potcar_functional"),
                    vasp_gd_kwargs=settings.get("vasp_gd_kwargs"),
                    vasp_dipole_correction=settings.get("vasp_dipole_correction", False),
                    vasp_relax_settings=settings.get("vasp_relax_settings"),
                    vasp_static_settings=settings.get("vasp_static_settings"),
                )
            )
            lines.append("VASP (patch_jobflow_jobs, resolved):")
            lines.append(_dump_json(_vk))
        except Exception:
            lines.append("VASP (resolved): (unavailable)")
        if settings.get("vasp_relax_settings") is not None:
            lines.append("vasp_relax_settings (deprecated bucket):")
            lines.append(_dump_json(settings.get("vasp_relax_settings")))
        if settings.get("vasp_static_settings") is not None:
            lines.append("vasp_static_settings (deprecated bucket):")
            lines.append(_dump_json(settings.get("vasp_static_settings")))
    if settings.get("lowest_energy_pairs_settings") is not None:
        lines.append("lowest_energy_pairs_settings:")
        lines.append(_dump_json(settings.get("lowest_energy_pairs_settings")))
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 80)
    lines.append(f"Local run dir: {result.get('local_workdir')}")
    lines.append(f"pairs_summary.txt: {result.get('pairs_summary_path')}")
    lines.append(f"opt_results.pkl: {result.get('opt_results_pkl')}")
    # Pair summary if present (text table)
    pairs_summary_path = result.get("pairs_summary_path")
    if not pairs_summary_path and result.get("pairs_dir"):
        pairs_summary_path = os.path.join(result.get("pairs_dir"), "pairs_summary.txt")
    # Fallback: search under local_workdir if not found
    if (not pairs_summary_path or not os.path.exists(pairs_summary_path)) and result.get("local_workdir"):
        for root, _, files in os.walk(result.get("local_workdir")):
            if "pairs_summary.txt" in files:
                pairs_summary_path = os.path.join(root, "pairs_summary.txt")
                break
    if pairs_summary_path and os.path.exists(pairs_summary_path):
        lines.append("")
        lines.append("Pairs Summary")
        lines.append("-" * 80)
        lines.append(f"(source: {pairs_summary_path})")
        try:
            with open(pairs_summary_path, "r", encoding="utf-8") as f:
                lines.append(f.read())
        except Exception:
            lines.append("Failed to read pairs_summary.txt")

    if result.get("server_submission"):
        lines.append("")
        lines.append("Server (login-node) submission")
        lines.append("-" * 80)
        ss = result["server_submission"]
        lines.append(f"success: {ss.get('success')}")
        lines.append(f"job_id: {ss.get('job_id')}")
        if ss.get("submit_workdir"):
            lines.append(f"submit_workdir: {ss.get('submit_workdir')}")
        if ss.get("error"):
            lines.append(f"error: {ss.get('error')}")
    lines.append("=" * 80)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_config_help() -> None:
    """
    Print a concise help table for BuildConfig parameters.
    """
    print("\n" + "=" * 80)
    print("BuildConfig 使用说明（先实例化了解参数，再传 prompt）")
    print("")
    print("【提交模式】")
    print("- 本地：LocalBuildConfig（run_locally，不走 jobflow-remote）")
    print("- 登录节点：ServerBuildConfig（当前机器 submit_flow + jf job info，无需在配置里写 SSH 账号密码）")
    print("")
    print("【本地模式必填】")
    print("- api_key, base_url")
    print("")
    print("【登录节点模式必填】")
    print("- api_key, base_url")
    print("- mlip_resources, vasp_resources（QResources）")
    print("- mlip_worker, vasp_worker, mlip_project（默认 std）")
    print("- server_pre_cmd：如 source ~/.bashrc && conda activate atomate2")
    print("- 环境自检：interoptimus-env")
    print("")
    print("【常用可选参数】")
    print("- settings.bulk.film_cif / substrate_cif: 本地 CIF 路径（归一化后覆盖 BuildConfig）")
    print("- output_flow_json: 输出 JSON 文件名")
    print("- relax_user_* / static_user_* / vasp_gd_kwargs / vasp_dipole_correction: 与 itworker.patch_jobflow_jobs 同名")
    print("- vasp_relax_settings / vasp_static_settings: 旧版 INCAR/KPOINTS 桶（仅当未设 user_* 时生效）")
    print("- do_mlip_gd: 是否做 MLIP 梯度下降")
    print("- do_vasp_gd: 是否做 VASP GD")
    print("- lowest_energy_pairs_settings: 筛选最佳 pairs 的参数（如 only_lowest_energy_each_plane）")
    print("- ckpt_path: MLIP checkpoint 路径")
    print("- mlip_calc: 强制 MLIP 后端（sevenn / orb-models / dpa / matris），优先级高于 prompt 推断")
    print("")
    print("【Prompt 关键词控制（更详细）】")
    print("- 真空层：")
    print("  - '无真空/不加真空/不要真空/without vacuum/no vacuum' → without_vacuum")
    print("  - '双界面/double interface/double_interface' → without_vacuum（强制）")
    print("- VASP：")
    print("  - 包含“VASP/DFT/电子结构/静态/弛豫”等关键词 → do_vasp=True")
    print("  - 本地模式下会强制 do_vasp=False（不跑 VASP）")
    print("- 梯度下降：")
    print("  - '不要/不做/不进行 梯度下降' 或 'no GD/disable GD' → do_mlip_gd=False 且 do_vasp_gd=False")
    print("  - 单界面默认：do_mlip_gd 开启，do_vasp_gd 关闭（不建议开启 VASP GD）")
    print("- 长度/角度应变：")
    print("  - '最大长度、角度应变不超过5%' → max_length_tol=max_angle_tol=0.05")
    print("  - '长度5%，角度3%' → max_length_tol=0.05, max_angle_tol=0.03")
    print("- MLIP 模型：")
    print("  - prompt 中含 SevenNet/sevennet/sevenn → global_minimization calc=sevenn（教程预设默认为 orb-models）")
    print("  - prompt 中包含 'ckpt_path=/path/to/xxx.ckpt' → 使用指定 checkpoint")
    print("- 其它常用（默认由单次 LLM 根据提示词生成；可选正则回退）：")
    print("  - use_regex_numeric_fallback=True 可在合并后再用正则补数字（旧行为，默认 False）")
    print("  - 匹配面积 / max_area、真空层、z_range、n_calls_density、fmax、steps、Miller 上限等")
    print("  - 仅 MLIP / 不要 VASP：'不要 VASP'、'仅 MLIP' → do_vasp=False")
    print("- 真空/双界面优先级：双界面 > 无真空 > 默认有真空")
    print("")
    print("【双界面 vs 单界面 有效参数提示】")
    print("- 双界面（without_vacuum / double_interface）：")
    print("  - 结构参数更关键：double_interface=True；预设 vacuum_over_film=5（可在 physics 覆盖）")
    print("  - 终止面筛选/厚度：termination_ftol, film_thickness, substrate_thickness")
    print("- 单界面（with_vacuum）：")
    print("  - 真空层参数有效：vacuum_over_film > 0")
    print("  - 终止面筛选/厚度：termination_ftol, film_thickness, substrate_thickness")
    print("  - 仅单界面更常用（含解释）：")
    print("    - do_mlip_gd：是否进行 MLIP 侧梯度下降（GD）")
    print("    - set_relax_thicknesses：设置薄膜/基底的放开厚度比例 (film, substrate)")
    print("    - relax_in_layers：按层放松（True 表示以层为单位选择可动原子）")
    print("    - relax_in_ratio：按比例放松（True 表示按厚度比例选可动原子）")
    print("=" * 80 + "\n")


@dataclass
class BaseBuildConfig:
    """Base configuration for building an IOMaker Flow from text."""

    # LLM
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-3.5-turbo"

    # Inputs
    # By default these are written/read in the CURRENT WORKING DIRECTORY.
    # (i.e., wherever the user runs the Python command)
    film_cif: str = DEFAULT_FILM_CIF
    substrate_cif: str = DEFAULT_SUBSTRATE_CIF

    # Output
    output_flow_json: str = "io_flow.json"

    # Print settings to stdout after building
    print_settings: bool = True

    # Execution target: "local" (in-process run_locally) or "server" (login node: submit_flow via jobflow-remote).
    submit_target: Literal["local", "server"] = "local"

    # --- Login-node / cluster (submit_target="server") ---
    # Optional parent directory for run folders; default is current working directory.
    server_run_parent: Optional[str] = None
    # Shell snippet before submit (e.g. ``source ~/.bashrc && conda activate atomate2``).
    server_pre_cmd: str = ""
    server_python: str = field(default_factory=_default_server_python)
    server_jf_bin: str = "jf"

    # Jobflow resources and workers
    # These are passed to IOMaker for jobflow_remote configuration
    mlip_resources: Optional[Callable] = None  # Function returning QResources for MLIP jobs
    vasp_resources: Optional[Callable] = None  # Function returning QResources for VASP jobs (required if do_vasp=True)
    mlip_worker: str = "std_worker"  # Worker name for MLIP jobs
    vasp_worker: str = "std_worker"  # Worker name for VASP jobs (required if do_vasp=True)
    mlip_project: str = "std"  # jobflow-remote project (submit_flow)
    # jobflow-remote exec_config for VASP sub-flow (e.g. {"pre_run": "module load VASP/6.4.3"}).
    # If None, IOMaker uses env INTEROPTIMUS_VASP_PRE_RUN or a built-in default.
    vasp_exec_config: Optional[Dict[str, Any]] = None

    # VASP: same kwargs as InterfaceWorker.patch_jobflow_jobs (itworker-style user_* on pymatgen input sets).
    relax_user_incar_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_settings: Optional[Dict[str, Any]] = None
    relax_user_kpoints_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_functional: Optional[Any] = "PBE_54"
    static_user_incar_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_settings: Optional[Dict[str, Any]] = None
    static_user_kpoints_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_functional: Optional[Any] = "PBE_54"
    vasp_gd_kwargs: Optional[Dict[str, Any]] = None
    #: None = fall back to settings["vasp_dipole_correction"]; else bool(cfg).
    vasp_dipole_correction: Optional[bool] = None
    #: Deprecated buckets (INCAR/KPOINTS/…); used only if all relax_user_* / static_user_* are unset.
    vasp_relax_settings: Optional[Dict[str, Any]] = None
    vasp_static_settings: Optional[Dict[str, Any]] = None
    # Whether to run MLIP gradient descent
    do_mlip_gd: Optional[bool] = None
    # Whether to run gradient descent for VASP jobs (IO -> VASP GD)
    do_vasp_gd: bool = False
    # Settings for get_lowest_energy_pairs_each_match
    lowest_energy_pairs_settings: Optional[Dict[str, Any]] = None

    # MLIP checkpoint path (optional).
    # If provided, this will be written into optimization_settings["ckpt_path"].
    # If None, the workflow will rely on env fallbacks inside InterfaceWorker.set_energy_calculator.
    ckpt_path: Optional[str] = None
    # If set, forces global_minimization_settings["calc"] (overrides tutorial preset default orb-models).
    mlip_calc: Optional[str] = None  # e.g. "sevenn", "orb-models", "dpa", "matris"
    # If True, apply legacy regex extraction for numeric hints after the single LLM merge (optional fallback).
    use_regex_numeric_fallback: bool = False

    def __post_init__(self) -> None:
        # No side effects on init. Use print_config_help() for guidance.
        return None


@dataclass
class LocalBuildConfig(BaseBuildConfig):
    """Local execution configuration."""

    submit_target: Literal["local", "server"] = "local"


@dataclass
class ServerBuildConfig(BaseBuildConfig):
    """Login-node configuration: submit to jobflow-remote on the current host."""

    submit_target: Literal["local", "server"] = "server"


# Backward compatible alias
BuildConfig = BaseBuildConfig


def _openai_client(api_key: str, base_url: str):
    """Create OpenAI-compatible client (lazy import)."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency 'openai'. Please install it (pip install openai)."
        ) from e
    return OpenAI(api_key=api_key, base_url=base_url)


def load_structures(
    film_cif: str,
    substrate_cif: str,
) -> Tuple[Structure, Structure, Dict[str, Any]]:
    """
    Load film and substrate from local CIF files on disk.

    Relative paths are resolved against the current working directory.
    Returns: (film_conv, substrate_conv, meta)
    """
    meta: Dict[str, Any] = {"source": "local_cif"}

    film_cif_path = film_cif if os.path.isabs(film_cif) else os.path.join(os.getcwd(), film_cif)
    substrate_cif_path = (
        substrate_cif if os.path.isabs(substrate_cif) else os.path.join(os.getcwd(), substrate_cif)
    )

    if not os.path.isfile(film_cif_path):
        raise FileNotFoundError(
            f"Film CIF not found: {film_cif_path!r}. Provide paths via settings.bulk or BuildConfig.film_cif."
        )
    if not os.path.isfile(substrate_cif_path):
        raise FileNotFoundError(
            f"Substrate CIF not found: {substrate_cif_path!r}. Provide paths via settings.bulk or BuildConfig.substrate_cif."
        )

    film_conv = Structure.from_file(film_cif_path)
    substrate_conv = Structure.from_file(substrate_cif_path)
    meta["film_cif"] = os.path.abspath(film_cif_path)
    meta["substrate_cif"] = os.path.abspath(substrate_cif_path)
    return film_conv, substrate_conv, meta


def _strip_llm_json_markdown(content: str) -> str:
    """Remove ```json ... ``` wrappers that models often emit despite instructions."""
    s = (content or "").strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_llm_json_response(content: str) -> Dict[str, Any]:
    raw = _strip_llm_json_markdown(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError(f"LLM did not return parseable JSON. Raw:\n{content!r}") from None
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM JSON root must be an object")
    return data


def _normalize_llm_iomaker_output(data: Dict[str, Any], user_prompt: str) -> None:
    """
    Ensure mode and inputs are usable. Models sometimes return inputs: {} or omit type / paths.
    """
    m = (data.get("mode") or "").strip()
    if m not in ("with_vacuum", "without_vacuum"):
        data["mode"] = _tutorial_mode_from_prompt(user_prompt)

    inp = data.get("inputs")
    if not isinstance(inp, dict):
        inp = {}
    inp["type"] = "local_cif"
    for _k in ("film_mp_id", "substrate_mp_id"):
        inp.pop(_k, None)
    if not inp.get("film_cif"):
        inp["film_cif"] = DEFAULT_FILM_CIF
    if not inp.get("substrate_cif"):
        inp["substrate_cif"] = DEFAULT_SUBSTRATE_CIF
    data["inputs"] = inp


def llm_generate_iomaker_settings(
    *,
    user_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    """
    Single LLM call: full InterOptimus IOMaker reference (defaults + parameter meanings) is embedded in the
    prompt; the model returns JSON for IOMaker (inputs, mode, optional nested overrides, do_vasp, mlip_calc, …).
    """
    schema_hint = {
        "inputs": {
            "type": "local_cif",
            "film_cif": "film.cif",
            "substrate_cif": "substrate.cif",
        },
        "mode": "with_vacuum | without_vacuum",
        "do_vasp": False,
        "mlip_calc": "orb-models | sevenn | dpa | matris | null",
        "lattice_matching_settings": "optional partial dict — only keys that differ from chosen mode defaults",
        "structure_settings": "optional partial dict",
        "optimization_settings": "optional partial dict",
        "global_minimization_settings": "optional partial dict",
        "relax_user_incar_settings": "optional; same as InterfaceWorker.patch_jobflow_jobs",
        "relax_user_potcar_settings": "optional",
        "relax_user_kpoints_settings": "optional dict for pymatgen DictSet (e.g. reciprocal_density, grid_density; not raw kpoints mesh)",
        "relax_user_potcar_functional": "optional",
        "static_user_incar_settings": "optional",
        "static_user_potcar_settings": "optional",
        "static_user_kpoints_settings": "optional dict (reciprocal_density / grid_density / …)",
        "static_user_potcar_functional": "optional",
        "vasp_gd_kwargs": "optional dict {tol: ...} for do_dft_gd",
        "vasp_dipole_correction": "optional bool",
        "vasp_relax_settings": "optional deprecated INCAR/KPOINTS/... bucket if user_* not set",
        "vasp_static_settings": "optional deprecated bucket",
    }

    ref = _iomaker_llm_reference_block()

    sys_msg = (
        "You are an expert in heteroepitaxy and interface modeling using InterOptimus (Python). "
        "The REFERENCE block in the user message contains: (1) a short pipeline summary, "
        "(2) full default parameter trees for with_vacuum and without_vacuum modes, "
        "(3) a glossary of every important key and its meaning.\n"
        "Your task: read the user's natural-language request and produce ONE JSON object that will be merged "
        "into those defaults (runtime starts from the preset matching your chosen mode, then applies your JSON).\n"
        "Return ONLY valid JSON. No markdown fences, no commentary.\n"
        "Rules:\n"
        "- Always include keys 'inputs' and 'mode'.\n"
        "- Choose mode from the user: with_vacuum unless they want bicrystal/double interface/no vacuum → without_vacuum.\n"
        "- Include nested partial dicts only for parameters the user explicitly constrains or that you must set "
        "(e.g. steps, fmax, max_area, z_range, n_calls_density, film_thickness, calc/mlip_calc).\n"
        "- steps: integer >= 1 (MLIP relaxation iteration count). Never use 0. Do not confuse with fmax (e.g. 0.05).\n"
        "- fmax: small positive float (e.g. 0.05) for force convergence — not step count.\n"
        "- mlip_calc / global_minimization_settings.calc: sevenn, orb-models, dpa, or matris as appropriate.\n"
        "- do_vasp: true only if the user clearly wants VASP/DFT in addition to MLIP.\n"
        "- inputs.type must be 'local_cif'; CIF paths default to film.cif / substrate.cif (override via settings.bulk when not using LLM-only JSON).\n"
    )

    user_msg = (
        "REFERENCE (authoritative defaults and parameter meanings):\n\n"
        f"{ref}\n\n"
        "---\n"
        "Output JSON schema shape (example keys, not literal values to copy):\n"
        f"{json.dumps(schema_hint, indent=2, ensure_ascii=False)}\n\n"
        "User requirement:\n"
        f"{user_prompt}\n"
    )

    client = _openai_client(api_key, base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM API request failed ({type(e).__name__}): {e}\n"
            "If you see context length / token limit errors, use a model with a larger context window "
            "or set a smaller user prompt."
        ) from e

    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        ref = getattr(msg, "refusal", None)
        if ref:
            raise ValueError(f"LLM refused to answer: {ref!r}")
        raise ValueError("LLM returned empty content (no JSON).")

    data = _parse_llm_json_response(content)
    _normalize_llm_iomaker_output(data, user_prompt)
    return data


def _sanitize_numeric_settings(settings: Dict[str, Any]) -> None:
    """Drop invalid step/fmax values that often come from LLM or regex mistakes."""
    opt = settings.get("optimization_settings")
    if not isinstance(opt, dict):
        return
    st = opt.get("steps")
    if isinstance(st, (int, float)) and float(st) <= 0:
        opt["steps"] = 200
    fm = opt.get("fmax")
    if isinstance(fm, (int, float)) and float(fm) <= 0:
        opt["fmax"] = 0.05


# Keys required for a **complete** settings dict (same shape as LLM normalized output).
_IOMAKER_FULL_SETTINGS_REQUIRED_KEYS = (
    "name",
    "mode",
    "inputs",
    "lattice_matching_settings",
    "structure_settings",
    "optimization_settings",
    "global_minimization_settings",
)


def uses_full_llm_style_settings_dict(d: Optional[Dict[str, Any]]) -> bool:
    """
    True when *d* has every section of a complete IOMaker settings dict
    (``Parameter settings (LLM output normalized)``), not a partial patch.
    """
    if not isinstance(d, dict):
        return False
    return all(k in d for k in _IOMAKER_FULL_SETTINGS_REQUIRED_KEYS)


def normalize_iomaker_settings_from_full_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate, copy, and lightly normalize a **complete** IOMaker settings dict.

    This is the no-LLM equivalent of the final ``settings`` object in
    :func:`build_iomaker_flow_from_prompt`: **no tutorial preset merge** — *raw* is the
    single source of truth for lattice / structure / optimization / global sections.
    """
    missing = [k for k in _IOMAKER_FULL_SETTINGS_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            "Full IOMaker settings dict missing keys: "
            f"{missing}. Expected the same keys as the LLM normalized "
            '"Parameter settings" block (name, mode, inputs, lattice_matching_settings, …).'
        )
    settings: Dict[str, Any] = deepcopy(raw)
    if not isinstance(settings.get("inputs"), dict):
        settings["inputs"] = {}
    settings["do_vasp"] = bool(settings.get("do_vasp", False))
    _bulk = settings.get("bulk")
    if not isinstance(_bulk, dict):
        _bulk = {}
    settings["bulk"] = {
        "film_cif": str(_bulk.get("film_cif") or DEFAULT_FILM_CIF),
        "substrate_cif": str(_bulk.get("substrate_cif") or DEFAULT_SUBSTRATE_CIF),
    }
    _sanitize_numeric_settings(settings)
    # Structures are always user-provided CIFs; strip legacy MP ids from inputs.
    _inp = settings.get("inputs")
    if isinstance(_inp, dict):
        _inp["type"] = "local_cif"
        _inp.pop("film_mp_id", None)
        _inp.pop("substrate_mp_id", None)
        settings["inputs"] = _inp
    return settings


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
        if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, bytearray)):
            try:
                return _clean_for_json_serialization(obj.tolist(), visited)
            except Exception:
                pass
        if hasattr(obj, "item") and not isinstance(obj, (str, bytes, bytearray)):
            try:
                return obj.item()
            except Exception:
                pass
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
    cfg: Optional[BaseBuildConfig] = None
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
    # Print jobflow configuration
    if cfg:
        print("\n⚙️  Jobflow configuration:")
        print(f"   MLIP worker: {cfg.mlip_worker}")
        print(f"   MLIP project: {cfg.mlip_project}")
        print(f"   MLIP resources: {'✅ Set' if cfg.mlip_resources else '❌ Not set (using defaults)'}")
        if settings.get("do_vasp"):
            print(f"   VASP worker: {cfg.vasp_worker}")
            print(f"   VASP resources: {'✅ Set' if cfg.vasp_resources else '❌ Not set'}")
            print(f"   VASP GD: {'✅ On' if settings.get('do_vasp_gd') else '❌ Off'}")
            vex = getattr(cfg, "vasp_exec_config", None) or settings.get("vasp_exec_config")
            if vex:
                print(f"   VASP exec_config (pre_run, …): {vex}")
            else:
                print("   VASP exec_config: (default / INTEROPTIMUS_VASP_PRE_RUN)")

            print("\n🧪 VASP input (InterfaceWorker.patch_jobflow_jobs / itworker-style):")
            try:
                from InterOptimus.jobflow import iomaker_resolve_patch_jobflow_vasp_kwargs

                _vk = iomaker_resolve_patch_jobflow_vasp_kwargs(
                    SimpleNamespace(
                        relax_user_incar_settings=settings.get("relax_user_incar_settings"),
                        relax_user_potcar_settings=settings.get("relax_user_potcar_settings"),
                        relax_user_kpoints_settings=settings.get("relax_user_kpoints_settings"),
                        relax_user_potcar_functional=settings.get("relax_user_potcar_functional"),
                        static_user_incar_settings=settings.get("static_user_incar_settings"),
                        static_user_potcar_settings=settings.get("static_user_potcar_settings"),
                        static_user_kpoints_settings=settings.get("static_user_kpoints_settings"),
                        static_user_potcar_functional=settings.get("static_user_potcar_functional"),
                        vasp_gd_kwargs=settings.get("vasp_gd_kwargs"),
                        vasp_dipole_correction=settings.get("vasp_dipole_correction", False),
                        vasp_relax_settings=settings.get("vasp_relax_settings"),
                        vasp_static_settings=settings.get("vasp_static_settings"),
                    )
                )
                print(json.dumps(_vk, indent=2, ensure_ascii=False))
            except Exception:
                print("   (could not resolve VASP kwargs preview)")
        
        if cfg.submit_target == "server":
            print("\n🖥️  Login-node (jobflow-remote) submission:")
            print(f"   Run parent dir: {cfg.server_run_parent or '(cwd)'}")
            print(f"   Pre-command: {cfg.server_pre_cmd or '(none)'}")
            print(f"   Python: {cfg.server_python}")
            print(f"   jf binary: {cfg.server_jf_bin}")

    structure_settings = settings.get("structure_settings") or {}
    optimization_settings = settings.get("optimization_settings") or {}
    if (
        isinstance(structure_settings, dict)
        and isinstance(optimization_settings, dict)
        and not bool(structure_settings.get("double_interface", False))
    ):
        print("\n🧩 Single-interface relax controls:")
        print(f"   set_relax_thicknesses: {optimization_settings.get('set_relax_thicknesses')}")
        print(f"   relax_in_layers: {optimization_settings.get('relax_in_layers')}")
        print(f"   relax_in_ratio: {optimization_settings.get('relax_in_ratio')}")
    
    print("\n🔧 Parameter settings (LLM output normalized):")
    print(json.dumps(_clean_for_json_serialization(settings), indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")


def execute_iomaker_pipeline(
    settings: Dict[str, Any],
    cfg: BaseBuildConfig,
    film_conv: Structure,
    substrate_conv: Structure,
    meta: Dict[str, Any],
    structures_info: Dict[str, Any],
    *,
    user_prompt: str = "",
) -> Dict[str, Any]:
    """
    Build IOMaker flow JSON, then run locally or submit on the login node per ``cfg.submit_target``.

    No LLM calls. Used by :func:`build_iomaker_flow_from_prompt` and the
    no-LLM :mod:`InterOptimus.agents.simple_iomaker` helper (requires a full ``settings`` dict plus ``execution``/``cluster`` only).
    """
    # Validate VASP requirements if do_vasp is True
    do_vasp = bool(settings.get("do_vasp", False))
    # Local mode: force-disable VASP
    if cfg.submit_target == "local" and do_vasp:
        do_vasp = False
        settings["do_vasp"] = False
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

    # Login node: jobflow-remote submit_flow
    if cfg.submit_target == "server":
        missing = []
        if do_vasp and cfg.vasp_resources is None:
            missing.append("vasp_resources")
        if cfg.mlip_resources is None:
            missing.append("mlip_resources")
        if do_vasp and not cfg.vasp_worker:
            missing.append("vasp_worker")
        if not cfg.mlip_worker:
            missing.append("mlip_worker")
        if missing:
            raise ValueError(
                "submit_target='server' requires: " + ", ".join(missing)
            )

    def _cfg_or_settings(key: str) -> Any:
        if hasattr(cfg, key) and getattr(cfg, key) is not None:
            return getattr(cfg, key)
        return settings.get(key)

    def _resolve_vasp_dipole() -> bool:
        c = getattr(cfg, "vasp_dipole_correction", None)
        if c is not None:
            return bool(c)
        return bool(settings.get("vasp_dipole_correction", False))

    # VASP: BuildConfig overrides settings (same names as InterfaceWorker.patch_jobflow_jobs).
    vasp_relax_settings = _cfg_or_settings("vasp_relax_settings")
    vasp_static_settings = _cfg_or_settings("vasp_static_settings")
    vasp_gd_kwargs_merged = _cfg_or_settings("vasp_gd_kwargs")
    vasp_dipole_merged = _resolve_vasp_dipole()

    # If user explicitly asks to disable gradient descent, also disable VASP GD.
    do_vasp_gd = bool(cfg.do_vasp_gd)
    if _disable_gd_from_prompt(user_prompt):
        do_vasp_gd = False

    # Record the effective VASP settings for printing/debugging (only matters when do_vasp=True)
    if do_vasp:
        settings["vasp_relax_settings"] = vasp_relax_settings
        settings["vasp_static_settings"] = vasp_static_settings
        settings["vasp_gd_kwargs"] = vasp_gd_kwargs_merged
        settings["vasp_dipole_correction"] = vasp_dipole_merged
        for _vk in (
            "relax_user_incar_settings",
            "relax_user_potcar_settings",
            "relax_user_kpoints_settings",
            "relax_user_potcar_functional",
            "static_user_incar_settings",
            "static_user_potcar_settings",
            "static_user_kpoints_settings",
            "static_user_potcar_functional",
        ):
            settings[_vk] = _cfg_or_settings(_vk)
        settings["do_vasp_gd"] = do_vasp_gd

    vasp_exec_config = getattr(cfg, "vasp_exec_config", None)
    if vasp_exec_config is None and isinstance(settings.get("vasp_exec_config"), dict):
        vasp_exec_config = settings.get("vasp_exec_config")
    if vasp_exec_config is None and settings.get("vasp_pre_run"):
        vasp_exec_config = {"pre_run": str(settings["vasp_pre_run"])}
    if do_vasp and vasp_exec_config is not None:
        settings["vasp_exec_config"] = vasp_exec_config

    # Auto-generate IOMaker name if user didn't specify one.
    # Use reduced_formula for film/substrate labels.
    if not settings.get("name") or settings.get("name") == "IO_llm":
        film_label = getattr(film_conv.composition, "reduced_formula", None) or "film"
        substrate_label = getattr(substrate_conv.composition, "reduced_formula", None) or "substrate"
        settings["name"] = _auto_iomaker_name(
            settings=settings,
            film_label=film_label,
            substrate_label=substrate_label,
        )

    # Run directory: local run_locally and/or server-side flow JSON + submit_flow cwd
    _slug = _local_run_workdir(settings.get("name", "IO_llm"))
    if cfg.submit_target == "server" and cfg.server_run_parent:
        local_workdir = os.path.join(cfg.server_run_parent, _slug)
    else:
        local_workdir = _slug
    local_workdir_abs = os.path.abspath(local_workdir)

    # Resolve callables to concrete QResources so Flow JSON does not embed @callable
    # references to arbitrary modules (e.g. simple_iomaker locals), which break
    # Flow.from_dict in the submit_flow subprocess.
    def _resources_or_callable(res: Any) -> Any:
        if res is None:
            return None
        return res() if callable(res) else res

    mlip_resolved = _resources_or_callable(cfg.mlip_resources)
    vasp_resolved = _resources_or_callable(cfg.vasp_resources)

    maker = IOMaker(
        name=settings.get("name", "IO_llm"),
        lattice_matching_settings=settings["lattice_matching_settings"],
        structure_settings=settings["structure_settings"],
        optimization_settings=settings["optimization_settings"],
        global_minimization_settings=settings["global_minimization_settings"],
        do_vasp=do_vasp,
        do_vasp_gd=do_vasp_gd,
        relax_user_incar_settings=_cfg_or_settings("relax_user_incar_settings"),
        relax_user_potcar_settings=_cfg_or_settings("relax_user_potcar_settings"),
        relax_user_kpoints_settings=_cfg_or_settings("relax_user_kpoints_settings"),
        relax_user_potcar_functional=_cfg_or_settings("relax_user_potcar_functional"),
        static_user_incar_settings=_cfg_or_settings("static_user_incar_settings"),
        static_user_potcar_settings=_cfg_or_settings("static_user_potcar_settings"),
        static_user_kpoints_settings=_cfg_or_settings("static_user_kpoints_settings"),
        static_user_potcar_functional=_cfg_or_settings("static_user_potcar_functional"),
        vasp_gd_kwargs=vasp_gd_kwargs_merged,
        vasp_dipole_correction=vasp_dipole_merged,
        vasp_relax_settings=vasp_relax_settings,
        vasp_static_settings=vasp_static_settings,
        lowest_energy_pairs_settings=cfg.lowest_energy_pairs_settings,
        pairs_output_dir=None,
        mlip_resources=mlip_resolved,
        vasp_resources=vasp_resolved,
        mlip_worker=cfg.mlip_worker,
        vasp_worker=cfg.vasp_worker,
        vasp_exec_config=vasp_exec_config,
    )

    flow = maker.make(film_conv, substrate_conv)

    # Write Flow to JSON file
    # Try to use jobflow's to_json() method first, which handles serialization properly
    os.makedirs(local_workdir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(local_workdir, cfg.output_flow_json))
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

    submit_dir = os.path.abspath(os.path.dirname(out_path))
    result = {
        "flow_json_path": out_path,
        "flow_dict": flow_dict,
        "settings": settings,
        "structures_meta": meta,
        "structures_info": structures_info,
        # Same directory as io_flow.json; contains io_interoptimus_task.json after submit.
        # Artifacts path; query APIs use job UUID (see mlip_job_uuid / iomaker_progress).
        "submit_workdir": submit_dir,
        "interoptimus_task_json_path": os.path.join(submit_dir, "io_interoptimus_task.json"),
    }

    # Run locally if configured
    if cfg.submit_target == "local":
        run_dir = local_workdir or _local_run_workdir(settings.get("name", "IO_llm"))
        _run_flow_locally(flow, run_dir)
        result["local_workdir"] = os.path.abspath(run_dir)
        result["pairs_summary_path"] = os.path.abspath(os.path.join(run_dir, "pairs_summary.txt"))
        result["opt_results_pkl"] = os.path.abspath(os.path.join(run_dir, "opt_results.pkl"))

    if cfg.submit_target == "server" and local_workdir_abs:
        result["local_workdir"] = local_workdir_abs

    # Submit on the current machine (cluster login node) — jobflow-remote
    if cfg.submit_target == "server":
        try:
            from .remote_submit import qresources_to_plain_dict

            submit_result = submit_io_flow_locally(
                out_path,
                workdir=local_workdir_abs,
                remote_python=cfg.server_python,
                pre_cmd=cfg.server_pre_cmd,
                submit_flow_worker=cfg.mlip_worker,
                submit_flow_project=cfg.mlip_project,
                submit_flow_resources_kwargs=(
                    qresources_to_plain_dict(cfg.mlip_resources())
                    if cfg.mlip_resources is not None
                    else None
                ),
            )
            result["server_submission"] = submit_result
            if submit_result.get("mlip_job_uuid"):
                result["mlip_job_uuid"] = submit_result["mlip_job_uuid"]
            if submit_result.get("vasp_job_uuids") is not None:
                result["vasp_job_uuids"] = list(submit_result.get("vasp_job_uuids") or [])
            if submit_result.get("flow_uuid"):
                result["flow_uuid"] = submit_result["flow_uuid"]

            if submit_result.get("success"):
                print(f"\n✅ Submitted to jobflow-remote on this host (login-node mode)")
                if submit_result.get("submit_workdir"):
                    print(f"   Submit cwd: {submit_result['submit_workdir']}")
                if submit_result.get("job_id"):
                    print(f"📋 submit_flow job id: {submit_result['job_id']}")
                if submit_result.get("mlip_job_uuid"):
                    print(f"📌 MLIP job UUID（query_interoptimus_task_progress 优先跟踪）: {submit_result['mlip_job_uuid']}")
                if submit_result.get("submit_id_note"):
                    print(f"   ℹ️  {submit_result['submit_id_note']}")
                if submit_result.get("vasp_job_uuids"):
                    print(f"📎 Planned VASP job UUID(s): {submit_result['vasp_job_uuids']}")

                reg_jf_id = str(
                    submit_result.get("job_id")
                    or submit_result.get("mlip_job_uuid")
                    or submit_result.get("flow_uuid")
                    or ""
                ).strip()

                if reg_jf_id:
                    try:
                        wd_reg = submit_result.get("submit_workdir") or local_workdir_abs or ""
                        rec = register_interoptimus_server_task(
                            task_name=str(settings.get("name", "IO_llm")),
                            jf_job_id=reg_jf_id,
                            submit_workdir=str(wd_reg),
                            do_vasp=bool(settings.get("do_vasp")),
                            mlip_job_uuid=submit_result.get("mlip_job_uuid"),
                            vasp_job_uuids=submit_result.get("vasp_job_uuids"),
                            flow_uuid=submit_result.get("flow_uuid"),
                        )
                        result["interoptimus_task_serial"] = rec["serial_id"]
                        result["interoptimus_task_record"] = rec
                        _mlip = (rec.get("mlip_job_uuid") or "").strip() or str(
                            submit_result.get("mlip_job_uuid") or ""
                        ).strip()
                        print(f"\n🔖 Registry serial (内部用）: {rec['serial_id']}")
                        if _mlip:
                            print(
                                "   查进度 / 拉结果请用 **job UUID**："
                                f"iomaker_progress(result) 或 query_interoptimus_task_progress({_mlip!r})"
                            )
                            print(
                                "   内核重启: export INTEROPTIMUS_JOB_ID="
                                f"{_mlip!r} 或 iomaker_progress(ref={_mlip!r})"
                            )
                            print(
                                "   fetch: fetch_interoptimus_task_results("
                                f"{_mlip!r}, copy_images_to=..., ...)"
                            )
                        else:
                            print(
                                "   查进度: iomaker_progress(result) 或 "
                                "query_interoptimus_task_progress(<submit_flow job id UUID>)"
                            )
                    except Exception as _reg_err:
                        result["interoptimus_task_register_error"] = str(_reg_err)
                        try:
                            persist_iomaker_task_index_fallback(
                                task_name=str(settings.get("name", "IO_llm")),
                                jf_job_id=reg_jf_id,
                                submit_workdir=str(
                                    submit_result.get("submit_workdir")
                                    or local_workdir_abs
                                    or ""
                                ),
                                do_vasp=bool(settings.get("do_vasp")),
                                mlip_job_uuid=submit_result.get("mlip_job_uuid"),
                                vasp_job_uuids=submit_result.get("vasp_job_uuids"),
                                flow_uuid=submit_result.get("flow_uuid"),
                            )
                        except Exception:
                            pass
            else:
                print(f"\n❌ Login-node submit_flow failed:")
                print(f"   Error: {submit_result.get('error', 'unknown')}")
                print(f"   Stderr: {submit_result.get('stderr', '')}")
        except Exception as e:
            print(f"\n⚠️  Server-side submission failed: {e}")
            result["server_submission_error"] = str(e)

    # Write summary report
    report_path = os.path.join(os.path.dirname(out_path), "io_report.txt")
    _write_run_report(report_path, settings=settings, structures_meta=meta, result=result)
    result["report_path"] = report_path

    return result


def execute_iomaker_from_settings(
    settings: Dict[str, Any],
    cfg: BaseBuildConfig,
    user_prompt: str = "",
) -> Dict[str, Any]:
    """
    Load structures from ``cfg`` / ``settings["inputs"]``, then run :func:`execute_iomaker_pipeline`.

    CIF paths: ``settings["bulk"]["film_cif"]`` / ``settings["bulk"]["substrate_cif"]`` (filled by
    :func:`normalize_iomaker_settings_from_full_dict`) override ``cfg.film_cif`` / ``cfg.substrate_cif``.
    """
    bulk = settings.get("bulk") if isinstance(settings.get("bulk"), dict) else {}
    if bulk:
        cfg = replace(
            cfg,
            film_cif=str(bulk.get("film_cif") or cfg.film_cif),
            substrate_cif=str(bulk.get("substrate_cif") or cfg.substrate_cif),
        )

    film_conv, substrate_conv, meta = load_structures(cfg.film_cif, cfg.substrate_cif)

    # Structure info for report (formula + spacegroup)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        film_sg = SpacegroupAnalyzer(film_conv).get_space_group_symbol()
        film_sg_num = SpacegroupAnalyzer(film_conv).get_space_group_number()
        sub_sg = SpacegroupAnalyzer(substrate_conv).get_space_group_symbol()
        sub_sg_num = SpacegroupAnalyzer(substrate_conv).get_space_group_number()
        structures_info = {
            "film": {
                "reduced_formula": film_conv.composition.reduced_formula,
                "pretty_formula": film_conv.composition.formula,
                "spacegroup_symbol": film_sg,
                "spacegroup_number": film_sg_num,
            },
            "substrate": {
                "reduced_formula": substrate_conv.composition.reduced_formula,
                "pretty_formula": substrate_conv.composition.formula,
                "spacegroup_symbol": sub_sg,
                "spacegroup_number": sub_sg_num,
            },
        }
    except Exception:
        structures_info = {}

    return execute_iomaker_pipeline(
        settings,
        cfg,
        film_conv,
        substrate_conv,
        meta,
        structures_info,
        user_prompt=user_prompt,
    )


def build_iomaker_flow_from_prompt(user_prompt: str, cfg: BaseBuildConfig) -> Dict[str, Any]:
    """
    Main entry: build an IOMaker Flow JSON from natural language.

    Returns:
      dict with keys:
        - flow_json_path
        - flow_dict
        - settings (llm output)
        - structures_meta
        - server_submission (optional): when ``submit_target="server"``
    """
    if not cfg.api_key or not cfg.base_url:
        raise ValueError("api_key and base_url are required to run the LLM. Initialize BuildConfig with them.")

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

    # LLM may return partial lattice/structure/optimization/global dicts (same keys as Tutorial presets)
    _merge_llm_partial_settings(settings, llm_out)

    for _vk in (
        "relax_user_incar_settings",
        "relax_user_potcar_settings",
        "relax_user_kpoints_settings",
        "relax_user_potcar_functional",
        "static_user_incar_settings",
        "static_user_potcar_settings",
        "static_user_kpoints_settings",
        "static_user_potcar_functional",
        "vasp_gd_kwargs",
    ):
        if _vk in llm_out and llm_out[_vk] is not None:
            settings[_vk] = llm_out[_vk]
    if "vasp_dipole_correction" in llm_out and isinstance(llm_out["vasp_dipole_correction"], bool):
        settings["vasp_dipole_correction"] = llm_out["vasp_dipole_correction"]
    if isinstance(llm_out.get("vasp_relax_settings"), dict):
        settings["vasp_relax_settings"] = llm_out["vasp_relax_settings"]
    if isinstance(llm_out.get("vasp_static_settings"), dict):
        settings["vasp_static_settings"] = llm_out["vasp_static_settings"]

    merged_calc_hint = llm_out.get("mlip_calc")
    if merged_calc_hint is None:
        merged_calc_hint = settings["global_minimization_settings"].get("calc")
    resolved_calc = _resolve_mlip_calc(
        cfg_mlip=cfg.mlip_calc,
        llm_mlip=merged_calc_hint if isinstance(merged_calc_hint, str) else None,
        user_prompt=user_prompt,
    )
    if resolved_calc is not None:
        settings["global_minimization_settings"]["calc"] = resolved_calc

    # Prompt beats LLM when user explicitly requests / forbids VASP
    do_vasp_hint = _infer_do_vasp_from_prompt(user_prompt)
    if do_vasp_hint is not None:
        settings["do_vasp"] = do_vasp_hint

    # Allow user to override MLIP checkpoint path via BuildConfig or prompt
    # (goes into InterfaceWorker.opt_kwargs via parse_optimization_params)
    if cfg.ckpt_path is not None:
        settings["optimization_settings"]["ckpt_path"] = cfg.ckpt_path
    else:
        prompt_ckpt = _extract_ckpt_path_from_prompt(user_prompt)
        if prompt_ckpt:
            settings["optimization_settings"]["ckpt_path"] = prompt_ckpt
    if settings["optimization_settings"].get("ckpt_path") == "":
        settings["optimization_settings"]["ckpt_path"] = None

    # If user explicitly asks to NOT do gradient descent, force-disable it.
    # This controls MLIP-side gradient descent in InterfaceWorker.parse_optimization_params().
    if _disable_gd_from_prompt(user_prompt):
        settings["optimization_settings"]["do_mlip_gd"] = False
    elif cfg.do_mlip_gd is not None:
        settings["optimization_settings"]["do_mlip_gd"] = cfg.do_mlip_gd

    # Optional legacy regex fallbacks (off by default; single LLM carries full reference + user prompt).
    if getattr(cfg, "use_regex_numeric_fallback", False):
        # If user specifies max length/angle strain tolerance, override both.
        len_tol, ang_tol = _extract_max_strain_tolerances(user_prompt)
        if len_tol is not None:
            settings["lattice_matching_settings"]["max_length_tol"] = len_tol
        if ang_tol is not None:
            settings["lattice_matching_settings"]["max_angle_tol"] = ang_tol

        # Max matching area from prompt
        max_area = _extract_max_area(user_prompt)
        if max_area is not None:
            settings["lattice_matching_settings"]["max_area"] = max_area

        # If user specifies interface thickness, set film/substrate thickness (can be split)
        film_thk, sub_thk = _extract_thicknesses(user_prompt)
        if film_thk is not None:
            settings["structure_settings"]["film_thickness"] = film_thk
        if sub_thk is not None:
            settings["structure_settings"]["substrate_thickness"] = sub_thk

        # Vacuum layer (single-interface / with_vacuum only)
        if mode == "with_vacuum":
            vac = _extract_vacuum_over_film(user_prompt)
            if vac is not None:
                settings["structure_settings"]["vacuum_over_film"] = vac

        zr = _extract_z_range_from_prompt(user_prompt)
        if zr is not None:
            settings["global_minimization_settings"]["z_range"] = [zr[0], zr[1]]

        ncd = _extract_n_calls_density_from_prompt(user_prompt)
        if ncd is not None:
            settings["global_minimization_settings"]["n_calls_density"] = ncd

        fm, st = _extract_fmax_steps_from_prompt(user_prompt)
        if fm is not None:
            settings["optimization_settings"]["fmax"] = fm
        if st is not None:
            settings["optimization_settings"]["steps"] = st

        fmx, smx = _extract_miller_max_from_prompt(user_prompt)
        if fmx is not None:
            settings["lattice_matching_settings"]["film_max_miller"] = fmx
        if smx is not None:
            settings["lattice_matching_settings"]["substrate_max_miller"] = smx

        sec = _extract_strain_E_correction_from_prompt(user_prompt)
        if sec is not None:
            settings["global_minimization_settings"]["strain_E_correction"] = sec

        r_layers, r_ratio = _extract_relax_mode_from_prompt(user_prompt)
        opt = settings["optimization_settings"]
        if r_layers is not None and "relax_in_layers" in opt:
            opt["relax_in_layers"] = r_layers
        if r_ratio is not None and "relax_in_ratio" in opt:
            opt["relax_in_ratio"] = r_ratio

    _sanitize_numeric_settings(settings)
    # Structures are always user-provided CIF files (paths from BuildConfig unless settings.bulk set).
    _b0 = settings.get("bulk") if isinstance(settings.get("bulk"), dict) else {}
    settings["bulk"] = {
        "film_cif": str(_b0.get("film_cif") or cfg.film_cif or DEFAULT_FILM_CIF),
        "substrate_cif": str(_b0.get("substrate_cif") or cfg.substrate_cif or DEFAULT_SUBSTRATE_CIF),
    }
    _inp = settings.get("inputs")
    if not isinstance(_inp, dict):
        _inp = {}
    _inp["type"] = "local_cif"
    _inp.pop("film_mp_id", None)
    _inp.pop("substrate_mp_id", None)
    settings["inputs"] = _inp

    return execute_iomaker_from_settings(settings, cfg, user_prompt=user_prompt)


def main():
    import argparse

    p = argparse.ArgumentParser(description="LLM -> IOMaker Flow builder")
    p.add_argument("--prompt", required=True, help="User natural language requirement")
    p.add_argument("--api-key", required=False, default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--base-url", required=False, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--model", required=False, default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    p.add_argument("--film-cif", required=False, default=DEFAULT_FILM_CIF)
    p.add_argument("--substrate-cif", required=False, default=DEFAULT_SUBSTRATE_CIF)
    p.add_argument("--out", required=False, default="io_flow.json")
    p.add_argument(
        "--submit-target",
        choices=("local", "server"),
        default="local",
        help="local=run_locally; server=jobflow-remote submit on this host (login node)",
    )
    p.add_argument(
        "--server-pre-cmd",
        default=os.getenv("INTEROPTIMUS_SERVER_PRE_CMD", ""),
        help="For --submit-target server: shell snippet before python (e.g. conda activate)",
    )
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key (or env OPENAI_API_KEY).")

    cfg = BuildConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        film_cif=args.film_cif,
        substrate_cif=args.substrate_cif,
        output_flow_json=args.out,
        submit_target=args.submit_target,
        server_pre_cmd=args.server_pre_cmd,
    )

    result = build_iomaker_flow_from_prompt(args.prompt, cfg)
    print("✅ Flow JSON written:", result["flow_json_path"])
    print("📦 Structures source:", result["structures_meta"].get("source"))


if __name__ == "__main__":
    main()

