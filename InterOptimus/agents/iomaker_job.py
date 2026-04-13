"""IOMaker job pipeline (JSON settings → Jobflow Flow, local or jobflow-remote).

This module is the **non-LLM** core used by :mod:`InterOptimus.agents.simple_iomaker`.
Natural-language → settings via LLM was removed from the ``pure-workflow`` branch;
use ``interoptimus-simple`` / :func:`run_simple_iomaker` with a full JSON/YAML config instead.
"""
from __future__ import annotations
from types import SimpleNamespace
import json
import os
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Literal, Callable
from pymatgen.core.structure import Structure
from ..jobflow import IOMaker
from .remote_submit import persist_iomaker_task_index_fallback, register_interoptimus_server_task, submit_io_flow_locally
DEFAULT_FILM_CIF = 'film.cif'
DEFAULT_SUBSTRATE_CIF = 'substrate.cif'
TUTORIAL_WITH_VACUUM = {'lattice_matching_settings': {'max_area': 20, 'max_length_tol': 0.03, 'max_angle_tol': 0.03, 'film_max_miller': 3, 'substrate_max_miller': 3, 'film_millers': None, 'substrate_millers': None}, 'structure_settings': {'termination_ftol': 0.15, 'film_thickness': 8, 'substrate_thickness': 8, 'double_interface': False, 'vacuum_over_film': 5}, 'optimization_settings': {'set_relax_thicknesses': (1 / 2, 1 / 2), 'relax_in_layers': False, 'relax_in_ratio': True, 'fmax': 0.05, 'steps': 200, 'device': 'cpu', 'discut': 0.8, 'ckpt_path': None, 'BO_coord_bin_size': 0.25, 'BO_energy_bin_size': 0.05, 'BO_rms_bin_size': 0.3, 'do_mlip_gd': True}, 'global_minimization_settings': {'n_calls_density': 4, 'z_range': [0.5, 3.0], 'calc': 'orb-models', 'strain_E_correction': True}}
TUTORIAL_WITHOUT_VACUUM = {'lattice_matching_settings': {'max_area': 25, 'max_length_tol': 0.03, 'max_angle_tol': 0.03, 'film_max_miller': 3, 'substrate_max_miller': 3, 'film_millers': None, 'substrate_millers': None}, 'structure_settings': {'termination_ftol': 0.15, 'film_thickness': 8, 'substrate_thickness': 8, 'double_interface': True, 'vacuum_over_film': 5}, 'optimization_settings': {'fmax': 0.05, 'steps': 200, 'device': 'cpu', 'discut': 0.8, 'ckpt_path': None}, 'global_minimization_settings': {'n_calls_density': 4, 'z_range': [0.5, 3.0], 'calc': 'orb-models', 'strain_E_correction': True}}
def _disable_gd_from_prompt(user_prompt: str) -> bool:
    """
    Return True if the user explicitly asks to disable gradient descent (GD).

    Examples:
    - 不要/不做/不进行 梯度下降
    - 不要 GD / 不用 GD
    - no gradient descent / disable GD
    """
    raw = user_prompt or ''
    t = raw.lower()
    if re.search('(不要|不做|不进行|不用|禁止).{0,8}(梯度下降|gd)', raw):
        return True
    if re.search("\\b(no|without|disable|dont|don't|do not)\\b.{0,16}\\b(gradient\\s*descent|gd)\\b", t):
        return True
    if re.search('\\bno\\s*gd\\b', t) or re.search('\\bdisable\\s*gd\\b', t):
        return True
    return False


def _slug(text: str, maxlen: int=32) -> str:
    """
    Make a filesystem-safe, compact token for naming.
    """
    s = (text or 'NA').strip()
    s = re.sub('\\s+', '-', s)
    s = re.sub('[^A-Za-z0-9._-]+', '', s)
    return s[:maxlen] or 'NA'

def _summarize_key_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select a stable subset of settings for name hashing.
    """
    lm = settings.get('lattice_matching_settings', {}) or {}
    opt = settings.get('optimization_settings', {}) or {}
    gm = settings.get('global_minimization_settings', {}) or {}
    return {'mode': settings.get('mode'), 'do_vasp': settings.get('do_vasp'), 'calc': gm.get('calc'), 'z_range': gm.get('z_range'), 'n_calls_density': gm.get('n_calls_density'), 'strain_E_correction': gm.get('strain_E_correction'), 'max_area': lm.get('max_area'), 'max_length_tol': lm.get('max_length_tol'), 'max_angle_tol': lm.get('max_angle_tol'), 'do_mlip_gd': opt.get('do_mlip_gd'), 'ckpt_path': opt.get('ckpt_path')}

def _auto_iomaker_name(settings: Dict[str, Any], film_label: str, substrate_label: str) -> str:
    """
    Build a deterministic, human-readable name for IOMaker jobs.
    """
    mode = settings.get('mode', 'with_vacuum')
    do_vasp = bool(settings.get('do_vasp', False))
    mode_tag = 'double' if mode == 'without_vacuum' else 'vac'
    gm = settings.get('global_minimization_settings', {}) or {}
    calc = gm.get('calc', 'mlip')
    calc_tag = {'orb-models': 'orb', 'sevenn': 'sevenn', 'matris': 'matris', 'dpa': 'dpa'}.get(calc, _slug(str(calc), 12))
    key_settings = _summarize_key_settings(settings)
    blob = json.dumps(key_settings, sort_keys=True, ensure_ascii=False)
    h8 = hashlib.sha1(blob.encode('utf-8')).hexdigest()[:8]
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'IO_{_slug(film_label)}-{_slug(substrate_label)}_m-{mode_tag}_v{(1 if do_vasp else 0)}_mlip-{calc_tag}_{ts}_{h8}'

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
            from jobflow import run_locally
            run_locally(flow, create_folders=True, ensure_success=True)
        except Exception:
            from jobflow.managers.local import run_locally
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
    lines.append('InterOptimus IOMaker Run Report')
    lines.append('=' * 80)
    lines.append(f"Flow JSON: {result.get('flow_json_path')}")
    lines.append(f"IOMaker name: {settings.get('name')}")
    lines.append(f"Mode: {settings.get('mode')}")
    lines.append(f"Do VASP: {settings.get('do_vasp')}")
    if settings.get('do_vasp'):
        lines.append(f"Do VASP GD: {settings.get('do_vasp_gd')}")
    lines.append('')
    lines.append('Structures')
    lines.append('-' * 80)
    lines.append(f"Source: {structures_meta.get('source')}")
    struct_info = result.get('structures_info', {})
    if struct_info:
        film_info = struct_info.get('film', {})
        sub_info = struct_info.get('substrate', {})
        if film_info:
            lines.append(f"Film: {film_info.get('reduced_formula')} | {film_info.get('pretty_formula')} | SG {film_info.get('spacegroup_symbol')} ({film_info.get('spacegroup_number')})")
        if sub_info:
            lines.append(f"Substrate: {sub_info.get('reduced_formula')} | {sub_info.get('pretty_formula')} | SG {sub_info.get('spacegroup_symbol')} ({sub_info.get('spacegroup_number')})")
    if structures_meta.get('film_cif'):
        lines.append(f"Film CIF: {structures_meta.get('film_cif')}")
    if structures_meta.get('substrate_cif'):
        lines.append(f"Substrate CIF: {structures_meta.get('substrate_cif')}")
    lines.append('')
    lines.append('Key Settings')
    lines.append('-' * 80)
    lm = settings.get('lattice_matching_settings', {})
    st = settings.get('structure_settings', {})
    opt = settings.get('optimization_settings', {})
    gm = settings.get('global_minimization_settings', {})
    lines.append(f"max_area: {lm.get('max_area')}")
    lines.append(f"max_length_tol: {lm.get('max_length_tol')}")
    lines.append(f"max_angle_tol: {lm.get('max_angle_tol')}")
    lines.append(f"film_thickness: {st.get('film_thickness')}")
    lines.append(f"substrate_thickness: {st.get('substrate_thickness')}")
    lines.append(f"vacuum_over_film: {st.get('vacuum_over_film')}")
    lines.append(f"calc: {gm.get('calc')}")
    ckpt = opt.get('ckpt_path')
    lines.append(f'ckpt_path: {ckpt}')
    if ckpt:
        lines.append(f'ckpt_model_name: {os.path.basename(str(ckpt))}')
    lines.append(f"do_mlip_gd: {opt.get('do_mlip_gd')}")
    lines.append('')
    lines.append('IOMaker Parameters (full)')
    lines.append('-' * 80)
    lines.append('lattice_matching_settings:')
    lines.append(_dump_json(settings.get('lattice_matching_settings', {})))
    lines.append('structure_settings:')
    lines.append(_dump_json(settings.get('structure_settings', {})))
    lines.append('optimization_settings:')
    lines.append(_dump_json(settings.get('optimization_settings', {})))
    lines.append('global_minimization_settings:')
    lines.append(_dump_json(settings.get('global_minimization_settings', {})))
    lines.append(f"do_vasp: {settings.get('do_vasp')}")
    lines.append(f"do_vasp_gd: {settings.get('do_vasp_gd')}")
    if settings.get('do_vasp'):
        try:
            from InterOptimus.jobflow import iomaker_resolve_patch_jobflow_vasp_kwargs
            _vk = iomaker_resolve_patch_jobflow_vasp_kwargs(SimpleNamespace(relax_user_incar_settings=settings.get('relax_user_incar_settings'), relax_user_potcar_settings=settings.get('relax_user_potcar_settings'), relax_user_kpoints_settings=settings.get('relax_user_kpoints_settings'), relax_user_potcar_functional=settings.get('relax_user_potcar_functional'), static_user_incar_settings=settings.get('static_user_incar_settings'), static_user_potcar_settings=settings.get('static_user_potcar_settings'), static_user_kpoints_settings=settings.get('static_user_kpoints_settings'), static_user_potcar_functional=settings.get('static_user_potcar_functional'), vasp_gd_kwargs=settings.get('vasp_gd_kwargs'), vasp_dipole_correction=settings.get('vasp_dipole_correction', False), vasp_relax_settings=settings.get('vasp_relax_settings'), vasp_static_settings=settings.get('vasp_static_settings')))
            lines.append('VASP (patch_jobflow_jobs, resolved):')
            lines.append(_dump_json(_vk))
        except Exception:
            lines.append('VASP (resolved): (unavailable)')
        if settings.get('vasp_relax_settings') is not None:
            lines.append('vasp_relax_settings (deprecated bucket):')
            lines.append(_dump_json(settings.get('vasp_relax_settings')))
        if settings.get('vasp_static_settings') is not None:
            lines.append('vasp_static_settings (deprecated bucket):')
            lines.append(_dump_json(settings.get('vasp_static_settings')))
    if settings.get('lowest_energy_pairs_settings') is not None:
        lines.append('lowest_energy_pairs_settings:')
        lines.append(_dump_json(settings.get('lowest_energy_pairs_settings')))
    lines.append('')
    lines.append('Outputs')
    lines.append('-' * 80)
    lines.append(f"Local run dir: {result.get('local_workdir')}")
    lines.append(f"pairs_summary.txt: {result.get('pairs_summary_path')}")
    lines.append(f"opt_results.pkl: {result.get('opt_results_pkl')}")
    pairs_summary_path = result.get('pairs_summary_path')
    if not pairs_summary_path and result.get('pairs_dir'):
        pairs_summary_path = os.path.join(result.get('pairs_dir'), 'pairs_summary.txt')
    if (not pairs_summary_path or not os.path.exists(pairs_summary_path)) and result.get('local_workdir'):
        for root, _, files in os.walk(result.get('local_workdir')):
            if 'pairs_summary.txt' in files:
                pairs_summary_path = os.path.join(root, 'pairs_summary.txt')
                break
    if pairs_summary_path and os.path.exists(pairs_summary_path):
        lines.append('')
        lines.append('Pairs Summary')
        lines.append('-' * 80)
        lines.append(f'(source: {pairs_summary_path})')
        try:
            with open(pairs_summary_path, 'r', encoding='utf-8') as f:
                lines.append(f.read())
        except Exception:
            lines.append('Failed to read pairs_summary.txt')
    if result.get('server_submission'):
        lines.append('')
        lines.append('Server (login-node) submission')
        lines.append('-' * 80)
        ss = result['server_submission']
        lines.append(f"success: {ss.get('success')}")
        lines.append(f"job_id: {ss.get('job_id')}")
        if ss.get('submit_workdir'):
            lines.append(f"submit_workdir: {ss.get('submit_workdir')}")
        if ss.get('error'):
            lines.append(f"error: {ss.get('error')}")
    lines.append('=' * 80)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def print_config_help() -> None:
    """Print a concise help table for BuildConfig parameters (JSON / simple_iomaker workflow)."""
    print("\n" + "=" * 80)
    print("BuildConfig（LocalBuildConfig / ServerBuildConfig）— 与 interoptimus-simple 配套")
    print("")
    print("【提交模式】")
    print("- 本地：LocalBuildConfig（jobflow run_locally，不走 jobflow-remote；do_vasp 会被强制 False）")
    print("- 登录节点：ServerBuildConfig（本机 jf submit_flow；需已配置 MongoDB、runner、worker）")
    print("")
    print("【登录节点模式通常必填】")
    print("- mlip_resources: 返回 QResources 的可调用对象（MLIP 作业）")
    print("- mlip_worker, mlip_project（worker 名与 JFREMOTE 项目名，需与 ~/.jfremote/*.yaml 一致）")
    print("- do_vasp=True 时还需: vasp_resources, vasp_worker，以及 settings 里 VASP/potcar 相关配置")
    print("- server_pre_cmd: 例如 source ~/.bashrc && conda activate <env>")
    print("- 环境自检: interoptimus-env")
    print("")
    print("【常用可选】")
    print("- film_cif / substrate_cif 或 settings.bulk 下的 CIF 路径")
    print("- output_flow_json, ckpt_path, relax_user_* / static_user_*（与 InterfaceWorker 一致）")
    print("- 详见 docs/simple_iomaker_parameters.md")
    print("=" * 80 + "\n")

@dataclass
class BaseBuildConfig:
    """Base configuration for building an IOMaker Flow from JSON settings (no LLM)."""
    film_cif: str = DEFAULT_FILM_CIF
    substrate_cif: str = DEFAULT_SUBSTRATE_CIF
    output_flow_json: str = 'io_flow.json'
    print_settings: bool = True
    submit_target: Literal['local', 'server'] = 'local'
    server_run_parent: Optional[str] = None
    server_pre_cmd: str = ''
    server_python: str = 'python'
    server_jf_bin: str = 'jf'
    mlip_resources: Optional[Callable] = None
    vasp_resources: Optional[Callable] = None
    mlip_worker: str = 'std_worker'
    vasp_worker: str = 'std_worker'
    mlip_project: str = 'std'
    vasp_exec_config: Optional[Dict[str, Any]] = None
    relax_user_incar_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_settings: Optional[Dict[str, Any]] = None
    relax_user_kpoints_settings: Optional[Dict[str, Any]] = None
    relax_user_potcar_functional: Optional[Any] = 'PBE_54'
    static_user_incar_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_settings: Optional[Dict[str, Any]] = None
    static_user_kpoints_settings: Optional[Dict[str, Any]] = None
    static_user_potcar_functional: Optional[Any] = 'PBE_54'
    vasp_gd_kwargs: Optional[Dict[str, Any]] = None
    vasp_dipole_correction: Optional[bool] = None
    vasp_relax_settings: Optional[Dict[str, Any]] = None
    vasp_static_settings: Optional[Dict[str, Any]] = None
    do_mlip_gd: Optional[bool] = None
    do_vasp_gd: bool = False
    lowest_energy_pairs_settings: Optional[Dict[str, Any]] = None
    ckpt_path: Optional[str] = None
    mlip_calc: Optional[str] = None
    use_regex_numeric_fallback: bool = False

    def __post_init__(self) -> None:
        return None

@dataclass
class LocalBuildConfig(BaseBuildConfig):
    """Local execution configuration."""
    submit_target: Literal['local', 'server'] = 'local'

@dataclass
class ServerBuildConfig(BaseBuildConfig):
    """Login-node configuration: submit to jobflow-remote on the current host."""
    submit_target: Literal['local', 'server'] = 'server'
BuildConfig = BaseBuildConfig

def load_structures(film_cif: str, substrate_cif: str) -> Tuple[Structure, Structure, Dict[str, Any]]:
    """
    Load film and substrate from local CIF files on disk.

    Relative paths are resolved against the current working directory.
    Returns: (film_conv, substrate_conv, meta)
    """
    meta: Dict[str, Any] = {'source': 'local_cif'}
    film_cif_path = film_cif if os.path.isabs(film_cif) else os.path.join(os.getcwd(), film_cif)
    substrate_cif_path = substrate_cif if os.path.isabs(substrate_cif) else os.path.join(os.getcwd(), substrate_cif)
    if not os.path.isfile(film_cif_path):
        raise FileNotFoundError(f'Film CIF not found: {film_cif_path!r}. Provide paths via settings.bulk or BuildConfig.film_cif.')
    if not os.path.isfile(substrate_cif_path):
        raise FileNotFoundError(f'Substrate CIF not found: {substrate_cif_path!r}. Provide paths via settings.bulk or BuildConfig.substrate_cif.')
    film_conv = Structure.from_file(film_cif_path)
    substrate_conv = Structure.from_file(substrate_cif_path)
    meta['film_cif'] = os.path.abspath(film_cif_path)
    meta['substrate_cif'] = os.path.abspath(substrate_cif_path)
    return (film_conv, substrate_conv, meta)

def _sanitize_numeric_settings(settings: Dict[str, Any]) -> None:
    """Drop invalid step/fmax values from malformed configs."""
    opt = settings.get('optimization_settings')
    if not isinstance(opt, dict):
        return
    st = opt.get('steps')
    if isinstance(st, (int, float)) and float(st) <= 0:
        opt['steps'] = 200
    fm = opt.get('fmax')
    if isinstance(fm, (int, float)) and float(fm) <= 0:
        opt['fmax'] = 0.05
_IOMAKER_FULL_SETTINGS_REQUIRED_KEYS = ('name', 'mode', 'inputs', 'lattice_matching_settings', 'structure_settings', 'optimization_settings', 'global_minimization_settings')

def uses_full_llm_style_settings_dict(d: Optional[Dict[str, Any]]) -> bool:
    """
    True when *d* has every section of a complete IOMaker settings dict
    (same shape as ``normalize_iomaker_settings_from_full_dict`` expects), not a partial patch.
    """
    if not isinstance(d, dict):
        return False
    return all((k in d for k in _IOMAKER_FULL_SETTINGS_REQUIRED_KEYS))

def normalize_iomaker_settings_from_full_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate, copy, and lightly normalize a **complete** IOMaker settings dict.

    **No tutorial preset merge** — *raw* is the single source of truth for lattice /
    structure / optimization / global sections (use ``simple_iomaker`` cost_preset if you need presets).
    """
    missing = [k for k in _IOMAKER_FULL_SETTINGS_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"Full IOMaker settings dict missing keys: {missing}. "
            "Required top-level keys: name, mode, inputs, lattice_matching_settings, "
            "structure_settings, optimization_settings, global_minimization_settings."
        )
    settings: Dict[str, Any] = deepcopy(raw)
    if not isinstance(settings.get('inputs'), dict):
        settings['inputs'] = {}
    settings['do_vasp'] = bool(settings.get('do_vasp', False))
    _bulk = settings.get('bulk')
    if not isinstance(_bulk, dict):
        _bulk = {}
    settings['bulk'] = {'film_cif': str(_bulk.get('film_cif') or DEFAULT_FILM_CIF), 'substrate_cif': str(_bulk.get('substrate_cif') or DEFAULT_SUBSTRATE_CIF)}
    _sanitize_numeric_settings(settings)
    _inp = settings.get('inputs')
    if isinstance(_inp, dict):
        _inp['type'] = 'local_cif'
        _inp.pop('film_mp_id', None)
        _inp.pop('substrate_mp_id', None)
        settings['inputs'] = _inp
    return settings

def _clean_for_json_serialization(obj: Any, visited: Optional[set]=None) -> Any:
    """
    Recursively remove non-serializable objects (like functions) from a dictionary.
    This is needed because jobflow Flow objects may contain function references
    that cannot be serialized to JSON.
    """
    if visited is None:
        visited = set()
    if isinstance(obj, (dict, list, tuple)):
        obj_id = id(obj)
        if obj_id in visited:
            return None
        visited.add(obj_id)
    try:
        if hasattr(obj, 'tolist') and (not isinstance(obj, (str, bytes, bytearray))):
            try:
                return _clean_for_json_serialization(obj.tolist(), visited)
            except Exception:
                pass
        if hasattr(obj, 'item') and (not isinstance(obj, (str, bytes, bytearray))):
            try:
                return obj.item()
            except Exception:
                pass
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if callable(value) and (not isinstance(value, type)):
                    continue
                try:
                    cleaned[key] = _clean_for_json_serialization(value, visited)
                except (TypeError, ValueError, RecursionError):
                    continue
            if isinstance(obj, dict):
                visited.discard(id(obj))
            return cleaned
        elif isinstance(obj, (list, tuple)):
            cleaned = []
            for item in obj:
                if callable(item) and (not isinstance(item, type)):
                    continue
                try:
                    cleaned.append(_clean_for_json_serialization(item, visited))
                except (TypeError, ValueError, RecursionError):
                    continue
            if isinstance(obj, (list, tuple)):
                visited.discard(id(obj))
            return cleaned if isinstance(obj, list) else tuple(cleaned)
        else:
            if callable(obj) and (not isinstance(obj, type)):
                return None
            return obj
    except (TypeError, ValueError, RecursionError):
        return None

def _print_final_settings(settings: Dict[str, Any], structures_meta: Dict[str, Any], flow_json_path: str, cfg: Optional[BaseBuildConfig]=None) -> None:
    """
    Print all settings for user visibility after building a job.
    """
    print('\n' + '=' * 80)
    print('✅ InterOptimus IOMaker job generated')
    print(f'📄 Flow JSON: {flow_json_path}')
    print(f"📦 Structures source: {structures_meta.get('source')}")
    if structures_meta.get('film_cif') and structures_meta.get('substrate_cif'):
        print(f"🧱 film.cif: {structures_meta.get('film_cif')}")
        print(f"🧱 substrate.cif: {structures_meta.get('substrate_cif')}")
    if cfg:
        print('\n⚙️  Jobflow configuration:')
        print(f'   MLIP worker: {cfg.mlip_worker}')
        print(f'   MLIP project: {cfg.mlip_project}')
        print(f"   MLIP resources: {('✅ Set' if cfg.mlip_resources else '❌ Not set (using defaults)')}")
        if settings.get('do_vasp'):
            print(f'   VASP worker: {cfg.vasp_worker}')
            print(f"   VASP resources: {('✅ Set' if cfg.vasp_resources else '❌ Not set')}")
            print(f"   VASP GD: {('✅ On' if settings.get('do_vasp_gd') else '❌ Off')}")
            vex = getattr(cfg, 'vasp_exec_config', None) or settings.get('vasp_exec_config')
            if vex:
                print(f'   VASP exec_config (pre_run, …): {vex}')
            else:
                print('   VASP exec_config: (default / INTEROPTIMUS_VASP_PRE_RUN)')
            print('\n🧪 VASP input (InterfaceWorker.patch_jobflow_jobs / itworker-style):')
            try:
                from InterOptimus.jobflow import iomaker_resolve_patch_jobflow_vasp_kwargs
                _vk = iomaker_resolve_patch_jobflow_vasp_kwargs(SimpleNamespace(relax_user_incar_settings=settings.get('relax_user_incar_settings'), relax_user_potcar_settings=settings.get('relax_user_potcar_settings'), relax_user_kpoints_settings=settings.get('relax_user_kpoints_settings'), relax_user_potcar_functional=settings.get('relax_user_potcar_functional'), static_user_incar_settings=settings.get('static_user_incar_settings'), static_user_potcar_settings=settings.get('static_user_potcar_settings'), static_user_kpoints_settings=settings.get('static_user_kpoints_settings'), static_user_potcar_functional=settings.get('static_user_potcar_functional'), vasp_gd_kwargs=settings.get('vasp_gd_kwargs'), vasp_dipole_correction=settings.get('vasp_dipole_correction', False), vasp_relax_settings=settings.get('vasp_relax_settings'), vasp_static_settings=settings.get('vasp_static_settings')))
                print(json.dumps(_vk, indent=2, ensure_ascii=False))
            except Exception:
                print('   (could not resolve VASP kwargs preview)')
        if cfg.submit_target == 'server':
            print('\n🖥️  Login-node (jobflow-remote) submission:')
            print(f"   Run parent dir: {cfg.server_run_parent or '(cwd)'}")
            print(f"   Pre-command: {cfg.server_pre_cmd or '(none)'}")
            print(f'   Python: {cfg.server_python}')
            print(f'   jf binary: {cfg.server_jf_bin}')
    structure_settings = settings.get('structure_settings') or {}
    optimization_settings = settings.get('optimization_settings') or {}
    if isinstance(structure_settings, dict) and isinstance(optimization_settings, dict) and (not bool(structure_settings.get('double_interface', False))):
        print('\n🧩 Single-interface relax controls:')
        print(f"   set_relax_thicknesses: {optimization_settings.get('set_relax_thicknesses')}")
        print(f"   relax_in_layers: {optimization_settings.get('relax_in_layers')}")
        print(f"   relax_in_ratio: {optimization_settings.get('relax_in_ratio')}")
    print("\n🔧 Parameter settings (normalized):")
    print(json.dumps(_clean_for_json_serialization(settings), indent=2, ensure_ascii=False))
    print('=' * 80 + '\n')

def execute_iomaker_pipeline(settings: Dict[str, Any], cfg: BaseBuildConfig, film_conv: Structure, substrate_conv: Structure, meta: Dict[str, Any], structures_info: Dict[str, Any], *, user_prompt: str='') -> Dict[str, Any]:
    """
    Build IOMaker flow JSON, then run locally or submit on the login node per ``cfg.submit_target``.

    Used by :mod:`InterOptimus.agents.simple_iomaker` (full ``settings`` plus ``execution`` / ``cluster``).
    """
    do_vasp = bool(settings.get('do_vasp', False))
    if cfg.submit_target == 'local' and do_vasp:
        do_vasp = False
        settings['do_vasp'] = False
    if do_vasp:
        if cfg.vasp_resources is None:
            raise ValueError('vasp_resources is required when do_vasp=True. Please provide a Callable that returns QResources for VASP jobs.')
        if not cfg.vasp_worker:
            raise ValueError('vasp_worker is required when do_vasp=True. Please provide the worker name for VASP jobs.')
    if cfg.submit_target == 'server':
        missing = []
        if do_vasp and cfg.vasp_resources is None:
            missing.append('vasp_resources')
        if cfg.mlip_resources is None:
            missing.append('mlip_resources')
        if do_vasp and (not cfg.vasp_worker):
            missing.append('vasp_worker')
        if not cfg.mlip_worker:
            missing.append('mlip_worker')
        if missing:
            raise ValueError("submit_target='server' requires: " + ', '.join(missing))

    def _cfg_or_settings(key: str) -> Any:
        if hasattr(cfg, key) and getattr(cfg, key) is not None:
            return getattr(cfg, key)
        return settings.get(key)

    def _resolve_vasp_dipole() -> bool:
        c = getattr(cfg, 'vasp_dipole_correction', None)
        if c is not None:
            return bool(c)
        return bool(settings.get('vasp_dipole_correction', False))
    vasp_relax_settings = _cfg_or_settings('vasp_relax_settings')
    vasp_static_settings = _cfg_or_settings('vasp_static_settings')
    vasp_gd_kwargs_merged = _cfg_or_settings('vasp_gd_kwargs')
    vasp_dipole_merged = _resolve_vasp_dipole()
    do_vasp_gd = bool(cfg.do_vasp_gd)
    if _disable_gd_from_prompt(user_prompt):
        do_vasp_gd = False
    if do_vasp:
        settings['vasp_relax_settings'] = vasp_relax_settings
        settings['vasp_static_settings'] = vasp_static_settings
        settings['vasp_gd_kwargs'] = vasp_gd_kwargs_merged
        settings['vasp_dipole_correction'] = vasp_dipole_merged
        for _vk in ('relax_user_incar_settings', 'relax_user_potcar_settings', 'relax_user_kpoints_settings', 'relax_user_potcar_functional', 'static_user_incar_settings', 'static_user_potcar_settings', 'static_user_kpoints_settings', 'static_user_potcar_functional'):
            settings[_vk] = _cfg_or_settings(_vk)
        settings['do_vasp_gd'] = do_vasp_gd
    vasp_exec_config = getattr(cfg, 'vasp_exec_config', None)
    if vasp_exec_config is None and isinstance(settings.get('vasp_exec_config'), dict):
        vasp_exec_config = settings.get('vasp_exec_config')
    if vasp_exec_config is None and settings.get('vasp_pre_run'):
        vasp_exec_config = {'pre_run': str(settings['vasp_pre_run'])}
    if do_vasp and vasp_exec_config is not None:
        settings['vasp_exec_config'] = vasp_exec_config
    if not settings.get('name') or settings.get('name') == 'IO_llm':
        film_label = getattr(film_conv.composition, 'reduced_formula', None) or 'film'
        substrate_label = getattr(substrate_conv.composition, 'reduced_formula', None) or 'substrate'
        settings['name'] = _auto_iomaker_name(settings=settings, film_label=film_label, substrate_label=substrate_label)
    _slug = _local_run_workdir(settings.get('name', 'IO_llm'))
    if cfg.submit_target == 'server' and cfg.server_run_parent:
        local_workdir = os.path.join(cfg.server_run_parent, _slug)
    else:
        local_workdir = _slug
    local_workdir_abs = os.path.abspath(local_workdir)

    def _resources_or_callable(res: Any) -> Any:
        if res is None:
            return None
        return res() if callable(res) else res
    mlip_resolved = _resources_or_callable(cfg.mlip_resources)
    vasp_resolved = _resources_or_callable(cfg.vasp_resources)
    maker = IOMaker(name=settings.get('name', 'IO_llm'), lattice_matching_settings=settings['lattice_matching_settings'], structure_settings=settings['structure_settings'], optimization_settings=settings['optimization_settings'], global_minimization_settings=settings['global_minimization_settings'], do_vasp=do_vasp, do_vasp_gd=do_vasp_gd, relax_user_incar_settings=_cfg_or_settings('relax_user_incar_settings'), relax_user_potcar_settings=_cfg_or_settings('relax_user_potcar_settings'), relax_user_kpoints_settings=_cfg_or_settings('relax_user_kpoints_settings'), relax_user_potcar_functional=_cfg_or_settings('relax_user_potcar_functional'), static_user_incar_settings=_cfg_or_settings('static_user_incar_settings'), static_user_potcar_settings=_cfg_or_settings('static_user_potcar_settings'), static_user_kpoints_settings=_cfg_or_settings('static_user_kpoints_settings'), static_user_potcar_functional=_cfg_or_settings('static_user_potcar_functional'), vasp_gd_kwargs=vasp_gd_kwargs_merged, vasp_dipole_correction=vasp_dipole_merged, vasp_relax_settings=vasp_relax_settings, vasp_static_settings=vasp_static_settings, lowest_energy_pairs_settings=cfg.lowest_energy_pairs_settings, pairs_output_dir=None, mlip_resources=mlip_resolved, vasp_resources=vasp_resolved, mlip_worker=cfg.mlip_worker, vasp_worker=cfg.vasp_worker, vasp_exec_config=vasp_exec_config)
    flow = maker.make(film_conv, substrate_conv)
    os.makedirs(local_workdir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(local_workdir, cfg.output_flow_json))
    flow_dict = None
    flow_json_str = None
    try:
        if hasattr(flow, 'to_json'):
            flow_json_str = flow.to_json()
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(flow_json_str)
            flow_dict = json.loads(flow_json_str)
    except (TypeError, AttributeError) as e:
        print(f'Warning: to_json() failed ({e}), attempting to clean and serialize manually...')
        try:
            flow_dict = flow.as_dict() if hasattr(flow, 'as_dict') else flow.to_dict()
            flow_dict = _clean_for_json_serialization(flow_dict)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(flow_dict, f, indent=2)
            flow_json_str = json.dumps(flow_dict, indent=2)
        except Exception as e2:
            raise RuntimeError(f'Failed to serialize Flow to JSON. to_json() error: {e}, manual serialization error: {e2}. This may be due to non-serializable objects (like functions) in the Flow. Please check that mlip_resources and vasp_resources are not stored in the Flow.') from e2
    if flow_dict is None:
        raise RuntimeError('Failed to serialize Flow to JSON')
    if cfg.print_settings:
        _print_final_settings(settings=settings, structures_meta=meta, flow_json_path=out_path, cfg=cfg)
    submit_dir = os.path.abspath(os.path.dirname(out_path))
    result = {'flow_json_path': out_path, 'flow_dict': flow_dict, 'settings': settings, 'structures_meta': meta, 'structures_info': structures_info, 'submit_workdir': submit_dir, 'interoptimus_task_json_path': os.path.join(submit_dir, 'io_interoptimus_task.json')}
    if cfg.submit_target == 'local':
        run_dir = local_workdir or _local_run_workdir(settings.get('name', 'IO_llm'))
        _run_flow_locally(flow, run_dir)
        result['local_workdir'] = os.path.abspath(run_dir)
        result['pairs_summary_path'] = os.path.abspath(os.path.join(run_dir, 'pairs_summary.txt'))
        result['opt_results_pkl'] = os.path.abspath(os.path.join(run_dir, 'opt_results.pkl'))
    if cfg.submit_target == 'server' and local_workdir_abs:
        result['local_workdir'] = local_workdir_abs
    if cfg.submit_target == 'server':
        try:
            from .remote_submit import qresources_to_plain_dict
            submit_result = submit_io_flow_locally(out_path, workdir=local_workdir_abs, remote_python=cfg.server_python, pre_cmd=cfg.server_pre_cmd, submit_flow_worker=cfg.mlip_worker, submit_flow_project=cfg.mlip_project, submit_flow_resources_kwargs=qresources_to_plain_dict(cfg.mlip_resources()) if cfg.mlip_resources is not None else None)
            result['server_submission'] = submit_result
            if submit_result.get('mlip_job_uuid'):
                result['mlip_job_uuid'] = submit_result['mlip_job_uuid']
            if submit_result.get('vasp_job_uuids') is not None:
                result['vasp_job_uuids'] = list(submit_result.get('vasp_job_uuids') or [])
            if submit_result.get('flow_uuid'):
                result['flow_uuid'] = submit_result['flow_uuid']
            if submit_result.get('success'):
                print(f'\n✅ Submitted to jobflow-remote on this host (login-node mode)')
                if submit_result.get('submit_workdir'):
                    print(f"   Submit cwd: {submit_result['submit_workdir']}")
                if submit_result.get('job_id'):
                    print(f"📋 submit_flow job id: {submit_result['job_id']}")
                if submit_result.get('mlip_job_uuid'):
                    print(f"📌 MLIP job UUID（query_interoptimus_task_progress 优先跟踪）: {submit_result['mlip_job_uuid']}")
                if submit_result.get('submit_id_note'):
                    print(f"   ℹ️  {submit_result['submit_id_note']}")
                if submit_result.get('vasp_job_uuids'):
                    print(f"📎 Planned VASP job UUID(s): {submit_result['vasp_job_uuids']}")
                reg_jf_id = str(submit_result.get('job_id') or submit_result.get('mlip_job_uuid') or submit_result.get('flow_uuid') or '').strip()
                if reg_jf_id:
                    try:
                        wd_reg = submit_result.get('submit_workdir') or local_workdir_abs or ''
                        rec = register_interoptimus_server_task(task_name=str(settings.get('name', 'IO_llm')), jf_job_id=reg_jf_id, submit_workdir=str(wd_reg), do_vasp=bool(settings.get('do_vasp')), mlip_job_uuid=submit_result.get('mlip_job_uuid'), vasp_job_uuids=submit_result.get('vasp_job_uuids'), flow_uuid=submit_result.get('flow_uuid'))
                        result['interoptimus_task_serial'] = rec['serial_id']
                        result['interoptimus_task_record'] = rec
                        _mlip = (rec.get('mlip_job_uuid') or '').strip() or str(submit_result.get('mlip_job_uuid') or '').strip()
                        print(f"\n🔖 Registry serial (内部用）: {rec['serial_id']}")
                        if _mlip:
                            print(f'   查进度 / 拉结果请用 **job UUID**：iomaker_progress(result) 或 query_interoptimus_task_progress({_mlip!r})')
                            print(f'   内核重启: export INTEROPTIMUS_JOB_ID={_mlip!r} 或 iomaker_progress(ref={_mlip!r})')
                            print(f'   fetch: fetch_interoptimus_task_results({_mlip!r}, copy_images_to=..., ...)')
                        else:
                            print('   查进度: iomaker_progress(result) 或 query_interoptimus_task_progress(<submit_flow job id UUID>)')
                    except Exception as _reg_err:
                        result['interoptimus_task_register_error'] = str(_reg_err)
                        try:
                            persist_iomaker_task_index_fallback(task_name=str(settings.get('name', 'IO_llm')), jf_job_id=reg_jf_id, submit_workdir=str(submit_result.get('submit_workdir') or local_workdir_abs or ''), do_vasp=bool(settings.get('do_vasp')), mlip_job_uuid=submit_result.get('mlip_job_uuid'), vasp_job_uuids=submit_result.get('vasp_job_uuids'), flow_uuid=submit_result.get('flow_uuid'))
                        except Exception:
                            pass
            else:
                print(f'\n❌ Login-node submit_flow failed:')
                print(f"   Error: {submit_result.get('error', 'unknown')}")
                print(f"   Stderr: {submit_result.get('stderr', '')}")
        except Exception as e:
            print(f'\n⚠️  Server-side submission failed: {e}')
            result['server_submission_error'] = str(e)
    report_path = os.path.join(os.path.dirname(out_path), 'io_report.txt')
    _write_run_report(report_path, settings=settings, structures_meta=meta, result=result)
    result['report_path'] = report_path
    return result

def execute_iomaker_from_settings(settings: Dict[str, Any], cfg: BaseBuildConfig, user_prompt: str='') -> Dict[str, Any]:
    """
    Load structures from ``cfg`` / ``settings["inputs"]``, then run :func:`execute_iomaker_pipeline`.

    CIF paths: ``settings["bulk"]["film_cif"]`` / ``settings["bulk"]["substrate_cif"]`` (filled by
    :func:`normalize_iomaker_settings_from_full_dict`) override ``cfg.film_cif`` / ``cfg.substrate_cif``.
    """
    bulk = settings.get('bulk') if isinstance(settings.get('bulk'), dict) else {}
    if bulk:
        cfg = replace(cfg, film_cif=str(bulk.get('film_cif') or cfg.film_cif), substrate_cif=str(bulk.get('substrate_cif') or cfg.substrate_cif))
    film_conv, substrate_conv, meta = load_structures(cfg.film_cif, cfg.substrate_cif)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        film_sg = SpacegroupAnalyzer(film_conv).get_space_group_symbol()
        film_sg_num = SpacegroupAnalyzer(film_conv).get_space_group_number()
        sub_sg = SpacegroupAnalyzer(substrate_conv).get_space_group_symbol()
        sub_sg_num = SpacegroupAnalyzer(substrate_conv).get_space_group_number()
        structures_info = {'film': {'reduced_formula': film_conv.composition.reduced_formula, 'pretty_formula': film_conv.composition.formula, 'spacegroup_symbol': film_sg, 'spacegroup_number': film_sg_num}, 'substrate': {'reduced_formula': substrate_conv.composition.reduced_formula, 'pretty_formula': substrate_conv.composition.formula, 'spacegroup_symbol': sub_sg, 'spacegroup_number': sub_sg_num}}
    except Exception:
        structures_info = {}
    return execute_iomaker_pipeline(settings, cfg, film_conv, substrate_conv, meta, structures_info, user_prompt=user_prompt)
if __name__ == '__main__':
    main()
