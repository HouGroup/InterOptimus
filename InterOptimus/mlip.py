"""
Machine Learning Interatomic Potential (MLIP) Interface Module

This module provides interfaces to various MLIP calculators including
ORB, SevenNet, MatRIS, and DPA models, along with ASE optimizer configurations.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from ase.filters import UnitCellFilter
from ase.constraints import FixAtoms
from pymatgen.core.structure import Structure


def default_mlip_checkpoint_dir() -> Path:
    """
    Directory where InterOptimus stores and resolves MLIP checkpoint files by default.

    Default: ``~/.cache/InterOptimus/checkpoints`` (e.g. ``/home/<user>/.cache/InterOptimus/checkpoints``).

    Override with environment variable ``INTEROPTIMUS_CHECKPOINT_DIR`` (absolute path recommended).
    """
    env = (os.environ.get("INTEROPTIMUS_CHECKPOINT_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "InterOptimus" / "checkpoints"


def ensure_mlip_checkpoint_dir() -> Path:
    """Create the default checkpoint directory if it does not exist."""
    d = default_mlip_checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_glob_patterns(calc: str) -> list[str]:
    if calc == "orb-models":
        return [
            "orb-v3-conservative-inf-omat-20250404.ckpt",
            "orb-v3-conservative-20-omat-20250404.ckpt",
            "orb-v3-*.ckpt",
            "orb-*.ckpt",
        ]
    if calc == "sevenn":
        return [
            "checkpoint_sevennet_mf_ompa.pth",
            "checkpoint_sevennet*.pth",
            "*sevennet*.pth",
            "*7net*.pth",
        ]
    if calc == "dpa":
        return [
            "dpa*.pth",
            "dpa*.pt",
            "dpa*.pb",
            "*deepmd*.pth",
            "*deepmd*.pb",
        ]
    return []


def _first_matching_checkpoint_in_dir(calc: str, directory: Path) -> str | None:
    """Newest matching checkpoint file under ``directory`` (non-recursive), or None."""
    if not directory.is_dir():
        return None
    seen: set[str] = set()
    for pattern in _checkpoint_glob_patterns(calc):
        for path in sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            resolved = str(path.resolve())
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            return resolved
    return None


def resolve_mlip_checkpoint(calc: str) -> str | None:
    """
    Pick the newest matching checkpoint file under :func:`default_mlip_checkpoint_dir`.

    Returns None if the directory is missing or no file matches.
    """
    ensure_mlip_checkpoint_dir()
    directory = default_mlip_checkpoint_dir()
    return _first_matching_checkpoint_in_dir(calc, directory)


def _normalize_ckpt_path(calc: str, user_settings: dict) -> None:
    if "ckpt_path" not in user_settings:
        user_settings["ckpt_path"] = None
        return
    cp = user_settings["ckpt_path"]
    if cp in ("", None):
        user_settings["ckpt_path"] = None
        return
    if isinstance(cp, str) and os.path.isdir(cp):
        resolved = _first_matching_checkpoint_in_dir(calc, Path(cp).expanduser().resolve())
        user_settings["ckpt_path"] = resolved
        return


def _apply_checkpoint_defaults(calc: str, user_settings: dict) -> None:
    """Fill ckpt_path from ~/.cache/InterOptimus/checkpoints when not set explicitly."""
    _normalize_ckpt_path(calc, user_settings)
    if user_settings.get("ckpt_path") is None and calc in (
        "orb-models",
        "sevenn",
        "dpa",
    ):
        ckpt = resolve_mlip_checkpoint(calc)
        if ckpt:
            user_settings["ckpt_path"] = ckpt


def _orb_conservative_omat_loader_for_checkpoint_path(ckpt_path: str):
    """
    ORB v3 conservative OMAT has two public checkpoints (different graph sizes).

    Pick the ``pretrained.orb_v3_conservative_*_omat`` loader that matches the filename
    so ``load_state_dict(strict=True)`` succeeds. Default / inf naming uses the
    ``conservative-inf-omat`` architecture.
    """
    from orb_models.forcefield import pretrained

    base = os.path.basename(ckpt_path).lower()
    if "conservative-20-omat" in base:
        return pretrained.orb_v3_conservative_20_omat
    return pretrained.orb_v3_conservative_inf_omat


def get_optimizer(optimizer):
    """
    Get ASE optimizer class by name.

    Args:
        optimizer (str): Name of the optimizer

    Returns:
        ASE optimizer class

    Supported optimizers:
    - BFGS, LBFGS, LBFGSLineSearch, GPMin, FIRE, MDMin
    - SciPyFminBFGS, SciPyFminCG, BasinHopping, MinimaHopping
    """
    if optimizer == 'BFGS':
        from ase.optimize import BFGS
        return BFGS
    elif optimizer == 'LBFGS':
        from ase.optimize import LBFGS
        return LBFGS
    elif optimizer == 'LBFGSLineSearch':
        from ase.optimize import LBFGSLineSearch
        return LBFGSLineSearch
    elif optimizer == 'GPMin':
        from ase.optimize import GPMin
        return GPMin
    elif optimizer == 'FIRE':
        from ase.optimize import FIRE
        return FIRE
    elif optimizer == 'MDMin':
        from ase.optimize import MDMin
        return MDMin
    elif optimizer == 'SciPyFminBFGS':
        from ase.optimize.sciopt import SciPyFminBFGS
        return SciPyFminBFGS
    elif optimizer == 'SciPyFminCG':
        from ase.optimize.sciopt import SciPyFminCG
        return SciPyFminCG
    elif optimizer == 'BasinHopping':
        from ase.optimize.basin import BasinHopping
        return BasinHopping
    elif optimizer == 'MinimaHopping':
        from ase.optimize.minimahopping import MinimaHopping
        return MinimaHopping

class MlipCalc:
    """
    Machine Learning Interatomic Potential Calculator Interface.

    Provides a unified interface for different MLIP calculators including
    ORB models, SevenNet, MatRIS, and Deep Potential models.

    Args:
        calc (str): Calculator type ('orb-models', 'sevenn', 'matris', 'dpa')
        user_settings (dict): Calculator-specific settings
            - device (str): Device for computation ('cpu' or 'cuda')
            - ckpt_path (str): Path to model checkpoint (optional)
            - model (str): Model identifier for calculators that expose named models
            - task (str): Property bundle for calculators that support task selection
    """

    def __init__(self, calc, user_settings=None):
        """
        Initialize MLIP calculator.

        Args:
            calc (str): Type of calculator to initialize
            user_settings (dict): Configuration settings for the calculator
        """
        user_settings = dict(user_settings or {})
        user_settings.setdefault('device', 'cpu')
        _apply_checkpoint_defaults(calc, user_settings)
        self.calc_type = calc
        self.user_settings = user_settings

        if calc == 'orb-models':
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            device = user_settings['device']
            print(f"Initializing ORB calculator on device: {device}")
            ckpt = user_settings.get('ckpt_path')

            # orb_models loads weights via cached_path; default weights_path is an HTTPS URL to S3.
            # If ckpt is set, failure must not silently fall back to that URL (common on air-gapped nodes).
            if ckpt:
                try:
                    orb_loader = _orb_conservative_omat_loader_for_checkpoint_path(ckpt)
                    orbff = orb_loader(
                        weights_path=ckpt,
                        device=device,
                        precision="float32-high",
                    )
                    self.calc = ORBCalculator(orbff, device=device)
                    print(
                        "ORB initialization success (checkpoint from "
                        "~/.cache/InterOptimus/checkpoints, INTEROPTIMUS_CHECKPOINT_DIR, or explicit path)"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ORB weights from ckpt_path={ckpt!r}. "
                        "Use a path to a .ckpt file that exists on this machine (including compute nodes), "
                        "or set INTEROPTIMUS_CHECKPOINT_DIR and place orb-v3-*.ckpt there. "
                        "Default ORB weights are downloaded from the network and were not used after this error."
                    ) from e
            else:
                try:
                    self.calc = ORBCalculator(
                        pretrained.orb_v3_conservative_inf_omat(device=device),
                        device=device
                    )
                    print(
                        "ORB initialization success (ORB v3 conservative inf OMAT default; requires network unless cached)"
                    )
                except Exception as e:
                    raise RuntimeError(
                        "ORB default weights load failed (no ckpt_path). orb_models uses a remote URL by default; "
                        "place a matching orb-v3 checkpoint under INTEROPTIMUS_CHECKPOINT_DIR / "
                        "~/.cache/InterOptimus/checkpoints or pass optimization_settings.ckpt_path."
                    ) from e

        elif calc == 'sevenn':
            from sevenn.calculator import SevenNetCalculator
            from pathlib import PurePath

            ckpt = user_settings.get('ckpt_path')
            if ckpt:
                try:
                    self.calc = SevenNetCalculator(PurePath(ckpt), modal='mpa')
                    print("SevenNet initialized with checkpoint from ~/.cache/InterOptimus/checkpoints or explicit path")
                except Exception:
                    self.calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
                    print("SevenNet initialized with default model (checkpoint load failed)")
            else:
                self.calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
                print("SevenNet initialized with default model (no checkpoint in cache)")

        elif calc == 'matris':
            from matris.applications.base import MatRISCalculator

            model = user_settings.get('model', 'matris_10m_oam')
            task = user_settings.get('task', 'efsm')
            device = user_settings['device']
            self.calc = MatRISCalculator(model=model, task=task, device=device)
            print(f"MatRIS initialized with model={model}, task={task}, device={device}")

        elif calc == 'dpa':
            from deepmd.calculator import DP
            from pathlib import PurePath

            ckpt = user_settings.get('ckpt_path')
            if not ckpt:
                cache = default_mlip_checkpoint_dir()
                raise ValueError(
                    f"No DPA checkpoint found. Place a model file in {cache} "
                    "or pass ckpt_path."
                )
            try:
                self.calc = DP(model=PurePath(ckpt))
                print("Deep Potential initialized with custom model")
            except Exception as e:
                raise ValueError(f'Invalid path for DPA checkpoint file: {e}')

        else:
            raise ValueError(f"Unsupported calculator type: {calc}")

    def calculate(self, structure):
        """
        Calculate potential energy of a structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            float: Potential energy in eV
        """
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calc
        return atoms.get_potential_energy()

    def optimize(self, structure, optimizer='BFGS', **kwargs):
        """
        Optimize structure using MLIP calculator.

        Args:
            structure: pymatgen Structure object to optimize
            optimizer (str): ASE optimizer name (default: 'BFGS')
            **kwargs: Optimization parameters including:
                - fmax (float): Maximum force threshold
                - steps (int): Maximum number of optimization steps
                - fix_cell_booleans (list): Cell constraint flags
                - viz_meta (dict, optional): If set and :func:`InterOptimus.viz_runtime.is_enabled`, emit per-step events.

        Returns:
            tuple: (optimized_structure, final_energy)
        """
        optimizer_class = get_optimizer(optimizer)
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calc

        kw = dict(kwargs)
        viz_meta = kw.pop("viz_meta", None)
        fmax = float(kw.get("fmax", 0.05))
        max_steps = int(kw.get("steps", 200))
        fix_b = kw["fix_cell_booleans"]

        ft = UnitCellFilter(atoms, fix_b)
        relax = optimizer_class(ft, logfile=None)

        from InterOptimus.viz_runtime import emit_event, is_enabled

        def _interface_gamma_j_m2(E_sup: float, meta: dict) -> Optional[float]:
            try:
                A = float(meta.get("match_area_A2") or 0.0)
                if A <= 0:
                    return None
                efr = float(meta.get("E_film_ref", 0.0))
                esr = float(meta.get("E_sub_ref", 0.0))
                di = bool(meta.get("double_interface", True))
                g = (float(E_sup) - efr - esr) / A * 16.02176634
                if di:
                    g /= 2.0
                return float(g)
            except Exception:
                return None

        if is_enabled() and isinstance(viz_meta, dict):
            pos0 = atoms.get_positions().copy()
            nat = int(len(atoms))
            for step in range(max_steps):
                relax.run(fmax=fmax, steps=1)
                E = float(atoms.get_potential_energy())
                pos = atoms.get_positions()
                rms = float(np.sqrt(np.mean((pos - pos0) ** 2)))
                payload = {
                    **viz_meta,
                    "event": "relax_step",
                    "step": int(step),
                    "energy": E,
                    "rms_displacement": rms,
                    "n_atoms": nat,
                    "energy_per_atom": float(E) / float(max(nat, 1)),
                }
                ig = _interface_gamma_j_m2(E, viz_meta)
                if ig is not None:
                    payload["interface_gamma_J_m2"] = ig
                emit_event(payload)
                conv = getattr(relax, "converged", None)
                if callable(conv):
                    try:
                        if conv():
                            break
                    except Exception:
                        pass
            flat = atoms.get_positions().flatten()
            nmax = min(len(flat), 4000 * 3)
            nfin = int(len(atoms))
            efin = float(atoms.get_potential_energy())
            fin = {
                **viz_meta,
                "event": "relax_final",
                "energy": efin,
                "energy_per_atom": float(efin) / float(max(nfin, 1)),
                "rms_displacement": float(np.sqrt(np.mean((atoms.get_positions() - pos0) ** 2))),
                "positions": flat[:nmax].tolist(),
                "n_atoms": nfin,
                "numbers": atoms.get_atomic_numbers().tolist(),
                "cell": atoms.get_cell().tolist(),
            }
            igf = _interface_gamma_j_m2(efin, viz_meta)
            if igf is not None:
                fin["interface_gamma_J_m2"] = igf
            emit_event(fin)
        else:
            relax.run(fmax=fmax, steps=max_steps)

        return Structure.from_ase_atoms(atoms), atoms.get_potential_energy()

    def close(self):
        """
        Clean up calculator resources if needed.
        """
        pass
