"""
Machine Learning Interatomic Potential (MLIP) Interface Module

This build exposes only the Eqnorm calculator, plus ASE optimizer configuration helpers.
"""

import os
import shutil
import sys
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


def _sync_eqnorm_ckpt_to_package_cache(user_settings: dict) -> None:
    """
    ``eqnorm.calculator.EqnormCalculator`` loads only from ``~/.cache/{model_name}/{model_variant}.pt``.
    Copy InterOptimus-resolved ``ckpt_path`` there when set so local / air-gapped runs work.
    """
    model_name = str(user_settings.get("model_name") or "eqnorm")
    model_variant = str(user_settings.get("model_variant") or "eqnorm-mptrj")
    dest_dir = Path.home() / ".cache" / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{model_variant}.pt"
    ckpt = user_settings.get("ckpt_path")
    if isinstance(ckpt, str) and ckpt.strip() and os.path.isfile(ckpt):
        shutil.copy2(os.path.expanduser(ckpt), dest)


def _patch_torch_jit_for_frozen_bundle() -> None:
    """
    PyInstaller bundles Python as bytecode in an archive; ``inspect.getsource`` fails.
    DeepMD uses ``@torch.jit.script`` on helpers; without sources TorchScript errors.
    Fall back to the uncompiled function when compilation cannot read source (frozen app).
    """
    if not getattr(sys, "frozen", False):
        return
    import torch

    if getattr(torch.jit, "_interoptimus_script_patch", False):
        return
    _orig = torch.jit.script

    def _script(fn: object) -> object:
        try:
            return _orig(fn)
        except OSError:
            # e.g. torch_geometric + PyInstaller: "Can't get source for <class ...>"
            return fn
        except Exception as e:
            msg = str(e).lower()
            if any(
                s in msg
                for s in (
                    "source",
                    "torchscript",
                    "getsourcelines",
                    "could not get source",
                    "can't get source",
                )
            ):
                return fn
            raise

    torch.jit.script = _script  # type: ignore[assignment]
    setattr(torch.jit, "_interoptimus_script_patch", True)


def _checkpoint_glob_patterns(calc: str) -> list[str]:
    if calc == "eqnorm":
        return [
            "eqnorm*.pt",
            "eqnorm*.pth",
            "*eqnorm*.pt",
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
    if user_settings.get("ckpt_path") is None and calc == "eqnorm":
        ckpt = resolve_mlip_checkpoint(calc)
        if ckpt:
            user_settings["ckpt_path"] = ckpt


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
    Eqnorm-only MLIP calculator wrapper for InterfaceWorker.

    Args:
        calc: Must be ``'eqnorm'`` in this build.
        user_settings: ``device``, optional ``ckpt_path``, ``model_name``, ``model_variant``, ``compile``.
    """

    def __init__(self, calc, user_settings=None):
        user_settings = dict(user_settings or {})
        user_settings.setdefault('device', 'cpu')
        # torch_geometric / Eqnorm may hit TorchScript edge cases without source in frozen apps.
        _patch_torch_jit_for_frozen_bundle()
        _apply_checkpoint_defaults(calc, user_settings)
        self.calc_type = calc
        self.user_settings = user_settings

        if calc != "eqnorm":
            raise ValueError(
                "This InterOptimus build supports only the Eqnorm MLIP backend. "
                f"Use calc='eqnorm', not {calc!r}."
            )
        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        from eqnorm.calculator import EqnormCalculator

        model_name = str(user_settings.get("model_name") or "eqnorm")
        model_variant = str(user_settings.get("model_variant") or "eqnorm-mptrj")
        device = str(user_settings.get("device") or "cpu")
        if device in ("gpu",):
            device = "cuda"
        compile_m = bool(user_settings.get("compile", False))
        _sync_eqnorm_ckpt_to_package_cache(user_settings)
        try:
            self.calc = EqnormCalculator(
                model_name=model_name,
                model_variant=model_variant,
                device=device,
                compile=compile_m,
            )
            print(
                f"Eqnorm initialized (model_name={model_name}, model_variant={model_variant}, device={device})"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize EqnormCalculator: {e}") from e

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
        # Eqnorm uses torch.autograd.grad internally for energy/forces; inference_mode breaks that.
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

        # Eqnorm needs autograd for forces during relaxation; do not use torch.inference_mode() here.

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
