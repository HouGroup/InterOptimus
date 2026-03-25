"""
Machine Learning Interatomic Potential (MLIP) Interface Module

This module provides interfaces to various MLIP calculators including
ORB, SevenNet, MatRIS, and DPA models, along with ASE optimizer configurations.
"""

import json
import os
from pathlib import Path
import subprocess

from ase.filters import UnitCellFilter
from ase.constraints import FixAtoms
from pymatgen.core.structure import Structure

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
        return FIRE
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
        self.calc_type = calc
        self.user_settings = user_settings
        self._external_env = self._resolve_external_env(calc, user_settings)

        if self._external_env:
            self.calc = None
            print(f"Using external conda env '{self._external_env}' for {calc}")
            return

        if calc == 'orb-models':
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            device = user_settings['device']
            print(f"Initializing ORB calculator on device: {device}")

            try:
                orbff = pretrained.orb_v3_conservative_20_omat(
                    weights_path=user_settings['ckpt_path'],
                    device=device,
                    precision="float32-high"
                )
                self.calc = ORBCalculator(orbff, device=device)
                print('ORB initialization success')
            except Exception as e:
                print(f"Failed to load custom ORB model: {e}")
                print("Using default ORB model")
                self.calc = ORBCalculator(
                    pretrained.orb_v3_conservative_20_omat(device=user_settings['device']),
                    device=user_settings['device']
                )

        elif calc == 'sevenn':
            from sevenn.calculator import SevenNetCalculator

            try:
                from pathlib import PurePath
                checkpoint_path = PurePath(user_settings['ckpt_path'])
                self.calc = SevenNetCalculator(checkpoint_path, modal='mpa')
                print("SevenNet initialized with custom checkpoint")
            except Exception:
                self.calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
                print("SevenNet initialized with default model")

        elif calc == 'matris':
            from matris.applications.base import MatRISCalculator

            model = user_settings.get('model', 'matris_10m_oam')
            task = user_settings.get('task', 'efsm')
            device = user_settings['device']
            self.calc = MatRISCalculator(model=model, task=task, device=device)
            print(f"MatRIS initialized with model={model}, task={task}, device={device}")

        elif calc == 'dpa':
            from deepmd.calculator import DP

            try:
                from pathlib import PurePath
                checkpoint_path = PurePath(user_settings['ckpt_path'])
                self.calc = DP(model=checkpoint_path)
                print("Deep Potential initialized with custom model")
            except Exception as e:
                raise ValueError(f'Invalid path for DPA checkpoint file: {e}')

        else:
            raise ValueError(f"Unsupported calculator type: {calc}")

    @staticmethod
    def _resolve_external_env(calc, user_settings):
        if user_settings.get('_force_local'):
            return None
        explicit = user_settings.get('conda_env')
        if explicit:
            return explicit
        env_map = {
            'dpa': os.getenv('DPA_CONDA_ENV', 'dpa'),
            'matris': os.getenv('MATRIS_CONDA_ENV', 'matris'),
        }
        return env_map.get(calc)

    def _run_external(self, operation, structure, *, optimizer='BFGS', **kwargs):
        payload = {
            'calc': self.calc_type,
            'operation': operation,
            'structure': structure.as_dict(),
            'user_settings': {**self.user_settings, '_force_local': True},
            'optimizer': optimizer,
            'kwargs': kwargs,
        }
        helper = Path(__file__).resolve().with_name('mlip_external_runner.py')
        repo_root = helper.parent.parent
        command = [
            os.environ.get('CONDA_EXE', 'conda'),
            'run',
            '--no-capture-output',
            '-n',
            self._external_env,
            'python',
            str(helper),
        ]
        timeout_seconds = kwargs.pop('external_timeout_seconds', None)
        if timeout_seconds is None:
            timeout_seconds = 120 if operation == 'calculate' else 600
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=str(repo_root),
            check=False,
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"External MLIP runner failed for {self.calc_type} in env {self._external_env}: {detail}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"External MLIP runner produced no output for {self.calc_type}")
        response = json.loads(lines[-1])
        if operation == 'calculate':
            return response['energy']
        return Structure.from_dict(response['structure']), response['energy']

    def calculate(self, structure):
        """
        Calculate potential energy of a structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            float: Potential energy in eV
        """
        if self._external_env:
            return self._run_external('calculate', structure)
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
                - steps (int): Maximum number of steps
                - fix_cell_booleans (list): Cell constraint flags

        Returns:
            tuple: (optimized_structure, final_energy)
        """
        if self._external_env:
            return self._run_external('optimize', structure, optimizer=optimizer, **kwargs)
        optimizer_class = get_optimizer(optimizer)
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calc

        # Cell constraints can be applied here if needed
        # if hasattr(structure, 'fatom_ids') and len(structure.fatom_ids) > 0:
        #     atoms.set_constraint([FixAtoms(indices=structure.fatom_ids)])

        ft = UnitCellFilter(atoms, kwargs['fix_cell_booleans'])
        relax = optimizer_class(ft, logfile=None)
        relax.run(fmax=kwargs['fmax'], steps=kwargs['steps'])

        return Structure.from_ase_atoms(atoms), atoms.get_potential_energy()

    def close(self):
        """
        Clean up calculator resources if needed.
        """
        pass
