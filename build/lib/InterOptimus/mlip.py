from ase.filters import UnitCellFilter
from ase.constraints import FixAtoms, FixedLine
from pymatgen.core.structure import Structure

def get_optimizer(optimizer):
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
    def __init__(self, calc, user_settings = {'device':'cpu'}):
        if calc == 'orb-models':
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            try:
                self.calc = ORBCalculator(pretrained.orb_v3_conservative_20_omat(weights_path = user_settings['ckpt_path'], device = user_settings['device']), device = user_settings['device'])
            except:
                self.calc = ORBCalculator(pretrained.orb_v3_conservative_20_omat(device = user_settings['device']), device = user_settings['device'])
        elif calc == 'sevenn':
            from sevenn.calculator import SevenNetCalculator
            try:
                from pathlib import PurePath
                p = PurePath(user_settings['ckpt_path'])
                self.calc = SevenNetCalculator(p, modal='mpa')
            except:
                self.calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
    
    def calculate(self, structure):
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calc
        return atoms.get_potential_energy()
    
    def optimize(self, structure, optimizer = 'BFGS', **kwargs):
        optimizer = get_optimizer(optimizer)
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calc
        atoms.set_constraint([FixAtoms(indices = structure.fatom_ids)])
        ft = UnitCellFilter(atoms, kwargs['fix_cell_booleans'])
        relax = optimizer(ft, logfile = None)
        relax.run(fmax = kwargs['fmax'], steps =  kwargs['steps'])
        return Structure.from_ase_atoms(atoms), atoms.get_potential_energy()
    
    def close(self):
        pass
