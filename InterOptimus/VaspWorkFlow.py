from fireworks import FireTaskBase, FWAction, explicit_serialize, Firework, Workflow, ScriptTask
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, ModifyIncar
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.analysis.interfaces.coherent_interfaces import get_rot_3d_for_2d, CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from scipy.linalg import polar
from InterOptimus.CNID import calculate_cnid_in_supercell
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from numpy import arange, ceil, savetxt, dot, meshgrid, array, inf, column_stack
from numpy.linalg import norm, inv
import pickle
import shutil
import os

def update_setting_dict(old_dict, new_dict):
    """
    update old dict by new dict

    Args:
    old_dict (dict)
    new_dict (dict)
    """
    if new_dict != None:
        for i in new_dict.keys():
            old_dict[i] = new_dict[i]
    return old_dict

def get_potcar(structure, functional):
    """
    get the pymatgen Potcar instance for a structure

    Args:
    structure (Structure)
    functional (str): functional set to use
    
    Return:
    (Potcar)
    """
    return Potcar([get_potcar_dict()[i.symbol] for i in structure.elements], functional)

def get_potcar_dict():
    """
    default potcar label for each element (used recommended ones by the vasp mannual for the PBE.54 functionals)

    Return:
    (dict): {element: potcar label}
    """
    return {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Ar': 'Ar', 'As': 'As', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be_sv', 'Bi': 'Bi_d', 'Br': 'Br', 'C': 'C', 'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cl': 'Cl', 'Co': 'Co', 'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu_2', 'F': 'F', 'Fe': 'Fe', 'Ga': 'Ga_d', 'Gd': 'Gd_3', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg_pv', 'Mn': 'Mn_pv', 'Mo': 'Mo_sv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_sv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni', 'Np': 'Np', 'O': 'O', 'Os': 'Os', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 'Pm': 'Pm_3', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Rb': 'Rb_sv', 'Re': 'Re', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_sv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_sv', 'W': 'W_sv', 'Xe': 'Xe', 'Y': 'Y_sv', 'Yb': 'Yb_2', 'Zn': 'Zn', 'Zr': 'Zr_sv', 'Po': 'Po_d',  'At':'At', 'Rn':'Rn', 'Fr':'Fr_sv', 'Ra':'Ra_sv', 'Am':'Am', 'Cm':'Cm'}

def get_default_incar_settings(name, **kwargs):
    """
    get default incar settings

    Args:
    name (str): what to calculate
    """
    if name == 'standard relax':
        return {
        "EDIFF": 1e-4,
        "EDIFFG": -0.05,
        "ALGO": "Normal",
        "ISIF": 3,
        "NELM": 500,
        "NSW": 300,
        "LWAVE": False,
        "NCORE": 12,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL":"Auto",
        "ADDGRID": True,
        "LDAU": False,
        }
    elif name == 'standard static':
        return {
        "EDIFF": 1e-5,
        "ALGO": "Normal",
        "NELM": 250,
        "LWAVE": True,
        "NCORE": 12,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL":"Auto",
        "ADDGRID": True,
        "LDAU": False,
        }
    elif name == 'interface static':
        incar_settings = get_default_incar_settings('standard static')
        if kwargs['LDIPOL']:
            incar_settings['LDIPOL'] = True
            incar_settings['IDIPOL'] = 3
        return incar_settings
    elif name == 'interface relax':
        incar_settings = get_default_incar_settings('standard relax')
        if kwargs['LDIPOL']:
            incar_settings['LDIPOL'] = True
            incar_settings['IDIPOL'] = 3
        if kwargs['c_periodic']:
            incar_settings['IOPTCELL'] = "0 0 0 0 0 0 0 0 1"
            incar_settings['ISIF'] = 3
        else:
            incar_settings['ISIF'] = 2
        return incar_settings
    else:
        raise ValueError("default inter settings only support 'standard relax', 'standard static', 'interface static' and 'interface relax'")
        
def get_vasp_input_settings(name, structure, update_incar_settings = None, update_potcar_functional = None,
                            update_potcar_settings = None, update_kpoints_settings = None, **kwargs):
    
    """
    get user vasp input settings
    
    Args:
    name (str): one of 'standard relax', 'standard static', 'interface static', 'interface relax'
    structure (Structure): structure for calculation
    ##ENCUT_scale (float): scaling factor of the maximum en_cut in potcars
    update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
    update_potcar_functional (str): which set of functional to use
    
    Return:
    (VaspInputSet)
    """
    default_incar_settings = get_default_incar_settings(name, **kwargs)
    default_potcar_settings = get_potcar_dict()
    #default_potcar_settings = {}
    default_kpoints_settings = {'reciprocal_density':100}
    user_incar_settings = update_setting_dict(default_incar_settings, update_incar_settings)
    user_potcar_settings = update_setting_dict(default_potcar_settings, update_potcar_settings)
    user_kpoints_settings = update_setting_dict(default_kpoints_settings, update_kpoints_settings)

    if update_potcar_functional == None:
        user_potcar_functional = 'PBE_64'
    else:
        user_potcar_functional = update_potcar_functional
    """
    potcar = get_potcar(structure, user_potcar_functional)
    max_encut = max(p.keywords['ENMAX'] for p in potcar)
    custom_encut = max_encut * ENCUT_scale
    user_incar_settings['ENCUT'] = custom_encut
    """
    if name == 'standard relax' or name == 'interface relax':
        return MPRelaxSet(structure, user_incar_settings = user_incar_settings, \
                                       user_potcar_functional=user_potcar_functional, \
                                       user_potcar_settings = user_potcar_settings, \
                                       user_kpoints_settings = user_kpoints_settings)
    else:
        return MPStaticSet(structure, user_incar_settings = user_incar_settings, \
                                       user_potcar_functional=user_potcar_functional, \
                                       user_potcar_settings = user_potcar_settings, \
                                       user_kpoints_settings = user_kpoints_settings, \
                                       reciprocal_density = user_kpoints_settings['reciprocal_density'])

class ItFireworkPatcher:
    """
    patch interface fireworks
    """
    def __init__(self, project_name, db_file, vasp_cmd,
                                user_incar_settings = None,
                                user_potcar_settings = None,
                                user_kpoints_settings = None,
                                user_potcar_functional = None):
        """
        Args:
        project_name (str): project name to be stored in mongodb database
        db_file (str): path to atomate mongodb config file
        vasp_cmd (str): command to run vasp
        update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
        update_potcar_functional (str): which set of functional to use
        """
        self.project_name = project_name
        self.db_file = db_file
        self.vasp_cmd = vasp_cmd
        self.user_incar_settings = user_incar_settings
        self.user_potcar_settings = user_potcar_settings
        self.user_kpoints_settings = user_kpoints_settings
        self.user_potcar_functional = user_potcar_functional
        
    def vasp_input_settings(self, name, structure, **kwargs):
        """
        get vasp input settings for a structure
        
        name (str): one of 'standard relax', 'standard static', 'interface static', 'interface relax'
        structure (Structure): structure to calculate
        
        Return:
        (VaspInputSet)
        """
        return get_vasp_input_settings(name, structure,
                                        update_incar_settings = self.user_incar_settings,
                                        update_potcar_functional = self.user_potcar_functional,
                                        update_potcar_settings = self.user_potcar_settings,
                                        update_kpoints_settings = self.user_kpoints_settings,
                                        **kwargs)

    def non_dipole_mod_fol_by_diple_mod(self, name, structure, additional_fields, launch_dir, dp = False, c_periodic = False):
        """
        a non-dipole corrected calculation firework (followed by a dipole-corrected calculation firework, optional)
        
        Args:
        name (str): 'interface static' or 'interface relax'
        structure (Structure): calculation structure
        additional_fields (dict): firework additional fields used to distinguish different calculations
        launch_dir (str): launch dictory
        dp (bool): whether to do dipole correction
        
        Return (list): [firework_1, firework_2]
        """
        additional_fields['pname'] = self.project_name
        additional_fields_ndp = additional_fields.copy()
        additional_fields_ndp['dp'] = 'f'
        if 'k' in additional_fields.keys():
            fw_name = f"{self.project_name}_{additional_fields['tp']}_{additional_fields['i']}_{additional_fields['j']}_{additional_fields['k']}"
        else:
            fw_name = f"{self.project_name}_{additional_fields['tp']}_{additional_fields['i']}_{additional_fields['j']}"
        fw1 = Firework(
                       tasks=[WriteVaspFromIOSet(vasp_input_set = self.vasp_input_settings('interface static', structure, LDIPOL = False),
                                                 structure = structure),
                              RunVaspCustodian(vasp_cmd = self.vasp_cmd, gzip_output = False),
                              VaspToDb(db_file = self.db_file, additional_fields = additional_fields_ndp)],
                        name = fw_name + "_ndp",
                        spec={"_launch_dir": launch_dir}
                       )
        #dipole correction
        if dp:
            additional_fields_dp = additional_fields.copy()
            additional_fields_dp['dp'] = 't'
            mod_incar_update = get_default_incar_settings(name, LDIPOL = True, c_periodic = c_periodic)
            mod_incar_update['LWAVE'] = False
            fw2 = Firework(
                           tasks=[ModifyIncar(incar_update = mod_incar_update),
                                  RunVaspCustodian(vasp_cmd = self.vasp_cmd, gzip_output = False),
                                  VaspToDb(db_file = self.db_file, additional_fields = additional_fields_dp),
                                  ScriptTask.from_str('rm WAVECAR'),
                                  ScriptTask.from_str('rm CHG'),
                                  ScriptTask.from_str('rm CHGCAR')],
                            name = fw_name + "_dp",
                            spec={"_launch_dir": launch_dir},
                            parents = fw1
                           )
            return [fw1, fw2]
        else:
            return [fw1]
    
    def get_fw(self, structure, additional_fields, launch_dir, name, **kwargs):
        """
        get a firework for a vasp job by current self settings
        
        Args
        structure (Structure): calculation structure
        additional_fields (dict): firework additional fields used to distinguish different calculations
        launch_dir (str): launch dictory
        name (str): standard relax or interface relax
        """
        additional_fields['pname'] = self.project_name
        return Firework(
                        tasks=[
                                WriteVaspFromIOSet(
                                                  vasp_input_set = self.vasp_input_settings(name, structure, **kwargs),
                                                  structure = structure
                                                  ),
                                RunVaspCustodian(vasp_cmd = self.vasp_cmd, gzip_output = False),
                                VaspToDb(db_file = self.db_file, additional_fields = additional_fields),
                                ScriptTask.from_str('rm WAVECAR'),
                                ScriptTask.from_str('rm CHG'),
                                ScriptTask.from_str('rm CHGCAR')],
                               name = f'{self.project_name}_NDP',
                               spec={"_launch_dir": launch_dir})
