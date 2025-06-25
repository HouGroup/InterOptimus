from InterOptimus.matching import interface_searching, EquiMatchSorter
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.tool import apply_cnid_rbt, trans_to_bottom, sort_list, get_it_core_indices, get_min_nb_distance, cut_vaccum, add_sele_dyn_slab, add_sele_dyn_it, get_non_strained_film, get_rot_strain
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
from numpy import array, dot, column_stack, argsort, zeros, mod, mean, ceil, concatenate, random, repeat, cross, inf, round, arccos, pi
from numpy.linalg import norm
from InterOptimus.CNID import calculate_cnid_in_supercell
from InterOptimus.VaspWorkFlow import ItFireworkPatcher
import os
import pandas as pd
from fireworks import Workflow
import json
import pickle
import warnings
import shutil
from ase.filters import UnitCellFilter
from ase.constraints import FixAtoms, FixedLine
from interfacemaster.cellcalc import get_normal_from_MI, get_primitive_hkl

def registration_minimizer(interfaceworker, n_calls, z_range):
    """
    baysian optimization for xyz registration
    
    Args:
    n_calls (int): num of optimization
    z_range (float): range of z sampling
    
    Return:
    optimization result
    """
    def trial_with_progress(func, n_calls, *args, **kwargs):
        with tqdm(total = n_calls, desc = "registration optimizing") as rgst_pbar:  # Initialize tqdm with total number of iterations
            def wrapped_func(*args, **kwargs):
                result = func(*args, **kwargs)
                rgst_pbar.update(1)  # Update progress bar by 1 after each function call
                return result
            return gp_minimize(wrapped_func, search_space, n_calls=n_calls, *args, **kwargs)
    search_space = [
        Real(0, 1, name='x'),
        Real(0, 1, name='y'),
        Real(z_range[0], z_range[1], name = 'z')
    ]
    # Run the optimization with progress bar
    result = trial_with_progress(interfaceworker.sample_xyz_energy, n_calls=n_calls, random_state=42)
    return result

class InterfaceWorker:
    """
    core class for the interface jobs
    """
    def __init__(self, film_conv, substrate_conv):
        """
        Args:
        film_conv (Structure): film conventional cell
        substrate_conv (Structure): substrate conventional cell
        """
        self.film_conv = film_conv
        self.substrate_conv = substrate_conv
        self.film = film_conv.get_primitive_structure()
        self.substrate = substrate_conv.get_primitive_structure()
    
    def lattice_matching(self, max_area = 47, max_length_tol = 0.03, max_angle_tol = 0.01,
                         film_max_miller = 3, substrate_max_miller = 3, film_millers = None, substrate_millers = None):
        """
        lattice matching by Zur and McGill

        Args:
        max_area (float), max_length_tol (float), max_angle_tol (float): searching tolerance parameters
        film_max_miller (int), substrate_max_miller (int): maximum miller index
        film_millers (None|array), substrate_millers (None|array): specified searching miller indices (optional)
        """
        sub_analyzer = SubstrateAnalyzer(max_area = max_area, max_length_tol = max_length_tol, max_angle_tol = max_angle_tol,
                                         film_max_miller = film_max_miller, substrate_max_miller = substrate_max_miller)
        self.unique_matches, \
        self.equivalent_matches, \
        self.unique_matches_indices_data,\
        self.equivalent_matches_indices_data,\
        self.areas = interface_searching(self.substrate_conv, self.film_conv, sub_analyzer, film_millers, substrate_millers)
        self.ems = EquiMatchSorter(self.film_conv, self.substrate_conv, self.equivalent_matches_indices_data, self.unique_matches)
    
    def screen_matches_by_parallel_planes(self, hkl_conv_film, hkl_conv_substrate, tol):
        hkl_prim_film = get_primitive_hkl(hkl = hkl_conv_film,
                                  C_lattice = self.film_conv.lattice.matrix.T, P_lattice = self.film.lattice.matrix.T)
        n_prim_film = get_normal_from_MI(self.film.lattice.matrix.T, hkl_prim_film)
        n_prim_film = n_prim_film / norm(n_prim_film)
        hkl_prim_substrate = get_primitive_hkl(hkl = hkl_conv_substrate,
                                       C_lattice = self.substrate_conv.lattice.matrix.T, P_lattice = self.substrate.lattice.matrix.T)
        n_prim_substrate = get_normal_from_MI(self.substrate.lattice.matrix.T, hkl_prim_substrate)
        n_prim_substrate = n_prim_substrate / norm(n_prim_substrate)
        screened_matches = []
        for i in range(len(self.equivalent_matches)):
            for j in range(len(self.equivalent_matches[i])):
                match = self.equivalent_matches[i][j]
                f_vs = match.film_sl_vectors
                s_vs = match.substrate_sl_vectors
                R,s = get_rot_strain(f_vs, s_vs)
                score = norm(cross(n_prim_substrate, dot(R, n_prim_film)))
                if score < tol:
                    screened_matches.append([i,j])
                    break
        return screened_matches
    
    def parse_interface_structure_params(self, termination_ftol = 0.01, c_periodic = False, \
                                        vacuum_over_film = 10, film_thickness = 10, substrate_thickness = 10, \
                                        shift_to_bottom = True):
        """
        parse necessary structure parameters for interface generation in the next steps

        Args:

        termination_ftol (float): tolerance of the c-fractional coordinates for termination atom clustering
        c_periodic (bool): whether to make double interface supercell
        vacuum_over_film (float): vacuum thickness over film
        film_thickness (float): film slab thickness
        substrate_thickness (float): substrate slab thickness
        shift_to_bottom (bool): whether to shift the supercell to the bottom
        """
        self.termination_ftol, self.c_periodic, self.vacuum_over_film, self.film_thickness, self.substrate_thickness, self.shift_to_bottom = \
        termination_ftol, c_periodic, vacuum_over_film, film_thickness, substrate_thickness, shift_to_bottom
        self.get_all_unique_terminations()
        self.calculate_thickness()
        self.do_opt = False
        self.dzs = {}
    
    def parse_optimization_params(self, do = True, set_fix_thicknesses = (0,0), fix_in_layers = False, whole_slab_fixed = True, **kwargs):
        self.do_opt = do
        self.set_fix_thicknesses = set_fix_thicknesses
        self.fix_in_layers = fix_in_layers
        self.whole_slab_fixed = whole_slab_fixed
        self.opt_kwargs = kwargs
        if self.c_periodic:
            self.opt_kwargs['fix_cell_booleans'] = [False, False, True, False, False, False]
        else:
            self.opt_kwargs['fix_cell_booleans'] = [False, False, False, False, False, False]
        
        if 'fmax' not in self.opt_kwargs:
            self.opt_kwargs['fmax'] = 0.05
        if 'steps' not in self.opt_kwargs:
            self.opt_kwargs['steps'] = 200
        if 'device' not in self.opt_kwargs:
            self.opt_kwargs['device'] = 'cpu'
        if 'ckpt_path' not in self.opt_kwargs:
            self.opt_kwargs['ckpt_path'] = None
            
        self.absolute_fix_thicknesses = []
        for i in range(len(self.unique_matches)):
            if self.fix_in_layers:
                if self.set_fix_thicknesses[0] == self.thickness_in_layers[i][0]:
                    raise ValueError(f'match {i}: the whole film in the interface will be fixed, please reset fix conditions')
                if self.set_fix_thicknesses[1] == self.thickness_in_layers[i][1]:
                    raise ValueError(f'match {i}: the whole substrate in the interface will be fixed, please reset fix conditions')
        
            fix_thickness_film, fix_thickness_substrate = self.get_specified_match_fix_thickness(i, 0)
            print(f'match {i}: fix thicknesses (film, substrate) ({round(fix_thickness_film, 2)} {round(fix_thickness_substrate, 2)})')
            if fix_thickness_film < 2:
                warnings.warn(f'match {i}: fixed film thickness {fix_thickness_film} is lower than 2 angstrom!')
            if fix_thickness_substrate < 2:
                warnings.warn(f'match {i}: fixed substrate thickness {fix_thickness_substrate} is lower than 2 angstrom!')
            self.absolute_fix_thicknesses.append([fix_thickness_film, fix_thickness_substrate])
    
    def get_specified_match_fix_thickness(self, match_id, term_id):
        if self.fix_in_layers:
            film_layer_thickness, substrate_layer_thickness = self.get_film_substrate_layer_thickness(match_id, term_id)

            return film_layer_thickness * self.set_fix_thicknesses[0] - 1e-6,\
                                   substrate_layer_thickness * self.set_fix_thicknesses[1] - 1e-6
        else:
            return self.set_fix_thicknesses[0], self.set_fix_thicknesses[1]

    def get_specified_match_cib(self, id):
        """
        get the CoherentInterfaceBuilder instance for a specified unique match

        Args:
        id (int): unique match index
        """
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[id].film_miller,
                               substrate_miller=self.unique_matches[id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[id]]
        return cib
    
    def get_unique_terminations(self, id):
        """
        get non-identical terminations for a specified unique match id

        Args:
        id (int): unique match index
        """
        unique_term_ids = get_non_identical_slab_pairs(self.film, self.substrate, self.unique_matches[id], \
                                                       ftol = self.termination_ftol, c_periodic = self.c_periodic)[0]
        cib = self.get_specified_match_cib(id)
        return [cib.terminations[i] for i in unique_term_ids]
    
    def get_all_unique_terminations(self):
        """
        get unique terminations for all the unique matches
        """
        all_unique_terminations = []
        for i in range(len(self.unique_matches)):
            all_unique_terminations.append(self.get_unique_terminations(i))
        self.all_unique_terminations = all_unique_terminations
    
    def calculate_thickness(self):
        self.thickness_in_layers = []
        self.absolute_thicknesses = []
        print('\n')
        for i in range(len(self.unique_matches)):
            film_l, substrate_l = self.get_film_substrate_layer_thickness(i, 0)
            film_thickness = int(ceil(self.film_thickness/film_l))
            substrate_thickness = int(ceil(self.substrate_thickness/substrate_l))
            self.thickness_in_layers.append((film_thickness, substrate_thickness))
            self.absolute_thicknesses.append((film_thickness * film_l, substrate_thickness * substrate_l))
            print(f'match {i}: thicknesses (film, substrate) ({round(film_thickness * film_l, 2)} {round(substrate_thickness * substrate_l, 2)}); num of unique terminations: {len(self.all_unique_terminations[i])}')
    
    def get_specified_interface(self, match_id, term_id, xyz = [0,0,2]):
        """
        get a specified interface by unique match index, unique termination index, and xyz registration

        Args:
        match_id (int): unique match index
        term_id (int): unique termination index
        xyz (array): xyz registration
        
        Return:
        (Interface)
        """
        x, y, z = xyz
        if self.c_periodic:
            gap = vacuum_over_film = z
        else:
            gap = z
            vacuum_over_film = self.vacuum_over_film
        cib = self.get_specified_match_cib(match_id)
        film_thickness, substrate_thickness = self.thickness_in_layers[match_id]
        interface_here = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness, film_thickness = film_thickness, \
                                       vacuum_over_film = vacuum_over_film, gap = gap, in_layers = True))[0]
        interface_here = apply_cnid_rbt(interface_here, x, y, 0)
        if self.shift_to_bottom:
            interface_here = trans_to_bottom(interface_here)
        return interface_here
    
    def set_energy_calculator(self, calc, user_settings = None):
        """
        set energy calculator docker container
        
        Args:
        calc (str): mace, orb-models, sevenn, chgnet, grace-2l
        """
        if calc == 'orb-models' or calc == 'sevenn':
            from InterOptimus.mlip import MlipCalc
        else:
            from mlipdockers.core import MlipCalc
        self.mc = MlipCalc(calc = calc, user_settings = user_settings)
    
    def close_energy_calculator(self):
        """
        close energy calculator docker container
        """
        self.mc.close()
    
    def sample_xyz_energy(self, params):
        """
        sample the predicted energy for a specified xyz registration of a initial interface
        
        Args:
        xyz: sampled xyz

        Return
        energy (float): predicted energy by chgnet
        """
        x,y,z = params
        xyz = [x,y,z]
        if self.c_periodic:
            interface_here = self.get_specified_interface(self.match_id_now, self.term_id_now, xyz = xyz)
        else:
            initial_interface = self.get_specified_interface(self.match_id_now, self.term_id_now, [0,0,2])
            xyz[2] = (xyz[2] - 2)/initial_interface.lattice.c
            interface_here = apply_cnid_rbt(initial_interface, xyz[0],xyz[1],xyz[2])
        self.opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
        term_atom_ids = self.get_interface_atom_indices(interface_here)
        for i in term_atom_ids:
            if get_min_nb_distance(i, interface_here, self.discut) < self.discut:
                return 0
        return self.mc.calculate(interface_here)
    
    def get_film_substrate_layer_thickness(self, match_id, term_id):
        """
        get single layer thickness
        """
        cib = self.get_specified_match_cib(match_id)
        
        delta_c = 0
        last_delta_c = 0
        initial_n = 2
        while last_delta_c == 0:
            last_delta_c = delta_c
            interface_film_1 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                           substrate_thickness = 2, film_thickness = initial_n, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            interface_film_2 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                           substrate_thickness = 2, film_thickness = initial_n + 5, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            delta_c = interface_film_2.lattice.c - interface_film_1.lattice.c
        film_delta_c = delta_c/5
            
        
        delta_c = 0
        last_delta_c = 0
        initial_n = 2
        while last_delta_c == 0:
            last_delta_c = delta_c
            interface_substrate_1 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                           substrate_thickness = initial_n, film_thickness = 2, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            interface_substrate_2 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                           substrate_thickness = initial_n + 5, film_thickness = 2, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            delta_c = interface_substrate_2.lattice.c - interface_substrate_1.lattice.c
        substrate_delta_c = delta_c/5
        
        
        return film_delta_c, substrate_delta_c
    
    def output_slabs(self, match_id, term_id):
        sgs, dbs = self.get_decomposition_slabs(match_id, term_id)
        sgs[0].to_file('fmsg_POSCAR')
        sgs[1].to_file('stsg_POSCAR')
        dbs[0].to_file('fmdb_POSCAR')
        dbs[1].to_file('stdb_POSCAR')

    def get_decomposition_slabs(self, match_id, term_id, get_double_it = False):
        """
        get decomposed film & substrate slabs to calculate binding energy

        Args:
        match_id (int): unique match index
        term_id (int): unique termination index

        Return:
        (single_film, single_substrate), (double_film, double_substrate) (tuple): single and double (film, substrate) pairs
        """
        
        cib = self.get_specified_match_cib(match_id)

        film_dz, substrate_dz = self.get_film_substrate_layer_thickness(match_id, term_id)
        self.dzs[(match_id, term_id)] = [film_dz, substrate_dz]

        film_thickness, substrate_thickness = self.thickness_in_layers[match_id]

        interface_single = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness, film_thickness = film_thickness, \
                                       vacuum_over_film = self.vacuum_over_film, gap = 2, in_layers = True))[0]
        
        n_film_s = len(interface_single.film)
        n_substrate_s = len(interface_single.substrate)
        
        db_substrate_thickness = substrate_thickness * 2
        db_film_thickness = film_thickness * 2
        
        interface_double = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = db_substrate_thickness, film_thickness = db_film_thickness, \
                                       vacuum_over_film = self.vacuum_over_film, gap = 2, in_layers = True))[0]
                                       
        n_film_d = len(interface_double.film)
        n_substrate_d = len(interface_double.substrate)
        
        while not (n_film_d == 2 * n_film_s and n_substrate_d == 2 * n_substrate_s):
            if n_substrate_d > 2 * n_substrate_s:
                db_substrate_thickness += -1
            if n_substrate_d < 2 * n_substrate_s:
                db_substrate_thickness += 1
            if n_film_d > 2 * n_film_s:
                db_film_thickness += -1
            if n_film_d < 2 * n_film_s:
                db_film_thickness += 1
            
            interface_double = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = db_substrate_thickness, film_thickness = db_film_thickness, \
                                       vacuum_over_film = self.vacuum_over_film, gap = 2, in_layers = True))[0]
            n_film_d = len(interface_double.film)
            n_substrate_d = len(interface_double.substrate)
        
        if get_double_it:
            return interface_double
        
        return (cut_vaccum(trans_to_bottom(interface_single.film), self.vacuum_over_film), \
                cut_vaccum(trans_to_bottom(interface_single.substrate), self.vacuum_over_film)), \
                (cut_vaccum(trans_to_bottom(interface_double.film), self.vacuum_over_film), \
                cut_vaccum(trans_to_bottom(interface_double.substrate), self.vacuum_over_film))
    
    def get_interface_energy_and_binding_energy_non_relax(self, supcl_E, match_id, term_id):
        area = self.unique_matches[match_id].match_area
        single_pair, double_pair = self.get_decomposition_slabs(match_id, term_id)

        film_single_E = self.mc.calculate(single_pair[0])
        film_double_E = self.mc.calculate(double_pair[0])
        substrate_single_E = self.mc.calculate(single_pair[1])
        substrate_double_E = self.mc.calculate(double_pair[1])

        E_it = (supcl_E - (film_double_E + substrate_double_E) / 2) / area * 16.02176634
        E_ch = (supcl_E - (film_single_E + substrate_single_E)) / area * 16.02176634
        if self.c_periodic:
            E_it = E_it/2
            E_ch = E_ch/2
        single_pair_E = (film_single_E, substrate_single_E)
        double_pair_E = (film_double_E, substrate_double_E)
        
        return E_it, E_ch, single_pair, double_pair, single_pair_E, double_pair_E
    
    def get_interface_energy_and_binding_energy_relax(self, supcl_E, match_id, term_id):
        area = self.unique_matches[match_id].match_area
        single_pair, double_pair = self.get_decomposition_slabs(match_id, term_id)
        
        #perform fix
        #0:no fix, 1: fix substrate, 2: fix both substrate and film
        
        #all mobile
        single_fixed_film_mobility_mtx, single_fixed_substrate_mobility_mtx, \
        double_fixed_film_mobility_mtx, double_fixed_substrate_mobility_mtx = \
        repeat(array([[True, True, True]]), len(single_pair[0]), axis = 0), \
        repeat(array([[True, True, True]]), len(single_pair[1]), axis = 0), \
        repeat(array([[True, True, True]]), len(double_pair[0]), axis = 0), \
        repeat(array([[True, True, True]]), len(double_pair[1]), axis = 0)
        
        if self.whole_slab_fixed:
            fix_thickness_film, fix_thickness_substrate = inf, inf
        else:
            fix_thickness_film, fix_thickness_substrate = self.absolute_fix_thicknesses[match_id]
        
        single_fixed_substrate, single_fixed_substrate_mobility_mtx = \
                                                                    add_sele_dyn_slab(single_pair[1], fix_thickness_substrate, 'left')
        double_fixed_substrate, double_fixed_substrate_mobility_mtx = \
                                                                    add_sele_dyn_slab(double_pair[1], fix_thickness_substrate, 'left')
    
        #print(fix_thickness_film, fix_thickness_substrate)
        single_fixed_film, single_fixed_film_mobility_mtx = \
                                                            add_sele_dyn_slab(single_pair[0], fix_thickness_film, 'right')
        double_fixed_film, double_fixed_film_mobility_mtx = \
                                                            add_sele_dyn_slab(double_pair[0], fix_thickness_film, 'right')
        
        single_pair = (single_fixed_film, single_fixed_substrate)
        double_pair = (double_fixed_film, double_fixed_substrate)
            
        #relax and calculate energy
        film_single, film_single_E = self.mc.optimize(single_pair[0], **self.opt_kwargs)
        film_double, film_double_E = self.mc.optimize(double_pair[0], **self.opt_kwargs)
        substrate_single, substrate_single_E = self.mc.optimize(single_pair[1], **self.opt_kwargs)
        substrate_double, substrate_double_E = self.mc.optimize(double_pair[1], **self.opt_kwargs)
        
        #parse structure
        single_pair, double_pair = (film_single, substrate_single), (film_double, substrate_double)
        
        #package
        E_it = (supcl_E - (film_double_E + substrate_double_E) / 2) / area * 16.02176634
        E_ch = (supcl_E - (film_single_E + substrate_single_E)) / area * 16.02176634
        single_pair_E = (film_single_E, substrate_single_E)
        double_pair_E = (film_double_E, substrate_double_E)
        single_pair_mobility_mtx = (single_fixed_film_mobility_mtx, single_fixed_substrate_mobility_mtx)
        double_pair_mobility_mtx = (double_fixed_film_mobility_mtx, double_fixed_substrate_mobility_mtx)
        
        #dyn_mtx to relaxed structure
        single_pair = (single_pair[0].add_site_property('selective_dynamics', single_pair_mobility_mtx[0]), \
                       single_pair[1].add_site_property('selective_dynamics', single_pair_mobility_mtx[1]))
        double_pair = (double_pair[0].add_site_property('selective_dynamics', double_pair_mobility_mtx[0]), \
                       double_pair[1].add_site_property('selective_dynamics', double_pair_mobility_mtx[1]))
        
        return E_it, E_ch, single_pair, double_pair, single_pair_E, double_pair_E, single_pair_mobility_mtx, double_pair_mobility_mtx
    
    def get_interface_atom_indices(self, interface):
        """
        get the indices of interface atoms
        
        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        
        Return:
        indices (array)
        """
        #interface atom indices
        ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max = get_it_core_indices(interface)
        if self.c_periodic:
            return concatenate((ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max))
        else:
            return concatenate((ids_film_min, ids_substrate_max))
    
    def optimize_specified_interface_by_mlip(self, match_id, term_id, n_calls = 50, z_range = (0.5, 3), calc = 'mace'):
        """
        apply bassian optimization for the xyz registration of a specified interface with the predicted
        interface energy by machine learning potential

        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc: MLIP calculator (str): mace, orb-models, sevenn, chgnet, grace-2l
        """
        #initialize opt info dict
        if not hasattr(self, 'opt_results'):
            self.opt_results = {}
        self.opt_results[(match_id,term_id)] = {}
        self.opt_results[(match_id,term_id)]['sampled_interfaces'] = []

        #set match&term id
        self.match_id_now = match_id
        self.term_id_now = term_id
        
        #optimize
        result = registration_minimizer(self, n_calls, z_range)
        xs = array(result.x_iters)
        ys = result.func_vals

        #rank xs by energy
        xs = xs[argsort(ys)]
        
        #list need to be ranked by special function
        self.opt_results[(match_id,term_id)]['sampled_interfaces'] = \
        sort_list(self.opt_results[(match_id,term_id)]['sampled_interfaces'], ys)

        #self.opt_results[(match_id,term_id)]['opt_results'] = result

        #rank energy
        ys = ys[argsort(ys)]

        #get cartesian xyzs
        interface = self.get_specified_interface(match_id, term_id)
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = column_stack((dot(interface.lattice.matrix.T, CNID),[0,0,0]))
        xs_cart = dot(CNID_cart, xs.T).T + column_stack((zeros(len(xs)), zeros(len(xs)), xs[:,2]))
        
        self.opt_results[(match_id,term_id)]['xyzs_ognl'] = xs
        self.opt_results[(match_id,term_id)]['xyzs_cart'] = xs_cart
        self.opt_results[(match_id,term_id)]['supcl_E'] = ys
    

    def global_minimization(self, n_calls = 50, z_range = (0.5, 3), calc = 'sevenn', discut = 0.8, rank_by = 'it', strain_E_correction = False):
        """
        apply bassian optimization for the xyz registration of all the interfaces with the predicted
        interface energy by machine learning potential, getting ranked interface energies

        Args:
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc (str): MLIP calculator: mace, orb-models, sevenn, chgnet, grace-2l
        discut: (float): allowed minimum atomic distance for searching
        rank_by (str): it interface energy, bd binding energy
        """

        self.opt_results = {}
        self.discut = discut
        columns = [r'$h_f$',r'$k_f$',r'$l_f$',
                  r'$h_s$',r'$k_s$',r'$l_s$',
                   r'$A$ (' + '\u00C5' + '$^2$)', r'$\epsilon$', r'$E_{it}$ $(J/m^2)$', r'$E_{bd}$ $(J/m^2)$', r'$E_{el}$ $(eV/\atom)$', r'$E_{sp}$',
                   r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$',
                   r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$',
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$',
                   r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$', r'$T$', r'$i_m$', r'$i_t$']
        formated_data = []
        #set docker container
        self.set_energy_calculator(calc, self.opt_kwargs)
        #scanning matches and terminations
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            #for i in range(1):
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    for j in range(len(self.all_unique_terminations[i])):
                    #for j in range(1):
                        #optimize
                        self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
                        
                        #formated data
                        m = self.unique_matches
                        idt = self.unique_matches_indices_data
                        
                        hkl_f, hkl_s = idt[i]['film_conventional_miller'], idt[i]['substrate_conventional_miller']
                        A, epsilon, E_sups = m[i].match_area, m[i].von_mises_strain, self.opt_results[(i,j)]['supcl_E']
                        uvw_f1, uvw_f2 = idt[i]['film_conventional_vectors']
                        uvw_s1, uvw_s2 = idt[i]['substrate_conventional_vectors']
                        
                        ##calculate adhesive & interface energy
                        it_Es, bd_Es, single_pair, double_pair, single_pair_E, double_pair_E = self.get_interface_energy_and_binding_energy_non_relax(array(E_sups), i, j)
                        
                        self.opt_results[(i,j)]['A'] = A
                        self.opt_results[(i,j)]['strain'] = epsilon
                        self.opt_results[(i,j)]['it_Es'] = it_Es
                        self.opt_results[(i,j)]['bd_Es'] = bd_Es
                        self.opt_results[(i,j)]['sup_Es'] = E_sups
                        ##save single double slabs
                        self.opt_results[(i,j)]['slabs'] = {}
                        self.opt_results[(i,j)]['slabs']['fmsg'] = {}
                        self.opt_results[(i,j)]['slabs']['fmsg']['structure'] = single_pair[0]
                        self.opt_results[(i,j)]['slabs']['fmsg']['e'] = single_pair_E[0]
                        
                        self.opt_results[(i,j)]['slabs']['stsg'] = {}
                        self.opt_results[(i,j)]['slabs']['stsg']['structure'] = single_pair[1]
                        self.opt_results[(i,j)]['slabs']['stsg']['e'] = single_pair_E[1]
                        
                        self.opt_results[(i,j)]['slabs']['fmdb'] = {}
                        self.opt_results[(i,j)]['slabs']['fmdb']['structure'] = double_pair[0]
                        self.opt_results[(i,j)]['slabs']['fmdb']['e'] = double_pair_E[0]
                        
                        self.opt_results[(i,j)]['slabs']['stdb'] = {}
                        self.opt_results[(i,j)]['slabs']['stdb']['structure'] = double_pair[1]
                        self.opt_results[(i,j)]['slabs']['stdb']['e'] = double_pair_E[1]

                        if self.do_opt:
                            #relax best interface & slabs
                            self.opt_results[(i,j)]['relaxed_slabs'] = {}
                            self.opt_results[(i,j)]['relaxed_slabs']['fmsg'] = {}
                            self.opt_results[(i,j)]['relaxed_slabs']['stsg'] = {}
                            self.opt_results[(i,j)]['relaxed_slabs']['fmdb'] = {}
                            self.opt_results[(i,j)]['relaxed_slabs']['stdb'] = {}
                            self.opt_results[(i,j)]['relaxed_best_interface'] = {}
                            
                            #no fix shell:
                            #fix interface
                            if not self.c_periodic:
                                fix_thickness_film, fix_thickness_substrate = self.absolute_fix_thicknesses[i]
                                
                                best_it, mobility_mtx = add_sele_dyn_it(self.opt_results[(i,j)]['sampled_interfaces'][0], fix_thickness_film, fix_thickness_substrate)
                            else:
                                best_it = self.opt_results[(i,j)]['sampled_interfaces'][0]
                                mobility_mtx = repeat(array([[True, True, True]]), len(best_it), axis = 0)
                                best_it.fatom_ids = []
                            #relax interface
                            relaxed_best_it, relaxed_best_sup_E = self.mc.optimize(best_it, **self.opt_kwargs)
                            relaxed_best_it = relaxed_best_it.add_site_property('selective_dynamics', mobility_mtx)
                            
                            #compute Eit Ech
                            it_E, bd_E, single_pair, double_pair, \
                            single_pair_E, double_pair_E, \
                            single_pair_mobility_mtx, double_pair_mobility_mtx, \
                            = self.get_interface_energy_and_binding_energy_relax(relaxed_best_sup_E, i, j)
                            
                            #parse relaxed slab structure & energy
                            self.opt_results[(i,j)]['relaxed_slabs']['stsg']['structure'] = single_pair[1]
                            self.opt_results[(i,j)]['relaxed_slabs']['stdb']['structure'] = double_pair[1]
                            self.opt_results[(i,j)]['relaxed_slabs']['fmsg']['structure'] = single_pair[0]
                            self.opt_results[(i,j)]['relaxed_slabs']['fmdb']['structure'] = double_pair[0]
                            
                            self.opt_results[(i,j)]['relaxed_slabs']['stsg']['e'] = single_pair_E[1]
                            self.opt_results[(i,j)]['relaxed_slabs']['stdb']['e'] = double_pair_E[1]
                            self.opt_results[(i,j)]['relaxed_slabs']['fmsg']['e'] = single_pair_E[0]
                            self.opt_results[(i,j)]['relaxed_slabs']['fmdb']['e'] = double_pair_E[0]
                            
                            #non-strained-film-slab
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg'] = {}
                            fmns_sg = cut_vaccum(get_non_strained_film(self.unique_matches[i], best_it), self.vacuum_over_film)
                            min_c, max_c = min(fmns_sg.cart_coords[:,2]), max(fmns_sg.cart_coords[:,2])
                            fmns_sg_L = ceil((max_c - min_c)/self.dzs[(i,j)][0]) * self.dzs[(i,j)][0]
                            fmns_sg_E = self.mc.calculate(fmns_sg)
                            strain_E = - (fmns_sg_E - single_pair_E[0])
                            strain_E_patom = strain_E/len(fmns_sg)
                            strain_E_mod = strain_E * fmns_sg_L / self.film_thickness / A * 16.02176634
                            self.opt_results[(i,j)]['A'] = A
                            self.opt_results[(i,j)]['fmns_sg_L'] = fmns_sg_L
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg']['structure'] = fmns_sg
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg']['e'] = fmns_sg_E
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg']['strain_e'] = strain_E
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg']['strain_e_per_atom'] = strain_E/len(fmns_sg)
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_sg']['e_correction'] = strain_E_mod
                            if strain_E_correction:
                                bd_E += strain_E_mod
                            
                            #non-strained-film-slab-double
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db'] = {}
                            fmns_db = cut_vaccum(get_non_strained_film(self.unique_matches[i], self.get_decomposition_slabs(i, j, True)), self.vacuum_over_film)
                            min_c, max_c = min(fmns_db.cart_coords[:,2]), max(fmns_db.cart_coords[:,2])
                            fmns_db_L = ceil((max_c - min_c)/self.dzs[(i,j)][0]) * self.dzs[(i,j)][0]
                            fmns_db_E = self.mc.calculate(fmns_db)
                            strain_E = - (fmns_db_E - double_pair_E[0])
                            strain_E_mod = strain_E / 2 * fmns_sg_L / self.film_thickness / A * 16.02176634
                            
                            self.opt_results[(i,j)]['fmns_db_L'] = fmns_db_L
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db']['structure'] = fmns_db
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db']['e'] = fmns_db_E
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db']['strain_e'] = strain_E
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db']['strain_e_per_atom'] = strain_E/len(fmns_db)
                            self.opt_results[(i,j)]['relaxed_slabs']['fmns_db']['e_correction'] = strain_E_mod
                            if strain_E_correction:
                                it_E += strain_E_mod
                            #parse relaxed best it structure and energy, Eit, Ech
                            self.opt_results[(i,j)]['relaxed_best_interface']['structure'] = relaxed_best_it
                            self.opt_results[(i,j)]['relaxed_best_interface']['e'] = relaxed_best_sup_E
                            self.opt_results[(i,j)]['relaxed_min_it_E'] = it_E
                            self.opt_results[(i,j)]['relaxed_min_bd_E'] = bd_E
                                
                            self.opt_results[(i,j)]['single_double_pairs'] = ((self.opt_results[(i,j)]['relaxed_slabs']['fmsg'],
                                                                              self.opt_results[(i,j)]['relaxed_slabs']['stsg']),
                                                                              (self.opt_results[(i,j)]['relaxed_slabs']['fmdb'],
                                                                              self.opt_results[(i,j)]['relaxed_slabs']['stdb']))
                            formated_data.append(
                                        [hkl_f[0], hkl_f[1], hkl_f[2],\
                                        hkl_s[0], hkl_s[1], hkl_s[2], \
                                        A, epsilon, it_E, bd_E, strain_E_patom,  relaxed_best_sup_E, \
                                        uvw_f1[0], uvw_f1[1], uvw_f1[2], \
                                        uvw_f2[0], uvw_f2[1], uvw_f2[2], \
                                        uvw_s1[0], uvw_s1[1], uvw_s1[2], \
                                        uvw_s2[0], uvw_s2[1], uvw_s2[2], self.all_unique_terminations[i][j], i, j])
                        else:
                             formated_data.append(
                                        [hkl_f[0], hkl_f[1], hkl_f[2],\
                                        hkl_s[0], hkl_s[1], hkl_s[2], \
                                        A, epsilon, it_Es[0], bd_Es[0], E_sups[0], \
                                        uvw_f1[0], uvw_f1[1], uvw_f1[2], \
                                        uvw_f2[0], uvw_f2[1], uvw_f2[2], \
                                        uvw_s1[0], uvw_s1[1], uvw_s1[2], \
                                        uvw_s2[0], uvw_s2[1], uvw_s2[2], self.all_unique_terminations[i][j], i, j])
                                        
                        term_pbar.update(1)
                match_pbar.update(1)
        self.global_optimized_data = pd.DataFrame(formated_data, columns = columns)
        if rank_by == 'it':
            self.global_optimized_data = self.global_optimized_data.sort_values(by = r'$E_{it}$ $(J/m^2)$')
        elif rank_by == 'bd':
            self.global_optimized_data = self.global_optimized_data.sort_values(by = r'$E_{bd}$ $(J/m^2)$')
        self.best_key = (self.global_optimized_data[r'$i_m$'][0], self.global_optimized_data[r'$i_t$'][0])
        #close docker container
        self.close_energy_calculator()
    
    def write_vasp_input(self, folder = 'it_E_vasp_input', user_incar_settings = {}, user_potcar_functional = 'PBE', e = 'bd'):
        try:
            shutil.rmtree(folder)
        except:
            pass
        os.mkdir(folder)
        names = ['it', 'film', 'film_non_strain', 'substrate']
        structures = {}
        structures['it'] = self.opt_results[self.best_key]['relaxed_best_interface']['structure']
        if e == 'bd':
            structures['film'] = self.opt_results[self.best_key]['relaxed_slabs']['fmsg']['structure']
            structures['film_non_strain'] = self.opt_results[self.best_key]['relaxed_slabs']['fmns_sg']['structure']
            structures['substrate'] = self.opt_results[self.best_key]['relaxed_slabs']['stsg']['structure']
        else:
            structures['film'] = self.opt_results[self.best_key]['relaxed_slabs']['fmdb']['structure']
            structures['film_non_strain'] = self.opt_results[self.best_key]['relaxed_slabs']['fmns_db']['structure']
            structures['substrate'] = self.opt_results[self.best_key]['relaxed_slabs']['stdb']['structure']
        
        for i in names:
            vis = MPRelaxSet(structures[i], user_incar_settings = user_incar_settings, user_potcar_functional = user_potcar_functional)
            vis.write_input(f'{folder}/{i}')
    
    def random_sampling_specified_interface(self, match_id, term_id, n_taget, n_max, sampling_min_displace, discut, set_seed = True, seed = 999):
        """
        perform random sampling of rigid body translation for a specified interface
        
        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        n_taget (int): target number of sampling
        n_max (int): max number of trials
        sampling_min_displace (float): sampled rigid body translation position are not allowed to be closer than this (angstrom)
        discut (float): the atoms are not allowed to be closer than this (angstrom)
        set_seed (bool): whether to set random seed
        seed (int): random seed
        
        Return:
        sampled_interfaces (list): list of sampled interfaces (json)
        xyzs (list): list of sampled xyz parameters
        rbt_carts: list of sampled RBT positions in cartesian coordinates
        """
        #get initial interface
        interface = self.get_specified_interface(match_id, term_id)
        #calculate cnid catesian
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = dot(interface.lattice.matrix.T, CNID)
        #sampling
        num_of_sampled = 1
        n_trials = 0
        rbt_carts = [[0,0,2]]
        xyzs = [[0,0,2]]
        ##interface atom indices
        sampled_interfaces = []
        sampled_interfaces.append(self.get_specified_interface(match_id, term_id, [0,0,2]).to_json())
        if set_seed == True:
            random.seed(seed)
        one_short_random = random.rand(n_max, 3)
        while num_of_sampled < n_taget and n_trials < n_max:
            #sampling from (0,0,0) to (1,1,1)
            x,y,z = one_short_random[n_trials]
            #z is cartesian
            z = z * 3
            #calculate cartesian RBT
            cart_here = x*CNID_cart[:,0] + y*CNID_cart[:,1] + [0,0,z]
            #calculate distances between this RBT position and already sampled RBT positions
            distwithbefore = norm(repeat([cart_here], num_of_sampled, axis = 0) - rbt_carts, axis = 1)
            #RBT position distance not too close
            if min(distwithbefore) > sampling_min_displace:
                #min atomic distance not too close
                interface_here = self.get_specified_interface(match_id, term_id, [x, y, z])
                existing_too_close_sites = False
                ##interface atomic indices
                it_atom_ids = self.get_interface_atom_indices(interface_here)
                for i in it_atom_ids:
                    if get_min_nb_distance(i, interface_here, discut) < discut:
                        existing_too_close_sites = True
                        break
                if not existing_too_close_sites:
                    #interface_here.to_file(f'op_its/{num_of_sampled}_POSCAR')
                    sampled_interfaces.append(interface_here.to_json())
                    rbt_carts.append(list(cart_here))
                    xyzs.append([x,y,z])
                    num_of_sampled += 1
            n_trials += 1
        
        return sampled_interfaces, xyzs, rbt_carts
    
    def calculate_itE_bdE(self, fmsg_E, fmdb_E, stsg_E, stdb_E, sup_E, A):
        it_E = (sup_E - 1/2 * (fmdb_E + stdb_E)) / A * 16.02176634
        bd_E = (sup_E - (fmsg_E + stsg_E)) / A * 16.02176634
        return it_E, bd_E
    
    def predict_global_random_sampling(self, mlip, user_settings = None):
        """
        predict all the sampled structures
        
        Args:
        mlip (str): which machine learning potential to use
        """
        keys = list(self.global_random_sample_dict.keys())
        self.set_energy_calculator(mlip, self.opt_kwargs)
        for i in keys:
            if 'predict' not in self.global_random_sample_dict[i].keys():
                self.global_random_sample_dict[i]['predict'] = {}
            self.global_random_sample_dict[i]['predict'][mlip] = {}
            
            #slabs
            for slb in ['fmsg','fmdb','stsg','stdb']:
                stct = Structure.from_dict(json.loads(self.global_random_sample_dict[i][slb]))
                self.global_random_sample_dict[i]['predict'][mlip][slb] = self.mc.calculate(stct)
                
            dct = self.global_random_sample_dict[i]['predict'][mlip]
            
            self.global_random_sample_dict[i]['predict'][mlip]['sup_Es'] = []
            self.global_random_sample_dict[i]['predict'][mlip]['it_Es'] = []
            self.global_random_sample_dict[i]['predict'][mlip]['bd_Es'] = []
            
            for k in self.global_random_sample_dict[i]['sampled_interfaces']:
                sup_E = self.mc.calculate(Structure.from_dict(json.loads(k)))
                self.global_random_sample_dict[i]['predict'][mlip]['sup_Es'].append(sup_E)
                
                it_E, bd_E = self.calculate_itE_bdE(dct['fmsg'], dct['fmdb'], dct['stsg'], dct['stdb'], sup_E, self.global_random_sample_dict[i]['A'])
                
                self.global_random_sample_dict[i]['predict'][mlip]['it_Es'].append(it_E)
                self.global_random_sample_dict[i]['predict'][mlip]['bd_Es'].append(bd_E)
                
        self.close_energy_calculator()
        
    def global_random_sampling(self, n_taget, n_max, sampling_min_displace, discut, set_seed = True, seed = 999, to_fireworks = False, **kwargs):
        """
        perform random sampling of rigid body translation for all the interface
        
        Args:
        n_taget (int): target number of sampling
        n_max (int): max number of trials
        sampling_min_displace (float): sampled rigid body translation position are not allowed to be closer than this (angstrom)
        discut (float): the atoms are not allowed to be closer than this (angstrom)
        set_seed (bool): whether to set random seed
        seed (int): random seed
        to_fireworks (bool): whether to generate firework workflow dict
        
        kwargs:
        project_name (str): project name to be stored in mongodb database
        db_file (str): path to atomate mongodb config file
        vasp_cmd (str): command to run vasp
        work_dir (str): working directory
        update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
        update_potcar_functional (str): which set of functional to use

        Return:
        (Workflow)
        """
        if to_fireworks:
            for st in ['user_incar_settings', 'user_potcar_settings', 'user_kpoints_settings', 'user_potcar_functional']:
                if st not in kwargs.keys():
                    kwargs[st] = None
            it_firework_patcher = ItFireworkPatcher(kwargs['project_name'], kwargs['db_file'], kwargs['vasp_cmd'],
                                                     user_incar_settings = kwargs['user_incar_settings'],
                                                     user_potcar_settings = kwargs['user_potcar_settings'],
                                                     user_kpoints_settings = kwargs['user_kpoints_settings'],
                                                     user_potcar_functional = kwargs['user_potcar_functional'])
            wf = []

        self.global_random_sample_dict = {}
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    #for j in range(1):
                    for j in range(len(self.all_unique_terminations[i])):
                        #print('modified')
                        key = f'{i}_{j}'
                        self.global_random_sample_dict[key] = {}
                        self.global_random_sample_dict[key]['A'] = self.unique_matches[i].match_area
                        self.global_random_sample_dict[key]['strain'] = self.unique_matches[i].von_mises_strain
                        
                        self.global_random_sample_dict[key]['sampled_interfaces'], \
                        self.global_random_sample_dict[key]['xyzs'], \
                        self.global_random_sample_dict[key]['rbt_carts'] \
                        = self.random_sampling_specified_interface(i, j, n_taget, n_max, \
                                                                    sampling_min_displace, discut, set_seed, seed)
                        single_pairs, double_pairs = self.get_decomposition_slabs(i, j)

                        if to_fireworks:

                            wf += get_slab_fireworks('random_sample', single_pairs, double_pairs, it_firework_patcher, i, j, kwargs['work_dir'], kwargs['dp'])
                            
                            #interface workflow
                            its = self.global_random_sample_dict[f'{i}_{j}']['sampled_interfaces']
                            for k in range(len(its)):
                                fws = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', Structure.from_dict(json.loads(its[k])),
                                                                                          {'i':i, 'j':j, 'k':k, 'tp':'it'},
                                                                                          os.path.join(kwargs['work_dir'], f'it_{i}_{j}_{k}'), kwargs['dp'])
                                wf += fws

                        #save slab info
                        self.global_random_sample_dict[f'{i}_{j}']['fmsg'], self.global_random_sample_dict[f'{i}_{j}']['fmdb'], \
                        self.global_random_sample_dict[f'{i}_{j}']['stsg'], self.global_random_sample_dict[f'{i}_{j}']['stdb'], = \
                        single_pairs[0].to_json(), double_pairs[0].to_json(), single_pairs[1].to_json(), double_pairs[1].to_json()
                        
                        term_pbar.update(1)
                match_pbar.update(1)
                
        if to_fireworks:
            with open('global_random_sampling.json','w') as f:
                json.dump(self.global_random_sample_dict, f)
            return Workflow(wf)
    
    def mlip_benchmark(self, mlips, n_calls = 50, z_range = (0.5, 3), discut = 0.8, **kwargs):
        """
        glabal minimization by different mlips, generate firework
        
        Args:
        mlips (list): mlips
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        discut: (float): allowed minimum atomic distance for searching
        
        kwargs:
        project_name (str): project name to be stored in mongodb database
        db_file (str): path to atomate mongodb config file
        vasp_cmd (str): command to run vasp
        work_dir (str): working directory
        update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
        update_potcar_functional (str): which set of functional to use
        user_settings: mlip docker container settings
        
        Return:
        (Workflow)
        """

        #print('modified')
        if 'work_dir' not in kwargs.keys():
            kwargs['work_dir'] = ''
        work_dir = kwargs['work_dir']
        for st in ['project_name', 'db_file', 'vasp_cmd', 'user_incar_settings', 'user_potcar_settings', 'user_kpoints_settings', 'user_potcar_functional', 'user_settings']:
            if st not in kwargs.keys():
                kwargs[st] = None
        it_firework_patcher = ItFireworkPatcher(kwargs['project_name'], kwargs['db_file'], kwargs['vasp_cmd'],
                                                 user_incar_settings = kwargs['user_incar_settings'],
                                                 user_potcar_settings = kwargs['user_potcar_settings'],
                                                 user_kpoints_settings = kwargs['user_kpoints_settings'],
                                                 user_potcar_functional = kwargs['user_potcar_functional'])
        slab_fws_added = False
        wf = []
        self.benchmk_dict = {}
        for mlip in mlips:
            self.global_minimization(n_calls = n_calls, z_range = z_range, calc = mlip, discut = discut, user_settings = kwargs['user_settings'])
            self.benchmk_dict[mlip] = {}
            #for i in range(1):
            for i in range(len(self.unique_matches)):
                for j in range(len(self.all_unique_terminations[i])):
                #for j in range(1):
                    self.benchmk_dict[mlip][(i,j)] = {}
                    self.benchmk_dict[mlip][(i,j)]['A'] = self.unique_matches[i].match_area
                    self.benchmk_dict[mlip][(i,j)]['strain'] = self.unique_matches[i].von_mises_strain
                    self.benchmk_dict[mlip][(i,j)]['slabs'] = self.opt_results[(i,j)]['slabs']
                    self.benchmk_dict[mlip][(i,j)]['relaxed_slabs'] = self.opt_results[(i,j)]['relaxed_slabs']
                    self.benchmk_dict[mlip][(i,j)]['best_it'] = {}
                    if self.do_opt:
                        self.benchmk_dict[mlip][(i,j)]['best_it']['structure'] = self.opt_results[(i,j)]['relaxed_best_interface']['structure']
                        self.benchmk_dict[mlip][(i,j)]['best_it']['sup_E'] = self.opt_results[(i,j)]['relaxed_best_interface']['e']
                        self.benchmk_dict[mlip][(i,j)]['best_it']['it_E'] = self.opt_results[(i,j)]['relaxed_min_it_E']
                        self.benchmk_dict[mlip][(i,j)]['best_it']['bd_E'] = self.opt_results[(i,j)]['relaxed_min_bd_E']
                    else:
                        self.benchmk_dict[mlip][(i,j)]['best_it']['structure'] = self.opt_results[(i,j)]['sampled_interfaces'][0]
                        self.benchmk_dict[mlip][(i,j)]['best_it']['sup_E'] = self.opt_results[(i,j)]['sup_Es'][0]
                        self.benchmk_dict[mlip][(i,j)]['best_it']['it_E'] = self.opt_results[(i,j)]['it_Es'][0]
                        self.benchmk_dict[mlip][(i,j)]['best_it']['bd_E'] = self.opt_results[(i,j)]['bd_Es'][0]
                    self.benchmk_dict[mlip][(i,j)]['opt_results'] = self.opt_results[(i,j)]
                    
                    work_dir = os.path.join(kwargs['work_dir'], mlip)
                    
                    if self.do_opt:
                        self.benchmk_dict[mlip][(i,j)]['relaxed_slabs'] = self.opt_results[(i,j)]['relaxed_slabs']
                        best_it_structure = self.opt_results[(i,j)]['relaxed_best_interface']['structure']
                        if not slab_fws_added:
                            single_pairs, double_pairs = self.opt_results[(i,j)]['single_double_pairs']
                            if kwargs['dp']:
                                wf += get_slab_fireworks_dp('interface relax', single_pairs, double_pairs, it_firework_patcher, i, j, work_dir, kwargs['dp'])
                            else:
                                wf += get_slab_fireworks_relax_direct(single_pairs, double_pairs, it_firework_patcher, i, j, work_dir, self.c_periodic)
                    else:
                        best_it_structure = self.benchmk_dict[mlip][(i,j)]['best_it']['structure']
                        if not slab_fws_added:
                            single_pairs, double_pairs = self.get_decomposition_slabs(i, j)
                            wf += get_slab_fireworks_dp('interface static', single_pairs, double_pairs, it_firework_patcher, i, j, work_dir, kwargs['dp'])
                            slab_fws_added = True
                    
                    if self.do_opt:
                        if kwargs['dp']:
                            wf += it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface relax',
                                                                                best_it_structure,
                                                                                {'mlip':mlip, 'i':i, 'j':j, 'tp':'it'},
                                                                                os.path.join(work_dir, f'it_{i}_{j}'), dp = kwargs['dp'], c_periodic = self.c_periodic)
                        else:
                            wf += [it_firework_patcher.get_fw(best_it_structure, {'mlip':mlip, 'i':i, 'j':j, 'tp':'it'}, os.path.join(work_dir, f'it_{i}_{j}'), 'interface relax', LDIPOL = False, c_periodic = self.c_periodic)]
                    else:
                        wf += it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static',
                                                                            best_it_structure,
                                                                            {'mlip':mlip, 'i':i, 'j':j, 'tp':'it'},
                                                                            os.path.join(work_dir, f'it_{i}_{j}'), dp = kwargs['dp'])
            slab_fws_added = True
        
        
        
        with open('benchmk.pkl','wb') as f:
            pickle.dump(self.benchmk_dict, f)
        return Workflow(wf)
    
    def conv_test(self, length_list, match_id, term_id, n_calls, calc = 'sevenn', discut = 0.8, conv = 'slab', slab_length = 10, user_settings = None):
        conv_dict = {}
        film_l, substrate_l = self.get_film_substrate_layer_thickness(match_id, term_id)
        fix_thickness_film, fix_thickness_substrate = self.absolute_fix_thicknesses[match_id]
        if fix_thickness_film < 2:
            warnings.warn(f'({i}, {j}) fixed film thickness is lower than 2 angstrom!')
        if fix_thickness_substrate < 2:
            warnings.warn(f'({i}, {j}) fixed substrate thickness is lower than 2 angstrom!')
            
        self.set_energy_calculator(calc, self.opt_kwargs)
        self.discut = discut
        for L in length_list:
            if conv == 'slab':
                self.parse_interface_structure_params(termination_ftol = self.termination_ftol, c_periodic = False, \
                                        vacuum_over_film = 10, film_thickness = L, \
                                        substrate_thickness = L, shift_to_bottom = True)
            else:
                self.parse_interface_structure_params(termination_ftol = self.termination_ftol, c_periodic = False, \
                                        vacuum_over_film = L, film_thickness = slab_length, \
                                        substrate_thickness = slab_length, shift_to_bottom = True)
            self.optimize_specified_interface_by_mlip(match_id, term_id, n_calls = n_calls, z_range = (0.5, 3), calc = calc)
            conv_dict[L] = {}
            conv_dict[L]['relaxed_slabs'] = {}
            conv_dict[L]['relaxed_slabs']['fmsg'] = {}
            conv_dict[L]['relaxed_slabs']['stsg'] = {}
            conv_dict[L]['relaxed_slabs']['fmdb'] = {}
            conv_dict[L]['relaxed_slabs']['stdb'] = {}
            conv_dict[L]['relaxed_best_interface'] = {}
            
            best_it = self.get_specified_interface(match_id=match_id, term_id=term_id, xyz=self.opt_results[(match_id,term_id)]['xyzs_ognl'][0])
            best_it, mobility_mtx = add_sele_dyn_it(best_it, fix_thickness_film, fix_thickness_substrate)
            
            #relax interface
            relaxed_best_it, relaxed_best_sup_E = self.mc.optimize(best_it, **self.opt_kwargs)
            #add site property
            relaxed_best_it = relaxed_best_it.add_site_property('selective_dynamics', mobility_mtx)
            
            #compute Eit Ech
            it_E, bd_E, single_pair, double_pair, \
            single_pair_E, double_pair_E, \
            single_pair_mobility_mtx, double_pair_mobility_mtx, \
            = self.get_interface_energy_and_binding_energy_relax(relaxed_best_sup_E, match_id, term_id)
            
            #parse relaxed slab structure & energy
            conv_dict[L]['relaxed_slabs']['stsg']['structure'] = single_pair[1]
            conv_dict[L]['relaxed_slabs']['stdb']['structure'] = double_pair[1]
            conv_dict[L]['relaxed_slabs']['fmsg']['structure'] = single_pair[0]
            conv_dict[L]['relaxed_slabs']['fmdb']['structure'] = double_pair[0]
            
            conv_dict[L]['relaxed_slabs']['stsg']['e'] = single_pair_E[1]
            conv_dict[L]['relaxed_slabs']['stdb']['e'] = double_pair_E[1]
            conv_dict[L]['relaxed_slabs']['fmsg']['e'] = single_pair_E[0]
            conv_dict[L]['relaxed_slabs']['fmdb']['e'] = double_pair_E[0]
            
            #parse relaxed best it structure and energy, Eit, Ech
            conv_dict[L]['relaxed_best_interface']['structure'] = relaxed_best_it
            conv_dict[L]['relaxed_best_interface']['e'] = relaxed_best_sup_E
            conv_dict[L]['relaxed_min_it_E'] = it_E
            conv_dict[L]['relaxed_min_bd_E'] = bd_E
        self.conv_dict = conv_dict
        self.close_energy_calculator()


def get_slab_fireworks(name, single_pairs, double_pairs, it_firework_patcher, i, j, workdir, dp):
    """
    get fireworks of a set of slabs
    """
    #slab workflow
    fws_fmsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, single_pairs[0],
                                                                        {'i':i, 'j':j, 'tp':'fmsg'},
                                                                        os.path.join(workdir, f'fmsg_{i}_{j}'), dp, c_periodic = False)
    fws_fmdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, double_pairs[0],
                                                                        {'i':i, 'j':j, 'tp':'fmdb'},
                                                                        os.path.join(workdir, f'fmdb_{i}_{j}'), dp, c_periodic = False)
    fws_stsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, single_pairs[1],
                                                                        {'i':i, 'j':j, 'tp':'stsg'},
                                                                        os.path.join(workdir, f'stsg_{i}_{j}'), dp, c_periodic = False)
    fws_stdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, double_pairs[1],
                                                                        {'i':i, 'j':j, 'tp':'stdb'},
                                                                        os.path.join(workdir, f'stdb_{i}_{j}'), dp, c_periodic = False)
    return fws_fmsg + fws_fmdb + fws_stsg + fws_stdb

def get_slab_fireworks_dp(name, single_pairs, double_pairs, it_firework_patcher, i, j, workdir, dp):
    """
    get fireworks of a set of slabs
    """
    #slab workflow
    fws_fmsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, single_pairs[0]['structure'],
                                                                        {'i':i, 'j':j, 'tp':'fmsg'},
                                                                        os.path.join(workdir, f'fmsg_{i}_{j}'), dp, c_periodic = False)
    fws_fmdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, double_pairs[0]['structure'],
                                                                        {'i':i, 'j':j, 'tp':'fmdb'},
                                                                        os.path.join(workdir, f'fmdb_{i}_{j}'), dp, c_periodic = False)
    fws_stsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, single_pairs[1]['structure'],
                                                                        {'i':i, 'j':j, 'tp':'stsg'},
                                                                        os.path.join(workdir, f'stsg_{i}_{j}'), dp, c_periodic = False)
    fws_stdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod(name, double_pairs[1]['structure'],
                                                                        {'i':i, 'j':j, 'tp':'stdb'},
                                                                        os.path.join(workdir, f'stdb_{i}_{j}'), dp, c_periodic = False)
    return fws_fmsg + fws_fmdb + fws_stsg + fws_stdb

def get_slab_fireworks_relax_direct(single_pairs, double_pairs, it_firework_patcher, i, j, workdir, c_periodic):
    """
    get fireworks of a set of slabs
    """
    #slab workflow
    fws_fmsg = it_firework_patcher.get_fw(single_pairs[0]['structure'], {'i':i, 'j':j, 'tp':'fmsg'},
                                            os.path.join(workdir, f'fmsg_{i}_{j}'), 'interface relax', LDIPOL = False, c_periodic = c_periodic)
                                            
    fws_fmdb = it_firework_patcher.get_fw(double_pairs[0]['structure'], {'i':i, 'j':j, 'tp':'fmdb'},
                                            os.path.join(workdir, f'fmdb_{i}_{j}'), 'interface relax', LDIPOL = False, c_periodic = c_periodic)
    
    fws_stsg = it_firework_patcher.get_fw(single_pairs[1]['structure'], {'i':i, 'j':j, 'tp':'stsg'},
                                            os.path.join(workdir, f'stsg_{i}_{j}'), 'interface relax', LDIPOL = False, c_periodic = c_periodic)
    
    fws_stdb = it_firework_patcher.get_fw(double_pairs[1]['structure'], {'i':i, 'j':j, 'tp':'stdb'},
                                            os.path.join(workdir, f'stdb_{i}_{j}'), 'interface relax', LDIPOL = False, c_periodic = c_periodic)
    return [fws_fmsg, fws_fmdb, fws_stsg, fws_stdb]
