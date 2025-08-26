from InterOptimus.matching import interface_searching, EquiMatchSorter
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.tool import apply_cnid_rbt, sort_list, get_it_core_indices, get_min_nb_distance, cut_vaccum, add_sele_dyn_slab, add_sele_dyn_it, get_non_strained_film, get_rot_strain
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
from numpy import array, dot, column_stack, argsort, zeros, mod, mean, ceil, concatenate, random, repeat, cross, inf, round, arccos, pi, where, unique
from numpy.linalg import norm
from InterOptimus.CNID import calculate_cnid_in_supercell
import os
import pandas as pd
from fireworks import Workflow
import json
import pickle
import warnings
import shutil
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
    
    def screen_matches_by_parallel_planes(self, hkl_conv_film, hkl_conv_substrate, tol = 0.1, max_area = 200, max_length_tol = 0.1, max_angle_tol = 0.1, film_max_miller = 3, substrate_max_miller = 3):
        hkl_prim_film = get_primitive_hkl(hkl = hkl_conv_film,
                                  C_lattice = self.film_conv.lattice.matrix.T, P_lattice = self.film.lattice.matrix.T)
        n_prim_film = get_normal_from_MI(self.film.lattice.matrix.T, hkl_prim_film)
        n_prim_film = n_prim_film / norm(n_prim_film)
        hkl_prim_substrate = get_primitive_hkl(hkl = hkl_conv_substrate,
                                       C_lattice = self.substrate_conv.lattice.matrix.T, P_lattice = self.substrate.lattice.matrix.T)
        n_prim_substrate = get_normal_from_MI(self.substrate.lattice.matrix.T, hkl_prim_substrate)
        n_prim_substrate = n_prim_substrate / norm(n_prim_substrate)
        screened_matches = []
        sub_analyzer = SubstrateAnalyzer(max_area = max_area, max_length_tol = max_length_tol, max_angle_tol = max_length_tol, film_max_miller = film_max_miller, substrate_max_miller = substrate_max_miller)
        matches = list(sub_analyzer.calculate(film=self.film, substrate=self.substrate))
        for i in range(len(matches)):
            #for j in range(len(matches[i])):
            match = matches[i]
            f_vs = match.film_sl_vectors
            s_vs = match.substrate_sl_vectors
            R,s = get_rot_strain(f_vs, s_vs)
            score = norm(cross(n_prim_substrate, dot(R, n_prim_film)))
            if score < tol:
                screened_matches.append(match)
                print(arccos(dot(n_prim_substrate,dot(R, n_prim_film))) / pi * 180, match.match_area)
                #break
        return screened_matches
    
    def parse_interface_structure_params(self, termination_ftol = 0.01, film_thickness = 10, substrate_thickness = 10, \
                                        shift_to_bottom = True):
        """
        parse necessary structure parameters for interface generation in the next steps

        Args:

        termination_ftol (float): tolerance of the c-fractional coordinates for termination atom clustering
        film_thickness (float): film slab thickness
        substrate_thickness (float): substrate slab thickness
        shift_to_bottom (bool): whether to shift the supercell to the bottom
        """
        self.termination_ftol, self.film_thickness, self.substrate_thickness, self.shift_to_bottom = \
        termination_ftol, film_thickness, substrate_thickness, shift_to_bottom
        self.get_all_unique_terminations()
        self.calculate_thickness()
        self.do_opt = False
        self.dzs = {}
    
    def parse_optimization_params(self, set_fix_thicknesses = (0,0), fix_in_layers = False, whole_slab_fixed = True, num_relax_bayesian = 0, discut = 0.8,  **kwargs):
        #number of relaxing steps during BO
        self.num_relax_bayesian = num_relax_bayesian
        #during BO, structures with minimum atomic distance lower than discut will be attached a zero energy
        self.discut = discut
        
        self.set_fix_thicknesses = set_fix_thicknesses
        self.fix_in_layers = fix_in_layers
        self.whole_slab_fixed = whole_slab_fixed
        self.opt_kwargs = kwargs

        self.opt_kwargs['fix_cell_booleans'] = [False, False, True, False, False, False]

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
                                                       ftol = self.termination_ftol, c_periodic = True)[0]
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

        cib = self.get_specified_match_cib(match_id)
        film_thickness, substrate_thickness = self.thickness_in_layers[match_id]
        interface_here = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness, film_thickness = film_thickness, \
                                       vacuum_over_film = z, gap = z, in_layers = True))[0]
        interface_here = apply_cnid_rbt(interface_here, x, y, 0)
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

        interface_here = self.get_specified_interface(self.match_id_now, self.term_id_now, xyz = xyz)

        term_atom_ids = self.get_interface_atom_indices(interface_here)
        for i in term_atom_ids:
            if get_min_nb_distance(i, interface_here, self.discut) < self.discut:
                return 0
        if self.num_relax_bayesian == 0:
            self.opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
            return self.mc.calculate(interface_here)
        else:
            fix_thickness_film, fix_thickness_substrate = self.absolute_fix_thicknesses[self.match_id_now]
            interface_here, mobility_mtx = add_sele_dyn_it(interface_here, fix_thickness_film, fix_thickness_substrate)
            interface_here_relaxed, e = self.mc.optimize(interface_here, fix_cell_booleans = self.opt_kwargs['fix_cell_booleans'], fmax = 0.05, steps = self.num_relax_bayesian)
            interface_here_relaxed.film = interface_here.film
            interface_here_relaxed.substrate = interface_here.substrate
            interface_here_relaxed.interface_properties = interface_here.interface_properties
            self.opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
            return e
    
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

        return concatenate((ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max))

    
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
    
    def screen_low_energy_static_interfaces(self, i, j):
        #select all the interfaces with low energies
        if self.tol_relax_bayesian > 0:
            indices = []
            ys = self.opt_results[(i,j)]['it_Es']
            ys = ys[(ys - min(ys)) < self.tol_relax_bayesian]
            print(ys)
            xs = self.opt_results[(i,j)]['xyzs_cart']
            for idx in range(len(ys)):
                if idx == 0:
                    indices.append(idx)
                else:
                    too_close = False
                    for i_idx in indices:
                        if norm(xs[i_idx] - xs[indices]) < 1:
                            too_close = True
                            break
                    if not too_close:
                        indices.append(idx)
        else:
            indices = [0]
        return [self.opt_results[(i,j)]['sampled_interfaces'][it] for it in indices]
    
    def get_i_j_static_best_it(self, i, j, n_calls, z_range, calc):
        self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
        best_it = self.opt_results[(i,j)]['sampled_interfaces'][0]
        best_sup_E = self.opt_results[(i,j)]['supcl_E'][0]
        area = best_it.lattice.a * best_it.lattice.b
        E_it = (best_sup_E - len(best_it.film)/len(self.film) * self.film_e - len(best_it.substrate)/len(self.substrate) * self.substrate_e) / area * 16.02176634 / 2
        return best_it, E_it
    
    def phase_stability_evaluation(self, n_calls = 50, z_range = (0.5, 3), calc = 'sevenn', discut = 0.8):
        self.set_energy_calculator(calc, self.opt_kwargs)
        self.film_e = self.mc.calculate(self.film)
        self.substrate_e = self.mc.calculate(self.substrate)
        self.discut = discut
        its, Es = [], []
        self.num_relax_bayesian = 0
        self.tol_relax_bayesian = 0
        self.opt_kwargs['fix_cell_booleans'] = [False, False, False, False, False, False]
        for i in range(len(self.unique_matches)):
            for j in range(len(self.all_unique_terminations[i])):
                it, E = self.get_i_j_static_best_it(i, j, n_calls, z_range, calc)
                its.append(it)
                Es.append(E)
        its = sort_list(its, Es)
        static_it = its[0]
        static_it.fatom_ids = []
        relaxed_it,e = self.mc.optimize(static_it, **self.opt_kwargs)
        dis_fracs = abs(relaxed_it.frac_coords - static_it.frac_coords)
        dis_fracs = abs(dis_fracs - round(dis_fracs))
        dis_carts = dot(relaxed_it.lattice.matrix.T, dis_fracs.T).T
        return static_it, relaxed_it,  sum(norm(dis_fracs, axis = 1))/len(relaxed_it), sum(norm(dis_carts, axis = 1))/len(relaxed_it)
    
    def global_minimization(self, n_calls = 50, z_range = (0.5, 3), calc = 'sevenn', strain_E_correction = False, name = ''):
        """
        apply bassian optimization for the xyz registration of all the interfaces with the predicted
        interface energy by machine learning potential, getting ranked interface energies

        Args:
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc (str): MLIP calculator: orb-models, sevenn
        """
        #optimization results
        self.opt_results = {}
        columns = [r'$h_f$',r'$k_f$',r'$l_f$',
                  r'$h_s$',r'$k_s$',r'$l_s$',
                   r'$A$ (' + '\u00C5' + '$^2$)', r'$\epsilon$', r'$E_{it}$ $(J/m^2)$', r'$E_{el}$ $(eV/\atom)$', r'$E_{sp}$',
                   r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$',
                   r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$',
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$',
                   r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$', r'$T$', r'$i_m$', r'$i_t$']
        formated_data = []
        #set mlip calculator
        self.set_energy_calculator(calc, self.opt_kwargs)
        self.film_e = self.mc.calculate(self.film)
        self.substrate_e = self.mc.calculate(self.substrate)
        #scanning matches and terminations
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            #for i in range(1):
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    #for j in range(1):
                    for j in range(len(self.all_unique_terminations[i])):
                        #optimize
                        self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
                        
                        #lattice matching data
                        m = self.unique_matches
                        idt = self.unique_matches_indices_data
                        hkl_f, hkl_s = idt[i]['film_conventional_miller'], idt[i]['substrate_conventional_miller']
                        epsilon = m[i].von_mises_strain
                        uvw_f1, uvw_f2 = idt[i]['film_conventional_vectors']
                        uvw_s1, uvw_s2 = idt[i]['substrate_conventional_vectors']
                        ltc = self.opt_results[(i,j)]['sampled_interfaces'][0].lattice
                        A = ltc.a * ltc.b

                        #layer thickness
                        film_dz, substrate_dz = self.get_film_substrate_layer_thickness(i, j)
                        self.dzs[(i, j)] = [film_dz, substrate_dz]
                        
                        #relax best interface & slabs
                        self.opt_results[(i,j)]['relaxed_best_interface'] = {}

                        #get lowest-energy static it
                        best_it = self.opt_results[(i,j)]['sampled_interfaces'][0]
                        
                        #relax interface
                        relaxed_best_it, relaxed_best_sup_E = self.mc.optimize(best_it, **self.opt_kwargs)
                        relaxed_best_it = Structure.from_dict(json.loads(  relaxed_best_it.to_json() ) )
                        relaxed_best_it.interface_properties = best_it.interface_properties
                        
                        #interface energy
                        it_E = (relaxed_best_sup_E - len(best_it.film)/len(self.film) * self.film_e - len(best_it.substrate)/len(self.substrate) * self.substrate_e) / A * 16.02176634 / 2
                        
                        #save to result dict
                        self.opt_results[(i,j)]['relaxed_best_interface']['structure'] = relaxed_best_it
                        self.opt_results[(i,j)]['relaxed_best_interface']['e'] = relaxed_best_sup_E
                        
                        #energy correction by film strain energy
                        if strain_E_correction:
                            
                            #film length
                            fmns_L = self.absolute_thicknesses[i][0]
                            
                            #get film with strain
                            match = self.unique_matches[i]
                            f_vs = match.film_sl_vectors
                            s_vs = match.substrate_sl_vectors
                            Rt, Sr = get_rot_strain(f_vs, s_vs)
                            DST = DeformStructureTransformation(Sr)
                            strain_film = DST.apply_transformation(self.film)
                            strain_E = (self.mc.calculate(strain_film) - self.film_e)/len(self.film)
                            
                            #energy correction
                            strain_E_mod = strain_E * len(best_it.film) * fmns_L / self.film_thickness / A * 16.02176634 / 2
                            it_E += strain_E_mod
                            self.opt_results[(i,j)]['strain_film'] = strain_film
                            self.opt_results[(i,j)]['strain_E'] = strain_E
                            self.opt_results[(i,j)]['length_factor'] = fmns_L / self.film_thickness
                            self.opt_results[(i,j)]['A'] = A

                        self.opt_results[(i,j)]['thicknesses'] = self.absolute_thicknesses[i]
                        self.opt_results[(i,j)]['relaxed_min_it_E'] = it_E
                        
                        formated_data.append(
                                [hkl_f[0], hkl_f[1], hkl_f[2],\
                                hkl_s[0], hkl_s[1], hkl_s[2], \
                                A, epsilon, it_E, strain_E,  relaxed_best_sup_E, \
                                uvw_f1[0], uvw_f1[1], uvw_f1[2], \
                                uvw_f2[0], uvw_f2[1], uvw_f2[2], \
                                uvw_s1[0], uvw_s1[1], uvw_s1[2], \
                                uvw_s2[0], uvw_s2[1], uvw_s2[2], self.all_unique_terminations[i][j], i, j])
                        
                        
                        term_pbar.update(1)
                    match_pbar.update(1)
        self.global_optimized_data = pd.DataFrame(formated_data, columns = columns)
        self.global_optimized_data = self.global_optimized_data.sort_values(by = r'$E_{it}$ $(J/m^2)$')

        self.best_key = (self.global_optimized_data[r'$i_m$'].to_numpy()[0], self.global_optimized_data[r'$i_t$'].to_numpy()[0])
        #close docker container
        self.close_energy_calculator()
        self.global_optimized_data.to_csv('all_data.csv')
        with open(f'opt_results_{name}.pkl','wb') as f:
            pickle.dump(self.opt_results, f)
    
    def get_selected_match_ids_per_plane(self):
        sub_millers = list(self.ems.matching_data[1].keys())
        type_lists_per_miller = []
        for i in sub_millers:
            type_lists_per_miller.append(self.ems.matching_data[1][i]['type_list'])
        selected_i_ms = []
        for i in type_lists_per_miller:
            for k in self.global_optimized_data['$i_m$'].to_numpy():
                if k in i:
                    selected_i_ms.append(k)
                    break
        self.selected_i_ms_substrate = unique(selected_i_ms)
        
        film_millers = list(self.ems.matching_data[0].keys())
        type_lists_per_miller = []
        for i in film_millers:
            type_lists_per_miller.append(self.ems.matching_data[1][i]['type_list'])
        selected_i_ms = []
        for i in type_lists_per_miller:
            for k in self.global_optimized_data['$i_m$'].to_numpy():
                if k in i:
                    selected_i_ms.append(k)
                    break
        self.selected_i_ms_film = unique(selected_i_ms)

    def patch_jobflow_jobs(self,
                            only_lowest_energy_each_sub_plane = False,
                            only_substrate = False,
                            
                            relax_user_incar_settings = None,
                            relax_user_potcar_settings = None,
                            relax_user_kpoints_settings = None,
                            relax_user_potcar_functional = None,
                            
                            static_user_incar_settings = None,
                            static_user_potcar_settings = None,
                            static_user_kpoints_settings = None,
                            static_user_potcar_functional = None,
                            
                            filter_name = 'my_IO_jobs',
                            ):
        """
        Patch JobFlow jobs, only for c_period = True
        """
        
        from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
        from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker
        from jobflow_remote import submit_flow
        from jobflow import Flow
        from atomate2.vasp.powerups import add_metadata_to_flow
        
        if only_lowest_energy_each_sub_plane:
            self.get_selected_match_ids_per_plane()
        pd = self.global_optimized_data
        ids = pd.index.to_numpy()
        i_s = pd['$i_m$'].to_numpy()
        j_s = pd['$i_t$'].to_numpy()
        match_ids = []
        pairs = []
        for i in range(len(i_s)):
            if only_lowest_energy_each_sub_plane:
                if only_substrate:
                    con = i_s[i] not in match_ids and i_s[i] in self.selected_i_ms_substrate
                else:
                    con = i_s[i] not in match_ids and (i_s[i] in self.selected_i_ms_substrate or
                                                        i_s[i] in self.selected_i_ms_film)
            else:
                con = i_s[i] not in match_ids
            if con:
                match_ids.append(i_s[i])
                pairs.append((i_s[i], j_s[i]))
        
        flows = []
        for num in range(2):
            structure = [self.film, self.substrate][num]
            job = ['film','job'][num]
            update_metadata = {'filter_name':filter_name, 'job': job}
            vasp_maker = RelaxMaker(
                                    input_set_generator = MPRelaxSet(
                                                        structure,
                                                        user_incar_settings = relax_user_incar_settings,
                                                         user_potcar_settings = relax_user_potcar_settings,
                                                          user_kpoints_settings = relax_user_kpoints_settings,
                                                           user_potcar_functional = relax_user_potcar_functional,
                                                           )
                                    )
            flows.append(add_metadata_to_flow(flow = vasp_maker.make(structure), additional_fields = update_metadata))

        for i in pairs:
            it_user_incar_settings = relax_user_incar_settings
            it_user_incar_settings['IOPTCELL'] = "0 0 0 0 0 0 0 0 1"
            #interface here
            it = self.opt_results[i]['relaxed_best_interface']['structure']
            update_metadata = {'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_it'}
            vasp_maker = RelaxMaker(
                                    input_set_generator = MPRelaxSet(
                                                                    it,
                                                                    user_incar_settings = it_user_incar_settings,
                                                                    user_potcar_settings = relax_user_potcar_settings,
                                                                    user_kpoints_settings = relax_user_kpoints_settings,
                                                                    user_potcar_functional = relax_user_potcar_functional,
                                                                    )
                                    )
            flows.append(add_metadata_to_flow(flow = vasp_maker.make(it), additional_fields = update_metadata))

            #strained film
            s_film = self.opt_results[i]['strain_film']
            update_metadata = {'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_sfilm'}
            vasp_maker = StaticMaker(
                                    input_set_generator = MPRelaxSet(
                                                                    s_film,
                                                                    user_incar_settings = static_user_incar_settings,
                                                                    user_potcar_settings = static_user_potcar_settings,
                                                                    user_kpoints_settings = static_user_kpoints_settings,
                                                                    user_potcar_functional = static_user_potcar_functional
                                                                    )
                                    )
            flows.append(add_metadata_to_flow(flow = vasp_maker.make(s_film), additional_fields = update_metadata))
        
        return flows
        
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
