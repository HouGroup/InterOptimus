"""
This module provides class to calculate the optimizing score of a interface structure by
comparing the terminating atom environment with those in materials project
"""
import numpy as np
from numpy import *
from numpy.linalg import norm
import pandas as pd
import sys
import time
from tqdm.notebook import tqdm
from InterOptimus.CNID import calculate_cnid_in_supercell
from dscribe.descriptors import SOAP
from pymatgen.core.periodic_table import Element
from skopt import gp_minimize
from skopt.space import Real
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from InterOptimus.MPsoap import to_ase, get_min_nb_distance, soap_data_generator, get_EN_diff_interface, get_delta_distances
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from InterOptimus.matching import interface_searching
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.matching import sort_list, get_area_match
import pickle
from scipy.stats import pearsonr
from scipy.stats.mstats import spearmanr
from skopt.space import Real, Integer
import math
from InterOptimus.VaspWorkFlow import RegistrationScan, ScoreRankerWF, AllMatchTermOPWF
from InterOptimus.tool import get_one_interface, read_key_item, get_it_core_indices

class interface_pre_optimizer:
    """
    interface pre-optimizer
    """
    def __init__(self, cib, termination, soap_data, slab_length = 10, c_periodic = False, vacuum_over_film = 0.01, \
                 kernel_factors = {'soap':1, 'rp':1, 'en':1}, en_cut = 1):
        """
        Args:
        cib (CoherentInterfaceBuilder).
        termination (string): termination.
        soap_data (soap_data_generator): SOAP data class with MP SOAP information.
        slab_length (float): minimum slab length.
        c_periodic (bool): whether set c-direction as periodic boundary condition.
        vacuum_over_film (float): length of vaccum over film.
        """
        self.interface_initial = list(cib.get_interfaces(termination = termination, \
                                   substrate_thickness = slab_length, \
                                   film_thickness = slab_length, \
                                   vacuum_over_film = 10, \
                                   gap = 10, \
                                   in_layers=False))[0]
        ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max = get_it_core_indices(self.interface_initial)
        if c_periodic:
            self.soap_ids = concatenate((ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max))
        else:
            self.soap_ids = concatenate((ids_film_min, ids_substrate_max))
            #self.soap_ids = ids_substrate_max
        #print(f'{len(self.soap_ids)} termination atoms')
        self.c_periodic = c_periodic
        self.cib = cib
        self.CNID = calculate_cnid_in_supercell(self.interface_initial)[0]
        self.soap_data = soap_data
        soap_params = soap_data.soap_params
        self.SOAP_analyzer = SOAP(
        species={el: Element(el).Z for el in soap_data.elements},
        r_cut=soap_params['r_cut'],
        n_max=soap_params['n_max'],
        l_max=soap_params['l_max'],
        periodic=True,
        sparse=False,
        weighting = soap_params['weighting'],
        )
        self.slab_length = slab_length
        self.cib = cib
        self.termination = termination
        self.vacuum_over_film = vacuum_over_film
        self.op_data = {}
        self.kernel_factors = kernel_factors
        self.en_cut = en_cut
    """
    def rpsv_pow_func(self, r):

        penalty too close atoms
        
        Args:
        r (float): 0 to 1, half scoring point
        
        Return:
        penalty factor

        return 1 - self.rpsv_pow['c']/(self.rpsv_pow['d'] + (r/self.rpsv_pow['rho'])**self.rpsv_pow['m'])
    """
    def trial(self, params):
        """
        score a structure with x, y, z interface registration
        
        Args:
        params (list): interface registration
        ksai (int): power
        
        Return:
        (list): scores for all terminating atoms
        """
        x,y,z = params
        if self.c_periodic:
            gap = vacuum_over_film = z
        else:
            gap = z
            vacuum_over_film = self.vacuum_over_film
        interface_here = list(self.cib.get_interfaces(termination= self.termination, \
                                       substrate_thickness = self.slab_length, \
                                       film_thickness=self.slab_length, \
                                       vacuum_over_film=vacuum_over_film, \
                                       gap=gap, \
                                       in_layers=False))[0]
        CNID_translation = TranslateSitesTransformation(interface_here.film_indices, x*self.CNID[:,0] + y*self.CNID[:,1])
        interface_here = CNID_translation.apply_transformation(interface_here)
        
        SOAP_discriptors = self.SOAP_analyzer.create(to_ase(interface_here), self.soap_ids)
        site_scores = []
        self.op_data[(x,y,z)] = {}
        for i in range(len(SOAP_discriptors)):
            #by_el_dict
            self.op_data[(x,y,z)][self.soap_ids[i]] = {}
            site = interface_here[self.soap_ids[i]]
            el_here = site.species.elements[0].name
            by_el_dict = self.soap_data.by_element_dict[el_here]
            
            #soap_kernels
            a = SOAP_discriptors[i]
            bs = by_el_dict['soap_descs']
            kernel_vals = soap_kernel(a, bs) ** self.kernel_factors['soap']
            self.op_data[(x,y,z)][self.soap_ids[i]]['it_term_soaps'] = a
            self.op_data[(x,y,z)][self.soap_ids[i]]['soap_kernels'] = kernel_vals.copy()
            """
            #repulsive kernel
            if self.rp_kernel:
                rp_kernels_here = RP_kernel(get_min_nb_distance(self.soap_ids[i], interface_here), \
                                                    by_el_dict['min_nb_distances'], self.rpsv_pow_func)
                self.op_data[(x,y,z)][self.soap_ids[i]]['rp_kernels'] = rp_kernels_here
                kernel_vals *= rp_kernels_here ** self.kernel_factors['rp']
                if all(rp_kernels_here < 1e-4):
                    existing_too_close_sites = True
            """
            rp_kernels = []
            min_ds_array = by_el_dict['min_nb_distances']
            for ref_ds in min_ds_array:
                close_ds = get_delta_distances(self.soap_ids[i], interface_here, ref_ds)
                rp_kernels.append(RP_kernel(ref_ds, close_ds, self.kernel_factors['rp']))
            rp_kernels = array(rp_kernels)
            kernel_vals += rp_kernels
            #Electronegtivity kernel
            EN_kernels = []
            r_ENs = []
            for j in range(len(by_el_dict['min_nb_distances'])):
                r_cut = by_el_dict['min_nb_distances'][j] * self.en_cut
                ref_EN = by_el_dict['EN_diffs'][j]
                t_EN = get_EN_diff_interface(interface_here, self.soap_ids[i], r_cut)
                r_ENs.append(ref_EN)
                EN_kernels.append(EN_kernel(t_EN, ref_EN))
            EN_kernels = array(EN_kernels)
            self.op_data[(x,y,z)][self.soap_ids[i]]['en_kernels'] = EN_kernels
            kernel_vals *= EN_kernels ** self.kernel_factors['en']
            
            site_scores.append(max(kernel_vals))
            self.op_data[(x,y,z)][self.soap_ids[i]]['site'] = interface_here[self.soap_ids[i]]
            self.op_data[(x,y,z)][self.soap_ids[i]]['EN'] = t_EN
            self.op_data[(x,y,z)][self.soap_ids[i]]['ref_ENs'] = r_ENs
            self.op_data[(x,y,z)][self.soap_ids[i]]['site_scores'] = kernel_vals
            self.op_data[(x,y,z)][self.soap_ids[i]]['structure_idx'] = by_el_dict['soap_struct_indices'][kernel_vals.argmax()]
            self.op_data[(x,y,z)][self.soap_ids[i]]['site_idx'] = by_el_dict['soap_site_indices'][kernel_vals.argmax()]
            #print(self.op_data[(x,y,z)][self.soap_ids[i]])
            self.this_trial_found_id = kernel_vals.argmax()
        #return dissimilarity for BO
        it_score = average(array(site_scores))
        self.op_data[(x,y,z)]['reg_score'] = it_score
        return 1 - it_score

def soap_kernel(a, bs):
    a = array(a)
    bs = array(bs)
    return dot(a, bs.T) / sqrt(dot(a,a) * diagonal(dot(bs,bs.T)))
"""
def RP_kernel(t_min_d, ref_ds, rpsv_pow_func):
    return rpsv_pow_func(t_min_d / ref_ds)
"""

def RP_kernel(ref_ds, close_ds, c):
    return - c * sum( ((ref_ds-close_ds)/ref_ds)**3 )**2

def EN_kernel(t_EN, ref_EN):
    """
    if t_EN * ref_EN > 0:
        if abs(t_EN) < abs(ref_EN):
            kns = t_EN/ref_EN
        else:
            kns = 1
    else:
        kns = 0
    return kns
    """
    kns = 1 - abs(t_EN - ref_EN)/abs(ref_EN)
    kns[kns < 0] = 0
    return kns

def registration_minimizer(itopt, n_calls):
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
        Real(1e-4, 2.5, name = 'z')
    ]
    # Run the optimization with progress bar
    result = trial_with_progress(itopt.trial, n_calls=n_calls, random_state=42)
    return result

class InputDataGenerator:
    def __init__(self):
        set_data = read_key_item('INTAR')
        substrate_conv = Structure.from_file('SBS.cif')
        film_conv = Structure.from_file('FLM.cif')
        sub_analyzer = SubstrateAnalyzer(max_area = set_data['MAXAREA'], max_length_tol = set_data['MAXLTOL'], max_angle_tol = set_data['MAXAGTOL'])
        #matching data
        self.unique_matches,\
        self.equivalent_matches,\
        self.unique_matches_indices_data,\
        self.equivalent_matches_indices_data,\
        self.areas = interface_searching(substrate_conv, film_conv, sub_analyzer)
        self.film = film_conv.get_primitive_structure()
        self.substrate = substrate_conv.get_primitive_structure()
    def dump_pickle(self):
        with open("unique_matches.pkl", "wb") as f:
            pickle.dump(self.unique_matches, f)
            
        with open("equivalent_matches.pkl", "wb") as f:
            pickle.dump(self.equivalent_matches, f)
            
        with open("unique_matches_indices_data.pkl", "wb") as f:
            pickle.dump(self.unique_matches_indices_data, f)
            
        with open("equivalent_matches_indices_data.pkl", "wb") as f:
            pickle.dump(self.equivalent_matches_indices_data, f)
            
        with open("areas.pkl", "wb") as f:
            pickle.dump(self.areas, f)

        with open("film.pkl", "wb") as f:
            pickle.dump(self.film, f)

        with open("substrate.pkl", "wb") as f:
            pickle.dump(self.substrate, f)

class WorkPatcher:
    def __init__(self, unique_matches, soap_data, film, substrate):
        self.unique_matches = unique_matches
        self.soap_data = soap_data
        self.film = film
        self.substrate = substrate
        
    @classmethod
    def from_dir(cls, dir):
        with open(f"{dir}/unique_matches.pkl", "rb") as f:
            unique_matches = pickle.load(f)
        film = Structure.from_file('FLM.cif').get_primitive_structure()
        substrate = Structure.from_file('SBS.cif').get_primitive_structure()
        return cls(unique_matches, None, film, substrate)
        
    def param_parse(self, project_name, termination_ftol, slab_length, c_periodic = False, vacuum_over_film = 0.01, \
                 kernel_factors = {'soap':1, 'rp':1, 'en':1}, en_cut = 1):
        self.project_name = project_name
        self.termination_ftol = termination_ftol
        self.slab_length = slab_length
        self.c_periodic = c_periodic
        self.vacuum_over_film = vacuum_over_film
        self.kernel_factors = kernel_factors
        self.en_cut = en_cut
        #self.get_all_unique_terminations()
        
    def get_unique_terminations(self, id):
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[id].film_miller,
                               substrate_miller=self.unique_matches[id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[id]]
        unique_term_ids = get_non_identical_slab_pairs(self.film, self.substrate, self.unique_matches[id], ftol = self.termination_ftol ,c_periodic = self.c_periodic)[0]
        #print(unique_term_ids)
        #print(cib.terminations)
        self.unique_terminations = [cib.terminations[i] for i in unique_term_ids]
        return self.unique_terminations
    
    def get_all_unique_terminations(self):
        all_unique_terminations = []
        for i in range(len(self.unique_matches)):
            all_unique_terminations.append(self.get_unique_terminations(i))
        self.all_unique_terminations = all_unique_terminations
        return all_unique_terminations
    
    def score_interfaces(self, match_id, termination_id, xyzs):
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[match_id].film_miller,
                               substrate_miller=self.unique_matches[match_id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[match_id]]
        itopt = interface_pre_optimizer(cib = cib, \
                                    termination = self.all_unique_terminations[match_id][termination_id], \
                                   soap_data = self.soap_data,\
                                    c_periodic = self.c_periodic, slab_length = self.slab_length, \
                                    vacuum_over_film = self.vacuum_over_film, \
                                    kernel_factors = self.kernel_factors, en_cut = self.en_cut)
        scores = []
        for i in xyzs:
            scores.append(1 - itopt.trial(i))
        return scores
    
    def get_interface_by_key(self, match_id, termination_id, xyz):
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[match_id].film_miller,
                               substrate_miller=self.unique_matches[match_id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[match_id]]
        return get_one_interface(cib, self.all_unique_terminations[match_id][termination_id], self.slab_length, xyz, \
                self.vacuum_over_film, self.c_periodic)
    
    def PatchRegistrationScan(self, match_id, termination_id, atom_non_closer_than, n_calls = 50, \
                                rbt_non_closer_than = 0.5, NCORE = 12, db_file = '', vasp_cmd = ''):
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[match_id].film_miller,
                               substrate_miller=self.unique_matches[match_id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[match_id]]
        self.cib = cib
        
        interface = list(cib.get_interfaces(termination = self.unique_terminations[termination_id], \
                                           substrate_thickness = self.slab_length, \
                                           film_thickness = self.slab_length, \
                                           vacuum_over_film = self.vacuum_over_film, \
                                           gap = 1.5, \
                                           in_layers=False))[0]
                                           
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = dot(interface.lattice.matrix.T, CNID)
        
        num_of_sampled = 1
        rbt_carts = [[0,0,2]]
        xyzs = [[0,0,2]]
        while num_of_sampled < n_calls:
            x,y,z = [random.random() for i in range(3)]
            z = z * 3
            cart_here = x*CNID_cart[:,0] + y*CNID_cart[:,1] + [0,0,z]
            distwithbefore = norm(repeat([cart_here], num_of_sampled, axis = 0) - rbt_carts, axis = 1)
            if min(distwithbefore) > rbt_non_closer_than:
                interface_here = get_one_interface(cib, self.unique_terminations[termination_id], self.slab_length, [x,y,z], \
                self.vacuum_over_film, self.c_periodic)
                
                ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max = get_it_core_indices(interface_here)
                if self.c_periodic:
                    term_atom_ids = concatenate((ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max))
                else:
                    term_atom_ids = concatenate((ids_film_min, ids_substrate_max))
                    
                existing_too_close_sites = False
                for i in term_atom_ids:
                    if get_min_nb_distance(i, interface_here) < atom_non_closer_than:
                        existing_too_close_sites = True
                        break
                if not existing_too_close_sites:
                    #interface_here.to_file(f'op_its/{num_of_sampled}_POSCAR')
                    rbt_carts.append(cart_here)
                    xyzs.append([x,y,z])
                    num_of_sampled += 1
        
        savetxt(f'{match_id}_{termination_id}_xyzs', xyzs)
        savetxt(f'{match_id}_{termination_id}_xyzs_carts', rbt_carts)
        return  RegistrationScan(self.cib, f'{match_id}_{termination_id}', xyzs, self.unique_terminations[termination_id], self.slab_length, self.vacuum_over_film, self.c_periodic, NCORE, db_file, vasp_cmd)

def unique_no_sort(array, axis):
    uniq, index = unique(array, return_index=True, axis = axis)
    return uniq[index.argsort()]

def cls_by_allclose(_multi, _unique):
    cl_ids = []
    for i in _multi:
        for j in range(len(_unique)):
            if allclose(i,_unique[j]):
                cl_ids.append(j)
                break
    return array(cl_ids)

def unique_no_sort(array, axis):
    uniq, index = unique(array, return_index=True, axis = axis)
    return uniq[index.argsort()]

def cls_by_allclose(_multi, _unique):
    cl_ids = []
    for i in _multi:
        for j in range(len(_unique)):
            if allclose(i,_unique[j]):
                cl_ids.append(j)
                break
    return array(cl_ids)

class interface_score_ranker:
    def __init__(self, IDG, soap_data, substrate, film):
        self.unique_matches_indices_data = IDG.unique_matches_indices_data
        self.unique_matches = IDG.unique_matches
        self.equivalent_matches_indices_data = IDG.equivalent_matches_indices_data
        self.areas = IDG.areas
        self.substrate = substrate
        self.film = film
        self.opt_info_dict = {}
        self.soap_data = soap_data
        
    def parse_opt_params(self, c_periodic = False, vacuum_over_film = 10, slab_length = 10, \
                 termination_ftol = 0.15, opt_num = 20, ct_ratio = 0.8, kernel_factors = {'soap':1, 'rp':1, 'en':1}, en_cut = 1):
        self.c_periodic = c_periodic
        self.vacuum_over_film = vacuum_over_film
        self.slab_length = slab_length
        self.termination_ftol = termination_ftol
        self.opt_num = opt_num
        self.ct_ratio = ct_ratio
        self.kernel_factors = {'soap':1, 'rp':1, 'en':1}
        self.get_match_term_idx()
        self.en_cut = en_cut

    def registration_optimizing(self, cib, termination, site_data_name):
        #registration by in cartesian coordinates
        itopt = interface_pre_optimizer(cib = cib, \
                                    termination = termination, \
                                   soap_data = self.soap_data,\
                                    c_periodic = self.c_periodic, slab_length = self.slab_length, \
                                    vacuum_over_film = self.vacuum_over_film, \
                                    kernel_factors = self.kernel_factors, en_cut = self.en_cut)
                    
        result = registration_minimizer(itopt, self.opt_num)
        with open(f'{site_data_name[0]}_{site_data_name[1]}.pkl', 'wb') as f:
            pickle.dump(itopt.op_data, f)
        cart_CNID = dot(itopt.interface_initial.lattice.matrix.T, itopt.CNID)
        cart_3D = dot(cart_CNID, array(result.x_iters)[:,:2].T).T
        cart_3D[:,2] = array(result.x_iters)[:,2]

        #selecting good regristrations and ranking
        values = 1 - array(result.func_vals)
        ct_value = min(values) + self.ct_ratio * (max(values) - min(values))
        selected_xs = array(result.x_iters)[values > ct_value]
        selected_cart_3Ds = cart_3D[values > ct_value]
        selected_values = values[values > ct_value]
        selected_xs = selected_xs[argsort(1-selected_values)]
        selected_cart_3Ds = selected_cart_3Ds[argsort(1-selected_values)]
        selected_values = selected_values[argsort(1-selected_values)]
        return selected_xs, selected_cart_3Ds, selected_values
    
    def get_match_term_idx(self):
        self.match_term_pairs = []
        for i in range(len(self.unique_matches)):
            unique_termination_ids = get_non_identical_slab_pairs(self.film, self.substrate, \
                                                                  self.unique_matches[i], ftol = self.termination_ftol)[0]
            for j in unique_termination_ids:
                self.match_term_pairs.append((i, j))
    
    def global_searching(self):
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            for i in range(len(self.unique_matches)):
                cib = CoherentInterfaceBuilder(film_structure=self.film,
                                       substrate_structure=self.substrate,
                                       film_miller=self.unique_matches[i].film_miller,
                                       substrate_miller=self.unique_matches[i].substrate_miller,
                                       zslgen=SubstrateAnalyzer(max_area=100),termination_ftol=self.termination_ftol,
                                       label_index=True,
                                       filter_out_sym_slabs=False)
    
                unique_termination_ids = get_non_identical_slab_pairs(self.film, self.substrate, \
                                                                      self.unique_matches[i], ftol = self.termination_ftol)[0]
                
                with tqdm(total = len(unique_termination_ids), desc = "unique terminations") as term_pbar:
                    unique_terminations = array(cib.terminations)[unique_termination_ids]
                    for j in range(len(unique_terminations)):
                        t_here = tuple(unique_terminations[j])
                        cib.zsl_matches = [self.unique_matches[i]]
                        self.interface = list(cib.get_interfaces(termination= t_here, \
                                                           substrate_thickness = self.slab_length, \
                                                           film_thickness=self.slab_length, \
                                                           vacuum_over_film=1, \
                                                           gap=1, \
                                                           in_layers=False))[0]
                        selected_xs, selected_cart_3Ds, selected_values = self.registration_optimizing(cib, t_here, (i,j))
                        term_pbar.update(1)
                        self.opt_info_dict[(i,j)] = {}
                        self.opt_info_dict[(i,j)]['termination'] = t_here
                        self.opt_info_dict[(i,j)]['registration_input'] = selected_xs
                        self.opt_info_dict[(i,j)]['registration_cart'] = selected_cart_3Ds
                        self.opt_info_dict[(i,j)]['score'] = selected_values
                        self.opt_info_dict[(i,j)]['best_registration'] = selected_xs[selected_values.argmax()]
                        self.opt_info_dict[(i,j)]['best_score'] = selected_values.max()
                        self.opt_info_dict[(i,j)]['atom_num'] = len(self.interface)
                match_pbar.update(1)
                
    def group_rank(self):
        #score array
        scores = []
        keys = self.opt_info_dict.keys()
        for i in keys:
            scores.append(self.opt_info_dict[i]['best_score'])
        scores = array(scores)
        self.scores = scores
        n_total = len(scores)
        out_ids = arange(n_total)
        #unique sub_millers
        sub_millers = array([self.unique_matches_indices_data[i[0]]['substrate_conventional_miller'] for i in keys])
        film_millers = array([self.unique_matches_indices_data[i[0]]['film_conventional_miller'] for i in keys])
        pairs = column_stack((sub_millers, film_millers))

        unique_pairs = unique_no_sort(pairs, axis = 0)
        unique_sub_millers = unique_no_sort(sub_millers, axis=0)
        unique_film_millers = unique_no_sort(film_millers, axis=0)

        #cluster by unique_sub_millers & unique_film_millers
        unique_pair_cl_ids = cls_by_allclose(pairs, unique_pairs)
        sub_miller_cl_ids = cls_by_allclose(sub_millers, unique_sub_millers)
        film_miller_cl_ids = cls_by_allclose(film_millers, unique_film_millers)
        
        #extract unique pairs
        filtered_ids = []
        pair_nums = []
        unique_pair_nums = []
        unique_match_arrays = array(list(self.opt_info_dict.keys()))[:,0]
        for i in range(len(unique_pairs)):
            ids_this_pair  = out_ids[unique_pair_cl_ids == i]
            included_UMA = unique_match_arrays[ids_this_pair]
            unique_pair_nums.append(len(unique(included_UMA)))
            pair_nums.append(len(ids_this_pair))
            scores_this_pair = scores[ids_this_pair]
            filtered_ids.append(ids_this_pair[scores_this_pair.argmax()])
        filtered_ids = array(filtered_ids)
        filtered_scores = scores[filtered_ids]
        filtered_sub_miller_cl_ids = sub_miller_cl_ids[filtered_ids]
        filtered_film_miller_cl_ids = film_miller_cl_ids[filtered_ids]
        
        unique_sub_miller_ids = arange(len(unique_sub_millers))

        groups = []
        group_max_scores = []
        group_termination_nums = []
        group_unique_match_nums = []
        unique_pair_count = 0
        
        for i in unique_sub_miller_ids:
            group_ids = filtered_ids[filtered_sub_miller_cl_ids == i]
            group_scores = scores[group_ids]
            group_ids = group_ids[argsort(1 - group_scores)]
            group_scores = group_scores[argsort(1 - group_scores)]
            groups.append(group_ids)
            group_max_scores.append(max(group_scores))
            group_termination_nums.append([])
            group_unique_match_nums.append([])
            for j in group_ids:
                group_termination_nums[i].append(pair_nums[unique_pair_count])
                group_unique_match_nums[i].append(unique_pair_nums[unique_pair_count])
                unique_pair_count += 1
        groups = sort_list(groups, 1 - array(group_max_scores))
        group_termination_nums = sort_list(group_termination_nums, 1 - array(group_max_scores))
        group_unique_match_nums = sort_list(group_unique_match_nums, 1 - array(group_max_scores))
        group_max_scores = sort_list(group_max_scores, 1 - array(group_max_scores))
        return groups, group_termination_nums, group_unique_match_nums, group_max_scores, argsort(1 - scores), sub_millers, film_millers

    def get_group_rank_info(self):
        group_columns = [r'$h_s$',r'$k_s$',r'$l_s$',r'$h_f$',r'$k_f$',r'$l_f$',r'$A^*$',r'$\epsilon^*$',r'$S^*$', r'$N_t$', r'$N_m$', r'id']
        group_data = pd.DataFrame(columns=group_columns)
        
        groups, group_termination_nums, group_unique_match_nums, group_max_scores, global_rank_ids, sub_millers, film_millers = self.group_rank()
        keys = array(list(self.opt_info_dict.keys()))
        for i in range(len(groups)):
            for j in range(len(groups[i])):
                id = groups[i][j]
                tnum = group_termination_nums[i][j]
                mnum = group_unique_match_nums[i][j]
                score = group_max_scores[i]
                A = get_area_match(self.unique_matches[keys[id][0]])
                strain = self.unique_matches[keys[id][0]].von_mises_strain
                s_i, s_j, s_k = sub_millers[id]
                f_i, f_j, f_k = film_millers[id]
                
                new_row = pd.DataFrame([[s_i, s_j, s_k, f_i, f_j, f_k, A, strain, score, tnum, mnum, id]], columns = group_columns)
                           
                if len(group_data) == 0:
                    group_data = new_row
                else:
                    group_data = pd.concat([group_data, new_row],ignore_index=True)
        group_data.index = group_data.index + 1
        return group_data

    def get_global_rank_info(self):
        groups, group_termination_nums, group_unique_match_nums, group_max_scores, global_rank_ids, sub_millers, film_millers = self.group_rank()
        columns = [r'$h_s$',r'$k_s$',r'$l_s$',r'$h_f$',r'$k_f$',r'$l_f$', r'$A$', r'$\epsilon$', r'$S$',
                     
                   
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$', r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$',\
                     r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$', r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$', r'TM', r'id']
        data = pd.DataFrame(columns=columns)
        keys = array(list(self.opt_info_dict.keys()))
        for i in global_rank_ids:
            term = self.opt_info_dict[(keys[i][0], keys[i][1])]['termination']
            s_i, s_j, s_k = sub_millers[i]
            
            f_i, f_j, f_k = film_millers[i]
            A = get_area_match(self.unique_matches[keys[i][0]])
            strain = self.unique_matches[keys[i][0]].von_mises_strain
            S = self.scores[i]
            s_vi1, s_vj1, s_vk1 = self.unique_matches_indices_data[keys[i][0]]['substrate_conventional_vectors'][0]
            s_vi2, s_vj2, s_vk2 = self.unique_matches_indices_data[keys[i][0]]['substrate_conventional_vectors'][1]
            
            f_vi1, f_vj1, f_vk1 = self.unique_matches_indices_data[keys[i][0]]['film_conventional_vectors'][0]
            f_vi2, f_vj2, f_vk2 = self.unique_matches_indices_data[keys[i][0]]['film_conventional_vectors'][1]
            
            new_row = pd.DataFrame([[s_i, s_j, s_k,
                                 f_i, f_j, f_k,
                                 A, strain, S,
                                 s_vi1, s_vj1, s_vk1,
                                 s_vi2, s_vj2, s_vk2,
                                 f_vi1, f_vj1, f_vk1,
                                 f_vi2, f_vj2, f_vk2, term, i]], columns = columns)
            if len(data) == 0:
                data = new_row
            else:
                data = pd.concat([data, new_row], ignore_index=True)
        data.index = data.index + 1
        return data
    
    def get_interface_by_id(self, id, optimized = True, x = 0, y = 0, z = 2):
        if optimized:
            key_arr = list(tuple(self.opt_info_dict.keys()))
            x, y, z = self.opt_info_dict[key_arr[id]]['best_registration']
        else:
            key_arr = self.match_term_pairs
            x, y, z = x, y, z
        if self.c_periodic:
            gap = vacuum_over_film = z
        else:
            gap = z
            vacuum_over_film = self.vacuum_over_film
        
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                                       substrate_structure=self.substrate,
                                       film_miller=self.unique_matches[key_arr[id][0]].film_miller,
                                       substrate_miller=self.unique_matches[key_arr[id][0]].substrate_miller,
                                       zslgen=SubstrateAnalyzer(max_area=200),termination_ftol=self.termination_ftol,
                                       label_index=True,
                                       filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[key_arr[id][0]]]
        if optimized:
            interface_here = list(cib.get_interfaces(termination= self.opt_info_dict[key_arr[id]]['termination'], \
                                       substrate_thickness = self.slab_length, \
                                       film_thickness=self.slab_length, \
                                       vacuum_over_film=vacuum_over_film, \
                                       gap=gap, \
                                       in_layers=False))[0]
            CNID = calculate_cnid_in_supercell(interface_here)[0]
            CNID_translation = TranslateSitesTransformation(interface_here.film_indices, x*CNID[:,0] + y*CNID[:,1])
            return CNID_translation.apply_transformation(interface_here)
            
        else:
            interface_here = list(cib.get_interfaces(termination= cib.terminations[key_arr[id][1]], \
                                       substrate_thickness = self.slab_length, \
                                       film_thickness=self.slab_length, \
                                       vacuum_over_film=vacuum_over_film, \
                                       gap=gap, \
                                       in_layers=False))[0]
            return interface_here

    def PatchHighThroughputWF(self, delta_score, project_name, NCORE, db_file, vasp_cmd):
        selected_its = []
        maxS = self.get_global_rank_info()['$S$'].to_numpy().max()
        for i in list(self.opt_info_dict.keys()):
            here_xyzs = self.opt_info_dict[i]['registration_input']
            here_xyz_carts = self.opt_info_dict[i]['registration_cart']
            here_Ss = self.opt_info_dict[i]['score']
            selected_xyzs = here_xyzs[here_Ss >= maxS - delta_score]
            selecte_xyz_carts = here_xyz_carts[here_Ss >= maxS - delta_score]
            selected_Ss = here_Ss[here_Ss >= maxS - delta_score]
            num_of_sampled = 0
            rbt_carts = []
            if len(selected_xyzs) > 0:
                for j in range(len(selected_xyzs)):
                    if num_of_sampled == 0:
                        selected_its.append([i, selected_xyzs[j], self.opt_info_dict[i]['termination'], selected_Ss[j]])
                        rbt_carts.append(here_xyz_carts[j])
                        num_of_sampled += 1
                    else:
                        cart_here = here_xyz_carts[j]
                        distwithbefore = norm(repeat([cart_here], num_of_sampled, axis = 0) - rbt_carts, axis = 1)
                        if min(distwithbefore) > 1:
                            selected_its.append([i, selected_xyzs[j], self.opt_info_dict[i]['termination'], selected_Ss[j]])
                            rbt_carts.append(here_xyz_carts[j])
                            num_of_sampled += 1
        return ScoreRankerWF(self, selected_its, project_name, NCORE, db_file, vasp_cmd)
        """
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = dot(interface.lattice.matrix.T, CNID)
        
        num_of_sampled = 1
        rbt_carts = [[0,0,2]]
        xyzs = [[0,0,2]]
        while num_of_sampled < n_calls:
            x,y,z = [random.random() for i in range(3)]
            z = z * 3
            cart_here = x*CNID_cart[:,0] + y*CNID_cart[:,1] + [0,0,z]
            distwithbefore = norm(repeat([cart_here], num_of_sampled, axis = 0) - rbt_carts, axis = 1)
            if min(distwithbefore) > rbt_non_closer_than:
        """
    def PatchAllMatchTermOPWF(self, project_name, NCORE, db_file, vasp_cmd):
        df = self.get_global_rank_info()
        its = []
        keys = array(list(self.opt_info_dict.keys()))
        for i in range(len(keys)):
            its.append(self.get_interface_by_id(i))
        return AllMatchTermOPWF(self, its, df, keys, project_name, NCORE, db_file, vasp_cmd)
        
def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def calculate_correlation(scores, energies):
    low_ids = where(energies < 1/2 * (min(energies)+ max(energies)))[0]
    sp_correlation_all = spearmanr(scores[low_ids], energies[low_ids]).correlation
    #ps_correlation_all, _ = pearsonr(scores[low_ids], energies[low_ids])
    #n = len(scores)
    #highest_20_ids = argsort(energies)[arange(int(n/2))]
    #scores_high = scores[highest_20_ids]
    #energies_high_score = energies[highest_20_ids]
    #sp_correlation_high = spearmanr(scores_high, energies_high_score).correlation
    #entropy_s = normalized_entropy_continuous(scores)
    correlation = sp_correlation_all
    #correlation = 0.3*sp_correlation_all + 0.7*ps_correlation_all
    if math.isnan(correlation):
        correlation = 1
    #print(f"cor: {correlation}, entropy: {entropy_s}, L: {correlation - 0.3 * entropy_s}")
    print(f"cor: {correlation}")
    #return correlation - 0.3 * entropy_s
    return correlation

def normalized_entropy_continuous(y, num_bins=100):
    counts, _ = np.histogram(y, bins=num_bins, range=(0, 1), density=True)
    probabilities = counts / np.sum(counts)
    
    probabilities = probabilities[probabilities > 0]

    H = -np.sum(probabilities * np.log2(probabilities))
    
    H_max = np.log2(num_bins)

    H_norm = H / H_max
    return H_norm

class HPtrainer:
    def __init__(self, substrate_conv, film_conv, sub_analyzer, DFT_results, slab_length = 5, termination_ftol = 0.1, \
                 vacuum_over_film = 10, c_period = False, structure_from_MP = True, training_y = 'binding_energies'):
        self.substrate_conv = substrate_conv
        self.film_conv = film_conv
        self.sub_analyzer = sub_analyzer
        self.DFT_results = DFT_results
        self.structure_from_MP = structure_from_MP
        self.termination_ftol = termination_ftol
        self.slab_length = slab_length
        self.c_period = c_period
        self.vacuum_over_film = vacuum_over_film
        self.film = film_conv.get_primitive_structure()
        self.substrate = substrate_conv.get_primitive_structure()
        self.soap_data = soap_data_generator.from_dir()
        self.soap_data.calculate_soaps(None, True)
        self.wp_initial = WorkPatcher.from_dir('.')
        self.wp_initial.param_parse(project_name = 'THP', termination_ftol = self.termination_ftol, slab_length = self.slab_length, \
                       c_periodic = self.c_period, vacuum_over_film = self.vacuum_over_film)
        self.all_unique_terminations = self.wp_initial.get_all_unique_terminations()
        self.training_y = training_y

    def trial(self, params):
        rcut, n_max, l_max, soapWr0, soapWc, soapWd, soapWm, KFsoap, KFrp, KFen, en_cut = params
        soap_params = {'r_cut':rcut, 'n_max':n_max, 'l_max':l_max, 'weighting':{"function":"pow", "r0":soapWr0, "c":soapWc, "d":soapWd, "m":soapWm}}
        self.soap_data.calculate_soaps(soap_params)
        wp = WorkPatcher(self.wp_initial.unique_matches, self.soap_data, self.film, self.substrate)
        wp.param_parse(project_name = 'THP', termination_ftol = self.termination_ftol, slab_length = self.slab_length, \
                       c_periodic = self.c_period, vacuum_over_film = self.vacuum_over_film, kernel_factors = {'soap':KFsoap, 'rp':KFrp, 'en':KFen}, en_cut = en_cut)
        wp.all_unique_terminations = self.all_unique_terminations
        print(params)
        if self.training_y == 'binding_energies':
            scores = []
            energies = []
            for i in self.DFT_results.keys():
                scores += wp.score_interfaces(i[0], i[1], self.DFT_results[i]['xyzs'])
                energies += list(self.DFT_results[i][self.training_y])
            scores = array(scores)
            energies = array(energies)
            return calculate_correlation(scores, energies)
        else:
            key_list = list(self.DFT_results.keys())
            correlations = []
            weightings = []
            for k in key_list:
                scores = wp.score_interfaces(k[0], k[1], self.DFT_results[k]['xyzs'])
                energies = self.DFT_results[k][self.training_y]
                num_this_match = len(scores)
                scores = array(scores)
                energies = array(energies)
                weightings.append(num_this_match)
                correlations.append(calculate_correlation(scores, energies))
            weightings, correlations = array(weightings), array(correlations)
            ave_corr = sum(weightings * correlations) / sum(weightings)
            print(ave_corr)
            return ave_corr
        

def HPoptimizer(hptrainer, n_calls):
    def trial_with_progress(func, n_calls, *args, **kwargs):
        with tqdm(total = n_calls, desc = "registration optimizing") as rgst_pbar:  # Initialize tqdm with total number of iterations
            def wrapped_func(*args, **kwargs):
                result = func(*args, **kwargs)
                rgst_pbar.update(1)  # Update progress bar by 1 after each function call
                return result
            return gp_minimize(wrapped_func, search_space, n_calls=n_calls, *args, **kwargs)
    search_space = [
        Real(5, 10, name='rcut'),
        Integer(2, 15, name='n_max'),
        Integer(2, 15, name='l_max'),
        Real(1, 5, name = 'soapWr0'),
        Real(0.5, 2, name = 'soapWc'),
        Real(0.5, 2, name = 'soapWd'),
        Integer(2,50, name = 'soapWm'),
        Real(0.5, 2, name = 'KFsoap'),
        Real(0, 0.1, name = 'KFrp'),
        Real(0.99, 1, name = 'KFen'),
        Real(1+1e-4, 1.5, name = 'en_cut'),
    ]
    # Run the optimization with progress bar
    result = trial_with_progress(hptrainer.trial, n_calls=n_calls, random_state=99)
    return result
