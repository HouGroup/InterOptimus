"""
InterOptimus Interface Worker Module

This module contains core functionality for interface analysis and optimization,
including gradient descent algorithms, interface structure generation,
and scoring functions for interface quality assessment.
"""

from InterOptimus.matching import interface_searching, EquiMatchSorter
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.tool import apply_cnid_rbt, sort_list, get_it_core_indices, get_min_nb_distance, cut_vaccum, add_sele_dyn_slab, add_sele_dyn_it, get_non_strained_film, get_rot_strain, trans_to_bottom, get_non_matching_structures, convert_dict_to_json
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
from numpy import array, dot, column_stack, argsort, zeros, mod, mean, ceil, concatenate, random, repeat, cross, inf, round, arccos, pi, where, unique
from numpy.linalg import norm
from InterOptimus.CNID import calculate_cnid_in_supercell
import os
from pathlib import Path
import pandas as pd
import json
import pickle
import warnings
import shutil
from interfacemaster.cellcalc import get_normal_from_MI, get_primitive_hkl


_DEFAULT_RELAX_INCAR_SETTINGS = {
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "EDIFF": 1e-4,
    "EDIFFG": -0.05,
    # Override MPRelaxSet.yaml (ALGO=FAST); VASP tag is case-insensitive.
    "ALGO": "Normal",
    # None removes MPRelaxSet PREC so it is not written (VASP built-in default).
    "PREC": None,
}

_DEFAULT_STATIC_INCAR_SETTINGS = {
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "EDIFF": 1e-4,
    "ALGO": "Normal",
    "PREC": None,
}

_DOUBLE_INTERFACE_LATTICE_CONSTRAINTS = ".FALSE. .FALSE. .TRUE."


def _merged_vasp_incar_settings(defaults, user_settings=None, *, extra_settings=None, num_atoms=None):
    """Merge stable default INCAR settings with user overrides without mutating inputs."""
    out = dict(defaults or {})
    if isinstance(user_settings, dict):
        out.update(user_settings)
    if isinstance(extra_settings, dict):
        out.update(extra_settings)
    if "EDIFF_PER_ATOM" in out:
        if num_atoms is not None:
            out["EDIFF"] = out["EDIFF_PER_ATOM"] * num_atoms
        del out["EDIFF_PER_ATOM"]
    return out


def default_relax_incar_settings(user_settings=None, *, extra_settings=None, num_atoms=None):
    """Default stable INCAR baseline for VASP relax jobs in InterOptimus."""
    return _merged_vasp_incar_settings(
        _DEFAULT_RELAX_INCAR_SETTINGS,
        user_settings,
        extra_settings=extra_settings,
        num_atoms=num_atoms,
    )


def default_static_incar_settings(user_settings=None, *, num_atoms=None):
    """Default stable INCAR baseline for VASP static jobs in InterOptimus."""
    return _merged_vasp_incar_settings(
        _DEFAULT_STATIC_INCAR_SETTINGS,
        user_settings,
        num_atoms=num_atoms,
    )


def interoptimus_vasp_run_kwargs() -> dict:
    """
    Kwargs for atomate2 ``BaseVaspMaker.run_vasp_kwargs``.

    Omits Custodian's ``LargeSigmaHandler`` so ``SIGMA`` is never auto-rescaled
    during a run (consistent smearing across all InterOptimus VASP jobs).
    """
    from atomate2.vasp.run import DEFAULT_HANDLERS
    from custodian.vasp.handlers import LargeSigmaHandler

    handlers = tuple(h for h in DEFAULT_HANDLERS if not isinstance(h, LargeSigmaHandler))
    return {"handlers": handlers}


def default_double_interface_relax_extra_settings():
    """Constrain double-interface relaxations so only the c-axis length is free."""
    return {"LATTICE_CONSTRAINTS": _DOUBLE_INTERFACE_LATTICE_CONSTRAINTS}


def effective_strain_E_correction(strain_E_correction: bool, *, double_interface: bool) -> bool:
    """
    Only enable elastic strain correction for single-interface workflows.

    For double-interface models, the current interface-energy expression already uses
    bulk reference energies directly and applying the legacy film-only strain correction
    is not considered reliable. Therefore it is disabled unconditionally here.
    """
    return bool(strain_E_correction) and not bool(double_interface)


def gradient_descend(sampling_function, dx, dim, tol, initial_r, initial_xy, min_steps, **kwargs):
    """
    Perform gradient descent optimization in multi-dimensional space.

    Uses finite difference method to compute gradients and adaptive step size
    for optimization. Continues until either tolerance is met or minimum steps
    are completed.

    Args:
        sampling_function: Function to evaluate at each point
        dx (float): Step size for finite difference gradient calculation
        dim (int): Dimensionality of the optimization space
        tol (float): Tolerance for convergence (change in function value)
        initial_r (float): Initial step size for gradient descent
        initial_xy (tuple): Initial point (x, y) to start optimization from
        min_steps (int): Minimum number of optimization steps to perform
        **kwargs: Additional arguments passed to sampling_function

    Returns:
        tuple: (xs, ys, rs) where:
            - xs: List of all x positions during optimization
            - ys: List of function values at each position
            - rs: List of step sizes used at each iteration
    """
    dy = inf
    g_n = zeros(dim)
    xs = []
    ys = []
    rs = []
    if initial_xy is not None:
        x_n, y_n = initial_xy
        xs.append(x_n)
        ys.append(y_n)
        rs.append(0)
    else:
        x_n = zeros(dim)
        y_n = sampling_function(x_n, is_x=True, **kwargs)
        xs.append(x_n)
        ys.append(y_n)
        rs.append(0)
    
    count = 0
    print('-----gradient descend------')
    while abs(dy) > tol or count < min_steps:
        g_n_1 = g_n.copy()
        for i in range(dim):
            pdx = zeros(dim)
            pdx[i] = dx
            g_n[i] = (sampling_function(x_n + pdx, **kwargs) - y_n) / dx
        
        if count == 0:
            r = initial_r
        else:
            #print(f'x_n {x_n} x_n_1 {x_n_1}')
            r = abs(dot((x_n - x_n_1), (g_n - g_n_1))) / norm(g_n - g_n_1) ** 2

        x_n_1, y_n_1 = x_n.copy(), y_n
        #print(f'dx {- r * g_n}')
        x_n += - r * g_n
        y_n = sampling_function(x_n, is_x = True, **kwargs)
        dy = y_n - y_n_1
        print(f'dy {dy} x_n {x_n[0]} {x_n[1]} {x_n[2]} g_n {g_n[0]} {g_n[1]} {g_n[2]}')
        xs.append(x_n.copy())
        ys.append(y_n)
        rs.append(r)
        count += 1
    
    return xs, ys, rs

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
            return gp_minimize(wrapped_func, search_space, n_calls = n_calls, n_random_starts = int(0.1 * n_calls), *args, **kwargs)
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

    def _global_min_debug_enabled(self):
        flag = None
        if hasattr(self, "opt_kwargs") and isinstance(self.opt_kwargs, dict):
            flag = self.opt_kwargs.get("debug_global_min")
        if flag is None:
            flag = os.environ.get("INTEROPTIMUS_DEBUG_GLOBAL_MIN", "")
        if isinstance(flag, str):
            return flag.strip().lower() not in ("", "0", "false", "no", "off")
        return bool(flag)

    def _global_min_debug(self, **payload):
        if not self._global_min_debug_enabled():
            return
        print(
            "[global_min_debug] "
            + json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
        )
    
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
    
    def parse_interface_structure_params(self, termination_ftol = 0.15, film_thickness = 15, substrate_thickness = 15, double_interface = False, vacuum_over_film = 5, charge_filter_settings = None):
        """
        parse necessary structure parameters for interface generation in the next steps

        Args:

        termination_ftol (float): tolerance of the c-fractional coordinates for termination atom clustering
        film_thickness (float): film slab thickness
        substrate_thickness (float): substrate slab thickness
        vacuum_over_film (float): vacuum over film
        charge_filter_settings (dict|bool|None): optional termination screening settings.
            When enabled, removes termination pairs with obviously unreasonable
            charge matching before MLIP energy estimation.
        """
        self.termination_ftol, self.film_thickness, self.substrate_thickness, self.double_interface, self.vacuum_over_film = \
        termination_ftol, film_thickness, substrate_thickness, double_interface, vacuum_over_film
        self.calculate_thickness()
        self.get_all_unique_terminations()
        self.charge_filter_settings = charge_filter_settings
        if charge_filter_settings is not None and charge_filter_settings is not False:
            if charge_filter_settings is True:
                charge_filter_settings = {}
            elif not isinstance(charge_filter_settings, dict):
                raise TypeError('charge_filter_settings must be None, bool, or dict')
            self.filter_terminations_by_charge_balance(**charge_filter_settings)
    
    def parse_optimization_params(self, set_relax_thicknesses = (0,0), relax_in_layers = False, relax_in_ratio = False, num_relax_bayesian = 0, discut = 0.8, BO_coord_bin_size = 0.5, BO_energy_bin_size = 0.01, BO_rms_bin_size = 0.5, do_mlip_gd = False, gd_tol = 5e-4, do_gd = None, **kwargs):
        #number of relaxing steps during BO
        self.num_relax_bayesian = num_relax_bayesian
        #during BO, structures with minimum atomic distance lower than discut will be attached a zero energy
        self.discut = discut
        
        self.set_relax_thicknesses = set_relax_thicknesses
        self.relax_in_layers = relax_in_layers
        self.opt_kwargs = kwargs
        self.BO_coord_bin_size = BO_coord_bin_size
        self.BO_energy_bin_size = BO_energy_bin_size
        self.BO_rms_bin_size = BO_rms_bin_size
        # Backward compatibility: do_gd -> do_mlip_gd
        if do_gd is not None:
            do_mlip_gd = do_gd
        self.do_gd = do_mlip_gd
        self.do_mlip_gd = do_mlip_gd
        self.gd_tol = gd_tol
        
        if self.double_interface:
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
            if self.relax_in_layers:
                fthk_film = (self.thickness_in_layers[i][0] - set_relax_thicknesses[0]) * self.layer_thicknesses[i][0] - 1e-6
                fthk_substrate = (self.thickness_in_layers[i][1] - set_relax_thicknesses[1]) * self.layer_thicknesses[i][1] - 1e-6
            else:
                if relax_in_ratio:
                    fthk_film = self.absolute_thicknesses[i][0] * (1-set_relax_thicknesses[0]) - 1e-6
                    fthk_substrate = self.absolute_thicknesses[i][1] * (1-set_relax_thicknesses[1]) - 1e-6
                else:
                    fthk_film = self.absolute_thicknesses[i][0] - set_relax_thicknesses[0] - 1e-6
                    fthk_substrate = self.absolute_thicknesses[i][1] - set_relax_thicknesses[1] - 1e-6
                
            if fthk_film <= 0:
                fthk_film = 0
                warnings.warn('match {i}: set relaxed film thicknesses exceeds the film slab length, the whole film will be relaxed')
            if fthk_substrate <= 0:
                fthk_substrate = 0
                warnings.warn('match {i}: set relaxed substrate thicknesses exceeds the film slab length, the whole film will be relaxed')
            
            if not self.double_interface:
                print(f'match {i}: fix/slab thicknesses (film, substrate) ({round(fthk_film, 2)}/{round(self.absolute_thicknesses[i][0], 2)}, {round(fthk_substrate, 2)}/{round(self.absolute_thicknesses[i][1],2)})')
            self.absolute_fix_thicknesses.append([fthk_film, fthk_substrate])

    def get_specified_match_cib(self, id, ftol_by_layer_thickness = True):
        """
        get the CoherentInterfaceBuilder instance for a specified unique match

        Args:
        id (int): unique match index
        """
        if ftol_by_layer_thickness:
            termination_ftol = self.ftol_termination_tuples[id]
            #layer_thks = self.layer_thicknesses[id]
            #layer_thks_0, layer_thks_1 = layer_thks[0] * self.termination_ftol, layer_thks[1] * self.termination_ftol
            #layer_thks = (layer_thks_0, layer_thks_1)
        else:
            termination_ftol = self.termination_ftol
        cib = CoherentInterfaceBuilder(
                                       film_structure=self.film,
                                       substrate_structure=self.substrate,
                                       film_miller=self.unique_matches[id].film_miller,
                                       substrate_miller=self.unique_matches[id].substrate_miller,
                                       zslgen=SubstrateAnalyzer(max_area=100),
                                       termination_ftol=termination_ftol,
                                       label_index=True,
                                       filter_out_sym_slabs=False,
                                       )
        cib.zsl_matches = [self.unique_matches[id]]
        return cib
    
    def get_unique_terminations(self, id):
        """
        get non-identical terminations for a specified unique match id

        Args:
        id (int): unique match index
        """
        unique_term_ids = get_non_identical_slab_pairs(self.film, self.substrate, self.unique_matches[id], \
                                                       ftol = self.ftol_termination_tuples[id], c_periodic = True)[0]
        print(f'\nmatch {id}: number of unique terminations: {len(unique_term_ids)}')
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

    def _apply_oxidation_states(self, oxidation_states):
        if oxidation_states is None:
            return
        for attr in ['film_conv', 'substrate_conv', 'film', 'substrate']:
            getattr(self, attr).add_oxidation_state_by_element(oxidation_states)

    def _ensure_oxidation_states_available(self):
        missing_info = []
        for label, structure in [('film', self.film), ('substrate', self.substrate)]:
            missing_elements = sorted({
                site.specie.symbol
                for site in structure
                if getattr(site.specie, 'oxi_state', None) is None
            })
            if len(missing_elements) > 0:
                missing_info.append(f"{label}: {', '.join(missing_elements)}")
        if len(missing_info) > 0:
            raise ValueError(
                'Charge-based termination screening requires oxidation states on both structures. '
                'Missing oxidation states for ' + '; '.join(missing_info) + '. '
                "Decorate the input structures first, or pass charge_filter_settings={'oxidation_states': {...}}."
            )

    def _get_site_charge_sign(self, site, charge_threshold):
        oxi_state = float(site.specie.oxi_state)
        if oxi_state > charge_threshold:
            return 1
        if oxi_state < -charge_threshold:
            return -1
        return 0

    def _summarize_layer_charge(self, structure, atom_ids, site_charge_threshold, layer_charge_threshold, dominant_ratio_threshold):
        atom_ids = list(atom_ids)
        oxi_states = [float(structure[i].specie.oxi_state) for i in atom_ids]
        signs = [self._get_site_charge_sign(structure[i], site_charge_threshold) for i in atom_ids]
        avg_oxi = mean(oxi_states)
        pos_ratio = sum(sign == 1 for sign in signs) / len(signs)
        neg_ratio = sum(sign == -1 for sign in signs) / len(signs)
        dominant_sign = 0
        dominant_ratio = max(pos_ratio, neg_ratio)
        if avg_oxi > layer_charge_threshold and pos_ratio >= dominant_ratio_threshold:
            dominant_sign = 1
        elif avg_oxi < -layer_charge_threshold and neg_ratio >= dominant_ratio_threshold:
            dominant_sign = -1
        return {
            'atom_ids': atom_ids,
            'avg_oxi': avg_oxi,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'dominant_sign': dominant_sign,
            'dominant_ratio': dominant_ratio,
        }

    def _get_contact_layer_pairs(self, interface):
        ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max = get_it_core_indices(interface)
        contact_pairs = [{
            'name': 'film_bottom_vs_substrate_top',
            'film_ids': list(ids_film_min),
            'substrate_ids': list(ids_substrate_max),
        }]
        if self.double_interface:
            contact_pairs.append({
                'name': 'film_top_vs_substrate_bottom',
                'film_ids': list(ids_film_max),
                'substrate_ids': list(ids_substrate_min),
            })
        return contact_pairs

    def evaluate_termination_charge_balance(self, match_id, term_id, initial_gap = 2.0,
                                            site_charge_threshold = 0.3,
                                            layer_charge_threshold = 0.3,
                                            dominant_ratio_threshold = 0.6,
                                            min_cross_interface_distance = 1.2,
                                            same_sign_pair_max_distance = 2.5,
                                            same_sign_pair_ratio_threshold = 0.6,
                                            pair_cutoff = 3.5,
                                            max_pairs = 6):
        interface = self.get_specified_interface(match_id, term_id, [0, 0, initial_gap])
        pair_reports = []
        reasons = []
        for contact in self._get_contact_layer_pairs(interface):
            film_summary = self._summarize_layer_charge(
                interface,
                contact['film_ids'],
                site_charge_threshold,
                layer_charge_threshold,
                dominant_ratio_threshold,
            )
            substrate_summary = self._summarize_layer_charge(
                interface,
                contact['substrate_ids'],
                site_charge_threshold,
                layer_charge_threshold,
                dominant_ratio_threshold,
            )

            all_pairs = []
            for film_id in contact['film_ids']:
                for substrate_id in contact['substrate_ids']:
                    all_pairs.append({
                        'film_id': int(film_id),
                        'substrate_id': int(substrate_id),
                        'distance': interface.get_distance(int(film_id), int(substrate_id)),
                        'film_sign': self._get_site_charge_sign(interface[int(film_id)], site_charge_threshold),
                        'substrate_sign': self._get_site_charge_sign(interface[int(substrate_id)], site_charge_threshold),
                    })
            all_pairs = sorted(all_pairs, key = lambda x: x['distance'])
            considered_pairs = [p for p in all_pairs if p['distance'] <= pair_cutoff][:max_pairs]
            if len(considered_pairs) == 0:
                considered_pairs = all_pairs[:max_pairs]
            repulsive_pairs = [
                p for p in considered_pairs
                if p['film_sign'] != 0 and
                p['film_sign'] == p['substrate_sign'] and
                p['distance'] <= same_sign_pair_max_distance
            ]
            min_distance = all_pairs[0]['distance'] if len(all_pairs) > 0 else inf
            repulsive_ratio = len(repulsive_pairs) / len(considered_pairs) if len(considered_pairs) > 0 else 0
            same_layer_sign = (
                film_summary['dominant_sign'] != 0 and
                film_summary['dominant_sign'] == substrate_summary['dominant_sign']
            )

            if same_layer_sign:
                layer_label = 'cation-cation' if film_summary['dominant_sign'] > 0 else 'anion-anion'
                reasons.append(f"{contact['name']}: dominant {layer_label} termination pairing")
            if min_cross_interface_distance is not None and min_distance < min_cross_interface_distance:
                reasons.append(
                    f"{contact['name']}: minimum cross-interface distance {round(min_distance, 3)} A "
                    f"is below {min_cross_interface_distance} A"
                )
            if repulsive_ratio >= same_sign_pair_ratio_threshold and len(repulsive_pairs) > 0:
                reasons.append(
                    f"{contact['name']}: {len(repulsive_pairs)}/{len(considered_pairs)} nearest pairs are "
                    f"same-sign within {same_sign_pair_max_distance} A"
                )

            pair_reports.append({
                'contact': contact['name'],
                'film_layer': film_summary,
                'substrate_layer': substrate_summary,
                'min_distance': min_distance,
                'repulsive_ratio': repulsive_ratio,
                'considered_pairs': considered_pairs,
            })

        return {
            'match_id': match_id,
            'term_id': term_id,
            'is_bad': len(reasons) > 0,
            'reasons': reasons,
            'contacts': pair_reports,
        }

    def filter_terminations_by_charge_balance(self, oxidation_states = None,
                                              initial_gap = 2.0,
                                              site_charge_threshold = 0.3,
                                              layer_charge_threshold = 0.3,
                                              dominant_ratio_threshold = 0.6,
                                              min_cross_interface_distance = 1.2,
                                              same_sign_pair_max_distance = 2.5,
                                              same_sign_pair_ratio_threshold = 0.6,
                                              pair_cutoff = 3.5,
                                              max_pairs = 6):
        """
        Screen out termination pairs with unreasonable electrostatic matching.

        This removes termination pairs before MLIP energy estimation using two
        simple heuristics:
        1. dominant contact layers on both sides have the same charge sign
        2. nearest cross-interface contacts are too short and/or dominated by
           same-sign ionic pairs
        """
        self._apply_oxidation_states(oxidation_states)
        self._ensure_oxidation_states_available()
        self.termination_charge_filter_log = {}
        total_before = 0
        total_after = 0
        for i in range(len(self.all_unique_terminations)):
            terminations_here = self.all_unique_terminations[i]
            kept_terminations = []
            kept_term_ids = []
            removed_reports = []
            total_before += len(terminations_here)
            for j in range(len(terminations_here)):
                report = self.evaluate_termination_charge_balance(
                    i,
                    j,
                    initial_gap = initial_gap,
                    site_charge_threshold = site_charge_threshold,
                    layer_charge_threshold = layer_charge_threshold,
                    dominant_ratio_threshold = dominant_ratio_threshold,
                    min_cross_interface_distance = min_cross_interface_distance,
                    same_sign_pair_max_distance = same_sign_pair_max_distance,
                    same_sign_pair_ratio_threshold = same_sign_pair_ratio_threshold,
                    pair_cutoff = pair_cutoff,
                    max_pairs = max_pairs,
                )
                if report['is_bad']:
                    removed_reports.append(report)
                else:
                    kept_terminations.append(terminations_here[j])
                    kept_term_ids.append(j)
            self.all_unique_terminations[i] = kept_terminations
            total_after += len(kept_terminations)
            self.termination_charge_filter_log[i] = {
                'num_before': len(terminations_here),
                'num_after': len(kept_terminations),
                'kept_term_ids': kept_term_ids,
                'removed': removed_reports,
            }
            print(
                f'match {i}: kept {len(kept_terminations)}/{len(terminations_here)} '
                'unique terminations after charge screening'
            )
            if len(kept_terminations) == 0:
                warnings.warn(
                    f'match {i}: all unique terminations were removed by charge screening; '
                    'this match will be skipped in global minimization'
                )
        print(f'charge screening kept {total_after}/{total_before} unique terminations')
        if total_after == 0:
            raise ValueError(
                'Charge screening removed all unique terminations. '
                'Please relax the thresholds or check the oxidation states.'
            )
    
    def calculate_thickness(self):
        self.thickness_in_layers = [] 
        self.absolute_thicknesses = []
        self.layer_thicknesses = []
        self.ftol_termination_tuples = []
        print('\n')
        for i in range(len(self.unique_matches)):
            film_l, substrate_l = self.get_film_substrate_layer_thickness(i)
            self.ftol_termination_tuples.append((film_l * self.termination_ftol, substrate_l * self.termination_ftol))
            self.layer_thicknesses.append((film_l, substrate_l))
            film_thickness = int(round(self.film_thickness/film_l))
            substrate_thickness = int(round(self.substrate_thickness/substrate_l))
            self.thickness_in_layers.append((film_thickness, substrate_thickness))
            self.absolute_thicknesses.append((film_thickness * film_l, substrate_thickness * substrate_l))
            print(f'match {i}: thicknesses (film, substrate) ({round(film_l,2)}, {round(substrate_l,2)}) ({film_thickness}, {substrate_thickness}) ({round(film_thickness * film_l, 2)} {round(substrate_thickness * substrate_l, 2)})')
    
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
        if self.double_interface:
            vacuum_over_film = z
        else:
            vacuum_over_film = self.vacuum_over_film

        cib = self.get_specified_match_cib(match_id)
        film_thickness, substrate_thickness = self.thickness_in_layers[match_id]
        interface_here = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness, film_thickness = film_thickness, \
                                       vacuum_over_film = vacuum_over_film, gap = z, in_layers = True))[0]
        interface_here = trans_to_bottom(apply_cnid_rbt(interface_here, x, y, 0))
        return interface_here
    
    def set_energy_calculator(self, calc, user_settings = None):
        """
        set energy calculator docker container
        
        Args:
        calc (str): orb-models, sevenn, matris, dpa
        """
        # NOTE: mlipdockers is deprecated in this project; always use InterOptimus.mlip.MlipCalc
        from InterOptimus.mlip import MlipCalc

        # Defensive copy & defaults
        if user_settings is None:
            user_settings = {}
        else:
            user_settings = dict(user_settings)

        if calc == "matris":
            user_settings.setdefault("model", "matris_10m_oam")
            user_settings.setdefault("task", "efsm")

        # Checkpoint resolution (INTEROPTIMUS_CHECKPOINT_DIR or ~/.cache/InterOptimus/checkpoints) is in MlipCalc.
        self.mc = MlipCalc(calc=calc, user_settings=user_settings)
    
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
        energy (float): predicted energy by mlip
        """
        x,y,z = params
        xyz = [x,y,z]

        interface_here = self.get_specified_interface(self.match_id_now, self.term_id_now, xyz = xyz)

        term_atom_ids = self.get_interface_atom_indices(interface_here)
        for i in term_atom_ids:
            if get_min_nb_distance(i, interface_here, self.discut) < self.discut:
                return 0
        #if self.num_relax_bayesian == 0:
        self.opt_results[(self.match_id_now, self.term_id_now)]['sampled_interfaces'].append(interface_here)
        return self.mc.calculate(interface_here)
        """
        else:
            fix_thickness_film, fix_thickness_substrate = self.absolute_fix_thicknesses[self.match_id_now]
            interface_here, mobility_mtx = add_sele_dyn_it(interface_here, fix_thickness_film, fix_thickness_substrate)
            interface_here_relaxed, e = self.mc.optimize(interface_here, fix_cell_booleans = self.opt_kwargs['fix_cell_booleans'], fmax = 0.05, steps = self.num_relax_bayesian)
            interface_here_relaxed.film = interface_here.film
            interface_here_relaxed.substrate = interface_here.substrate
            interface_here_relaxed.interface_properties = interface_here.interface_properties
            self.opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
            return e
        """
    def get_film_substrate_layer_thickness(self, match_id):
        """
        get single layer thickness
        """
        cib = self.get_specified_match_cib(match_id, False)
        
        delta_c = 0
        last_delta_c = 0
        initial_n = 2
        while last_delta_c == 0:
            last_delta_c = delta_c
            interface_film_1 = list(cib.get_interfaces(termination = cib.terminations[0], \
                                           substrate_thickness = 2, film_thickness = initial_n, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            interface_film_2 = list(cib.get_interfaces(termination = cib.terminations[0], \
                                           substrate_thickness = 2, film_thickness = initial_n + 5, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            delta_c = interface_film_2.lattice.c - interface_film_1.lattice.c
        film_delta_c = delta_c/5
            
        
        delta_c = 0
        last_delta_c = 0
        initial_n = 2
        while last_delta_c == 0:
            last_delta_c = delta_c
            interface_substrate_1 = list(cib.get_interfaces(termination = cib.terminations[0], \
                                           substrate_thickness = initial_n, film_thickness = 2, \
                                           vacuum_over_film = 1, gap = 1, in_layers = True))[0]
            interface_substrate_2 = list(cib.get_interfaces(termination = cib.terminations[0], \
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
        calc: MLIP calculator (str): orb-models, sevenn, matris, dpa
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
        
        self.opt_results[(match_id,term_id)]['original_xs'] = xs
        self.opt_results[(match_id,term_id)]['original_ys'] = ys
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
        xs_carts = dot(CNID_cart, xs.T).T + column_stack((zeros(len(xs)), zeros(len(xs)), xs[:,2]))
        
        self.opt_results[(match_id,term_id)]['xyzs_frac'] = xs
        self.opt_results[(match_id,term_id)]['xyzs_cart'] = xs_carts
        self.opt_results[(match_id,term_id)]['supcl_E'] = ys
        
        selected_ids = []
        selected_its = []
        
        ltc = self.opt_results[(match_id,term_id)]['sampled_interfaces'][0].lattice
        A = ltc.a * ltc.b
        
        for i in range(len(ys)):
            if i == 0:
                selected_ids.append(i)
                selected_its.append(self.opt_results[(match_id,term_id)]['sampled_interfaces'][i])
            else:
                if abs(ys[i] - ys[0])/A*16.02176634 < self.BO_energy_bin_size:
                    dz = (self.opt_results[(match_id,term_id)]['xyzs_frac'][selected_ids] - self.opt_results[(match_id,term_id)]['xyzs_frac'][i])[:,2]
                    dxdy = (self.opt_results[(match_id,term_id)]['xyzs_frac'][selected_ids] - self.opt_results[(match_id,term_id)]['xyzs_frac'][i])[:,0:2]
                    dxdy = abs(round(dxdy) - dxdy)
                    dxyz = dot(dxdy, dot(interface.lattice.matrix.T, CNID).T)
                    dxyz[:,2] = dz
                    #print(min(norm(dxyz, axis = 1)), abs(ys[i] - ys[0]))
                    if min(norm(dxyz, axis = 1)) > self.BO_coord_bin_size:
                        selected_ids.append(i)
                        selected_its.append(self.opt_results[(match_id,term_id)]['sampled_interfaces'][i])
        self.opt_results[(match_id,term_id)]['selected_its'] = selected_its
        smt = StructureMatcher(ltol = 0.01, stol = 0.5, angle_tol=0.01, primitive_cell=False, scale = True)
        selected_ids = array(selected_ids)
        selected_ids = selected_ids[get_non_matching_structures(selected_its, self.BO_rms_bin_size, smt)]
        self.opt_results[(match_id,term_id)]['BO_selected_ids'] = selected_ids
        print(f'num of selected low-energy its: {len(selected_ids)}')
    
    def get_displaced_relaxed_interface(self, displacement, interface, is_x = False):
        site_properties = interface.site_properties
        interface_properties = interface.interface_properties
        film_indices = interface.film_indices
        substrate_indices = interface.substrate_indices
        if len(self.gradient_descend_interfaces) == 0:
            #print(f'True disp: {displacement}')
            tt = TranslateSitesTransformation(interface.film_indices, displacement, False)
            interface = tt.apply_transformation(interface)
        else:
            #print(f'True disp: {displacement - self.gradient_descend_disps[-1]}')
            tt = TranslateSitesTransformation(interface.film_indices, displacement - self.gradient_descend_disps[-1], False)
            interface = tt.apply_transformation(self.gradient_descend_interfaces[-1])
        
        
        interface, e = self.mc.optimize(interface, **self.opt_kwargs)
        interface = Structure.from_dict(json.loads(  interface.to_json() ) )
        interface.interface_properties = interface_properties
        interface.film_indices, interface.substrate_indices = film_indices, substrate_indices
        if is_x:
            for k in site_properties:
                interface.add_site_property(k, site_properties[k])
            #print(f'updated x: {displacement}')
            self.gradient_descend_interfaces.append(interface)
            self.gradient_descend_disps.append(displacement.copy())
        
        return e
    
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
        relaxed_it,e = self.mc.optimize(static_it, **self.opt_kwargs)
        dis_fracs = abs(relaxed_it.frac_coords - static_it.frac_coords)
        dis_fracs = abs(dis_fracs - round(dis_fracs))
        dis_carts = dot(relaxed_it.lattice.matrix.T, dis_fracs.T).T
        return static_it, relaxed_it,  sum(norm(dis_fracs, axis = 1))/len(relaxed_it), sum(norm(dis_carts, axis = 1))/len(relaxed_it)
    
    def thickness_conv_test(self, i, j, thickness_list, n_calls_density = 1, z_range = (0,3), type = 'slab', calc = 'sevenn'):
        self.set_energy_calculator(calc, self.opt_kwargs)
        self.strain_E_correction = True
        self.thk_conv_test_results = {}
        self.film_e = self.mc.calculate(self.film)
        self.substrate_e = self.mc.calculate(self.substrate)
        for thk in thickness_list:
            if type == 'slab':
                self.film_thickness, self.substrate_thickness = thk, thk
            else:
                self.vacuum_over_film = thk
            
            _A = self.unique_matches[i].match_area
            if int(_A * n_calls_density) < 10:
                n_calls = 10
            else:
                n_calls = int(_A * n_calls_density)
            self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
            ltc = self.opt_results[(i,j)]['sampled_interfaces'][0].lattice
            A = ltc.a * ltc.b

            if self.double_interface:
                it_bd_E, strain_E = self.post_bayesian_process_double_interface(i,j,A)
            else:
                it_bd_E, strain_E = self.post_bayesian_process(i,j,A)
                
            self.thk_conv_test_results[thk] = self.opt_results[(i,j)]
        with open(f'conv_test_results.pkl','wb') as f:
            pickle.dump(self.thk_conv_test_results, f)
    
    def patch_conv_jobflow_jobs(self,
                            relax_user_incar_settings = None,
                            relax_user_potcar_settings = None,
                            relax_user_kpoints_settings = None,
                            relax_user_potcar_functional = 'PBE_54',
                            
                            static_user_incar_settings = None,
                            static_user_potcar_settings = None,
                            static_user_kpoints_settings = None,
                            static_user_potcar_functional = 'PBE_54',
                            
                            filter_name = 'my_conv_jobs',
                            do_dft_gd = False,
                            gd_kwargs = {},
                            ):
        """
        Patch JobFlow jobs, only for c_period = True
        """
        
        from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
        from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker
        from jobflow import Flow
        
        flows = []
        for num in range(2):
            structure = [self.film, self.substrate][num]
            static_incar_here = default_static_incar_settings(
                static_user_incar_settings,
                num_atoms=len(structure),
            )
            vasp_maker = StaticMaker(
                                    input_set_generator = MPStaticSet(
                                                        structure,
                                                        user_incar_settings = static_incar_here,
                                                         user_potcar_settings = static_user_potcar_settings,
                                                          user_kpoints_settings = static_user_kpoints_settings,
                                                           user_potcar_functional = static_user_potcar_functional,
                                                           ),
                                    run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                    )
            job = vasp_maker.make(structure)
            job.update_metadata({'filter_name':filter_name, 'job': ['film','substrate'][num]})
            flows.append(job)

        for thk in self.thk_conv_test_results.keys():
            if self.double_interface:
                relax_extra_settings = default_double_interface_relax_extra_settings()
            else:
                relax_extra_settings = {'ISIF':2}
            #interface here
            for it_id in range(len(self.thk_conv_test_results[thk]['relaxed_interfaces'])):
                it = self.thk_conv_test_results[thk]['relaxed_interfaces'][it_id]
                it_user_incar_settings = default_relax_incar_settings(
                    relax_user_incar_settings,
                    extra_settings=relax_extra_settings,
                    num_atoms=len(it),
                )
                vasp_maker = RelaxMaker(
                                        input_set_generator = MPRelaxSet(
                                                                        user_incar_settings = it_user_incar_settings,
                                                                        user_potcar_settings = relax_user_potcar_settings,
                                                                        user_kpoints_settings = relax_user_kpoints_settings,
                                                                        user_potcar_functional = relax_user_potcar_functional,
                                                                        ),
                                        run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                        )
                if do_dft_gd:
                    flow = GDVaspMaker(
                                        initial_it = it,
                                        film_indices = it.film_indices,
                                        metadata = {'filter_name':filter_name, 'job': f'{thk}_it_{it_id}'},
                                        relax_maker = vasp_maker,
                                        **gd_kwargs,
                                        ).make()
                    flows += flow
                else:
                    job = vasp_maker.make(it)
                    job.update_metadata({'filter_name':filter_name, 'job': f'{thk}_it_{it_id}'})
                    flows.append(job)

            #strained film
            s_film = self.thk_conv_test_results[thk]['strain_film']
            static_incar_here = default_static_incar_settings(
                static_user_incar_settings,
                num_atoms=len(s_film),
            )
            vasp_maker = StaticMaker(
                                    input_set_generator = MPStaticSet(
                                                                    s_film,
                                                                    user_incar_settings = static_incar_here,
                                                                    user_potcar_settings = static_user_potcar_settings,
                                                                    user_kpoints_settings = static_user_kpoints_settings,
                                                                    user_potcar_functional = static_user_potcar_functional
                                                                    ),
                                    run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                    )
            job = vasp_maker.make(s_film)
            job.update_metadata({'filter_name':filter_name, 'job': f'{thk}_sfilm'})
            flows.append(job)
            
            #film & substrate slab
            if not self.double_interface:
                for slab in ['film', 'substrate']:
                    slab_structure = self.thk_conv_test_results[thk]['slabs'][slab]['structure']
                    slab_incar_here = default_relax_incar_settings(
                        relax_user_incar_settings,
                        extra_settings=relax_extra_settings,
                        num_atoms=len(slab_structure),
                    )
                    vasp_maker = RelaxMaker(
                                            input_set_generator = MPRelaxSet(
                                                                            slab_structure,
                                                                            user_incar_settings = slab_incar_here,
                                                                            user_potcar_settings = relax_user_potcar_settings,
                                                                            user_kpoints_settings = relax_user_kpoints_settings,
                                                                            user_potcar_functional = relax_user_potcar_functional,
                                                                            ),
                                            run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                            )
                    job = vasp_maker.make(slab_structure)
                    job.update_metadata({'filter_name':filter_name, 'job': f'{thk}_{slab}_slab'})
                    flows.append(job)
 
        return flows
        
    def post_bayesian_process_double_interface(self, i, j, A):
        #layer thickness
        self.opt_results[(i,j)]['layer_thickness'] = self.layer_thicknesses[i]

        #fix thickness
        fthk_film, fthk_substrate = self.absolute_fix_thicknesses[i][0], self.absolute_fix_thicknesses[i][1]

        #relax best interface & slabs
        self.opt_results[(i,j)]['relaxed_best_interface'] = {}

        #get lowest-energy relaxed it
        relaxed_its, relaxed_Es = [], []
        for s_id in self.opt_results[(i,j)]['BO_selected_ids']:
            relaxed_it, relaxed_E = self.relax_with_selective_dyn_it(self.opt_results[(i,j)]['sampled_interfaces'][s_id], fthk_film, fthk_substrate)
            relaxed_its.append(relaxed_it)
            relaxed_Es.append(relaxed_E)

        relaxed_min_id = relaxed_Es.index(min(relaxed_Es))
        best_it, relaxed_best_sup_E = relaxed_its[relaxed_min_id], relaxed_Es[relaxed_min_id]
        self.opt_results[(i,j)]['relaxed_interfaces'] = relaxed_its
        self.opt_results[(i,j)]['relaxed_interface_sup_Es'] = relaxed_Es
        site_properties = self.opt_results[(i,j)]['sampled_interfaces'][0].site_properties
        for k in site_properties:
            best_it.add_site_property(k, site_properties[k])

        #interface energy
        it_E = (relaxed_best_sup_E - len(best_it.film_indices)/len(self.film) * self.film_e - len(best_it.substrate_indices)/len(self.substrate) * self.substrate_e) / A * 16.02176634 / 2
        
        #save to result dict
        self.opt_results[(i,j)]['relaxed_best_interface']['structure'] = best_it
        self.opt_results[(i,j)]['relaxed_best_interface']['e'] = relaxed_best_sup_E
        try:
            film_indices = [int(idx) for idx in best_it.film_indices]
            substrate_indices = [int(idx) for idx in best_it.substrate_indices]
            self.opt_results[(i, j)]['film_indices'] = film_indices
            self.opt_results[(i, j)]['substrate_indices'] = substrate_indices
            self.opt_results[(i, j)]['relaxed_best_interface']['film_indices'] = film_indices
            self.opt_results[(i, j)]['relaxed_best_interface']['substrate_indices'] = substrate_indices
        except Exception:
            pass
        # atom counts for reporting
        try:
            self.opt_results[(i, j)]['film_atom_count'] = len(best_it.film)
            self.opt_results[(i, j)]['substrate_atom_count'] = len(best_it.substrate)
        except Exception:
            try:
                self.opt_results[(i, j)]['film_atom_count'] = len(best_it.film_indices)
                self.opt_results[(i, j)]['substrate_atom_count'] = len(best_it.substrate_indices)
            except Exception:
                pass
        
        #energy correction by film strain energy
        if self.strain_E_correction and not self.double_interface:
            
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
            strain_E_mod = strain_E * len(best_it.film_indices) * fmns_L / self.film_thickness / A * 16.02176634 / 2
            it_E += strain_E_mod
            self.opt_results[(i,j)]['strain_film'] = strain_film
            self.opt_results[(i,j)]['strain_E'] = strain_E
            self.opt_results[(i,j)]['length_factor'] = fmns_L / self.film_thickness
            self.opt_results[(i,j)]['A'] = A
        else:
            strain_E = 0

        self.opt_results[(i,j)]['thicknesses'] = self.absolute_thicknesses[i]
        self.opt_results[(i,j)]['relaxed_min_it_E'] = it_E
        self._global_min_debug(
            stage="post_bayesian_double_interface",
            match_id=int(i),
            term_id=int(j),
            area=float(A),
            relaxed_best_sup_E=float(relaxed_best_sup_E),
            relaxed_min_it_E=float(it_E),
            strain_E=float(strain_E),
            film_atom_count=int(len(best_it.film_indices)),
            substrate_atom_count=int(len(best_it.substrate_indices)),
        )
        return it_E, strain_E
    
    def relax_with_selective_dyn_it(self, it, film_shell, substrate_shell):
        interface_properties = it.interface_properties
        film_indices, substrate_indices = it.film_indices, it.substrate_indices
        if not self.double_interface:
            it = cut_vaccum(it, self.vacuum_over_film)
            it, mblt_mtx = add_sele_dyn_it(it, film_shell, substrate_shell)
        if not self.double_interface:
            it.add_site_property('selective_dynamics', mblt_mtx)
        it, relaxed_it_sup_E = self.mc.optimize(it, **self.opt_kwargs)
        it = Structure.from_dict(json.loads(  it.to_json() ) )
        if not self.double_interface:
            it.add_site_property('selective_dynamics', mblt_mtx)
        it.interface_properties = interface_properties
        it.film_indices = film_indices
        it.substrate_indices = substrate_indices
        return it, relaxed_it_sup_E
    
    def relax_with_selective_dyn_slab(self, slab, shell, left_or_right):
        slab = cut_vaccum(slab, self.vacuum_over_film)
        slab, mblt_mtx = add_sele_dyn_slab(slab, shell, left_or_right)
        slab.add_site_property('selective_dynamics', mblt_mtx)
        slab, slab_sup_E = self.mc.optimize(slab, **self.opt_kwargs)
        slab = Structure.from_dict(json.loads(  slab.to_json() ) )
        slab.add_site_property('selective_dynamics', mblt_mtx)
        return slab, slab_sup_E
    
    def post_bayesian_process(self, i, j, A):
        #layer thickness
        self.opt_results[(i,j)]['layer_thickness'] = self.layer_thicknesses[i]
        
        #fix thickness
        fthk_film, fthk_substrate = self.absolute_fix_thicknesses[i][0], self.absolute_fix_thicknesses[i][1]
        
        #relax best interface & slabs
        self.opt_results[(i,j)]['relaxed_best_interface'] = {}

        #get lowest-energy relaxed it
        relaxed_its, relaxed_Es = [], []
        self.gradient_descend_disps, self.gradient_descend_interfaces = [], []
        self.opt_results[(i,j)]['film_indices'] = self.opt_results[(i,j)]['sampled_interfaces'][0].film_indices
        self.opt_results[(i,j)]['substrate_indices'] = self.opt_results[(i,j)]['sampled_interfaces'][0].substrate_indices
        for s_id in self.opt_results[(i,j)]['BO_selected_ids']:
            relaxed_it, relaxed_E = self.relax_with_selective_dyn_it(self.opt_results[(i,j)]['sampled_interfaces'][s_id], fthk_film, fthk_substrate)
            
            if self.do_gd:
                gd_xs, gd_Es, gd_gs = gradient_descend(sampling_function = self.get_displaced_relaxed_interface,
                                                     dx = 0.05,
                                                     dim = 3,
                                                     tol = self.gd_tol * len(relaxed_it),
                                                     initial_r = 0.1,
                                                     initial_xy = [array([0.0,0.0,0.0]), relaxed_E],
                                                     min_steps = 5,
                                                    interface = relaxed_it)
                                                    
                relaxed_its.append(self.gradient_descend_interfaces[-1])
                relaxed_Es.append(gd_Es[-1])
            else:
                relaxed_its.append(relaxed_it)
                relaxed_Es.append(relaxed_E)
            
        relaxed_min_id = relaxed_Es.index(min(relaxed_Es))
        best_it, relaxed_best_sup_E = relaxed_its[relaxed_min_id], relaxed_Es[relaxed_min_id]
        self.opt_results[(i,j)]['relaxed_interfaces'] = relaxed_its
        self.opt_results[(i,j)]['relaxed_interface_sup_Es'] = relaxed_Es

        #relax slabs
        film_slab, substrate_slab = trans_to_bottom(self.opt_results[(i,j)]['sampled_interfaces'][0].film), trans_to_bottom(self.opt_results[(i,j)]['sampled_interfaces'][0].substrate)
        film_slab, film_slab_E = self.relax_with_selective_dyn_slab(film_slab, fthk_film, 'right')
        substrate_slab, substrate_slab_E = self.relax_with_selective_dyn_slab(substrate_slab, fthk_substrate, 'left')
        
        #binding energy
        bd_E = (relaxed_best_sup_E - film_slab_E - substrate_slab_E) / A * 16.02176634
        
        #save to result dict
        self.opt_results[(i,j)]['relaxed_best_interface']['structure'] = best_it
        self.opt_results[(i,j)]['relaxed_best_interface']['e'] = relaxed_best_sup_E
        # atom counts for reporting
        try:
            self.opt_results[(i, j)]['film_atom_count'] = len(best_it.film)
            self.opt_results[(i, j)]['substrate_atom_count'] = len(best_it.substrate)
        except Exception:
            try:
                self.opt_results[(i, j)]['film_atom_count'] = len(best_it.film_indices)
                self.opt_results[(i, j)]['substrate_atom_count'] = len(best_it.substrate_indices)
            except Exception:
                pass
        
        self.opt_results[(i,j)]['slabs'] = {}
        
        self.opt_results[(i,j)]['slabs']['film'] = {}
        self.opt_results[(i,j)]['slabs']['film']['structure'] = film_slab
        self.opt_results[(i,j)]['slabs']['film']['e'] = film_slab_E
        
        self.opt_results[(i,j)]['slabs']['substrate'] = {}
        self.opt_results[(i,j)]['slabs']['substrate']['structure'] = substrate_slab
        self.opt_results[(i,j)]['slabs']['substrate']['e'] = substrate_slab_E
        
        #energy correction by film strain energy
        if self.strain_E_correction:
            
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
            strain_E_mod = strain_E * len(best_it.film_indices) * fmns_L / self.film_thickness / A * 16.02176634
            bd_E += strain_E_mod
            self.opt_results[(i,j)]['strain_film'] = strain_film
            self.opt_results[(i,j)]['strain_E'] = strain_E
            self.opt_results[(i,j)]['length_factor'] = fmns_L / self.film_thickness
            self.opt_results[(i,j)]['A'] = A
        else:
            strain_E = 0

        self.opt_results[(i,j)]['thicknesses'] = self.absolute_thicknesses[i]
        self.opt_results[(i,j)]['relaxed_min_bd_E'] = bd_E
        return bd_E, strain_E

    def global_minimization(self, n_calls_density = 4, z_range = (0.5, 3), calc = 'sevenn', strain_E_correction = False, term_screen_tol = 1, name = ''):
        """
        apply bassian optimization for the xyz registration of all the interfaces with the predicted
        interface energy by machine learning potential, getting ranked interface energies

        Args:
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc (str): MLIP calculator: orb-models, sevenn, matris, dpa
        strain_E_correction (bool): whether to correct it/cohesive energy by elastic energy
        term_screen_tol (float): tolerance to screen out terminations for structure optimization; terminations with unrelaxed energy higher than the lowest one by this value will be eliminated
        name (str): suffix for saved files
        """
        #optimization results
        self.opt_results = {}
        if self.double_interface:
            it_energetic_label = r'$E_{it}$ $(J/m^2)$'
        else:
            it_energetic_label = r'$E_{bd}$ $(J/m^2)$'
        columns = [r'$h_f$',r'$k_f$',r'$l_f$',
                  r'$h_s$',r'$k_s$',r'$l_s$',
                   r'$A$ (' + '\u00C5' + '$^2$)', r'$\epsilon$', it_energetic_label, r'$E_{el}$ $(eV/atom)$',
                   r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$',
                   r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$',
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$',
                   r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$', r'$T$', r'$i_m$', r'$i_t$']
        self.formated_data = []
        self.strain_E_correction = effective_strain_E_correction(
            strain_E_correction,
            double_interface=self.double_interface,
        )
        if self.double_interface and strain_E_correction:
            warnings.warn(
                "strain_E_correction is disabled for double-interface models; using the uncorrected "
                "double-interface energy instead.",
                stacklevel=2,
            )
        #set mlip calculator
        
        self.set_energy_calculator(calc, self.opt_kwargs)
        self.film_e = self.mc.calculate(self.film)
        self.substrate_e = self.mc.calculate(self.substrate)
        self._global_min_debug(
            stage="global_minimization_start",
            calc=calc,
            resolved_ckpt_path=(getattr(self.mc, "user_settings", {}) or {}).get("ckpt_path"),
            film_e=float(self.film_e),
            substrate_e=float(self.substrate_e),
            film_primitive_sites=int(len(self.film)),
            substrate_primitive_sites=int(len(self.substrate)),
            double_interface=bool(self.double_interface),
            strain_E_correction=bool(self.strain_E_correction),
            z_range=list(z_range),
            n_calls_density=float(n_calls_density),
        )
        #scanning matches and terminations
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            #for i in range(1):
            #lattice matching data
            m = self.unique_matches
            idt = self.unique_matches_indices_data
            for i in range(len(self.unique_matches)):
            #for i in range(1):
                hkl_f, hkl_s = idt[i]['film_conventional_miller'], idt[i]['substrate_conventional_miller']
                epsilon = m[i].von_mises_strain
                uvw_f1, uvw_f2 = idt[i]['film_conventional_vectors']
                uvw_s1, uvw_s2 = idt[i]['substrate_conventional_vectors']
                _A = m[i].match_area
                num_terms = len(self.all_unique_terminations[i])
                if num_terms == 0:
                    warnings.warn(f'match {i}: no unique terminations remain, skipping this match')
                    match_pbar.update(1)
                    continue
                
                if int(_A * n_calls_density) < 10:
                    n_calls = 10
                else:
                    n_calls = int(_A * n_calls_density)
                    
                with tqdm(total = num_terms, desc = "unique terminations") as term_pbar:
                    e_labels = []
                    #for j in range(1):
                    for j in range(num_terms):
                        #optimize
                        self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
                        it = self.opt_results[(i,j)]['sampled_interfaces'][0]
                        A = it.lattice.a * it.lattice.b
                        supcl_E0 = float(self.opt_results[(i,j)]['supcl_E'][0])
                        if self.double_interface:
                            film_scale = len(it.film_indices)/len(self.film)
                            substrate_scale = len(it.substrate_indices)/len(self.substrate)
                            it_E = (supcl_E0 - film_scale * self.film_e - substrate_scale * self.substrate_e) / A * 16.02176634 / 2
                            e_labels.append(it_E)
                            self._global_min_debug(
                                stage="prescreen_double_interface",
                                match_id=int(i),
                                term_id=int(j),
                                area=float(A),
                                supcl_E0=supcl_E0,
                                prescreen_it_E=float(it_E),
                                film_scale=float(film_scale),
                                substrate_scale=float(substrate_scale),
                                film_atom_count=int(len(it.film_indices)),
                                substrate_atom_count=int(len(it.substrate_indices)),
                            )
                        else:
                            film_slab = it.film
                            substrate_slab = it.substrate
                            film_slab_E = float(self.mc.calculate(film_slab))
                            substrate_slab_E = float(self.mc.calculate(substrate_slab))
                            bd_E = (supcl_E0 - film_slab_E - substrate_slab_E) / A * 16.02176634
                            e_labels.append(bd_E)
                            self._global_min_debug(
                                stage="prescreen_single_interface",
                                match_id=int(i),
                                term_id=int(j),
                                area=float(A),
                                supcl_E0=supcl_E0,
                                film_slab_E=film_slab_E,
                                substrate_slab_E=substrate_slab_E,
                                prescreen_bd_E=float(bd_E),
                            )
                    print(e_labels)
                    for j in range(num_terms):
                        if e_labels[j] < min(e_labels) + term_screen_tol:
                        
                            ltc = self.opt_results[(i,j)]['sampled_interfaces'][0].lattice
                            A = ltc.a * ltc.b
                            
                            if self.double_interface:
                                it_bd_E, strain_E = self.post_bayesian_process_double_interface(i,j,A)
                            else:
                                it_bd_E, strain_E = self.post_bayesian_process(i,j,A)
                            
                            self.formated_data.append(
                                    [hkl_f[0], hkl_f[1], hkl_f[2],\
                                    hkl_s[0], hkl_s[1], hkl_s[2], \
                                    A, epsilon, it_bd_E, strain_E, \
                                    uvw_f1[0], uvw_f1[1], uvw_f1[2], \
                                    uvw_f2[0], uvw_f2[1], uvw_f2[2], \
                                    uvw_s1[0], uvw_s1[1], uvw_s1[2], \
                                    uvw_s2[0], uvw_s2[1], uvw_s2[2], \
                                    self.all_unique_terminations[i][j], i, j])
                        
                        term_pbar.update(1)
                    
                    match_pbar.update(1)
        self.global_optimized_data = pd.DataFrame(self.formated_data, columns = columns)
        if len(self.global_optimized_data) == 0:
            self.best_key = None
            self.close_energy_calculator()
            self.global_optimized_data.to_csv(f'all_data_{name}.csv')
            with open(f'opt_results_{name}.pkl','wb') as f:
                pickle.dump(self.opt_results, f)
            self.opt_results = convert_dict_to_json(self.opt_results)
            raise ValueError('No interfaces remain after termination screening and energy pre-screening.')
        self.global_optimized_data = self.global_optimized_data.sort_values(by = it_energetic_label)

        self.best_key = (self.global_optimized_data[r'$i_m$'].to_numpy()[0], self.global_optimized_data[r'$i_t$'].to_numpy()[0])
        #close docker container
        self.close_energy_calculator()
        self.global_optimized_data.to_csv(f'all_data_{name}.csv')
        with open(f'opt_results_{name}.pkl','wb') as f:
            pickle.dump(self.opt_results, f)
        self.opt_results = convert_dict_to_json(self.opt_results)
    
    def get_selected_match_ids_per_plane(self):
        ranked_pairs = list(zip(
            self.global_optimized_data[r'$i_m$'].to_numpy(),
            self.global_optimized_data[r'$i_t$'].to_numpy(),
        ))

        def _get_type_lists_per_miller(tuple_id):
            miller_to_type_ids = {}
            for match_key, type_data in self.ems.all_matche_data.items():
                miller = match_key[tuple_id]
                if miller not in miller_to_type_ids:
                    miller_to_type_ids[miller] = set()
                miller_to_type_ids[miller].update(type_data.keys())
            return {
                miller: array(sorted(type_ids))
                for miller, type_ids in miller_to_type_ids.items()
            }

        def _select_best_pairs_per_miller(tuple_id):
            selected_pairs = {}
            type_lists_per_miller = _get_type_lists_per_miller(tuple_id)
            for miller, type_ids in type_lists_per_miller.items():
                for pair in ranked_pairs:
                    if pair[0] in type_ids:
                        selected_pairs[miller] = pair
                        break
            return selected_pairs

        self.selected_pairs_substrate_by_miller = _select_best_pairs_per_miller(1)
        self.selected_pairs_film_by_miller = _select_best_pairs_per_miller(0)

        self.selected_pairs_substrate = list(dict.fromkeys(self.selected_pairs_substrate_by_miller.values()))
        self.selected_pairs_film = list(dict.fromkeys(self.selected_pairs_film_by_miller.values()))

        self.selected_i_ms_substrate = unique([i[0] for i in self.selected_pairs_substrate])
        self.selected_i_ms_film = unique([i[0] for i in self.selected_pairs_film])
    
    def get_lowest_energy_pairs_each_match(self,
                                only_lowest_energy = False,
                            only_lowest_energy_each_plane = False,
                            only_substrate = False):
        
        pd = self.global_optimized_data
        i_s = pd['$i_m$'].to_numpy()
        j_s = pd['$i_t$'].to_numpy()
        ranked_pairs = list(zip(i_s, j_s))
        pair_order = {pair: idx for idx, pair in enumerate(ranked_pairs)}
        pairs = []
        if only_lowest_energy:
            pairs.append((i_s[0], j_s[0]))
        elif only_lowest_energy_each_plane:
            self.get_selected_match_ids_per_plane()
            if only_substrate:
                pairs = self.selected_pairs_substrate.copy()
            else:
                pairs = list(dict.fromkeys(self.selected_pairs_substrate + self.selected_pairs_film))
            pairs = sorted(pairs, key=lambda pair: pair_order[pair])
        else:
            match_ids = []
            for i in range(len(i_s)):
                con = i_s[i] not in match_ids
                if con:
                    match_ids.append(i_s[i])
                    pairs.append((i_s[i], j_s[i]))
        
        return pairs

    def get_lowest_energy_pairs_per_plane(self, only_substrate = False):
        """
        Get the lowest-energy optimized pair for each film/substrate Miller plane.

        Returns:
        dict: {
            'film': {(h, k, l): (i_m, i_t), ...},
            'substrate': {(h, k, l): (i_m, i_t), ...},
        }
        """
        self.get_selected_match_ids_per_plane()

        def _normalize_mapping(mapping):
            normalized = {}
            for miller, pair in mapping.items():
                normalized[tuple(int(v) for v in miller)] = (int(pair[0]), int(pair[1]))
            return normalized

        data = {
            'substrate': _normalize_mapping(self.selected_pairs_substrate_by_miller),
        }
        if not only_substrate:
            data['film'] = _normalize_mapping(self.selected_pairs_film_by_miller)
        return data
    
    def visualize_minimization_results(self, film_name, substrate_name):
        
        pairs = self.get_lowest_energy_pairs_each_match(only_lowest_energy_each_plane = True)
        selected_pair_by_match = {}
        selected_energy_by_match = {}
        for key in pairs:
            selected_pair_by_match[key[0]] = key
            if self.double_interface:
                selected_energy_by_match[key[0]] = self.opt_results[key]['relaxed_min_it_E']
            else:
                selected_energy_by_match[key[0]] = self.opt_results[key]['relaxed_min_bd_E']
        from InterOptimus.matching import get_area_match
        data = []
        for i in self.ems.all_matche_data.keys():
            here = self.ems.all_matche_data[i]
            low_E = inf
            low_pair = None
            for tp in here.keys():
                try:
                    if selected_energy_by_match[tp] < low_E:
                        low_E = selected_energy_by_match[tp]
                        low_pair = selected_pair_by_match[tp]
                    mt = self.unique_matches[tp]
                except:
                    pass
            if low_pair is None:
                continue
            data.append([i[0][0], i[0][1], i[0][2], i[1][0], i[1][1], i[1][2],
                         get_area_match(mt), mt.von_mises_strain, low_E, low_pair[0], low_pair[1]])
        from numpy import savetxt
        savetxt('area_strain',data,fmt = '(%i %i %i) (%i %i %i) %.4f %.4f %.4f %i %i')
        
        from InterOptimus.matching import visualize_minimization_results
        if self.double_interface:
            title = 'Interface Energy'
        else:
            title = 'Cohesive Energy'
        visualize_minimization_results(film_name, substrate_name, title)
    
    def patch_jobflow_jobs(self,
                            only_lowest_energy = False,
                            only_lowest_energy_each_plane = False,
                            only_substrate = False,
                            
                            relax_user_incar_settings = None,
                            relax_user_potcar_settings = None,
                            relax_user_kpoints_settings = None,
                            relax_user_potcar_functional = 'PBE_54',
                            
                            static_user_incar_settings = None,
                            static_user_potcar_settings = None,
                            static_user_kpoints_settings = None,
                            static_user_potcar_functional = 'PBE_54',
                            
                            filter_name = 'my_IO_jobs',
                            do_dft_gd = False,
                            gd_kwargs = {},
                            dipole_correction = False,
                            ):
        """
        Patch JobFlow jobs, only for c_period = True
        """
        
        from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
        from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker
        from jobflow import Flow
        
        if do_dft_gd:
            from InterOptimus.jobflow import GDVaspMaker
        
        pairs = self.get_lowest_energy_pairs_each_match(only_lowest_energy = only_lowest_energy,
                            only_lowest_energy_each_plane = only_lowest_energy_each_plane,
                            only_substrate = only_substrate)
        
        flows = []
        for num in range(2):
            structure = [self.film, self.substrate][num]
            static_incar_here = default_static_incar_settings(
                static_user_incar_settings,
                num_atoms=len(structure),
            )
            vasp_maker = StaticMaker(
                                    input_set_generator = MPStaticSet(
                                                        user_incar_settings = static_incar_here,
                                                         user_potcar_settings = static_user_potcar_settings,
                                                          user_kpoints_settings = static_user_kpoints_settings,
                                                           user_potcar_functional = static_user_potcar_functional,
                                                           ),
                                    run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                    )
            job = vasp_maker.make(structure)
            job.update_metadata({'filter_name':filter_name, 'job': ['film','substrate'][num]})
            flows.append(job)

        for i in pairs:
            if self.double_interface:
                relax_extra_settings = default_double_interface_relax_extra_settings()
            else:
                relax_extra_settings = {'ISIF':2}
            #interface here
            it = Structure.from_dict(json.loads(self.opt_results[i]['relaxed_best_interface']['structure']))
            it_user_incar_settings_here = default_relax_incar_settings(
                relax_user_incar_settings,
                extra_settings=relax_extra_settings,
                num_atoms=len(it),
            )
            if not with_LDAUU(it) and 'LDAU' in it_user_incar_settings_here.keys():
                del it_user_incar_settings_here['LDAU']
                del it_user_incar_settings_here['LDAUU']
                del it_user_incar_settings_here['LDAUJ']
                del it_user_incar_settings_here['LDAUL']
                del it_user_incar_settings_here['LDAUTYPE']
                del it_user_incar_settings_here['LDAUPRINT']
                
            vasp_maker = RelaxMaker(
                                    input_set_generator = MPRelaxSet(
                                                                    user_incar_settings = it_user_incar_settings_here,
                                                                    user_potcar_settings = relax_user_potcar_settings,
                                                                    user_kpoints_settings = relax_user_kpoints_settings,
                                                                    user_potcar_functional = relax_user_potcar_functional,
                                                                    ),
                                    run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                    )
            if do_dft_gd:
                if dipole_correction:
                    it_user_incar_settings_dp = it_user_incar_settings_here.copy()
                    it_user_incar_settings_dp['IDIPOL'] = 3
                    it_user_incar_settings_dp['LDIPOL'] = True
                    vasp_maker_dp = RelaxMaker(
                                            input_set_generator = MPRelaxSet(
                                                                            user_incar_settings = it_user_incar_settings_dp,
                                                                            user_potcar_settings = relax_user_potcar_settings,
                                                                            user_kpoints_settings = relax_user_kpoints_settings,
                                                                            user_potcar_functional = relax_user_potcar_functional,
                                                                            ),
                                            run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                            )
                else:
                    vasp_maker_dp = None

                flow = GDVaspMaker(
                                    initial_it = it,
                                    film_indices = self.opt_results[i]['film_indices'],
                                    metadata = {'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_it'},
                                    relax_maker = vasp_maker,
                                    relax_maker_dp = vasp_maker_dp,
                                    **gd_kwargs,
                                    ).make()
                
                
                flows += flow
            else:
                job = vasp_maker.make(it)
                job.update_metadata({'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_it'})
                flows.append(job)
                
                if dipole_correction:
                    it_user_incar_settings_dp = it_user_incar_settings_here.copy()
                    it_user_incar_settings_dp['IDIPOL'] = 3
                    it_user_incar_settings_dp['LDIPOL'] = True
                    vasp_maker = RelaxMaker(
                                            input_set_generator = MPRelaxSet(
                                                                            user_incar_settings = it_user_incar_settings_dp,
                                                                            user_potcar_settings = relax_user_potcar_settings,
                                                                            user_kpoints_settings = relax_user_kpoints_settings,
                                                                            user_potcar_functional = relax_user_potcar_functional,
                                                                            ),
                                            run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                            )
                    job_dp = vasp_maker.make(it, prev_dir = job.output.dir_name)
                    job_dp.update_metadata({'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_it_dp'})
                    flows.append(job_dp)
            
            if self.strain_E_correction:
                #strained film
                s_film = Structure.from_dict(json.loads(self.opt_results[i]['strain_film']))
                static_incar_here = default_static_incar_settings(
                    static_user_incar_settings,
                    num_atoms=len(s_film),
                )
                vasp_maker = StaticMaker(
                                        input_set_generator = MPStaticSet(
                                                                        s_film,
                                                                        user_incar_settings = static_incar_here,
                                                                        user_potcar_settings = static_user_potcar_settings,
                                                                        user_kpoints_settings = static_user_kpoints_settings,
                                                                        user_potcar_functional = static_user_potcar_functional
                                                                        ),
                                        run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                        )
                job = vasp_maker.make(s_film)
                job.update_metadata({'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_sfilm'})
                flows.append(job)
                
            #film & substrate slab
            if not self.double_interface:
                for slab in ['film', 'substrate']:
                    slab_structure = Structure.from_dict(json.loads(self.opt_results[i]['slabs'][slab]['structure']))
                    it_user_incar_settings_here = default_relax_incar_settings(
                        relax_user_incar_settings,
                        extra_settings=relax_extra_settings,
                        num_atoms=len(slab_structure),
                    )
                    if not with_LDAUU(slab_structure) and 'LDAU' in it_user_incar_settings_here.keys():
                        del it_user_incar_settings_here['LDAU']
                        del it_user_incar_settings_here['LDAUU']
                        del it_user_incar_settings_here['LDAUJ']
                        del it_user_incar_settings_here['LDAUL']
                        del it_user_incar_settings_here['LDAUTYPE']
                        del it_user_incar_settings_here['LDAUPRINT']
                    
                    vasp_maker = RelaxMaker(
                                            input_set_generator = MPRelaxSet(
                                                                            user_incar_settings = it_user_incar_settings_here,
                                                                            user_potcar_settings = relax_user_potcar_settings,
                                                                            user_kpoints_settings = relax_user_kpoints_settings,
                                                                            user_potcar_functional = relax_user_potcar_functional,
                                                                            ),
                                            run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                            )
                    job = vasp_maker.make(slab_structure)
                    job.update_metadata({'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_{slab}_slab'})
                    flows.append(job)
                    
                    if dipole_correction:
                        it_user_incar_settings_dp = it_user_incar_settings_here.copy()
                        it_user_incar_settings_dp['IDIPOL'] = 3
                        it_user_incar_settings_dp['LDIPOL'] = True
                        vasp_maker = RelaxMaker(
                                                input_set_generator = MPRelaxSet(
                                                                                user_incar_settings = it_user_incar_settings_dp,
                                                                                user_potcar_settings = relax_user_potcar_settings,
                                                                                user_kpoints_settings = relax_user_kpoints_settings,
                                                                                user_potcar_functional = relax_user_potcar_functional,
                                                                                ),
                                                run_vasp_kwargs=interoptimus_vasp_run_kwargs(),
                                                )
                        job_dp = vasp_maker.make(slab_structure, prev_dir = job.output.dir_name)
                        job_dp.update_metadata({'filter_name':filter_name, 'job': f'{i[0]}_{i[1]}_{slab}_slab_dp'})
                        flows.append(job_dp)
                    
                    
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

def with_LDAUU(structure):
    """
    Check if a structure requires LDA+U corrections in DFT calculations.

    Determines whether the structure contains elements that typically require
    Hubbard U corrections when F or O are present (transition metals).

    Args:
        structure: pymatgen Structure object

    Returns:
        bool: True if LDA+U corrections are recommended, False otherwise

    Notes:
        LDA+U is recommended when both oxygen/fluorine and certain transition
        metals (Co, Cr, Fe, Mn, Mo, Ni, V, W) are present in the structure.
    """
    elements = [el.symbol for el in structure.elements]
    with_LDAUU = False
    if 'F' in elements or 'O' in elements:
        for el in ['Co', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'V', 'W']:
            if el in elements:
                with_LDAUU = True
                break
    return with_LDAUU
