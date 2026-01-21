"""
InterOptimus Jobflow Module

This module defines jobflow-based workflows for interface optimization
calculations, including gradient descent optimization and VASP calculations
with automatic resource management.
"""

from pymatgen.core.structure import Structure
from jobflow import Flow, Response, job, Maker
from qtoolkit.core.data_objects import QResources
import os
from InterOptimus.itworker import InterfaceWorker
from dataclasses import dataclass
from atomate2.vasp.jobs.core import RelaxMaker
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.interface import Interface
from typing import Callable, Dict, Any, Optional, List
import numpy as np
import pickle
from jobflow_remote import set_run_config
import base64
import json

@dataclass
class GDVaspMaker(Maker):
    name: str = 'it_gradient_descend'
    dx: float = 0.05
    tol: float = 5e-4
    initial_r: float = 0.1
    initial_it: Interface = None
    film_indices: List = None
    min_steps: int = 3
    relax_maker: RelaxMaker = None
    relax_maker_dp: RelaxMaker = None
    metadata: Dict[str, Any] = None
    
    @job
    def update_n_p_1(self, y_dns, saved_data):
        #calculate g_n
        g_n = (np.array(y_dns) - np.array(saved_data['ys'][-1])) / self.dx
        if saved_data['n'] == 0:
            r = 0.01
        else:
            #x_n & x_n_1 have been updated in saved_data
            x_n, x_n_1 = np.array(saved_data['xs'][-1]), np.array(saved_data['xs'][-2])
            #g_n_1 has not been updated
            g_n_1 = np.array(saved_data['gs'][-1])
            r = abs(np.dot((x_n - x_n_1), (g_n - g_n_1))) / np.linalg.norm(g_n - g_n_1) ** 2
        
        #translate it_n
        tt = TranslateSitesTransformation(self.film_indices, - r * g_n, False)
        it = tt.apply_transformation(saved_data['its'][-1])
        #vasp_job = self.relax_maker.make(structure = it, prev_dir = saved_data['vasp_dir'])
        vasp_job = self.relax_maker.make(structure = it)
        saved_data['n'] += 1
        ask_gradient_job = self.ask_gradient(
                               saved_data,
                                np.array(saved_data['xs'][-1]) - r * g_n,
                                vasp_job.output.output.energy,
                                g_n,
                                vasp_job.output.output.structure,
                                vasp_job.output.dir_name
                                )
        return Response(replace = Flow([vasp_job, ask_gradient_job]))
        
    @job
    def save_final_data(self, saved_data):
        if self.relax_maker_dp is None:
            return saved_data
        else:
            #stct = Structure.from_dict(saved_data['its'][-1])
            job = self.relax_maker_dp.make(saved_data['its'][-1], prev_dir = saved_data['vasp_dir'])
            metadata = self.metadata.copy()
            metadata['job'] += '_dp'
            job.update_metadata(metadata)
            return Response(output = saved_data, replace = Flow([job]))
        
    @job
    #def ask_gradient(self, saved_data, x_n, y_n, g_n_1, it_n):
    def ask_gradient(self, saved_data, x_n, y_n, g_n_1, it_n, vasp_dir):
        # if first round, save vasp running dictionary
        saved_data['vasp_dir'] = vasp_dir
        # update n_p_1 data
        saved_data['xs'].append(x_n)
        saved_data['ys'].append(y_n)
        saved_data['gs'].append(g_n_1)
        saved_data['its'].append(it_n)
        
        print(type(it_n))
        #it_n_structure = Structure.from_dict(it_n)
        it_n_structure = it_n
        
        if saved_data['n'] > self.min_steps and abs(saved_data['ys'][-1] - saved_data['ys'][-2]) < self.tol * len(it_n_structure):
            save_job = self.save_final_data(saved_data)
            save_job.update_metadata(self.metadata)
            return Response(replace = Flow([save_job]))
            
        # jobs to calculate g_n
        jobs = []
        for i in range(3):
            pdx = np.zeros(3)
            pdx[i] = self.dx
            tt = TranslateSitesTransformation(self.film_indices, pdx, False)
            it = tt.apply_transformation(it_n_structure)
            #jobs.append(self.relax_maker.make(structure = it, prev_dir = vasp_dir))
            jobs.append(self.relax_maker.make(structure = it))
        
        # job to calculate r, x_n_p_1 & y_n_p_1
        n_p_1_job = self.update_n_p_1([jobs[0].output.output.energy,
                                        jobs[1].output.output.energy,
                                        jobs[2].output.output.energy],
                                        saved_data)
        return Response(replace = Flow(jobs + [n_p_1_job]))
    
    def make(self):
        saved_data = {}
        saved_data['xs'] = []
        saved_data['ys'] = []
        saved_data['gs'] = []
        saved_data['its'] = []
        saved_data['n'] = 0
        #first vasp job
        vasp_job = self.relax_maker.make(structure = self.initial_it)
        #update saved_data with the first-vasp-job output
        ask_gradient_job = self.ask_gradient(
                            saved_data = saved_data,
                            x_n = np.array([0,0,0]),
                            y_n = vasp_job.output.output.energy,
                            g_n_1 = 0,
                            it_n = vasp_job.output.output.structure,
                            vasp_dir = vasp_job.output.dir_name
                            )
        return [vasp_job, ask_gradient_job]

@dataclass
class IOMaker(Maker):
    name: str = 'IO_std'
    lattice_matching_settings: Dict[str, Any] = None
    structure_settings: Dict[str, Any] = None
    optimization_settings: Dict[str, Any] = None
    global_minimization_settings: Dict[str, Any] = None
    do_vasp: bool = False
    do_vasp_gd: bool = False
    vasp_relax_settings: Dict[str, Any] = None
    vasp_static_settings: Dict[str, Any] = None
    mlip_resources: Callable = None
    vasp_resources: Callable = None
    mlip_worker: str = 'std_worker'
    vasp_worker: str = 'std_worker'
    
    @job(data="IO_results")
    def IO_HT_job(self, film_conv, substrate_conv):
        iw = InterfaceWorker(film_conv, substrate_conv)
        iw.lattice_matching(**self.lattice_matching_settings)
        iw.ems.plot_unique_matches()
        iw.ems.plot_matching_data(['film', 'substrate'],'project.jpg', show_millers = True, show_legend = False)
        iw.parse_interface_structure_params(**self.structure_settings)
        iw.parse_optimization_params(**self.optimization_settings)
        iw.global_minimization(**self.global_minimization_settings)
        
        results = {}
        results['unique_matches'] = iw.unique_matches
        results['all_match_data'] = iw.ems.all_matche_data
        results['match_data'] = iw.ems.matching_data
        results['opt_results'] = iw.opt_results
        io_results_bytes = pickle.dumps(results)
        io_results_b64 = base64.b64encode(io_results_bytes).decode('utf-8')
        if self.do_vasp:
            # Prepare VASP settings, use None (defaults to MPRelaxSet/MPStaticSet) if not provided
            relax_incar = self.vasp_relax_settings.get('INCAR') if self.vasp_relax_settings else None
            relax_kpoints = self.vasp_relax_settings.get('KPOINTS') if self.vasp_relax_settings else None
            relax_potcar_func = self.vasp_relax_settings.get('POTCAR_FUNCTIONAL') if self.vasp_relax_settings else None
            relax_potcar = self.vasp_relax_settings.get('POTCAR') if self.vasp_relax_settings else None
            
            static_incar = self.vasp_static_settings.get('INCAR') if self.vasp_static_settings else None
            static_kpoints = self.vasp_static_settings.get('KPOINTS') if self.vasp_static_settings else None
            static_potcar_func = self.vasp_static_settings.get('POTCAR_FUNCTIONAL') if self.vasp_static_settings else None
            static_potcar = self.vasp_static_settings.get('POTCAR') if self.vasp_static_settings else None
            
            # Handle do_dft_gd and gd_kwargs (default to False if not provided)
            gd_kwargs = {'tol': self.vasp_relax_settings.get('GDTOL', 5e-4)} if self.vasp_relax_settings else {'tol': 5e-4}
            
            flow = iw.patch_jobflow_jobs(filter_name=self.name,
                            only_lowest_energy_each_plane = True,
                            relax_user_incar_settings=relax_incar,
                            relax_user_kpoints_settings=relax_kpoints,
                            relax_user_potcar_functional=relax_potcar_func,
                            relax_user_potcar_settings=relax_potcar,
                            
                            static_user_incar_settings=static_incar,
                            static_user_kpoints_settings=static_kpoints,
                            static_user_potcar_functional=static_potcar_func,
                            static_user_potcar_settings=static_potcar,
                            
                            do_dft_gd=self.do_vasp_gd,
                            gd_kwargs=gd_kwargs)
            
            return {'IO_results':io_results_b64, 'flow': Flow(flow).to_json()}
        else:
            return {'IO_results':io_results_b64}
    
    @job
    def vasp_job(self, flow_json):
        return Response(replace = Flow.from_dict(json.loads(flow_json)))
        
    def make(self, film_conv, substrate_conv):
        if self.do_vasp:
            IO_job = self.IO_HT_job(film_conv, substrate_conv)
            
            # Call resources function to get QResources object (not function)
            mlip_res = self.mlip_resources() if callable(self.mlip_resources) else self.mlip_resources
            
            IO_job = set_run_config(IO_job, worker = self.mlip_worker,
                              resources = mlip_res,
                              priority=10,
                              dynamic = False)
            
            vasp_job = self.vasp_job(IO_job.output['flow'])
            
            # Call resources function to get QResources object (not function)
            vasp_res = self.vasp_resources() if callable(self.vasp_resources) else self.vasp_resources
            
            vasp_job = set_run_config(vasp_job, worker = self.vasp_worker,
                              resources = vasp_res,
                              priority=0,
                              exec_config={'pre_run':'module load VASP/6.4.3-optcell'},
                              dynamic = True)
            
            return Flow([IO_job, vasp_job])
        else:
            # NOTE: use the passed-in conventional structures
            IO_job = self.IO_HT_job(film_conv, substrate_conv)
            # Apply MLIP resources if provided
            if self.mlip_resources is not None:
                mlip_res = self.mlip_resources() if callable(self.mlip_resources) else self.mlip_resources
                IO_job = set_run_config(IO_job, worker = self.mlip_worker,
                                  resources = mlip_res,
                                  priority=10,
                                  dynamic = False)
            return Flow([IO_job])
        
@job
def check_it_phase_stability(film_conv, substrate_conv, device='cpu',
                                                        fmax=0.5,
                                                        steps=500,
                                                        n_calls=30,
                                                        calc='sevenn',
                                                        ckpt_path_ENV='SEVENN_CHECKPOINT'):
    """
    Evaluate interface phase stability using MLIP optimization.

    Performs lattice matching, interface structure generation, and phase
    stability evaluation using machine learning interatomic potentials.
    Compares static and relaxed interface structures to assess stability.

    Args:
        film_conv: Conventional unit cell of the film material
        substrate_conv: Conventional unit cell of the substrate material
        device (str): Device for MLIP calculations ('cpu' or 'cuda')
        fmax (float): Maximum force threshold for relaxation
        steps (int): Maximum steps for geometry optimization
        n_calls (int): Number of optimization calls for interface registration
        calc (str): MLIP calculator to use ('sevenn', 'orb-models', etc.)
        ckpt_path_ENV (str): Environment variable name for model checkpoint path

    Returns:
        dict: Results containing:
            - static_it: JSON representation of static interface
            - relaxed_it: JSON representation of relaxed interface
            - rms_frac: RMS displacement in fractional coordinates
            - rms_cart: RMS displacement in Cartesian coordinates
    """
    
    iw = InterfaceWorker(film_conv, substrate_conv)
    
    iw.lattice_matching(max_area = 100,
                        max_length_tol = 0.03,
                        max_angle_tol = 0.03,
                        film_max_miller = 4,
                        substrate_max_miller = 4)
                        
    iw.parse_interface_structure_params(termination_ftol = 2, c_periodic = True, \
                                    vacuum_over_film = 10, film_thickness = 15, \
                                    substrate_thickness = 15, shift_to_bottom = True)
                                    
    iw.parse_optimization_params(do = True,
                             set_fix_thicknesses = (0,0),
                             fix_in_layers = True,
                             whole_slab_fixed = False,
                             fmax = fmax,
                             steps = steps,
                             device = device,
                             ckpt_path = os.getenv(ckpt_path_ENV))
    
    static_it, relaxed_it, dx_frac, dx_cart = iw.phase_stability_evaluation(
                                                                            n_calls = n_calls,
                                                                            z_range = (0, 3),
                                                                            calc = calc,
                                                                            discut = 0.5,
                                                                            )
    
    return {'static_it': static_it.to_json(),
            'relaxed_it': relaxed_it.to_json(),
            'rms_frac': dx_frac,
            'rms_cart': dx_cart}
