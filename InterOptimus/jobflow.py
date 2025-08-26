from pymatgen.core.structure import Structure
from jobflow import job
from qtoolkit.core.data_objects import QResources
import os
from InterOptimus.itworker import InterfaceWorker

@job
def check_it_phase_stability(film_conv, substrate_conv, device = 'cpu',
                                                        fmax = 0.5,
                                                        steps = 500,
                                                        n_calls = 30,
                                                        calc = 'sevenn',
                                                        ckpt_path_ENV = 'SEVENN_CHECKPOINT'):
    
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
