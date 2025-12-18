"""
InterOptimus Tool Module

This module provides utility functions for crystal interface analysis,
structure manipulation, data processing, and visualization.
"""

from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.surface import SlabGenerator
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.interfaces.zsl import fast_norm
from InterOptimus.CNID import triple_dot, calculate_cnid_in_supercell
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import spearmanr
from scipy.stats import pearsonr

def convert_dict_to_json(obj):
    """
    Recursively convert objects to JSON-serializable format.

    Converts numpy arrays to lists, pymatgen Structures to JSON,
    and recursively processes dictionaries and lists.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the input object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_dict_to_json(v)
        return obj
    elif isinstance(obj, list):
        for k in range(len(obj)):
            obj[k] = convert_dict_to_json(obj[k])
        return obj
    elif isinstance(obj, Structure):
        return obj.to_json()
    else:
        return obj

def get_min_nb_distance(atom_index, structure, cutoff):
    """
    Get the minimum neighboring distance for a specific atom in a structure.

    Args:
        atom_index (int): Index of the atom in the structure
        structure (Structure): pymatgen Structure object
        cutoff (float): Maximum distance to search for neighbors

    Returns:
        float: Minimum neighboring distance, or np.inf if no neighbors found
    """
    neighbors = structure.get_neighbors(structure[atom_index], r=cutoff)
    if len(neighbors) == 0:
        return np.inf
    else:
        return min([neighbor[1] for neighbor in neighbors])

def sort_list(array_to_sort, keys):
    """
    Sort a list of arrays based on corresponding keys.

    Args:
        array_to_sort (list): List of arrays/items to sort
        keys (list): Keys to sort by (same length as array_to_sort)

    Returns:
        list: Sorted version of array_to_sort
    """
    combined_array = []
    for idx, row in enumerate(array_to_sort):
        combined_array.append((keys[idx], row))
    combined_array_sorted = sorted(combined_array, key=lambda x: x[0])
    keys_sorted, array_sorted = zip(*combined_array_sorted)
    return list(array_sorted)

def apply_cnid_rbt(interface, x, y, z):
    """
    Apply rigid body translation to an interface using CNID coordinates.

    Args:
        interface: Interface object to translate
        x (float): Fractional CNID x-coordinate
        y (float): Fractional CNID y-coordinate
        z (float): Fractional coordinate in c-direction

    Returns:
        Interface object after rigid body translation
    """
    CNID = calculate_cnid_in_supercell(interface)[0]
    CNID_translation = TranslateSitesTransformation(
        interface.film_indices,
        x * CNID[:, 0] + y * CNID[:, 1] + [0, 0, z]
    )
    return CNID_translation.apply_transformation(interface)

def existfilehere(filename):
    """
    Check if a file exists in the current working directory.

    Args:
        filename (str): Name of the file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(os.path.join(os.getcwd(), filename))

def get_termination_indices(slab, ftol=0.25):
    """
    Get the terminating atom indices of a slab.

    Uses hierarchical clustering to identify atoms at the top and bottom
    terminations of a slab based on their c-coordinate distances.

    Args:
        slab (Structure): Slab structure to analyze
        ftol (float): Distance tolerance for clustering (default: 0.25)

    Returns:
        tuple: (bottom_indices, top_indices) arrays of atom indices
    """
    frac_coords = slab.frac_coords
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    h = slab.lattice.c

    # Calculate distances in c-direction considering periodic boundary
    for ii, jj in combinations(list(range(n)), 2):
        if ii != jj:
            cdist = frac_coords[ii][2] - frac_coords[jj][2]
            cdist = abs(cdist - np.round(cdist)) * h
            dist_matrix[ii, jj] = cdist
            dist_matrix[jj, ii] = cdist

    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, ftol, criterion="distance")

    clustered_sites = {c: [] for c in clusters}
    for idx, cluster in enumerate(clusters):
        clustered_sites[cluster].append(slab[idx])

    plane_heights = {
        np.mean(np.mod([s.frac_coords[2] for s in sites], 1)): c
        for c, sites in clustered_sites.items()
    }

    term_cluster_min = min(plane_heights.items(), key=lambda x: x[0])[1]
    term_cluster_max = max(plane_heights.items(), key=lambda x: x[0])[1]

    return np.where(clusters == term_cluster_min)[0], np.where(clusters == term_cluster_max)[0]

def get_termination_indices_shell(slab, shell=1.5):
    """
    Get terminating atom indices using a shell-based approach.

    Identifies atoms within a specified distance (shell) from the top and
    bottom surfaces of the slab.

    Args:
        slab (Structure): Slab structure to analyze
        shell (float): Shell thickness in Angstroms (default: 1.5)

    Returns:
        tuple: (bottom_indices, top_indices) arrays of atom indices
    """
    frac_coords_z = slab.cart_coords[:, 2]
    low = min(frac_coords_z)
    high = max(frac_coords_z)
    return (np.where(frac_coords_z < low + shell)[0],
            np.where(frac_coords_z > high - shell)[0])
    
def get_it_core_indices(interface):
    """
    Get the terminating atom indices of an interface structure.

    Identifies the top and bottom terminating atoms for both film and
    substrate components in an interface structure.

    Args:
        interface: Interface object containing film and substrate indices

    Returns:
        tuple: (film_bottom_indices, film_top_indices,
                substrate_bottom_indices, substrate_top_indices)
    """
    ids = np.array(interface.film_indices)
    slab = interface.film
    ids_film_min, ids_film_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    
    ids = np.array(interface.substrate_indices)
    slab = interface.substrate
    ids_substrate_min, ids_substrate_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    return ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max


def convert_value(value):
    """
    Convert string values to appropriate Python data types.

    Parses configuration file values and converts them to bool, int, float,
    or arrays as appropriate.

    Args:
        value (str): String value to convert

    Returns:
        Converted value (bool, int, float, or array)

    Conversion rules:
        - '.TRUE.'/TRUE -> True
        - '.FALSE.'/FALSE -> False
        - Comma-separated values -> numpy array of ints
        - Values with '.' -> float
        - Other numeric values -> int
        - Everything else -> string (unchanged)
    """
    if value.upper() == '.TRUE.' or value.upper() == 'TRUE':
        return True
    elif value.upper() == '.FALSE.' or value.upper() == 'FALSE':
        return False
    if '/' in value:
        return value
    if ',' in value:
        return np.array(value.split(','), dtype=int)
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

def read_key_item(filename):
    """
    Read configuration parameters from a key-value file.

    Parses a configuration file with format "KEY = VALUE" and converts
    values to appropriate Python types. Skips comments and empty lines.

    Args:
        filename (str): Path to configuration file

    Returns:
        dict: Dictionary of configuration parameters with converted values

    Default values set if not specified:
    - THEORETICAL: False
    - STABLE: True
    - NOELEM: True
    - STCTMP: True
    """
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            if '=' in line:
                tag, value = line.split('=', 1)
                tag = tag.strip()
                value = value.strip()
                data[tag] = convert_value(value)

    # Set default values
    if 'THEORETICAL' not in data.keys():
        data['THEORETICAL'] = False
    if 'STABLE' not in data.keys():
        data['STABLE'] = True
    if 'NOELEM' not in data.keys():
        data['NOELEM'] = True
    if 'STCTMP' not in data.keys():
        data['STCTMP'] = True

    return data
    
def get_one_interface(cib, termination, slab_length, xyz, vacuum_over_film, c_periodic=False):
    """
    Generate a single interface structure with specified parameters.

    Creates an interface structure using the CoherentInterfaceBuilder
    with given termination, slab thickness, and rigid body translation.

    Args:
        cib: CoherentInterfaceBuilder object
        termination: Termination specification for the interface
        slab_length (float): Length of the slab in Angstroms
        xyz (tuple): Rigid body translation coordinates (x, y, z)
        vacuum_over_film (float): Vacuum thickness over the film
        c_periodic (bool): Whether to use periodic boundary conditions in c-direction

    Returns:
        Interface: Generated interface structure
    """
    x,y,z = xyz
    if c_periodic:
        gap = vacuum_over_film = z
    else:
        gap = z
        vacuum_over_film = vacuum_over_film
    interface_here = list(cib.get_interfaces(termination= termination, \
                                   substrate_thickness = slab_length, \
                                   film_thickness=slab_length, \
                                   vacuum_over_film=vacuum_over_film, \
                                   gap=gap, \
                                   in_layers=False))[0]
    CNID = calculate_cnid_in_supercell(interface_here)[0]
    CNID_translation = TranslateSitesTransformation(interface_here.film_indices, x*CNID[:,0] + y*CNID[:,1])
    return CNID_translation.apply_transformation(interface_here)

def get_rot_strain(film_matrix, sub_matrix) -> np.ndarray:
    """
    Find transformation matrix that rotates and strains film to match substrate.

    Calculates the rotation and strain transformation matrix that will align
    the film lattice with the substrate lattice while preserving the c-axis.
    Uses singular value decomposition to find the optimal transformation.

    Args:
        film_matrix: Film lattice matrix (3x3)
        sub_matrix: Substrate lattice matrix (3x3)

    Returns:
        tuple: (rotation_matrix, strain_matrix) for transforming film to substrate
    """
    film_matrix = np.array(film_matrix)
    film_matrix = film_matrix.tolist()[:2]
    film_matrix.append(np.cross(film_matrix[0], film_matrix[1]))
    film_matrix[2] = film_matrix[2]/np.linalg.norm(film_matrix[2])
    # Generate 3D lattice vectors for substrate super lattice
    # Out of plane substrate super lattice has to be same length as
    # Film out of plane vector to ensure no extra deformation in that
    # direction
    sub_matrix = np.array(sub_matrix)
    sub_matrix = sub_matrix.tolist()[:2]
    temp_sub = np.cross(sub_matrix[0], sub_matrix[1]).astype(float)  # conversion to float necessary if using numba
    temp_sub *= fast_norm(np.array(film_matrix[2], dtype=float))  # conversion to float necessary if using numba
    sub_matrix.append(temp_sub)
    sub_matrix[2] = sub_matrix[2]/np.linalg.norm(sub_matrix[2])

    A = np.transpose(np.linalg.solve(film_matrix, sub_matrix))
    U, sigma, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    
    S = np.dot(R.T, A)
    return R, S

def get_non_strained_film(match, film):
    """
    Get film structure without strain applied for a given match.

    Calculates the rotation transformation to align the film with the
    substrate lattice vectors without applying strain, preserving the
    film's original lattice parameters.

    Args:
        match: Lattice match object containing film and substrate vectors
        film: Film structure to transform

    Returns:
        Structure: Transformed film structure aligned but not strained
    """
    f_vs = match.film_sl_vectors
    s_vs = match.substrate_sl_vectors
    R_21, s_21 = get_rot_strain(f_vs, s_vs)
    R_1it, _ = get_rot_strain(s_vs, film.lattice.matrix[:2])
    trans_f = np.dot(R_1it, R_21)
    trans_b = np.dot(np.linalg.inv(R_21), np.linalg.inv(R_1it))
    trans = triple_dot(trans_f, np.linalg.inv(s_21), trans_b)
    DST = DeformStructureTransformation(trans)
    return trans_to_bottom(DST.apply_transformation(film))

def trans_to_bottom(stct):
    """
    Translate structure so that the lowest atom is at z=0.

    Shifts the entire structure in the z-direction so that the atom with
    the lowest z-coordinate is positioned at z = 1e-6 (slightly above 0).

    Args:
        stct: Structure object to translate

    Returns:
        Structure: Translated structure with lowest atom at z ≈ 0
    """
    ids = np.arange(len(stct))
    min_fc = stct.frac_coords[:,2].min()
    TST = TranslateSitesTransformation(ids, [0,0,-min_fc+1e-6])
    return TST.apply_transformation(stct)

def trans_to_top(stct):
    """
    Translate structure so that the highest atom is at the top of the cell.

    Shifts the structure so that the highest atom is positioned just below
    the top of the unit cell, with a small vacuum gap.

    Args:
        stct: Structure object to translate

    Returns:
        Structure: Translated structure with highest atom near cell top
    """
    ids = np.arange(len(stct))
    max_fc = stct.frac_coords[:,2].max()
    TST = TranslateSitesTransformation(ids, [0,0,1-max_fc])
    nstct = TST.apply_transformation(stct)
    TST = TranslateSitesTransformation(ids, [0,0,-0.1], vector_in_frac_coords = False)
    return TST.apply_transformation(nstct)

def get_film_length(match, film, it):
    """
    Calculate the length of film slab in the interface.

    Determines the film thickness in the interface structure based on
    the number of layers and the projected height of the film slab.

    Args:
        match: Lattice match object
        film: Film structure
        it: Interface structure

    Returns:
        float: Film length/thickness in Angstroms
    """
    film_sg = SlabGenerator(
            film,
            match.film_miller,
            min_slab_size=1,
            min_vacuum_size=10,
            in_unit_planes=False,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )
    return film_sg._proj_height * it.film_layers

def add_sele_dyn_slab(slab, shell=0, lr='left'):
    """
    Add selective dynamics constraints to a slab structure.

    Sets atoms within a specified shell distance from the surface to be
    fixed during geometry optimization.

    Args:
        slab: Slab structure to modify
        shell (float): Shell thickness in Angstroms (0 = no constraints)
        lr (str): Direction to fix ('left' for bottom, 'right' for top)

    Returns:
        tuple: (modified_slab, mobility_matrix) where mobility_matrix
               indicates which degrees of freedom are free (True) or fixed (False)
    """
    slab.fatom_ids = []
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(slab), axis = 0)
    coords = np.array([i.coords[2] for i in slab])
    if shell <= 0:
        return slab, mobility_mtx
    elif shell == np.inf:
        fix_indices = np.arange(len(coords))
    else:
        if lr == 'left':
            fix_indices = np.where(coords < min(coords) + shell)[0]
        elif lr == 'right':
            fix_indices = np.where(coords > max(coords) - shell)[0]
        else:
            raise ValueError('fix left: lr = left, fix right: lr = right')
    slab.fatom_ids = fix_indices.tolist()
    mobility_mtx[fix_indices] = [False, False, False]
    return slab, mobility_mtx

def add_sele_dyn_it(it, film_shell, sub_shell):
    """
    Add selective dynamics constraints to an interface structure.

    Fixes atoms in the substrate bottom and film top regions during
    geometry optimization of the interface.

    Args:
        it: Interface structure to modify
        film_shell (float): Shell thickness for film top atoms
        sub_shell (float): Shell thickness for substrate bottom atoms

    Returns:
        tuple: (modified_interface, mobility_matrix) where mobility_matrix
               indicates free (True) or fixed (False) degrees of freedom
    """
    coords = np.array([i.coords[2] for i in it])
    it.fatom_ids = []
    sub_bot_indices = np.where(coords < min(coords) + sub_shell)[0]
    film_top_indices = np.where(coords > max(coords) - film_shell)[0]
    #print(max(coords), film_top_indices)
    it.fatom_ids = []
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(it), axis = 0)
    if len(sub_bot_indices) > 0:
        it.fatom_ids += sub_bot_indices.tolist()
        mobility_mtx[sub_bot_indices] = [False, False, False]
    if len(film_top_indices) > 0:
        it.fatom_ids += film_top_indices.tolist()
        mobility_mtx[film_top_indices] = [False, False, False]
    return it, mobility_mtx

def cut_vaccum(structure, c):
    """
    Adjust the c-lattice parameter to remove excess vacuum.

    Modifies the unit cell c-dimension so that the vacuum gap above
    the highest atom is exactly the specified thickness.

    Args:
        structure: Structure to modify
        c (float): Desired vacuum thickness above the highest atom

    Returns:
        Structure: Structure with adjusted c-lattice parameter
    """
    lps = structure.lattice.parameters
    carts = [i.coords for i in structure]
    max_z = max(np.array(carts)[:,2])
    lps = list(lps)
    lps[2] = c + max_z
    return Structure(Lattice.from_parameters(*lps), structure.species, [i.coords for i in structure], coords_are_cartesian = True)

def plot_bcmk(mlips, name):
    """
    Plot benchmark comparison between DFT and MLIP calculations.

    Creates comprehensive benchmark plots comparing interface energies
    and atomic displacements between DFT and MLIP predictions for
    different interface configurations.

    Args:
        mlips (list): List of MLIP model names to compare
        name (str): Name/prefix for the benchmark study

    Note:
        Reads data from 'benchmk.pkl' and 'dft_output.pkl' files.
        Generates POSCAR files and creates comparison plots.
    """
    mlip_name_dict = {'orb-models':'ORB', 'sevenn':'SevenNet'}
    with open('benchmk.pkl','rb') as f:
        bcdata = pickle.load(f)

    with open('dft_output.pkl', 'rb') as f:
        dftdata = pickle.load(f)
    
    try:
        shutil.rmtree('poscars')
    except:
        pass
    os.mkdir('poscars')
    
    os.mkdir(f'poscars/dft')
    all_dps = []
    all_itEs = []
    for mlip in mlips:
        os.mkdir(f'poscars/{mlip}')
        os.mkdir(f'poscars/{mlip}/mlip')
        os.mkdir(f'poscars/{mlip}/dft')
        for key in bcdata[mlip].keys():
            os.mkdir(f'poscars/{mlip}/mlip/{key}')
            os.mkdir(f'poscars/{mlip}/dft/{key}')
            bcdata[mlip][key]['displacement'] = {}
            for tp in ['fmsg', 'fmdb', 'stsg', 'stdb']:
                mlip_structure = bcdata[mlip][key]['slabs'][tp]['structure']
                dft_structure = Structure.from_dict(dftdata['slabs'][key][tp]['structure'])
                mlip_structure.to_file(f'poscars/{mlip}/mlip/{key}/{tp}_POSCAR')
                dft_structure.to_file(f'poscars/{mlip}/dft/{key}/{tp}_POSCAR')
                if 'selective_dynamics' in mlip_structure.site_properties.keys():
                    moving_atom_num = len(np.where(np.all(array(mlip_structure.site_properties['selective_dynamics']), axis = 1) == True)[0])
                else:
                    moving_atom_num = len(mlip_structure)
                dis_fracs = abs(mlip_structure.frac_coords - dft_structure.frac_coords)
                dis_fracs = abs(dis_fracs - np.round(dis_fracs))
                dis_fracs = dot(mlip_structure.lattice.matrix.T, dis_fracs.T).T
                bcdata[mlip][key]['displacement'][tp] = sum(norm(dis_fracs, axis = 1))/moving_atom_num
    
            mlip_structure = bcdata[mlip][key]['best_it']['structure']
            dft_structure = Structure.from_dict(dftdata['ht'][mlip][key]['structure'])
            mlip_structure.to_file(f'poscars/{mlip}/mlip/{key}/ht_POSCAR')
            dft_structure.to_file(f'poscars/{mlip}/dft/{key}/ht_POSCAR')
            if 'selective_dynamics' in mlip_structure.site_properties.keys():
                moving_atom_num = len(np.where(np.all(array(mlip_structure.site_properties['selective_dynamics']), axis = 1) == True)[0])
            else:
                moving_atom_num = len(mlip_structure)
            dis_fracs = abs(mlip_structure.frac_coords - dft_structure.frac_coords)
            dis_fracs = abs(dis_fracs - np.round(dis_fracs))
            dis_fracs = dot(mlip_structure.lattice.matrix.T, dis_fracs.T).T
            bcdata[mlip][key]['displacement']['ht'] = sum(norm(dis_fracs, axis = 1))/moving_atom_num
            all_dps.append(bcdata[mlip][key]['displacement']['ht'])
            all_itEs.append(dftdata['derived'][mlip][key]['it'])

    num_keys = len(bcdata['orb-models'].keys())
    num_mlips = len(mlips)
    fig, axs = plt.subplots(num_mlips, 2, figsize=(num_keys*0.5, num_mlips*1.5))
    count = 0
    plt.subplots_adjust(wspace=0.5)
    for mlip in mlips:
        keys = []
        mlip_it_Es = []
        dft_it_Es = []
        disps = []
        for key in bcdata[mlip].keys():
            if dftdata['derived'][mlip][key]['it'] != 0:
                mlip_it_Es.append(bcdata[mlip][key]['best_it']['it_E'])
                dft_it_Es.append(dftdata['derived'][mlip][key]['it'])
                keys.append(f'({key[0]},{key[1]})')
                disps.append(bcdata[mlip][key]['displacement']['ht'])
        disps = sort_list(disps, dft_it_Es)
        mlip_it_Es = sort_list(mlip_it_Es, dft_it_Es)
        keys = sort_list(keys, dft_it_Es)
        dft_it_Es = sort_list(dft_it_Es, dft_it_Es)
        axs[count][0].bar(keys, dft_it_Es, alpha = 0.5)
        axs[count][0].bar(keys, mlip_it_Es, alpha = 0.5)
        axs[count][0].set_ylim(0, max(all_itEs)+0.1)
        axs[count][0].set_ylabel(r'$E_{it}$' + ' J/m$^2$', fontsize = 15)
        axs[count][0].set_yticklabels(axs[count][0].get_yticklabels(), fontsize = 10)
        axs[count][1].bar(keys, disps, alpha = 1)
        axs[count][1].set_ylim(0, max(all_dps)+0.01)
        axs[count][1].set_ylabel(r'$\Delta X $' + ' $\AA$', fontsize = 15)
        axs[count][1].yaxis.grid(True)
        axs[count][0].yaxis.grid(True)
        axs[count][0].text(0.05, 0.88, f'{name}({mlip_name_dict[mlip]})',
            transform=axs[count][0].transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        for i in range(2):
            axs[count][i].set_xticklabels(axs[count][0].get_xticklabels(), rotation=45, fontsize = 8)
            axs[count][i].tick_params(axis = 'x', which='major', pad=0)
            axs[count][i].tick_params(axis = 'y', which='major', pad=0)
            #axs[count][i].yaxis.set_label_coords(-0.5, 0.5)
        count+=1
    plt.tight_layout()
    fig.savefig('mlip_bcmk.jpg', format ='jpg', dpi = 600)

def get_average_distance(stct_1, stct_2, smt):
    """
    Calculate average atomic distance between two structures.

    Computes the root mean square distance between corresponding atoms
    in two structures after optimal alignment using StructureMatcher.

    Args:
        stct_1: First structure
        stct_2: Second structure
        smt: StructureMatcher object for alignment

    Returns:
        float: RMS distance between structures, or inf if alignment fails
    """
    stct_2 = smt.get_s2_like_s1(stct_1, stct_2)
    if stct_2 is None:
        return np.inf
    else:
        distances = np.dot(stct_1.lattice.matrix.T, (stct_2.frac_coords - stct_1.frac_coords).T).T
        return (np.sum(distances**2)/len(distances))**0.5

def get_non_matching_structures(stcts, tol, smt):
    nms = []
    for i in range(len(stcts)):
        if len(nms) == 0:
            nms.append(i)
        else:
            existing = False
            for j in nms:
                if get_average_distance(stcts[j], stcts[i], smt) < tol:
                    existing = True
                    break
            if not existing:
                nms.append(i)
    return nms
