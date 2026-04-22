"""
InterOptimus Matching Module

This module provides classes and functions to analyze lattice matching
between crystal structures using pymatgen's SubstrateAnalyzer. It includes
symmetry analysis to identify equivalent matches and terminations.
"""

from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces import CoherentInterfaceBuilder
from InterOptimus.equi_term import get_non_identical_slab_pairs, co_point_group_operations
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from interfacemaster.cellcalc import get_primitive_hkl
from interfacemaster.hetero_searching import round_int, apply_function_to_array, \
float_to_rational, rational_to_float, get_rational_mtx, plane_set_transform, plane_set
from numpy import *
import numpy as np
from numpy.linalg import *
from pymatgen.analysis.structure_matcher import StructureMatcher
# from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from InterOptimus.equi_term import pair_fit
from InterOptimus.tool import sort_list
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.linalg import polar
from pymatgen.util.coord import in_coord_list
from interfacemaster.cellcalc import MID
from numpy.typing import ArrayLike

from collections.abc import Sequence
import itertools
from pymatgen.core.surface import _is_in_miller_family

import re
import json
import builtins
from pathlib import Path


def get_symmetrically_equivalent_miller_indices(
    structure: Structure,
    miller_index: tuple[int, ...],
) -> list:
    """Symmetrically equivalent Miller indices as 3-tuple (h, k, l).

    Uses ``structure.lattice.get_recp_symmetry_operation()`` only (no trigonal
    primitive-cell branch). Output is always hkl; there is no hkil conversion.
    """
    # Convert to hkl if hkil, because in_coord_list only handles tuples of 3
    if len(miller_index) >= 3:
        _miller_index: tuple[int, ...] = (
            miller_index[0],
            miller_index[1],
            miller_index[-1],
        )
    else:
        _miller_index = (miller_index[0], miller_index[1], miller_index[2])

    max_idx = max(np.abs(miller_index))
    idx_range = list(range(-max_idx, max_idx + 1))
    idx_range.reverse()

    # Skip crystal system analysis if already given
    symm_ops = structure.lattice.get_recp_symmetry_operation()

    equivalent_millers: list[tuple[int, int, int]] = [_miller_index]  # type: ignore[list-item]
    for miller in itertools.product(idx_range, idx_range, idx_range):
        if miller == _miller_index:
            continue

        if builtins.any(idx != 0 for idx in miller):
            if _is_in_miller_family(miller, equivalent_millers, symm_ops):
                equivalent_millers += [miller]

            # Include larger Miller indices in the family of planes
            if (
                all(max_idx > i for i in np.abs(miller))
                and not in_coord_list(equivalent_millers, miller)
                and _is_in_miller_family(max_idx * np.array(miller), equivalent_millers, symm_ops)
            ):
                equivalent_millers += [miller]

    return equivalent_millers

def get_identical_pairs(match, film, substrate):
    """
    Get all symmetrically equivalent matching pairs for given Miller indices.

    Generates all combinations of symmetrically equivalent Miller indices
    for both film and substrate that produce equivalent interfaces.

    Args:
        match (tuple): (film_miller, substrate_miller) pair
        film (Structure): Film structure
        substrate (Structure): Substrate structure

    Returns:
        list: List of tuples containing equivalent (film_miller, substrate_miller) pairs
    """
    film_idtc_millers = get_symmetrically_equivalent_miller_indices(film, match[0])
    substrate_idtc_millers = get_symmetrically_equivalent_miller_indices(substrate, match[1])

    combs = []
    for i in film_idtc_millers:
        for j in substrate_idtc_millers:
            combs.append((i, j))

    return combs

class equi_directions_identifier:
    """
    Identify whether two directions in a crystal structure are equivalent.

    Uses symmetry operations to determine if two direction vectors are
    equivalent under the crystal's space group symmetry.
    """

    def __init__(self, structure):
        """
        Initialize with a crystal structure.

        Args:
            structure (Structure): pymatgen Structure object
        """
        analyzer = SpacegroupAnalyzer(structure)
        self.symmetry_operations = analyzer.get_symmetry_operations(cartesian=True)

    def identify(self, v1, v2):
        """
        Check if two direction vectors are equivalent under symmetry.

        Args:
            v1 (array): First direction vector
            v2 (array): Second direction vector

        Returns:
            bool: True if directions are equivalent, False otherwise
        """
        direction1 = v1 / norm(v1)
        direction2 = v2 / norm(v2)
        are_equivalent = False

        for operation in self.symmetry_operations:
            transformed_direction1 = operation.operate(direction1)
            if norm(cross(transformed_direction1, direction2)) < 1e-2:
                are_equivalent = True
                break

        return are_equivalent

class equi_match_identifier:
    """
    Determine whether two lattice matches are identical under symmetry.

    Analyzes matching pairs between substrate and film slabs to identify
    equivalent configurations that produce the same interface structure.
    """

    def __init__(self, substrate, film, substrate_conv, film_conv):
        """
        Initialize with substrate and film structures.

        Args:
            substrate: Substrate slab structure
            film: Film slab structure
            substrate_conv: Conventional substrate unit cell
            film_conv: Conventional film unit cell
        """
        self.film = film
        self.substrate = substrate
        self.film_conv = film_conv
        self.substrate_conv = substrate_conv
        self.substrate_equi_directions_identifier = equi_directions_identifier(substrate_conv)
        self.film_equi_directions_identifier = equi_directions_identifier(film_conv)
    
    def identify_by_indices_matching(self, match_1, match_2):
        """
        Check if two matches are equivalent based on lattice vector indices.

        Compares whether two matching configurations produce the same interface
        by checking if their substrate and film supercell vectors are equivalent
        under symmetry operations.

        Args:
            match_1: First match object to compare
            match_2: Second match object to compare

        Returns:
            bool: True if matches are equivalent, False otherwise
        """
        equivalent = False
        substrate_set_1, substrate_set_2 = match_1.substrate_sl_vectors, match_2.substrate_sl_vectors
        film_set_1, film_set_2 = match_1.film_sl_vectors, match_2.film_sl_vectors
        """
        substrate_set_1 = around(dot(inv(self.substrate_conv.lattice.matrix.T), \
                                                        match_1.substrate_sl_vectors.T),8).T
        substrate_set_2 = around(dot(inv(self.substrate_conv.lattice.matrix.T), \
                                                        match_2.substrate_sl_vectors.T),8).T
        film_set_1 = around(dot(inv(self.film_conv.lattice.matrix.T), \
                                                        match_1.film_sl_vectors.T),8).T
        film_set_2 = around(dot(inv(self.film_conv.lattice.matrix.T), \
                                                        match_2.film_sl_vectors.T),8).T
        """
        if (
            self.substrate_equi_directions_identifier.identify(substrate_set_1[0], substrate_set_2[0]) \
            and self.substrate_equi_directions_identifier.identify(substrate_set_1[1], substrate_set_2[1]) \
            and self.film_equi_directions_identifier.identify(film_set_1[0], film_set_2[0]) \
            and self.film_equi_directions_identifier.identify(film_set_1[1], film_set_2[1])
            ) or (
            self.substrate_equi_directions_identifier.identify(substrate_set_1[0], substrate_set_2[1]) \
            and self.substrate_equi_directions_identifier.identify(substrate_set_1[1], substrate_set_2[0]) \
            and self.film_equi_directions_identifier.identify(film_set_1[0], film_set_2[1]) \
            and self.film_equi_directions_identifier.identify(film_set_1[1], film_set_2[0])
            ):
            equivalent = True
        return equivalent
    
    def identify_by_stct_matching(self, match_1, match_2):
        """
        Check if two matches produce equivalent interface structures.

        Uses StructureMatcher to compare the actual interface structures
        generated from two different matches to determine if they are equivalent.

        Args:
            match_1: First match object to compare
            match_2: Second match object to compare

        Returns:
            bool: True if the interface structures are equivalent, False otherwise
        """
        #matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True, scale = True)
        matcher = StructureMatcher(primitive_cell=True)
        matches = [match_1, match_2]
        its = []
        for i in range(2):
            cib = CoherentInterfaceBuilder(film_structure=self.film,
                                   substrate_structure=self.substrate,
                                   film_miller=matches[i].film_miller,
                                   substrate_miller=matches[i].substrate_miller,
                                   zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=0.1, label_index=True,\
                                   filter_out_sym_slabs=False)
            #print(cib.terminations)
            cib.zsl_matches = [matches[i]]
            its.append(list(cib.get_interfaces(termination = cib.terminations[0], substrate_thickness = 3,
                                                           film_thickness = 3,
                                                           vacuum_over_film=10,
                                                           gap=1))[0])
        return matcher.fit(its[0], its[1])

def get_cos(v1, v2):
    """
    Calculate cosine of the angle between two vectors.

    Computes the cosine similarity between two vectors, which represents
    the angle between them normalized to [-1, 1].

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        float: Cosine of the angle between the vectors
    """
    return dot(v1, v2) / (norm(v1) * norm(v2))
    
def get_area_match(match):
    """
    Calculate the area of the matching interface.

    Computes the area of the supercell formed by the substrate supercell
    vectors, which represents the interface area for the given match.

    Args:
        match: Match object containing substrate supercell vectors

    Returns:
        float: Interface area in Angstroms squared
    """
    return norm(cross(match.substrate_sl_vectors[0], match.substrate_sl_vectors[1]))


def match_search(substrate, film, substrate_conv, film_conv, sub_analyzer, film_millers, substrate_millers):
    """
    Search for lattice matches between substrate and film structures.

    Performs comprehensive lattice matching analysis using SubstrateAnalyzer
    to find all possible epitaxial relationships. Groups equivalent matches
    that produce the same interface under symmetry operations.

    Args:
        substrate (Structure): Primitive structure of the substrate material
        film (Structure): Primitive structure of the film material
        substrate_conv: Conventional substrate unit cell
        film_conv: Conventional film unit cell
        sub_analyzer: SubstrateAnalyzer with matching parameters
        film_millers: List of Miller indices for film surfaces
        substrate_millers: List of Miller indices for substrate surfaces

    Returns:
        tuple: (unique_matches, equivalent_matches, unique_areas)
            - unique_matches: List of non-identical matches
            - equivalent_matches: Clustered equivalent matches
            - unique_areas: List of matching areas for unique matches
    """
    matches = list(sub_analyzer.calculate(film=film, substrate=substrate, film_millers = film_millers, substrate_millers = substrate_millers))
    print(len(matches))
    areas = []
    for i in matches:
        areas.append(get_area_match(i))
    vstrains = []
    for i in matches:
        vstrains.append(i.von_mises_strain)
    matches = sort_list(matches, vstrains)
    unique_angles = []
    unique_matches = []
    equivalent_matches = []
    unique_areas = []
    ins_equi_match_identifier = equi_match_identifier(substrate, film, substrate_conv, film_conv)
    from tqdm.notebook import tqdm
    with tqdm(total = len(matches), desc = "checking matching identity") as rgst_pbar:
        for i in range(len(matches)):
            angle_here = get_cos(matches[i].substrate_sl_vectors[0],\
                                                           matches[i].substrate_sl_vectors[1])
            if i == 0:
                unique_matches.append(matches[i])
                equivalent_matches.append([matches[i]])
                unique_angles.append(angle_here)
                unique_areas.append(get_area_match(matches[i]))
            else:
                equivalent = False
                same_angle_ids = where(abs(array(unique_angles) - angle_here) < 1e-1)[0]
                if len(same_angle_ids) > 0:
                    for j in same_angle_ids:
                        #indices matching firstly
                        if ins_equi_match_identifier.identify_by_indices_matching(matches[i], unique_matches[j]):
                            equivalent = True
                        #if indices match, check structure match
                        else:
                            equivalent = ins_equi_match_identifier.identify_by_stct_matching(matches[i], unique_matches[j])
                                         
                        if equivalent:
                            equivalent_matches[j].append(matches[i])
                            equivalent = True
                            break
                if not equivalent:
                    unique_matches.append(matches[i])
                    equivalent_matches.append([matches[i]])
                    unique_angles.append(angle_here)
                    unique_areas.append(get_area_match(matches[i]))
            rgst_pbar.update(1)
    return unique_matches, equivalent_matches, unique_areas

class MatchIdentifier:
    def __init__(self, substrate_conv, film_conv):
        self.film_symops = SpacegroupAnalyzer(film_conv).get_point_group_operations()
        self.substrate_symops = SpacegroupAnalyzer(substrate_conv).get_point_group_operations()
        self.prod_symops = co_point_group_operations(self.film_symops, self.substrate_symops)
    def is_equivalent(self, normal_1, MR_1, normal_2, MR_2, tol=0.1):
        """
        Check if two interface orientations are equivalent under symmetry.

        Determines whether two interface configurations defined by their
        normal vectors and rotation matrices are equivalent considering
        the combined point group symmetry of film and substrate.

        Args:
            normal_1: Normal vector of first interface
            MR_1: Rotation matrix of first interface
            normal_2: Normal vector of second interface
            MR_2: Rotation matrix of second interface
            tol (float): Tolerance for matrix comparison

        Returns:
            bool: True if interfaces are symmetrically equivalent
        """
        disorient = dot(MR_1, inv(MR_2))
        for symop_out in self.prod_symops:
            if np.allclose(disorient, symop_out.rotation_matrix, atol=tol):
                for symop_in in self.substrate_symops:
                    if np.allclose(normal_1, dot(symop_in.rotation_matrix, normal_2), atol=tol):
                        return True
        return False
"""
def match_search(substrate, film, substrate_conv, film_conv, sub_analyzer, film_millers, substrate_millers):
    #
    given substrate, film lattice structures, \
    get non-identical matches and identical match groups
    
    Args:
    substrate (Structure): primitive structure of the substrate material
    film (Structure): primitive structure of the film material
    
    Return:
    unique_matches (list): list of non-identical matches.
    equivalent_matches (list): clustered identical matches.
    unique_areas (list): list of matching areas of non-identical matches
    #
    matches = list(sub_analyzer.calculate(film=film, substrate=substrate, film_millers = film_millers, substrate_millers = substrate_millers))
    print(len(matches))
    areas = []
    for i in matches:
        areas.append(get_area_match(i))
    matches = sort_list(matches, areas)
    #unique_normals = []
    #unique_MRs = []
    unique_matches = []
    equivalent_matches = []
    unique_areas = []
    #match_identifier = MatchIdentifier(substrate_conv, film_conv)
    emi = equi_match_identifier(substrate, film, substrate_conv, film_conv)
    from tqdm.notebook import tqdm
    with tqdm(total = len(matches), desc = "checking matching identity") as rgst_pbar:
        for i in range(len(matches)):
            #normal_here = cross(matches[i].substrate_sl_vectors[0], matches[i].substrate_sl_vectors[1])
            #normal_here = normal_here/norm(normal_here)
            #MR_here, T_here = polar(matches[i].match_transformation)
        
            if i == 0:
                unique_matches.append(matches[i])
                equivalent_matches.append([matches[i]])
                unique_areas.append(get_area_match(matches[i]))
                #unique_normals.append(normal_here)
                #unique_MRs.append(MR_here)
            else:
                if len(unique_matches) == 0:
                    unique_matches.append(matches[i])
                    equivalent_matches.append([matches[i]])
                    unique_areas.append(get_area_match(matches[i]))
                    #unique_normals.append(normal_here)
                    #unique_MRs.append(MR_here)
                else:
                    equivalent = False
                    for j in range(len(unique_matches)):
                        #
                        normal_comp = cross(matches[j].substrate_sl_vectors[0], matches[j].substrate_sl_vectors[1])
                        normal_comp = normal_comp/norm(normal_comp)
                        MR_comp, T_comp = polar(matches[j].match_transformation)
                        
                        equivalent = match_identifier.is_equivalent(normal_here, MR_here, normal_comp, MR_comp)
                        if equivalent:
                            equivalent_matches[j].append(matches[i])
                            break
                        
                        else:
                        #
                        if emi.identify_by_stct_matching(matches[i], unique_matches[j]):
                            equivalent = True
                            break
                        
                    if not equivalent:
                        unique_matches.append(matches[i])
                        equivalent_matches.append([matches[i]])
                        unique_areas.append(get_area_match(matches[i]))
                        #unique_normals.append(normal_here)
                        #unique_MRs.append(MR_here)
            rgst_pbar.update(1)
    return unique_matches, equivalent_matches, unique_areas
"""
class convert_info_forma:
    """
    class to generate matching indices information
    """
    def __init__(self, substrate_conv, film_conv):
        """
        Args:
        substrate_conv (Structure): conventional substrate structure
        film_conv (Structure): conventional film structure
        """
        substrate_prim = substrate_conv.get_primitive_structure()
        film_prim = film_conv.get_primitive_structure()
        self.substrate_conv_lattice = substrate_conv.lattice.matrix.T
        self.film_conv_lattice = film_conv.lattice.matrix.T
        self.substrate_prim_lattice = substrate_prim.lattice.matrix.T
        self.film_prim_lattice = film_prim.lattice.matrix.T
        
    def convert_to_conv(self, match):
        """
        convert primitive indices into conventional indices
        
        Args:
        match (dict): matching information
        
        Return:
        (dict): matching information by indices represented in both primitive & conventional structures
        """
        substrate_prim_sl_vecs_int = around(dot(inv(self.substrate_prim_lattice), \
                                                match.substrate_sl_vectors.T),8).T
        film_prim_sl_vecs_int = around(dot(inv(self.film_prim_lattice), \
                                           match.film_sl_vectors.T),8).T
                                           
        substrate_conv_sl = dot(inv(self.substrate_conv_lattice), \
                                                match.substrate_sl_vectors.T).T
        film_conv_sl = dot(inv(self.film_conv_lattice), \
                                                match.film_sl_vectors.T).T
        
        substrate_prim_plane_set = plane_set(self.substrate_prim_lattice, match.substrate_miller, \
                                             substrate_prim_sl_vecs_int[0], substrate_prim_sl_vecs_int[1])
        film_prim_plane_set = plane_set(self.film_prim_lattice, match.film_miller, \
                                             film_prim_sl_vecs_int[0], film_prim_sl_vecs_int[1])
        substrate_conv_plane_set = plane_set_transform(substrate_prim_plane_set, self.substrate_conv_lattice, 'rational')
        film_conv_plane_set = plane_set_transform(film_prim_plane_set, self.film_conv_lattice, 'rational')
        return {
        'substrate_primitive_miller':substrate_prim_plane_set.hkl,
        'film_primitive_miller':film_prim_plane_set.hkl,
            
        'substrate_conventional_miller':substrate_conv_plane_set.hkl,
        'film_conventional_miller':film_conv_plane_set.hkl,
        
        'substrate_primitive_vectors':vstack((substrate_prim_plane_set.v1, substrate_prim_plane_set.v2)),
        'film_primitive_vectors':vstack((film_prim_plane_set.v1, film_prim_plane_set.v2)),
        
        'substrate_conventional_vectors':vstack((substrate_conv_plane_set.v1, substrate_conv_plane_set.v2)),
        'film_conventional_vectors':vstack((film_conv_plane_set.v1, film_conv_plane_set.v2)),
        
        'substrate_conventional_vectors_float': substrate_conv_sl,
        'film_conventional_vectors_float': film_conv_sl}
        
def get_area(v1, v2):
    """
    Calculate the area spanned by two vectors.

    Computes the magnitude of the cross product of two vectors,
    which gives the area of the parallelogram they span.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        float: Area of the parallelogram spanned by v1 and v2
    """
    return norm(cross(v1,v2))

def interface_searching(substrate_conv, film_conv, sub_analyzer, film_millers=None, substrate_millers=None):
    """
    Perform comprehensive interface searching between substrate and film.

    Conducts lattice matching analysis and generates all possible interface
    configurations. Converts matching results to both primitive and conventional
    Miller indices representations.

    Args:
        substrate_conv: Conventional substrate unit cell
        film_conv: Conventional film unit cell
        sub_analyzer: SubstrateAnalyzer with matching parameters
        film_millers: List of Miller indices for film surfaces (optional)
        substrate_millers: List of Miller indices for substrate surfaces (optional)

    Returns:
        tuple: (unique_matches, equivalent_matches, unique_matches_indices_data,
                equivalent_matches_indices_data, areas)
            - unique_matches: List of non-identical matches
            - equivalent_matches: Clustered equivalent matches
            - unique_matches_indices_data: Miller indices data for unique matches
            - equivalent_matches_indices_data: Clustered Miller indices data
            - areas: List of matching areas for unique matches
    """
    unique_matches, equivalent_matches, areas = \
    match_search(substrate_conv.get_primitive_structure(),\
                 film_conv.get_primitive_structure(),\
                 substrate_conv,\
                 film_conv,\
                 sub_analyzer, film_millers, substrate_millers)
    unique_matches_indices_data = []
    equivalent_matches_indices_data = []
    
    is_convert_info_forma = convert_info_forma(substrate_conv, film_conv)
    for i in unique_matches:
        #areas.append(get_area(i.substrate_sl_vectors[0], i.substrate_sl_vectors[1]))
        unique_matches_indices_data.append(is_convert_info_forma.convert_to_conv(i))
    id = 0
    for i in equivalent_matches:
        equivalent_matches_indices_data.append([])
        for j in i:
            equivalent_matches_indices_data[id].append(is_convert_info_forma.convert_to_conv(j))
        id += 1
    
    #unique_matches_sorted = sort_list(unique_matches, areas)
    #unique_matches_indices_data_sorted = sort_list(unique_matches_indices_data, areas)
    #equivalent_matches_indices_data_sorted = sort_list(equivalent_matches_indices_data, areas)
    #areas_sorted = sort_list(areas, areas)

    return  unique_matches, \
            equivalent_matches, \
            unique_matches_indices_data,\
            equivalent_matches_indices_data,\
            areas

def miller_to_cartesian(miller, lattice):
    """
    Convert Miller indices to Cartesian normal vectors.

    Transforms crystallographic Miller indices to Cartesian coordinates
    using the reciprocal lattice vectors of the given lattice.

    Args:
        miller (tuple): Miller indices (h, k, l)
        lattice: pymatgen Lattice object

    Returns:
        ndarray: Normalized Cartesian normal vector
    """
    h, k, l = miller
    recip_lattice = lattice.reciprocal_lattice_crystallographic
    normal = h * recip_lattice.matrix[0] + k * recip_lattice.matrix[1] + l * recip_lattice.matrix[2]
    return normal/np.linalg.norm(normal)

def stereographic_projection(normal):
    """
    Project a normal vector onto a 2D stereographic projection plane.

    Performs stereographic projection of a 3D unit vector onto a 2D plane,
    commonly used for visualizing crystal orientations and pole figures.

    Args:
        normal (tuple): 3D unit vector (x, y, z)

    Returns:
        tuple: (X, Y) coordinates in the stereographic projection
    """
    x, y, z = normal
    if abs(np.around(z, 4)) == 1:
        X,Y = x,y
        
    elif z < 0:
        X = x / (1 - z)
        Y = y / (1 - z)
    else:
        X = x / (1 + z)
        Y = y / (1 + z)
    return X, Y

def format_miller_index(miller_index):
    """
    Format Miller indices for LaTeX/matplotlib display.

    Converts Miller indices to LaTeX format with overbars for negative values,
    suitable for mathematical rendering in plots.

    Args:
        miller_index (tuple): Miller indices (h, k, l)

    Returns:
        str: LaTeX-formatted Miller index string
    """
    h, k, l = miller_index
    def format_component(c):
        if c < 0:
            return r"\bar{" + f"{abs(c)}" + r"}"
        else:
            return str(c)
    return r"$(" + format_component(h) + format_component(k) + format_component(l) + r")$"

def scatter_by_miller_dict(millers, dict, tuple_id, lattice, strains):
    """
    Organize matching data for stereographic projection plotting.

    Processes matching data dictionary to extract stereographic projection
    coordinates and associated strain information for visualization.

    Args:
        millers (list): List of Miller indices to process
        dict (dict): Matching data dictionary
        tuple_id (int): Index in the tuple key (0 for film, 1 for substrate)
        lattice: Crystal lattice for coordinate transformation
        strains (array): Strain values corresponding to match types

    Returns:
        dict: Processed data with stereographic coordinates and strain info
    """
    found_data = {}
    for miller in millers:
        for i in list(dict.keys()):
            if allclose(miller, i[tuple_id]):
                if miller not in list(found_data.keys()):
                    found_data[miller] = {'type_list':list(dict[i].keys())}
                    found_data[miller]['XY'] = stereographic_projection(miller_to_cartesian(miller, lattice))
                else:
                    for j in list(dict[i].keys()):
                        if j not in found_data[miller]['type_list']:
                            found_data[miller]['type_list'].append(j)
    for i in found_data.keys():
        found_data[i]['type_list'] = array(found_data[i]['type_list'])[argsort(found_data[i]['type_list'])]
        found_data[i]['strains'] = strains[found_data[i]['type_list']]
    return found_data

def draw_circles(ax, data, existing_label, dotscatter):
    """
    Draw circles on stereographic projection for matching data.

    Creates circular markers on the stereographic projection plot to
    represent different types of lattice matches.

    Args:
        ax: Matplotlib axes object
        data (dict): Matching data with stereographic coordinates
        existing_label (list): List of already plotted labels
        dotscatter (bool): Whether to use dot scatter style

    Returns:
        tuple: (updated_existing_label, circle_size)
    """
    for i in range(len(data['type_list'])):
        if dotscatter:
            center_c = f"C{data['type_list'][i]+3}"
            center_s = 300
            center_ap = 0.5
        else:
            center_c = 'none'
            center_s = ((i+1)*16)**2
            center_ap = 0.7
        if data['type_list'][i] not in existing_label:
            ax.scatter(around(data['XY'][0],3), around(data['XY'][1],3), c=center_c,marker='o',edgecolors=f"C{data['type_list'][i]+3}", \
                       s = center_s, label = f"Type {data['type_list'][i]}", linewidths =7, alpha = center_ap)
            existing_label.append(data['type_list'][i])
        else:
            ax.scatter(around(data['XY'][0],3), around(data['XY'][1],3), c=center_c,marker='o',edgecolors=f"C{data['type_list'][i]+3}", s = center_s, linewidths =7, alpha = center_ap)
        if dotscatter:
            ax.scatter(around(data['XY'][0],3), around(data['XY'][1],3), c='none',marker='o', s = 10, alpha = 1)
            ax.scatter(around(data['XY'][0],3), around(data['XY'][1],3), c='none',marker='o',edgecolors=f"C{data['type_list'][i]+3}", s = center_s, linewidths =7, alpha = 1)
    return existing_label, ((i+1)*16)**2
    

def plot_matching_data(matching_data, titles, save_filename, show_millers, show_legend, show_title, special):
    """
    Create stereographic projection plots for lattice matching results.

    Generates comprehensive visualization of lattice matching data using
    stereographic projection, showing film and substrate orientations
    with corresponding match types and Miller indices.

    Args:
        matching_data (list): List of matching data dictionaries for film/substrate
        titles (list): Titles for film and substrate subplots
        save_filename (str): Output filename for the plot
        show_millers (bool): Whether to display Miller indices on the plot
        show_legend (bool): Whether to show the legend
        show_title (bool): Whether to show subplot titles
        special (bool): Special plotting mode flag
    """
    fig, ax = plt.subplots(1, 2, figsize=(20*1.25, 12*1.25))
    # Opt out of global figure.autolayout (see InterOptimus.__init__): manual tight_layout below.
    fig.set_layout_engine("none")
    #plt.rc('font', family='arial')
    #plt.rc('text', usetex=True)
    plt.subplots_adjust(wspace=0.01)
    for i in range(2):
        XYs = []
        existing_label = []
        existing_label_ids = []
        for k in list(matching_data[i].keys()):
           XYs.append([matching_data[i][k]['XY'][0], matching_data[i][k]['XY'][1]])
        XYs = np.array(around(XYs,3))
        projected = []
        already_done = []
        sampled_Xt_Yt = []
        sampled_X_Y = []
        for j in matching_data[i].keys():
            X, Y = matching_data[i][j]['XY']
            X = around(X,3)
            Y = around(Y,3)
            if abs(Y) < 1e-2:
                Y_t = Y + 0.11
                #Y_t = Y
            else:
                Y_t = Y + Y/abs(Y)*0.11
                #Y_t = Y
            if abs(X) < 1e-2:
                X_t = X
            else:
                X_t = X
            n = len(XYs[(abs(XYs[:,0] - X)<1e-2) & (abs(XYs[:,1] - Y)<1e-2)])
            #print(np.linalg.norm([X, Y]))
            #print(XYs[(abs(XYs[:,0] - X)<1e-2) & (abs(XYs[:,1] - Y)<1e-2)])
            #print(abs(XYs[:,0] - X), abs(XYs[:,0] - Y))
            if n < 2:
                if show_millers:
                    ax[i].text(X, Y_t, format_miller_index(j), fontsize=25, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.15'))
                else:
                    if show_millers or (abs(X) < 1e-2 and abs(Y) < 1e-2):
                        ax[i].text(X, Y_t, format_miller_index(j), fontsize=25, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.15'))
                existing_label, circle_s = draw_circles(ax[i], matching_data[i][j], existing_label, False)
            else:
                if [around(X,2), around(Y,2)] not in already_done:
                    if show_millers and (np.linalg.norm([X, Y]) > 0.9 or np.linalg.norm([X, Y]) < 0.01):
                        ax[i].text(X_t+0.12, Y_t, format_miller_index(j), fontsize=25, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.15'))
                    existing_label, circle_s = draw_circles(ax[i], matching_data[i][j], existing_label, False)
                    already_done.append([around(X,2), around(Y,2)])
                    sampled_Xt_Yt.append([X_t, Y_t])
                    sampled_X_Y.append([X, Y])
                else:
                    dis = norm(array(sampled_Xt_Yt) - array([X, Y]), axis = 1)
                    X_t_h, Y_t_h = array(sampled_Xt_Yt)[argsort(dis)[0]]
                    if show_millers and (np.linalg.norm([X, Y]) > 0.9  or (abs(X) < 1e-2 and abs(Y) < 1e-2)):
                        ax[i].text(X_t_h-0.12, Y_t_h, format_miller_index(j), fontsize=25, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.15'))
                    #ax[i].text(X_t, Y_t, ', ', fontsize=15, ha='center', va='center')
                    #existing_label = draw_circles(ax[i], matching_data[i][j], existing_label)
                    #existing_label, circle_s = draw_circles(ax[i], matching_data[i][j], existing_label, False)
            #ax[i].text(X_t, Y_t, format_miller_index(j), fontsize=20, ha='center', va='center')
            projected.append([X, Y])
            if X == 0 and Y == 0:
                have_zero = True
        projected = np.array(projected)

        ax[i].set_aspect('equal')
        ax[i].set_xlim([-1.25, 1.25])
        ax[i].set_ylim([-1.25, 1.25])
        
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        add_stereographic_guides(ax[i], projected)
        #ax[i].set_frame_on(False)
        if show_title:
            ax[i].set_title(titles[i], fontsize = 40)
        # 获取图形中的句柄和标签
        handles, labels = ax[i].get_legend_handles_labels()
        # 根据标签的字母顺序进行排序
        sorted_handles_labels = sorted(zip(labels, handles, existing_label), key=lambda x: x[2])
        # 解压缩排序后的句柄和标签
        sorted_labels, sorted_handles, existing_label = zip(*sorted_handles_labels)
        # 设置 legend，并按照排序后的顺序显示
        if show_legend:
            custom_labels = []
            for tp_num in range(len(sorted_labels)):
                custom_labels.append(
                                         Line2D([0], [0], marker='o', color = 'w', \
                                                label=f'Type {tp_num}', markerfacecolor='none', \
                                                markeredgecolor=f"C{tp_num+3}", markersize=32, markeredgewidth =7, alpha=0.7)
                                        )
                
            #ax[i].legend(sorted_handles, sorted_labels, fontsize = 12, labelspacing=0.5, ncol=int(len(sorted_labels)/2), loc='upper center', bbox_to_anchor=(0.5, 1.05))
            if i == 0:
                try:
                    ax[i].legend(
                                handles=custom_labels,
                                fontsize = 30,
                                labelspacing=0.5,
                                loc='lower center',
                                bbox_to_anchor=(0.5, -0.15),
                                ncol=int(len(sorted_labels)/2),
                                columnspacing=0.1,
                                handletextpad=0.05
                                )
                except:
                    pass

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(f'{save_filename}_all.jpg', dpi=600)



class EquiMatchSorter:
    def __init__(self, film, substrate, equivalent_matches_indices_data, unique_matches):
        self.film = film
        self.substrate = substrate
        self.equivalent_matches_indices_data = equivalent_matches_indices_data
        self.strains = []
        for i in unique_matches:
            self.strains.append(i.von_mises_strain)
        self.strains = array(self.strains)
        self.sort_zsl_match_results()
        self.generate_all_match_data()
        self.get_indices_map()
        self.unique_matches = unique_matches
    def sort_zsl_match_results(self):
        """
        Sort and organize ZSL matching results by match types.

        Processes the equivalent matches data to create a dictionary
        mapping Miller index pairs to their corresponding match type IDs.
        """
        type_id = 0
        all_matche_data = {}
        for i in self.equivalent_matches_indices_data:
            for j in i:
                match = (tuple(j['film_conventional_miller']), tuple(j['substrate_conventional_miller']))
                if match not in all_matche_data.keys():
                    all_matche_data[match] = {type_id: 1}
                else:
                    if type_id not in all_matche_data[match].keys():
                        all_matche_data[match][type_id] = 1
                    else:
                        all_matche_data[match][type_id] += 1
            type_id += 1
        self.unique_matche_data = all_matche_data
    def generate_all_match_data(self):
        """
        Generate complete matching data including symmetrically equivalent pairs.

        Expands the unique match data to include all symmetrically equivalent
        Miller index combinations that produce the same interface.
        """
        new_dict = {}
        for i in self.unique_matche_data.keys():
            combs = get_identical_pairs(i, self.film, self.substrate)
            for j in combs:
                if j not in new_dict.keys():
                    new_dict[j] = self.unique_matche_data[i]
                else:
                    for k in self.unique_matche_data[i].keys():
                        new_dict[j][k] = self.unique_matche_data[i][k]
        self.all_matche_data = new_dict
        #print(new_dict)
    def get_indices_map(self):
        """
        Create mapping dictionaries for Miller indices.

        Generates lookup dictionaries mapping Miller indices to integer IDs
        for both film and substrate materials.
        """
        film_millers = []
        substrate_millers = []
        for i in self.all_matche_data.keys():
            film_millers.append(i[0])
            substrate_millers.append(i[1])
        film_millers = list(set(film_millers))
        substrate_millers = list(set(substrate_millers))
        self.film_map = {m_id:id for id, m_id in enumerate(film_millers)}
        self.substrate_map = {m_id:id for id, m_id in enumerate(substrate_millers)}
    def plot_matching_data(self, names=['film', 'substrate'], save_filename='stereographic_projection.jpg',
                          show_millers=True, show_legend=True, show_title=True, special=False):
        """
        Create stereographic projection visualization of matching results.

        Generates and saves stereographic projection plots showing the
        distribution of lattice matches for both film and substrate materials.

        Args:
            names (list): Names for film and substrate materials
            save_filename (str): Output filename for the plot
            show_millers (bool): Whether to display Miller indices
            show_legend (bool): Whether to show the legend
            show_title (bool): Whether to show subplot titles
            special (bool): Special plotting mode flag
        """
        film_matching_data = scatter_by_miller_dict(list(self.film_map.keys()), self.all_matche_data, 0, self.film.lattice, self.strains)
        substrate_matching_data = scatter_by_miller_dict(list(self.substrate_map.keys()), self.all_matche_data, 1, self.substrate.lattice, self.strains)
        matching_data = [film_matching_data, substrate_matching_data]
        self.matching_data = matching_data
        data = []
        with open(f'{names[0]}_matching_data','w') as f:
            f.write(f'(h k l) (X Y) [types]\n')
            for i in film_matching_data.keys():
                X, Y = film_matching_data[i]['XY']
                f.write(f"{i[0]} {i[1]} {i[2]} {X} {Y} {film_matching_data[i]['type_list']}\n" )
        with open(f'{names[1]}_matching_data','w') as f:
            f.write(f'(h k l) (X Y) [types]\n')
            for i in substrate_matching_data.keys():
                X, Y = substrate_matching_data[i]['XY']
                f.write(f"{i[0]} {i[1]} {i[2]} {X} {Y} {substrate_matching_data[i]['type_list']}\n" )
        plot_matching_data(matching_data, names, save_filename, show_millers, show_legend, show_title, special)
        #plot_matching_data_num(matching_data, names, save_filename)
        #plot_matching_data_strain(matching_data, names, save_filename)

    def plot_unique_matches(self, filename='unique_matches.jpg'):
        """
        Create bar plot showing properties of unique matches.

        Generates a dual-axis bar plot displaying matching areas and
        von Mises strains for all unique lattice matches.

        Args:
            filename (str): Output filename for the plot
        """
        x = []
        strains = []
        areas = []
        ct = 0
        for i in self.unique_matches:
            strains.append(i.von_mises_strain)
            areas.append(norm(cross(i.substrate_sl_vectors[0], i.substrate_sl_vectors[1])))
            x.append(ct)
            ct+=1

        #plt.rc('font', family='arial')
        #plt.rc('text', usetex=False)
        x = x
        y1 = areas
        y2 = strains

        width = 0.35
        x_pos = np.arange(len(x))
        offset = 0.1
        fig, ax1 = plt.subplots(figsize = (len(x)*2,5))
        ax1.bar(x_pos - width/2 + offset, y1, width, alpha=0.6, label='matching area', color ='C00')

        ax2 = ax1.twinx()

        ax2.bar(x_pos + width/2 + offset, array(y2)*100, width, alpha=0.6, label='strain', color ='C01')

        ax1.set_xlabel('Type', fontsize = 30)
        ax1.set_ylabel('Matching area ($\mathregular{\AA}^2$)', color='C00', fontsize = 30)
        ax2.set_ylabel('Strain (%)', color='C01', fontsize = 30)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20, color = 'C00', labelcolor = 'C00')
        ax2.tick_params(axis='y', labelsize=20, color = 'C01', labelcolor = 'C01')

        #fig.legend(loc='upper left', bbox_to_anchor=(0.1, 1.25), fontsize = 25)
        plt.tight_layout()
        fig.savefig(filename, dpi = 600, format='jpg')

#!/usr/bin/env python3
"""
Stereographic Projection Plot Program
For analyzing interfacial matching and binding energy distribution between two materials
"""

import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_area_strain_records(filename):
    """
    Parse ``area_strain`` rows into structured records.

    Supported formats:
    - ``(h1 k1 l1) (h2 k2 l2) area strain energy match_id``
    - ``(h1 k1 l1) (h2 k2 l2) area strain energy match_id term_id``
    - v2 (MLIP): ``... match_id term_id stereo_winner`` where ``stereo_winner`` is 0 or 1
    - VASP merge: optional ``dft_status`` string; if ``stereo_winner`` is present it is 0/1 and
      ``dft_status`` may follow as a 7th token.
    """
    records = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                first_open = line.find('(')
                first_close = line.find(')')
                second_open = line.find('(', first_close + 1)
                second_close = line.find(')', second_open + 1)

                if first_open == -1 or second_open == -1:
                    continue

                plane1_str = line[first_open + 1:first_close]
                plane1 = [float(x) for x in plane1_str.split()]

                plane2_str = line[second_open + 1:second_close]
                plane2 = [float(x) for x in plane2_str.split()]

                values_part = line[second_close + 1:].strip()
                values = values_part.split()
                if len(values) < 4:
                    continue

                area = float(values[0])
                strain = float(values[1])
                binding_energy = float(values[2])
                match_id = int(values[3])
                term_id = int(values[4]) if len(values) >= 5 else None
                stereo_winner = 1
                dft_status = None
                if len(values) >= 6:
                    v5 = str(values[5]).strip()
                    if v5 in ('0', '1'):
                        stereo_winner = int(v5)
                        if len(values) >= 7:
                            dft_status = str(values[6]).strip().lower() or None
                    else:
                        dft_status = v5.lower() or None

                rec = {
                    "material1_plane": plane1,
                    "material2_plane": plane2,
                    "area": area,
                    "strain": strain,
                    "binding_energy": binding_energy,
                    "match_id": match_id,
                    "term_id": term_id,
                    "stereo_winner": stereo_winner,
                }
                if dft_status:
                    rec["dft_status"] = dft_status
                records.append(rec)
            except (ValueError, IndexError) as e:
                print(f"警告: 第{line_num}行解析失败: {e}")
                continue

    return records


def parse_area_strain_data(filename):
    """
    Parse area_strain data file

    Data format:
    index (h1 k1 l1) (h2 k2 l2) angle strain binding_energy match_score

    Parameters:
    filename: path to area_strain file

    Returns:
    material1_planes: crystal planes for material 1 (N, 3)
    material2_planes: crystal planes for material 2 (N, 3)
    binding_energies: binding energies array (N,)
    """
    records = parse_area_strain_records(filename)
    material1_planes = [r["material1_plane"] for r in records]
    material2_planes = [r["material2_plane"] for r in records]
    binding_energies = [r["binding_energy"] for r in records]

    return (np.array(material1_planes),
            np.array(material2_planes),
            np.array(binding_energies))


def enrich_area_strain_records_with_summary(records, summary_path='opt_results_summary.json'):
    """
    Fill legacy ``area_strain`` records missing ``term_id`` by reading
    ``opt_results_summary.json`` in the same directory.
    """
    if not records:
        return records
    if builtins.all(record.get("term_id") is not None for record in records):
        return records
    if not Path(summary_path).is_file():
        return records

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception:
        return records

    best_term_by_match = {}
    for item in summary.values():
        if not isinstance(item, dict):
            continue
        match_id = item.get("match_id")
        term_id = item.get("term_id")
        if match_id is None or term_id is None:
            continue
        energy = item.get("relaxed_min_it_E")
        if energy is None:
            energy = item.get("relaxed_min_bd_E")
        if energy is None:
            continue

        current = best_term_by_match.get(int(match_id))
        if current is None or float(energy) < current["energy"]:
            best_term_by_match[int(match_id)] = {
                "term_id": int(term_id),
                "energy": float(energy),
            }

    enriched = []
    for record in records:
        updated = dict(record)
        if updated.get("term_id") is None:
            best = best_term_by_match.get(int(updated["match_id"]))
            if best is not None:
                updated["term_id"] = best["term_id"]
        enriched.append(updated)
    return enriched


def miller_to_spherical(miller_indices):
    """
    Convert Miller indices to spherical coordinates (theta, phi)

    Parameters:
    miller_indices: Miller indices array (N, 3)

    Returns:
    theta: polar angle [0, pi]
    phi: azimuthal angle [-pi, pi]
    """
    h, k, l = miller_indices.T

    # 归一化向量
    norm = np.sqrt(h**2 + k**2 + l**2)
    norm = np.where(norm == 0, 1, norm)  # 避免除零错误

    x = h / norm
    y = k / norm
    z = l / norm

    # 转换为球坐标
    theta = np.arccos(np.clip(z, -1, 1))  # 极角 [0, pi]
    phi = np.arctan2(y, x)  # 方位角 [-pi, pi]

    return theta, phi


def spherical_to_stereographic(theta, phi):
    """
    Convert spherical coordinates to stereographic projection coordinates
    Lower hemisphere (theta > pi/2) is mapped back inside the unit circle

    Parameters:
    theta: polar angle [0, pi]
    phi: azimuthal angle [-pi, pi]

    Returns:
    x_proj, y_proj: stereographic projection coordinates
    """
    # For upper hemisphere (theta <= pi/2): normal projection
    # For lower hemisphere (theta > pi/2): map back inside unit circle
    upper_mask = theta <= np.pi / 2
    lower_mask = theta > np.pi / 2
    
    x_proj = np.zeros_like(theta)
    y_proj = np.zeros_like(theta)
    
    # Upper hemisphere: R = tan(theta/2)
    R_upper = np.tan(theta[upper_mask] / 2)
    x_proj[upper_mask] = R_upper * np.cos(phi[upper_mask])
    y_proj[upper_mask] = R_upper * np.sin(phi[upper_mask])
    
    # Lower hemisphere: map to inside unit circle
    # Use complementary angle: theta' = pi - theta
    # R = tan(theta'/2) = tan((pi - theta)/2) = cot(theta/2)
    # For lower hemisphere, theta' < pi/2, so R < 1 (inside unit circle)
    # Reverse phi direction to map back inside
    theta_complement = np.pi - theta[lower_mask]
    R_lower = np.tan(theta_complement / 2)
    # Reverse phi direction (add pi) to map back inside
    phi_reversed = phi[lower_mask] + np.pi
    x_proj[lower_mask] = R_lower * np.cos(phi_reversed)
    y_proj[lower_mask] = R_lower * np.sin(phi_reversed)

    return x_proj, y_proj


def _format_pair_label(match_id, term_id):
    if term_id is None:
        return f"({int(match_id)}, ?)"
    return f"({int(match_id)}, {int(term_id)})"


def _format_plane_hover_label(plane):
    h, k, l = [int(v) for v in plane]
    return f"({h},{k},{l})"


def _formula_label_plain(label):
    """
    Normalize a material label to a plain reduced-formula-like string.
    """
    text = str(label or "").strip()
    if not text:
        return ""
    text = re.sub(r"\$_(\d+)\$", r"\1", text)
    text = text.replace("$", "")
    return text


def _formula_label_html(label):
    """
    Format a material label for Plotly subplot titles.
    """
    text = _formula_label_plain(label)
    if not text:
        return ""
    return re.sub(r"(\d+)", r"<sub>\1</sub>", text)


def _group_projected_points(planes, binding_energies, theta, x_proj, y_proj, records, rounding=2):
    """
    Group projected points by displayed position.

    For overlapping upper/lower hemisphere points, keep the upper-hemisphere points.
    For multiple records at one displayed position, keep the lowest energy for coloring
    and retain all records in hover metadata.
    """
    position_groups = {}
    for idx, (x, y) in enumerate(zip(x_proj, y_proj)):
        pos_key = (round(float(x), rounding), round(float(y), rounding))
        position_groups.setdefault(pos_key, []).append(idx)

    grouped = []
    for indices in position_groups.values():
        upper_indices = [idx for idx in indices if theta[idx] <= np.pi / 2]
        candidate_indices = upper_indices if upper_indices else indices
        cand_arr = np.asarray(candidate_indices, dtype=int)
        energies_here = binding_energies[cand_arr]
        finite = np.isfinite(energies_here)
        if not np.any(finite):
            chosen_idx = int(cand_arr[0])
        else:
            sub_idx = cand_arr[finite]
            sub_e = binding_energies[sub_idx]
            chosen_idx = int(sub_idx[int(np.argmin(sub_e))])
        grouped.append(
            {
                "index": chosen_idx,
                "record_indices": candidate_indices,
            }
        )
    return grouped


def _rollup_dft_group_status(record_indices, dft_status_arr):
    """
    Roll up per-record DFT export status for one stereographic group.

    Priority: any failed -> failed; else any pending -> pending; else complete.
    ``mlip`` means no DFT was scheduled for that termination (show MLIP energy); treated like complete for coloring.
    When *dft_status_arr* is None, returns ``\"complete\"`` (legacy MLIP-only plots).
    """
    if dft_status_arr is None:
        return "complete"
    statuses = [str(dft_status_arr[i]).strip().lower() for i in record_indices]
    if builtins.any(s == "failed" for s in statuses):
        return "failed"
    if builtins.any(s == "pending" for s in statuses):
        return "pending"
    return "complete"


def _compute_stereographic_guides(projected, tolerance=0.01):
    """
    Compute the unique guide radii and angles for a stereographic plot.
    """
    if len(projected) == 0:
        return [1.0], []

    radii = []
    for r in np.linalg.norm(projected, axis=1):
        if all(abs(r - np.array(radii)) > tolerance):
            radii.append(float(r))
    if all(abs(1 - np.array(radii)) > tolerance):
        radii.append(1.0)

    angles = []
    for angle in np.arctan2(projected[:, 1], projected[:, 0]):
        if all(abs(angle - np.array(angles)) > tolerance):
            angles.append(float(angle))

    return radii, angles


def add_stereographic_guides(ax, projected):
    """
    Draw the same stereographic guide lines used by ``plot_matching_data``.

    The guide set consists of:
    - concentric dashed circles at the radii occupied by projected points
    - dashed radial lines from the origin at the angles occupied by points
    - the unit circle boundary
    """
    radii, angles = _compute_stereographic_guides(projected)
    for r in radii:
        wulff_circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_artist(wulff_circle)
    for angle in angles:
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], color='gray', linestyle='--', alpha=0.3)


def label_outer_ring_planes(ax, planes, x_proj, y_proj, theta):
    """
    Label only the outer-ring upper-hemisphere Miller indices with offsets.
    """
    n = len(x_proj)
    if n == 0:
        return

    distances = np.sqrt(x_proj**2 + y_proj**2)
    upper_hemisphere = theta <= np.pi / 2
    outer_ring = (distances >= 0.85) & (distances <= 1.0) & upper_hemisphere
    outer_indices = np.where(outer_ring)[0]
    if len(outer_indices) == 0:
        return

    position_groups = {}
    for idx in outer_indices:
        pos_key = (round(float(x_proj[idx]), 4), round(float(y_proj[idx]), 4))
        position_groups.setdefault(pos_key, []).append(idx)

    unique_indices = []
    for indices in position_groups.values():
        upper_indices = [idx for idx in indices if theta[idx] <= np.pi / 2]
        if upper_indices:
            unique_indices.append(upper_indices[0])

    if len(unique_indices) == 0:
        return

    angles = np.arctan2(y_proj[unique_indices], x_proj[unique_indices])
    sort_order = np.argsort(angles)
    sorted_indices = np.array(unique_indices)[sort_order]

    def format_single(value):
        val_int = int(value)
        if val_int < 0:
            return rf'\overline{{{abs(val_int)}}}'
        return str(val_int)

    total = len(sorted_indices)
    for order_idx, idx in enumerate(sorted_indices):
        x, y = x_proj[idx], y_proj[idx]
        h, k, l = planes[idx]
        label = f'$({format_single(h)},{format_single(k)},{format_single(l)})$'

        angle = np.arctan2(y, x)
        radius = np.hypot(x, y)
        radial_offset = -13 if radius > 0.95 else -10
        tangential_offset = 5 if order_idx % 2 == 0 else -5
        if total <= 4:
            tangential_offset = 3 if order_idx % 2 == 0 else -3

        # Pull labels inward so they stay clear of the frame, then stagger tangentially.
        offset_x = radial_offset * np.cos(angle) - tangential_offset * np.sin(angle)
        offset_y = radial_offset * np.sin(angle) + tangential_offset * np.cos(angle)

        if abs(x) > 0.9:
            offset_x *= 0.8
        if abs(y) > 0.9:
            offset_y *= 0.8

        ha = 'left' if offset_x > 3 else 'right' if offset_x < -3 else 'center'
        va = 'bottom' if offset_y > 3 else 'top' if offset_y < -3 else 'center'

        ax.annotate(
            label,
            (x, y),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=5,
            alpha=0.9,
            zorder=20,
            color='black',
            ha=ha,
            va=va,
            clip_on=False,
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='white',
                edgecolor='black',
                alpha=0.8,
                linewidth=0.5,
            ),
        )


def label_planes_in_arc(ax, planes, x_proj, y_proj, theta):
    """
    Label crystal planes in 360 degrees, only outer ring and center point
    Only label upper hemisphere if upper and lower hemisphere overlap
    
    Parameters:
    ax: matplotlib axis object
    planes: crystal plane Miller indices array (N, 3)
    x_proj, y_proj: projected coordinates (N,)
    theta: polar angles (N,) - used to identify upper/lower hemisphere
    """
    n = len(x_proj)
    if n == 0:
        return
    
    # Calculate distances from center
    distances = np.sqrt(x_proj**2 + y_proj**2)
    
    # Only label upper hemisphere (theta <= pi/2)
    upper_hemisphere = theta <= np.pi / 2
    
    # Find outer ring points (close to unit circle boundary)
    outer_ring = (distances >= 0.85) & (distances <= 1.0) & upper_hemisphere
    
    # Find center point (closest to origin)
    center_mask = upper_hemisphere & (distances <= 1.0)
    if np.any(center_mask):
        center_idx = np.argmin(distances[center_mask])
        # Get the actual index in the full array
        center_indices = np.where(center_mask)[0]
        center_point_idx = center_indices[center_idx]
    else:
        center_point_idx = None
    
    # Get outer ring indices
    outer_indices = np.where(outer_ring)[0]
    
    # Combine outer ring and center point
    final_indices = list(outer_indices)
    if center_point_idx is not None:
        final_indices.append(center_point_idx)
    
    if len(final_indices) == 0:
        return
    
    # Remove duplicates
    final_indices = list(set(final_indices))
    
    # For points that might overlap (same x, y position), prefer upper hemisphere
    # Group by position and keep only upper hemisphere ones
    position_groups = {}
    for idx in final_indices:
        pos_key = (round(x_proj[idx], 4), round(y_proj[idx], 4))
        if pos_key not in position_groups:
            position_groups[pos_key] = []
        position_groups[pos_key].append(idx)
    
    # Keep only upper hemisphere points for each position
    unique_indices = []
    for pos_key, indices in position_groups.items():
        # Filter to keep only upper hemisphere
        upper_indices = [idx for idx in indices if theta[idx] <= np.pi / 2]
        if len(upper_indices) > 0:
            # If multiple points at same position, take the first one
            unique_indices.append(upper_indices[0])
    
    if len(unique_indices) == 0:
        return
    
    # Calculate angles for sorting
    angles = np.arctan2(y_proj[unique_indices], x_proj[unique_indices])
    
    # Sort by angle for organized labeling
    sort_order = np.argsort(angles)
    sorted_indices = np.array(unique_indices)[sort_order]
    
    # Helper function to format Miller index with LaTeX bar notation for negative numbers
    def format_miller_index(value):
        """Format Miller index using LaTeX format, adding bar (overline) for negative numbers"""
        try:
            val_int = int(value)
            if val_int < 0:
                # Use LaTeX overline for negative numbers
                abs_val = abs(val_int)
                return rf'\overline{{{abs_val}}}'
            else:
                return str(val_int)
        except ValueError:
            return f'{value:.1f}'
    
    # Label each point with dynamic positioning based on location
    for idx in sorted_indices:
        x, y = x_proj[idx], y_proj[idx]
        h, k, l = planes[idx]
        
        # Format label with LaTeX bar notation for negative indices
        h_str = format_miller_index(h)
        k_str = format_miller_index(k)
        l_str = format_miller_index(l)
        # Combine into LaTeX math format
        label = f'$({h_str},{k_str},{l_str})$'
        
        # Determine label position based on point location
        # Use origin (0, 0) as center since stereographic projection is centered
        x_center = 0
        y_center = 0
        
        # Determine offset direction: choose primary direction based on distance from center
        offset_x = 0
        offset_y = 0
        ha = 'center'
        va = 'center'
        
        abs_x = abs(x)
        abs_y = abs(y)
        
        # For points at leftmost or rightmost edges, always place label below
        # Check if point is near the left or right edge (x close to ±1.0)
        is_at_edge = abs_x > 0.85
        
        if is_at_edge:
            # Force label below for edge points
            offset_y = -5
            va = 'top'
            ha = 'center'
        elif abs_y >= abs_x:
            # Use vertical positioning
            if y > y_center:
                # Upper half: label above
                offset_y = 5
                va = 'bottom'
            else:
                # Lower half: label below
                offset_y = -5
                va = 'top'
            ha = 'center'
        else:
            # Use horizontal positioning
            if x < x_center:
                # Left half: label to the left
                offset_x = -5
                ha = 'right'
            else:
                # Right half: label to the right
                offset_x = 5
                ha = 'left'
            va = 'center'
        
        # Place label with dynamic positioning
        ax.annotate(label, (x, y),
                   xytext=(offset_x, offset_y),
                   textcoords='offset points',
                   fontsize=5,
                   alpha=0.9,
                   zorder=20,
                   color='black',
                   ha=ha,
                   va=va,
                   bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='black',
                            alpha=0.8,
                            linewidth=0.5))


def _shared_energy_color_limits(binding_energies):
    """
    Stable shared color limits for stereographic plots.

    When all energies are identical, matplotlib/plotly color normalization becomes
    singular. Expand the range slightly so both subplots use the same visible color.
    """
    arr = np.asarray(binding_energies, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        pad = builtins.max(1e-9, abs(vmin) * 1e-9, 1e-6)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def create_stereographic_plot(
    planes,
    binding_energies,
    material_name,
    ax,
    *,
    vmin=None,
    vmax=None,
    dft_status=None,
):
    """
    Create stereographic projection plot

    Parameters:
    planes: crystal plane Miller indices array (N, 3)
    binding_energies: binding energies array (N,); may contain NaN when ``dft_status`` marks pending/failed DFT
    material_name: material name
    ax: matplotlib axis object
    dft_status: optional length-N sequence with values ``complete`` / ``pending`` / ``failed`` (DFT export state)

    Returns:
    im: mappable for colorbar (may be a ScalarMappable stub if no finite-energy points)
    scatter: colored scatter for finite DFT energies, or None
    """
    theta, phi = miller_to_spherical(planes)
    x_proj, y_proj = spherical_to_stereographic(theta, phi)
    be = np.asarray(binding_energies, dtype=float)
    n = len(x_proj)
    d_arr = None
    if dft_status is not None and len(dft_status) == n:
        d_arr = np.array([str(s).strip().lower() if s is not None else "complete" for s in dft_status], dtype=object)

    valid_mask = np.isfinite(x_proj) & np.isfinite(y_proj)
    if d_arr is None:
        valid_mask &= np.isfinite(be)
    else:
        # ``mlip`` = DFT was not scheduled for this termination; show MLIP energy (same as legacy MLIP-only).
        valid_mask &= np.isin(d_arr, ["complete", "pending", "failed", "mlip"])

    x_proj = x_proj[valid_mask]
    y_proj = y_proj[valid_mask]
    binding_energies_clean = be[valid_mask]
    planes_clean = planes[valid_mask]
    theta_clean = theta[valid_mask]
    if d_arr is not None:
        d_arr = d_arr[valid_mask]

    if len(x_proj) == 0:
        print(f"警告: {material_name} 没有有效的数据点")
        return None, None

    grouped = _group_projected_points(
        planes_clean,
        binding_energies_clean,
        theta_clean,
        x_proj,
        y_proj,
        records=None,
    )
    idxs_label = np.array([item["index"] for item in grouped], dtype=int)
    projected = np.column_stack([x_proj[idxs_label], y_proj[idxs_label]])
    add_stereographic_guides(ax, projected)

    x_c, y_c, e_c = [], [], []
    x_p, y_p = [], []
    x_f, y_f = [], []
    for item in grouped:
        chosen = int(item["index"])
        xc = float(x_proj[chosen])
        yc = float(y_proj[chosen])
        ev = binding_energies_clean[chosen]
        roll = _rollup_dft_group_status(item["record_indices"], d_arr)
        if roll == "complete" and np.isfinite(ev):
            x_c.append(xc)
            y_c.append(yc)
            e_c.append(float(ev))
        elif roll == "failed" or (roll == "complete" and not np.isfinite(ev)):
            x_f.append(xc)
            y_f.append(yc)
        else:
            x_p.append(xc)
            y_p.append(yc)

    if e_c:
        if vmin is None or vmax is None:
            vmin, vmax = _shared_energy_color_limits(np.array(e_c, dtype=float))
        scatter = ax.scatter(
            x_c,
            y_c,
            c=e_c,
            cmap="viridis",
            s=50,
            edgecolors="black",
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax,
            zorder=10,
        )
        im = scatter
    else:
        scatter = None
        if vmin is None or vmax is None:
            vmin, vmax = 0.0, 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        im = ScalarMappable(norm=norm, cmap="viridis")

    ms = 55
    for x, y in zip(x_p, y_p):
        ax.scatter(
            [x],
            [y],
            s=ms,
            facecolors="none",
            edgecolors="black",
            linewidth=1.0,
            zorder=11,
        )
        ax.text(x, y, "?", fontsize=8, ha="center", va="center", zorder=12, color="black")
    for x, y in zip(x_f, y_f):
        ax.scatter(
            [x],
            [y],
            s=ms,
            facecolors="none",
            edgecolors="black",
            linewidth=1.0,
            zorder=11,
        )
        ax.text(x, y, "\u00d7", fontsize=10, ha="center", va="center", zorder=12, color="black")

    label_outer_ring_planes(ax, planes_clean[idxs_label], x_proj[idxs_label], y_proj[idxs_label], theta_clean[idxs_label])

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_title(_formula_label_plain(material_name), fontsize=15, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    return im, scatter


def _add_plotly_guides(fig, projected, row, col):
    radii, angles = _compute_stereographic_guides(projected)
    import plotly.graph_objects as go

    for r in radii:
        t = np.linspace(0, 2 * np.pi, 240)
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(t),
                y=r * np.sin(t),
                mode="lines",
                line=dict(color="rgba(120,120,120,0.35)", dash="dash", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    for angle in angles:
        fig.add_trace(
            go.Scatter(
                x=[0, np.cos(angle)],
                y=[0, np.sin(angle)],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.35)", dash="dash", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def create_interactive_stereographic_plot(
    planes,
    binding_energies,
    pair_labels,
    material_name,
    fig,
    row,
    col,
    coloraxis="coloraxis",
    dft_status=None,
):
    theta, phi = miller_to_spherical(planes)
    x_proj, y_proj = spherical_to_stereographic(theta, phi)
    be = np.asarray(binding_energies, dtype=float)
    n = len(x_proj)
    d_arr = None
    if dft_status is not None and len(dft_status) == n:
        d_arr = np.array([str(s).strip().lower() if s is not None else "complete" for s in dft_status], dtype=object)

    valid_mask = np.isfinite(x_proj) & np.isfinite(y_proj)
    if d_arr is None:
        valid_mask &= np.isfinite(be)
    else:
        valid_mask &= np.isin(d_arr, ["complete", "pending", "failed", "mlip"])

    x_proj = x_proj[valid_mask]
    y_proj = y_proj[valid_mask]
    binding_energies_clean = be[valid_mask]
    planes_clean = planes[valid_mask]
    theta_clean = theta[valid_mask]
    pair_labels_clean = np.array(pair_labels, dtype=object)[valid_mask]
    if d_arr is not None:
        d_arr = d_arr[valid_mask]

    grouped = _group_projected_points(
        planes_clean,
        binding_energies_clean,
        theta_clean,
        x_proj,
        y_proj,
        records=pair_labels_clean,
    )
    if len(grouped) == 0:
        return

    import plotly.graph_objects as go

    def _hover_for_group(record_indices):
        seen = set()
        lines = []
        for ridx in record_indices:
            plane_label = _format_plane_hover_label(planes_clean[ridx])
            pair_label = pair_labels_clean[ridx]
            text = f"{plane_label}, {pair_label}"
            if text not in seen:
                seen.add(text)
                lines.append(text)
        return "<br>".join(lines)

    x_c, y_c, e_c, h_c = [], [], [], []
    x_p, y_p, h_p = [], [], []
    x_f, y_f, h_f = [], [], []
    for item in grouped:
        chosen = int(item["index"])
        xc = float(x_proj[chosen])
        yc = float(y_proj[chosen])
        ev = binding_energies_clean[chosen]
        roll = _rollup_dft_group_status(item["record_indices"], d_arr)
        ht0 = _hover_for_group(item["record_indices"])
        if roll == "complete" and np.isfinite(ev):
            x_c.append(xc)
            y_c.append(yc)
            e_c.append(float(ev))
            h_c.append(ht0)
        elif roll == "failed" or (roll == "complete" and not np.isfinite(ev)):
            x_f.append(xc)
            y_f.append(yc)
            h_f.append(ht0 + "<br>DFT: failed (no energy)")
        else:
            x_p.append(xc)
            y_p.append(yc)
            h_p.append(ht0 + "<br>DFT: pending / not finished")

    idxs = np.array([item["index"] for item in grouped], dtype=int)
    projected = np.column_stack([x_proj[idxs], y_proj[idxs]])
    _add_plotly_guides(fig, projected, row, col)

    if x_c:
        fig.add_trace(
            go.Scatter(
                x=x_c,
                y=y_c,
                mode="markers",
                marker=dict(
                    size=10,
                    color=e_c,
                    coloraxis=coloraxis,
                    line=dict(color="black", width=1),
                ),
                text=h_c,
                hovertemplate="%{text}<br>Energy=%{marker.color:.2f} J/m^2<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    if x_p:
        fig.add_trace(
            go.Scatter(
                x=x_p,
                y=y_p,
                mode="markers",
                marker=dict(symbol="circle-open", size=14, line=dict(color="black", width=1)),
                hovertext=h_p,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_p,
                y=y_p,
                mode="text",
                text=["?"] * len(x_p),
                textposition="middle center",
                textfont=dict(size=10, color="black"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    if x_f:
        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=y_f,
                mode="markers",
                marker=dict(symbol="circle-open", size=14, line=dict(color="black", width=1)),
                hovertext=h_f,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=y_f,
                mode="text",
                text=["\u00d7"] * len(x_f),
                textposition="middle center",
                textfont=dict(size=12, color="black"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    axis_suffix = "" if (row == 1 and col == 1) else "2"
    fig.update_xaxes(range=[-1.25, 1.25], visible=False, row=row, col=col)
    fig.update_yaxes(
        range=[-1.25, 1.25],
        visible=False,
        scaleanchor=f"x{axis_suffix}",
        scaleratio=1,
        row=row,
        col=col,
    )


def plot_binding_energy_analysis(
    material1_planes,
    material2_planes,
    binding_energies,
    film_name,
    substrate_name,
    title,
    dft_status=None,
):
    """
    Create stereographic projection plots for binding energy analysis

    Parameters:
    material1_planes: crystal planes for material 1 (N, 3)
    material2_planes: crystal planes for material 2 (N, 3)
    binding_energies: binding energies array (N,); may contain NaN when ``dft_status`` is set
    dft_status: optional length-N DFT state labels: ``complete`` / ``pending`` / ``failed``
    """
    # Set matplotlib parameters
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True

    be = np.asarray(binding_energies, dtype=float)
    fin = np.isfinite(be)
    if dft_status is not None and len(dft_status) == len(be):
        ds = np.array([str(s).strip().lower() for s in dft_status], dtype=object)
        fin &= np.isin(ds, ["complete", "mlip"])
    fe = be[fin]
    if fe.size:
        vmin, vmax = _shared_energy_color_limits(fe)
    else:
        vmin, vmax = 0.0, 1.0

    # Create figure - each subplot is 3x3, so total is 6x3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    # Opt out of global figure.autolayout: colorbar uses fixed axes + tight_layout(rect=...).
    fig.set_layout_engine("none")

    # Plot stereographic projection for material 1
    im1, scatter1 = create_stereographic_plot(
        material1_planes,
        binding_energies,
        film_name,
        ax1,
        vmin=vmin,
        vmax=vmax,
        dft_status=dft_status,
    )

    # Plot stereographic projection for material 2
    im2, scatter2 = create_stereographic_plot(
        material2_planes,
        binding_energies,
        substrate_name,
        ax2,
        vmin=vmin,
        vmax=vmax,
        dft_status=dft_status,
    )

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar_src = scatter2 if scatter2 is not None else scatter1
    if cbar_src is None:
        cbar_src = im2 if im2 is not None else im1
    cbar = fig.colorbar(cbar_src, cax=cbar_ax)
    cbar.set_label(f'{title} J/m$^2$', rotation=270, labelpad=15, fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout(rect=[0, 0, 0.85, 1], pad=0.1)
    return fig


def plot_binding_energy_analysis_interactive(
    material1_planes,
    material2_planes,
    binding_energies,
    film_pairs,
    substrate_pairs,
    film_name,
    substrate_name,
    title,
    dft_status=None,
):
    from plotly.subplots import make_subplots

    be = np.asarray(binding_energies, dtype=float)
    fin = np.isfinite(be)
    if dft_status is not None and len(dft_status) == len(be):
        ds = np.array([str(s).strip().lower() for s in dft_status], dtype=object)
        fin &= np.isin(ds, ["complete", "mlip"])
    fe = be[fin]
    if fe.size:
        vmin, vmax = _shared_energy_color_limits(fe)
    else:
        vmin, vmax = 0.0, 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(_formula_label_html(film_name), _formula_label_html(substrate_name)),
        horizontal_spacing=0.04,
    )
    create_interactive_stereographic_plot(
        material1_planes,
        binding_energies,
        film_pairs,
        film_name,
        fig,
        row=1,
        col=1,
        coloraxis="coloraxis",
        dft_status=dft_status,
    )
    create_interactive_stereographic_plot(
        material2_planes,
        binding_energies,
        substrate_pairs,
        substrate_name,
        fig,
        row=1,
        col=2,
        coloraxis="coloraxis",
        dft_status=dft_status,
    )
    fig.update_layout(
        width=1200,
        height=700,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(title=f"{title} J/m^2", tickformat=".2f"),
            cmin=vmin,
            cmax=vmax,
        ),
    )
    return fig

def visualize_minimization_results(film_name, substrate_name, title = 'Cohesive Energy'):
    records = parse_area_strain_records('area_strain')
    records = enrich_area_strain_records_with_summary(records)
    # Stereographic plots: only rows with stereo_winner!=0 (default 1). Extra area_strain rows
    # (competing matches per bin, equivalent Miller labels) use stereo_winner=0.
    plot_recs = [r for r in records if int(r.get("stereo_winner", 1)) != 0]
    if not plot_recs:
        plot_recs = records
    material1_planes = np.array([r["material1_plane"] for r in plot_recs])
    material2_planes = np.array([r["material2_plane"] for r in plot_recs])
    binding_energies = np.array([r["binding_energy"] for r in plot_recs])
    film_pairs = [_format_pair_label(r['match_id'], r['term_id']) for r in plot_recs]
    substrate_pairs = [_format_pair_label(r['match_id'], r['term_id']) for r in plot_recs]
    dft_status = None
    if plot_recs and any(r.get("dft_status") for r in plot_recs):
        dft_status = [r.get("dft_status") for r in plot_recs]
    fig = plot_binding_energy_analysis(
        material1_planes,
        material2_planes,
        binding_energies,
        film_name,
        substrate_name,
        title,
        dft_status=dft_status,
    )
    fig.savefig('stereographic.jpg', dpi=600, bbox_inches='tight', format='jpg')
    try:
        import plotly.io as pio

        interactive_fig = plot_binding_energy_analysis_interactive(
            material1_planes,
            material2_planes,
            binding_energies,
            film_pairs,
            substrate_pairs,
            film_name,
            substrate_name,
            title,
            dft_status=dft_status,
        )
        interactive_div = pio.to_html(
            interactive_fig,
            include_plotlyjs=True,
            full_html=False,
            default_width="100%",
            default_height="700px",
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stereographic Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fff; color: #111; }}
  </style>
</head>
<body>
  {interactive_div}
</body>
</html>
"""
        with open('stereographic_interactive.html', 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception as e:
        print(f"Warning: failed to write stereographic_interactive.html: {e}")
    try:
        import matplotlib

        if str(matplotlib.get_backend()).lower() != "agg":
            plt.show()
    except Exception:
        pass
