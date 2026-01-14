"""
InterOptimus Matching Module

This module provides classes and functions to analyze lattice matching
between crystal structures using pymatgen's SubstrateAnalyzer. It includes
symmetry analysis to identify equivalent matches and terminations.
"""

from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces import CoherentInterfaceBuilder
from InterOptimus.equi_term import get_non_identical_slab_pairs, co_point_group_operations
from pymatgen.core.structure import Structure, IStructure
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from interfacemaster.cellcalc import get_primitive_hkl
from interfacemaster.hetero_searching import round_int, apply_function_to_array, \
float_to_rational, rational_to_float, get_rational_mtx, plane_set_transform, plane_set
from numpy import *
import numpy as np
from numpy.linalg import *
from pymatgen.analysis.structure_matcher import StructureMatcher
#from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
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

from functools import reduce
import math

def get_meshgrid(lim):
    x = np.arange(-lim, lim, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)[0]]
    indice_0 = indice_0[np.argsort(np.linalg.norm(indice_0, axis = 1))]
    return indice_0[np.where(np.gcd.reduce(indice_0, axis=1) == 1)[0]]

def get_symmetrically_distinct_miller_indices(
    structure: Structure | IStructure,
    max_index: int,
) -> list:
    """Find all symmetrically distinct indices below a certain max-index
    for a given structure. Analysis is based on the symmetry of the
    reciprocal lattice of the structure.

    Args:
        structure (Structure): The input structure.
        max_index (int): The maximum index. For example, 1 means that
            (100), (110), and (111) are returned for the cubic structure.
            All other indices are equivalent to one of these.
    """
    # Get a list of all hkls for conventional (including equivalent)
    rng = list(range(-max_index, max_index + 1))[::-1]
    conv_hkl_list = get_meshgrid(max_index)

    # Sort by the maximum absolute values of Miller indices so that
    # low-index planes come first. This is important for trigonal systems.
    conv_hkl_list = sorted(conv_hkl_list, key=lambda x: max(np.abs(x)))

    # Get distinct hkl planes from the rhombohedral setting if trigonal
    spg_analyzer = SpacegroupAnalyzer(structure)
    miller_list = conv_hkl_list
    symm_ops = structure.lattice.get_recp_symmetry_operation()

    unique_millers: list = []
    unique_millers_conv: list = []

    for idx, miller in enumerate(miller_list):
        denom = abs(reduce(math.gcd, miller))  # type: ignore[arg-type]
        if not _is_in_miller_family(miller, unique_millers, symm_ops):
            unique_millers.append(miller)
            unique_millers_conv.append(miller)

    return unique_millers_conv

def get_symmetrically_equivalent_miller_indices(
    structure: Structure,
    miller_index: tuple[int, ...],
    return_hkil: bool = True
) -> list:
    """Get indices for all equivalent sites within a given structure.
    Analysis is based on the symmetry of its reciprocal lattice.

    Args:
        structure (Structure): Structure to analyze.
        miller_index (tuple): Designates the family of Miller indices
            to find. Can be hkl or hkil for hexagonal systems.
        return_hkil (bool): Whether to return hkil (True) form of Miller
            index for hexagonal systems, or hkl (False).
        system: The crystal system of the structure.
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

        if any(idx != 0 for idx in miller):
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
    film_idtc_millers = get_symmetrically_equivalent_miller_indices(film, match[0], return_hkil=False)
    substrate_idtc_millers = get_symmetrically_equivalent_miller_indices(substrate, match[1], return_hkil=False)

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

        existing_radii = []
        radii = np.linalg.norm(projected, axis=1)
        for r in radii:
            if all(abs(r - np.array(existing_radii))>0.01):
                wulff_circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.3)
                ax[i].add_artist(wulff_circle)
                existing_radii.append(r)
        if all(abs(1 - np.array(existing_radii))>0.01):
            ax[i].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3))

        existing_angles = []
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        for angle in angles:
            if all(abs(angle - np.array(existing_angles))>0.01):
                x = np.cos(angle)
                y = np.sin(angle)
                ax[i].plot([0, x], [0, y], color='gray', linestyle='--', alpha=0.3)
                existing_angles.append(angle)
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay


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
    material1_planes = []
    material2_planes = []
    binding_energies = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 分割行数据
            parts = line.split()

            # 跳过序号，提取晶面信息
            # 格式: (h1 k1 l1) (h2 k2 l2) ...
            try:
                # 找到两个晶面的括号位置
                first_open = line.find('(')
                first_close = line.find(')')
                second_open = line.find('(', first_close + 1)
                second_close = line.find(')', second_open + 1)

                if first_open == -1 or second_open == -1:
                    continue

                # 提取第一个晶面
                plane1_str = line[first_open+1:first_close]
                plane1 = [float(x) for x in plane1_str.split()]

                # 提取第二个晶面
                plane2_str = line[second_open+1:second_close]
                plane2 = [float(x) for x in plane2_str.split()]

                # 提取数值部分（结合能是倒数第二列）
                values_part = line[second_close+1:].strip()
                values = values_part.split()

                if len(values) >= 2:
                    # 倒数第二列是结合能
                    binding_energy = float(values[-2])
                else:
                    continue

                material1_planes.append(plane1)
                material2_planes.append(plane2)
                binding_energies.append(binding_energy)

            except (ValueError, IndexError) as e:
                print(f"警告: 第{line_num}行解析失败: {e}")
                continue

    return (np.array(material1_planes),
            np.array(material2_planes),
            np.array(binding_energies))


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


def draw_plane_connections(ax, planes, x_proj, y_proj):
    """
    Draw connections between adjacent crystal planes using Delaunay triangulation
    Each point only connects to its nearest neighbors, avoiding line crossings
    
    Parameters:
    ax: matplotlib axis object
    planes: crystal plane Miller indices array (N, 3)
    x_proj, y_proj: projected coordinates (N,)
    """
    # Draw equator circle (outermost circle) - this is the boundary
    phi_eq = np.linspace(0, 2*np.pi, 200)
    theta_eq = np.pi/2 * np.ones_like(phi_eq)
    x_eq, y_eq = spherical_to_stereographic(theta_eq, phi_eq)
    ax.plot(x_eq, y_eq, 'k-', linewidth=1.5, alpha=0.7, zorder=2)
    
    n = len(x_proj)
    if n < 3:
        return  # Need at least 3 points for triangulation
    
    # Prepare points for Delaunay triangulation
    points = np.column_stack([x_proj, y_proj])
    
    # Filter points inside or near unit circle to avoid issues with far points
    # Only use points within reasonable range
    distances = np.sqrt(x_proj**2 + y_proj**2)
    valid_mask = distances <= 1.5  # Include points slightly outside unit circle
    
    if np.sum(valid_mask) < 3:
        valid_mask = np.ones(n, dtype=bool)  # Use all points if too few valid
    
    valid_points = points[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_points) < 3:
        return
    
    try:
        # Perform Delaunay triangulation
        tri = Delaunay(valid_points)
        
        # Draw edges of triangles (each edge connects two adjacent points)
        # Use a set to avoid drawing the same edge twice
        edges_drawn = set()
        
        for simplex in tri.simplices:
            # Each simplex is a triangle with 3 vertices
            for i in range(3):
                v1_idx = simplex[i]
                v2_idx = simplex[(i + 1) % 3]
                
                # Create a unique key for the edge (smaller index first)
                edge_key = tuple(sorted([v1_idx, v2_idx]))
                
                if edge_key not in edges_drawn:
                    edges_drawn.add(edge_key)
                    
                    # Get actual indices in original array
                    actual_idx1 = valid_indices[v1_idx]
                    actual_idx2 = valid_indices[v2_idx]
                    
                    # Draw the edge
                    ax.plot([x_proj[actual_idx1], x_proj[actual_idx2]],
                           [y_proj[actual_idx1], y_proj[actual_idx2]],
                           'k-', linewidth=0.5, alpha=0.3, zorder=1)
    except Exception as e:
        # If Delaunay triangulation fails, skip drawing connections
        print(f"Warning: Triangulation failed: {e}")
        pass


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


def create_stereographic_plot(planes, binding_energies, material_name, ax):
    """
    Create stereographic projection plot

    Parameters:
    planes: crystal plane Miller indices array (N, 3)
    binding_energies: binding energies array (N,)
    material_name: material name
    ax: matplotlib axis object

    Returns:
    im: interpolated image object
    scatter: scatter plot object
    """
    # Convert to spherical coordinates
    theta, phi = miller_to_spherical(planes)

    # Convert to stereographic projection coordinates
    x_proj, y_proj = spherical_to_stereographic(theta, phi)

    # Data cleaning: remove invalid values
    valid_mask = np.isfinite(x_proj) & np.isfinite(y_proj) & np.isfinite(binding_energies)
    x_proj = x_proj[valid_mask]
    y_proj = y_proj[valid_mask]
    binding_energies_clean = binding_energies[valid_mask]
    planes_clean = planes[valid_mask]
    theta_clean = theta[valid_mask]

    if len(x_proj) == 0:
        print(f"警告: {material_name} 没有有效的数据点")
        return None, None

    # Remove overlapping points: keep only the point with lowest binding energy
    # Points are considered overlapping if they are within a small distance threshold
    distance_threshold = 0.01  # Threshold for considering points as overlapping
    
    # Group points by their rounded positions
    position_groups = {}
    for i in range(len(x_proj)):
        # Round to 2 decimal places for grouping
        pos_key = (round(x_proj[i], 2), round(y_proj[i], 2))
        if pos_key not in position_groups:
            position_groups[pos_key] = []
        position_groups[pos_key].append(i)
    
    # For each group, keep only the point with lowest binding energy
    unique_indices = []
    for pos_key, indices in position_groups.items():
        if len(indices) == 1:
            unique_indices.append(indices[0])
        else:
            # Multiple points at same position, keep the one with lowest energy
            energies = binding_energies_clean[indices]
            min_energy_idx = indices[np.argmin(energies)]
            unique_indices.append(min_energy_idx)
    
    # Filter data to keep only unique points
    unique_indices = np.array(unique_indices)
    x_proj_unique = x_proj[unique_indices]
    y_proj_unique = y_proj[unique_indices]
    binding_energies_unique = binding_energies_clean[unique_indices]
    planes_unique = planes_clean[unique_indices]
    theta_unique = theta_clean[unique_indices]

    # Draw connections between crystal planes first
    draw_plane_connections(ax, planes_unique, x_proj_unique, y_proj_unique)
    
    # Create color mapping (lower binding energy = darker color = stronger binding)
    vmin = np.min(binding_energies_unique)
    vmax = np.max(binding_energies_unique)

    # Plot data points only (no interpolation)
    # Use Nature-recommended colormap: 'viridis' (colorblind-friendly, widely used in Nature)
    scatter = ax.scatter(x_proj_unique, y_proj_unique, c=binding_energies_unique,
                        cmap='viridis', s=50, edgecolors='black',
                        linewidth=0.5, vmin=vmin, vmax=vmax, zorder=10)
    
    # Create a dummy image for colorbar (using scatter data)
    im = scatter

    # Label planes in 360 degrees, only outer ring and center point
    # Only label upper hemisphere if overlap
    label_planes_in_arc(ax, planes_unique, x_proj_unique, y_proj_unique, theta_unique)

    # Set plot properties with smaller margins
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.set_title(f'{material_name}', fontsize=15, pad=10)
    
    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    return im, scatter


def plot_binding_energy_analysis(material1_planes, material2_planes, binding_energies,
                                film_name, substrate_name, title):
    """
    Create stereographic projection plots for binding energy analysis

    Parameters:
    material1_planes: crystal planes for material 1 (N, 3)
    material2_planes: crystal planes for material 2 (N, 3)
    binding_energies: binding energies array (N,)
    """
    # Set matplotlib parameters
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True

    # Create figure - each subplot is 3x3, so total is 6x3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Plot stereographic projection for material 1
    im1, scatter1 = create_stereographic_plot(material1_planes, binding_energies,
                                                                             film_name, ax1)

    # Plot stereographic projection for material 2
    im2, scatter2 = create_stereographic_plot(material2_planes, binding_energies,
                                                                             substrate_name, ax2)

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter2, cax=cbar_ax)
    cbar.set_label(f'{title} J/m$^2$', rotation=270, labelpad=15, fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout(rect=[0, 0, 0.85, 1], pad=0.1)
    return fig

def visualize_minimization_results(film_name, substrate_name, title = 'Cohesive Energy'):
    material1_planes, material2_planes, binding_energies = parse_area_strain_data('area_strain')
    fig = plot_binding_energy_analysis(material1_planes,
                                material2_planes,
                                binding_energies,
                                film_name,
                                substrate_name,
                                title)
    fig.savefig('stereographic.jpg', dpi=600, bbox_inches='tight', format='jpg')
    plt.show()
