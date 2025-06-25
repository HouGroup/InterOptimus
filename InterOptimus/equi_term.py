"""This module provides functions to determine whether two sets of slabs with
different terminations making identical interfaces"""

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
import sys
import time

def is_colinear(a, b, directional=False,tol=1e-5):
    """
    determine whether two vectors are colinear
    
    Args:
    a, g (array or list): two vectors
    directional (bool): whether considering the direction
    tol: tolerance
    Return:
    (bool): True if colinear
    """
    if not directional and (1 - abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) < tol):
        return True
    elif directional and (1 - abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) < tol) and np.dot(a,b) > 0:
        return True
    else:
        return False

def get_rotation_from_match(lattice_by_row_vectors, S):
    """
    convert supercell match matrix into rotation matrix
    
    Args:
    lattice_by_row_vectors (array): lattice to transform by row vectors.
    S (array): supercell matching matrix.
    
    Return:
    (array): rotation matrix
    """
    return np.dot((np.dot(S, lattice_by_row_vectors)).T, np.linalg.inv(lattice_by_row_vectors.T))

def co_point_group_operations(sym_ops_s1, sym_ops_s2):
    """
    given two structures, get the their union point group
    
    Args:
    sym_ops_s1, sym_ops_s2 (lists): lists of point group operations
    
    Return:
    co_group (list): list of symmetry operations
    """
    #co-group
    co_group = []
    for op1 in sym_ops_s1:
        for op2 in sym_ops_s2:
            new_op = op1 * op2
            if len(co_group) == 0:
                co_group.append(new_op)
            if not any(np.allclose(new_op.rotation_matrix, existing_op.rotation_matrix) and
                       np.allclose(new_op.translation_vector, existing_op.translation_vector) for existing_op in co_group):
                co_group.append(new_op)
            new_op = op2 * op1
            if not any(np.allclose(new_op.rotation_matrix, existing_op.rotation_matrix) and
                       np.allclose(new_op.translation_vector, existing_op.translation_vector) for existing_op in co_group):
                co_group.append(new_op)
    return co_group

def pair_fit(film_slab_fit, sub_slab_fit, film_slab, sub_slab, matcher, c_periodic):
    """
    determine whether two pairs of film/substrate pairs are identical
    
    Args:
    film_slab_fit, sub_slab_fit (Structure): reference pair
    film_slab, sub_slab (Structure): comparing pair
    c_periodic: whether the periodic boundary condition is set along the c-direction of the interface supercell, which means no vacuum layers are included.
    
    Return:
    (bool): whether identical
    """
    #relative disorientation of slab and film compared with the existing one
    if film_slab_fit == film_slab:
        film_map = SymmOp.from_rotation_and_translation(np.eye(3), [0,0,0])
    else:
        film_transformation = matcher.get_transformation(film_slab, film_slab_fit)[0]
        film_rotation = get_rotation_from_match(film_slab_fit.lattice.matrix, film_transformation)
        film_map = SymmOp.from_rotation_and_translation(film_rotation, [0,0,0])
    if sub_slab_fit == sub_slab:
        sub_map = SymmOp.from_rotation_and_translation(np.eye(3), [0,0,0])
    else:
        sub_transformation = matcher.get_transformation(sub_slab, sub_slab_fit)[0]
        sub_rotation = get_rotation_from_match(sub_slab_fit.lattice.matrix, sub_transformation)
        sub_map = SymmOp.from_rotation_and_translation(sub_transformation, [0,0,0])
    film_over_sub = film_map * sub_map.inverse
    
    sym_ops_film = SpacegroupAnalyzer(film_slab_fit).get_point_group_operations(cartesian = True)
    sym_ops_sub = SpacegroupAnalyzer(sub_slab_fit).get_point_group_operations(cartesian = True)
    #co-group of film and slab to compare with
    co_group = co_point_group_operations(sym_ops_film, sym_ops_sub)

    #conditions to be identical film/slab pairs
    #condition 1: disorientation is a symmetry operation in co-group
    con1 = False
    for symm_op in co_group:
        if np.allclose(film_over_sub.rotation_matrix, symm_op.rotation_matrix, atol=1e-5):
            con1 = True
    #condition 2: transformed plane normals are related back to represent the same plane
    #by their own symmetry operatons
    rotated_film_normal = film_map.apply_rotation_only([0,0,1])
    rotated_sub_normal = sub_map.apply_rotation_only([0,0,1])
    #if no vacuum layers
    if c_periodic:
        con2 = any(is_colinear(sym_op.apply_rotation_only(rotated_film_normal), [0,0,1]) \
                   for sym_op in sym_ops_film) and \
               any(is_colinear(sym_op.apply_rotation_only(rotated_sub_normal), [0,0,1]) \
                   for sym_op in sym_ops_sub)
    else:
        #condition 2*: with vacuum layers, close to condition 2,
        #but also requiring the film and slab normals to be identical
        con2 = False
        for op_film in sym_ops_film:
            for op_sub in sym_ops_sub:
                if is_colinear(op_film.apply_rotation_only(rotated_film_normal), [0,0,1], directional = True) and \
                    is_colinear(op_sub.apply_rotation_only(rotated_sub_normal), [0,0,1], directional = True):
                        if np.allclose(op_film.apply_rotation_only(rotated_film_normal), \
                                       op_sub.apply_rotation_only(rotated_sub_normal), rtol = 1e-3, atol=1e-5):
                            con2 = True
                            break
            if con2:
                break
    return (con1 and con2)

def slab_pair_cluster(film_slabs, sub_slabs, c_periodic = True):
    """
    clarify film-substrate slab pairs so that those clustered into the same group generate identical termination conditions
    
    Args:
    film_slabs(list): list of film slabs.
    sub_slabs(list): list of substrate slabs.
    c_periodic(bool): whether the periodic boundary condition is set along the c-direction of the interface supercell, which means no vacuum layers are included.
    
    Return:
    slab_pair_groups, slab_pair_id_groups(list): groups of slab pairs and pair ids([film_id, slab_id, loop_id])
    """
    matcher = StructureMatcher(primitive_cell=False, scale=False)
    non_identical_slab_pairs = []
    non_identical_slab_id_pairs = []
    slab_pair_id_groups = []
    slab_pair_groups = []
    t_id = 0
    items = len(film_slabs)*(len(sub_slabs))
    for i in range(len(film_slabs)):
        for j in range(len(sub_slabs)):
            if i == 0 and j == 0:
                non_identical_slab_pairs.append([film_slabs[i],sub_slabs[j]])
                non_identical_slab_id_pairs.append([i,j])
                slab_pair_groups.append([[film_slabs[i], sub_slabs[j]]])
                slab_pair_id_groups.append([[i,j,t_id]])
            else:
                identical_interface_exist = False
                for k in range(len(non_identical_slab_pairs)):
                    #check whether both films and substrates match
                    film_slab_fit = non_identical_slab_pairs[k][0]
                    sub_slab_fit = non_identical_slab_pairs[k][1]
                    ase_SEC = SymmetryEquivalenceCheck()
                    if ase_SEC.compare(film_slab_fit.to_ase_atoms(), film_slabs[i].to_ase_atoms()) and \
                    ase_SEC.compare(sub_slab_fit.to_ase_atoms(), sub_slabs[j].to_ase_atoms()):
                        #print('fit')
                        if pair_fit(film_slab_fit, sub_slab_fit, film_slabs[i], sub_slabs[j], matcher, c_periodic):
                            identical_interface_exist = True
                            slab_pair_id_groups[k].append([i,j,t_id])
                            slab_pair_groups[k].append([film_slabs[i], sub_slabs[j]])
                            #print('identical!')
                            break
                if not identical_interface_exist:
                    #print(i,j)
                    non_identical_slab_pairs.append([film_slabs[i],sub_slabs[j]])
                    non_identical_slab_id_pairs.append([i,j])
                    slab_pair_groups.append([[film_slabs[i], sub_slabs[j]]])
                    slab_pair_id_groups.append([[i,j,t_id]])
            percentage = int(round( (i*len(sub_slabs)+j+1) / items, 2) * 100)
            print("\r", end="")
            print("symmetry checking progress: {}%: ".format(percentage), "▋" * (percentage // 2), end="")
            t_id += 1
    return slab_pair_groups, slab_pair_id_groups

def get_non_identical_slab_pairs(film, substrate, match, ftol = 1e-1, c_periodic = False):
    """
    get the ids of the non-identical slab pairs
    
    Args:
    film (Slab): film slab
    substrate (Slab): substrate slab
    match (dict): lattice match information
    ftol (float): toleration for flat cluster to distinguish atomic planes
    c_periodic (bool): whether c-direction is considered periodic
    
    Return:
    list of non-identical slab pairs
    """
    film_sg = SlabGenerator(
            film,
            match.film_miller,
            min_slab_size=1,
            min_vacuum_size=4,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )
    film_slabs = film_sg.get_slabs(ftol=ftol, filter_out_sym_slabs=False)
    sub_sg = SlabGenerator(
                substrate,
                match.substrate_miller,
                min_slab_size=1,
                min_vacuum_size=4,
                in_unit_planes=True,
                center_slab=True,
                primitive=True,
                reorient_lattice=False,  # This is necessary to not screw up the lattice
            )
    substrate_slabs = sub_sg.get_slabs(ftol=ftol, filter_out_sym_slabs=False)
    slab_pair_groups, slab_pair_id_groups = slab_pair_cluster(film_slabs, substrate_slabs, c_periodic)
    ids = []
    for i in slab_pair_id_groups:
        ids.append(i[0][-1])
    return ids, slab_pair_groups, slab_pair_id_groups
