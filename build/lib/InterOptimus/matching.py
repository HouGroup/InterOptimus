"""
This module provide classes to get lattice matching results by sorting the
results from pymatgen's SubstrateAnalyzer
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
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
#from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from InterOptimus.equi_term import pair_fit
from InterOptimus.tool import sort_list
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.linalg import polar

def get_identical_pairs(match, film, substrate):
    film_idtc_millers = get_symmetrically_equivalent_miller_indices(film, match[0], return_hkil = False)
    substrate_idtc_millers = get_symmetrically_equivalent_miller_indices(substrate, match[1], return_hkil = False)
    combs = []
    for i in film_idtc_millers:
        for j in substrate_idtc_millers:
            combs.append((i,j))
    return combs

class equi_directions_identifier:
    """
    identify whether two vectors of a structure are identical
    """
    def __init__(self, structure):
        """
        Args:
        
        structure (Structure)
        """
        analyzer = SpacegroupAnalyzer(structure)
        self.symmetry_operations = analyzer.get_symmetry_operations(cartesian = True)
    def identify(self, v1, v2):
        """
        Args:
        
        v1, v2 (array): two directions to determine equivalency
        
        Return:
        (bool) : whether being equivalent
        """
        direction1 = v1/norm(v1)
        direction2 = v2/norm(v2)
        are_equivalent = False
        for operation in self.symmetry_operations:
            transformed_direction1 = operation.operate(direction1)
            if norm(cross(transformed_direction1, direction2)) < 1e-2:
                are_equivalent = True
                break
        return are_equivalent

class equi_match_identifier:
    """
    determine whether two matches are identical
    """
    def __init__(self, substrate, film, substrate_conv, film_conv):
        """
        Args:
        
        substrate (slab): substrate slab.
        film (slab): film slabs.
        """
        self.film = film
        self.substrate = substrate
        self.film_conv = film_conv
        self.substrate_conv = substrate_conv
        self.substrate_equi_directions_identifier = equi_directions_identifier(substrate_conv)
        self.film_equi_directions_identifier = equi_directions_identifier(film_conv)
    
    def identify_by_indices_matching(self, match_1, match_2):
        """
        Args
        
        match_1, match_2 (arrays): two matches to compare
        
        Return
        (bool): whether equivalent
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
        Args
        
        match_1, match_2 (arrays): two matches to compare
        
        Return
        (bool): whether equivalent
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
    cosine distance
    """
    return dot(v1, v2) / (norm(v1) * norm(v2))
    
def get_area_match(match):
    """
    matching area
    """
    return norm(cross(match.substrate_sl_vectors[0], match.substrate_sl_vectors[1]))


def match_search(substrate, film, substrate_conv, film_conv, sub_analyzer, film_millers, substrate_millers):
    """
    given substrate, film lattice structures, \
    get non-identical matches and identical match groups
    
    Args:
    substrate (Structure): primitive structure of the substrate material
    film (Structure): primitive structure of the film material
    
    Return:
    unique_matches (list): list of non-identical matches.
    equivalent_matches (list): clustered identical matches.
    unique_areas (list): list of matching areas of non-identical matches
    """
    matches = list(sub_analyzer.calculate(film=film, substrate=substrate, film_millers = film_millers, substrate_millers = substrate_millers))
    print(len(matches))
    areas = []
    for i in matches:
        areas.append(get_area_match(i))
    matches = sort_list(matches, areas)
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
    def is_equivalent(self, normal_1, MR_1, normal_2, MR_2, tol = 0.1):
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
    get the areas of included by two basic vectors
    """
    return norm(cross(v1,v2))

def interface_searching(substrate_conv, film_conv, sub_analyzer, film_millers = None, substrate_millers = None):
    """
    given substrate, film lattice structures, \
    get non-identical matches and identical match groups
    
    Args:
    substrate (Structure): primitive structure of the substrate material
    film (Structure): primitive structure of the film material
    sub_analyzer (SubstrateAnalyzer): SubstrateAnalyzer setting lattice matching parameters
    
    Return:
    unique_matches (list): list of non-identical matches.
    equivalent_matches (list): clustered identical matches.
    unique_matches_indices_data (list): indices information of non-identical matches.
    equivalent_matches_indices_data (list): clustered indices information of identical matches
    areas (list): list of matching areas of non-identical matches
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
    """Convert Miller indices to Cartesian normal vectors"""
    h, k, l = miller
    recip_lattice = lattice.reciprocal_lattice_crystallographic
    normal = h * recip_lattice.matrix[0] + k * recip_lattice.matrix[1] + l * recip_lattice.matrix[2]
    return normal/np.linalg.norm(normal)

def stereographic_projection(normal):
    """将法向量 (x, y, z) 投影到二维平面"""
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
    h, k, l = miller_index
    def format_component(c):
        if c < 0:
            return r"\bar{" + f"{abs(c)}" + r"}"
        else:
            return str(c)
    return r"$(" + format_component(h) + format_component(k) + format_component(l) + r")$"

def scatter_by_miller_dict(millers, dict, tuple_id, lattice, strains):
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
        film_millers = []
        substrate_millers = []
        for i in self.all_matche_data.keys():
            film_millers.append(i[0])
            substrate_millers.append(i[1])
        film_millers = list(set(film_millers))
        substrate_millers = list(set(substrate_millers))
        self.film_map = {m_id:id for id, m_id in enumerate(film_millers)}
        self.substrate_map = {m_id:id for id, m_id in enumerate(substrate_millers)}
    def plot_matching_data(self, names = ['film', 'substrate'], save_filename = 'stereographic_projection.jpg', show_millers = True, show_legend = True, show_title = True, special = False):
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

    def plot_unique_matches(self, filename = 'unique_matches.jpg'):
        x = []
        strains = []
        areas = []
        ct = 0
        for i in self.unique_matches:
            strains.append(i.von_mises_strain)
            areas.append(i.match_area)
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
