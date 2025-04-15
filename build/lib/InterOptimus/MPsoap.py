"""This module provides class to extract SOAP descriptors of the non-identical sites in the crystalline materials in Materials Project."""
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from dscribe.descriptors import SOAP
from ase.io import read as aR
from numpy import *
from pymatgen.core.periodic_table import Element
from mp_api.client import MPRester
import os
import itertools
import shutil
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm
import pickle
import time
from InterOptimus.tool import read_key_item, existfilehere

def get_Z(struct):
    """given structure, get element names.
    
    Args:
        struct (Structure).
        
    Return:
        list of atomic numbers.
    """
    
    return [i.Z for i in struct.elements]
    
def generate_combinations(elements):
    """given elements, get all possible combinations.
    
    Args:
        elements (list): list of elements.
        
    Return:
        combinations (list): list of combinations.
    """
    combinations = []
    
    for i in range(1, len(elements) + 1):
        for combo in itertools.combinations(elements, i):
            combinations.append('-'.join(combo))
    return combinations
    
def get_elements(struct):
    """given structure, get atomic number list.
    
    Args:
        struct (Structure).
        
    Return:
        list of elements.
    """
    return [i.symbol for i in struct.elements]

def to_ase(pymatgen_struct):
    """given pymatgen structure, get ase Atoms.
    
    Args:
        struct (Structure).
        
    return:
        ase Atoms.
    """
    pymatgen_struct.to_file('POSCAR_tt')
    ase_struct = aR('POSCAR_tt')
    os.remove('POSCAR_tt')
    return ase_struct

def MPsearch(elements, API_KEY, theoretical = False, is_stable = True, filter_elemental_materials = True):
    """searching for synthesized structures including at least a set of elements from Materials Project.
    
    Args:
        elements (list): list of elments included at least.
        API_KEY (str): API key.
    
    return:
        docs (list): list of searching results.
    """
    #print(generate_combinations(elements))
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=generate_combinations(elements), \
            fields=["material_id", "structure", "nelements"], \
            theoretical=False,
            is_stable=is_stable,
        )
    if theoretical:
        with MPRester(API_KEY) as mpr:
            docs.extend(mpr.materials.summary.search(
                chemsys=generate_combinations(elements), \
                fields=["material_id", "structure", "nelements"], \
                theoretical=True,
                is_stable=is_stable,
            ))
    if filter_elemental_materials:
        docs = [i for i in docs if i.nelements > 1]
    return docs

class stct_help_class:
    def __init__(self, structure):
        self.structure = structure

class soap_data_generator:
    """generate soap data from MP database.
    """
    def __init__(self, \
    elements, \
    API_KEY, \
    theoretical, is_stable, filter_elemental_materials, structure_from_MP, film, substrate, from_dir = False):
        self.elements = elements
        self.theoretical = theoretical
        self.structure_from_MP = structure_from_MP
        if not from_dir:
            if self.structure_from_MP:
                self.docs = MPsearch(elements, API_KEY, theoretical, is_stable, filter_elemental_materials)
            else:
                self.docs = [stct_help_class(film), stct_help_class(substrate)]
        else:
            with open('MPdocs.pkl', 'rb') as file:
                docs = pickle.load(file)
            self.docs = []
            for i in docs.keys():
                self.docs.append(stct_help_class(Structure.from_dict(docs[i])))
    """
    Args:
        elements (list): list of elements to consider.
        API_KEY (string): API key for using Materials Project.
        theoretical (bool): whether to consider theoretical materials (not synthesized yet).
    """
    @classmethod
    def from_dir(cls):
        set_data = read_key_item('INTAR')
        substrate_conv = Structure.from_file('SBS.cif')
        film_conv = Structure.from_file('FLM.cif')
        elements = list(set([i.symbol for i in film_conv.elements]).union([i.symbol for i in substrate_conv.elements]))
        return cls(elements, set_data['APIKEY'], set_data['THEORETICAL'], set_data['STABLE'], set_data['NOELEM'],\
                                set_data['STCTMP'], film_conv, substrate_conv, True)
    
    def calculate_soaps(self, soap_params = None, output_sym_stct = False):
        """
        get soap descriptors for all the searched materials.
        
        Args:
        soap_params (dict): SOAP parameters.
        """
        soap_params_default = {'r_cut':6, 'n_max':7, 'l_max':7, \
        'weighting':{"function":"pow", "r0":4, "c":1, "d":1,
        "m":20}}
        if soap_params == None or len(soap_params) == 0:
            self.soap_params = soap_params_default
        else:
            for j in soap_params.keys():
                soap_params_default[j] = soap_params[j]
            self.soap_params = soap_params_default
        self.soap_elements = []
        self.soap_struct_indices = []
        self.sym_structures = []
        self.soap_site_indices = []
        self.soap_descs = []
        self.min_nb_distances = []
        self.EN_diffs = []
        #soap analyzer initialization
        with tqdm(total=len(self.docs), desc="calculating SOAPs", leave=False) as struct_bar:
            for i in range(len(self.docs)):
                my_soap_analyzer = soap_analyzer(self.elements, self.docs[i].structure, i, self.soap_params)
            
                #extract soap for each element
                my_soap_analyzer.extract_soap_for_searching_elements(self.elements)
                #update symmetrized structure info
                self.sym_structures.append(my_soap_analyzer.struct)
                #update soap structured data
                for j in my_soap_analyzer.soap_infos:
                    if len(self.soap_descs) == 0:
                        self.soap_descs = j.vector
                    else:
                        self.soap_descs = vstack((self.soap_descs, j.vector))
                    self.soap_elements.append(j.center_element)
                    self.soap_struct_indices.append(j.belonging_structure_index)
                    self.soap_site_indices.append(j.site_index)
                    self.min_nb_distances.append(j.min_nb_distance)
                    self.EN_diffs.append(j.EN_diff)
                struct_bar.update(1)
        self.soap_elements, self.soap_struct_indices, \
        self.soap_site_indices, self.min_nb_distances, self.EN_diffs = \
                                                            array(self.soap_elements), array(self.soap_struct_indices), \
                                                            array(self.soap_site_indices), array(self.min_nb_distances),\
                                                            array(self.EN_diffs)
        
        
        self.cluster_by_element()
        
        if output_sym_stct:
            try:
                shutil.rmtree('docs_sym_structures')
            except:
                print('generate searched structures')
            os.mkdir('docs_sym_structures')
            for i in range(len(self.sym_structures)):
                self.sym_structures[i].to_file(f'docs_sym_structures/{i}_POSCAR')
        
    def cluster_by_element(self):
        """
        cluster the soap descriptors by element names.
        """
        self.by_element_dict = {}
        min_dists_saved = existfilehere('min_dists.dat')
        for i in self.elements:
            self.by_element_dict[i] = {}
            self.by_element_dict[i]['soap_descs'] = \
            self.soap_descs[self.soap_elements == i]
            
            self.by_element_dict[i]['soap_struct_indices'] = \
             self.soap_struct_indices[self.soap_elements == i]
            
            self.by_element_dict[i]['soap_site_indices'] = \
             self.soap_site_indices[self.soap_elements == i]
            
            self.by_element_dict[i]['min_nb_distances'] = \
             self.min_nb_distances[self.soap_elements == i]
            if not min_dists_saved:
                with open('min_dists.dat','a') as f:
                    for distance in self.min_nb_distances[self.soap_elements == i]:
                        f.write(f'{distance} ')
                    f.write(f'\n')
            
            self.by_element_dict[i]['EN_diffs'] = \
             self.EN_diffs[self.soap_elements == i]
            
            self.by_element_dict[i]['min_nb_distance'] = \
             min(self.by_element_dict[i]['min_nb_distances'])
            
            self.by_element_dict[i]['pd'] =\
             pd.DataFrame(columns=['elements','struct_id','site_id'])
            
            for j in range(len(self.by_element_dict[i]['soap_descs'])):
                self.by_element_dict[i]['pd'].loc[j] =\
                 [self.docs[self.by_element_dict[i]['soap_struct_indices'][j]].structure.elements,\
                    self.by_element_dict[i]['soap_struct_indices'][j],\
                    self.by_element_dict[i]['soap_site_indices'][j]]
    
    def get_distances(self):
        """
        get the distances(dissimilarities) of all the descriptors
        
        Return:
        distance_pdist (array): distance list.
        """
        distance_pdist = {}
        for i in self.elements:
            dis_list = pdist(self.by_element_dict[i]['soap_descs'], \
            metric = 'cosine')
            distance_pdist[i] = dis_list
        return distance_pdist
        
class soap_info:
    """
    soap descriptor information
    
    Args:
    vector (array): soap descripor.
    center_element (string): center element name.
    belonging_structure_index (int): which structure it belongs to.
    site_index (int): which site it is.
    min_nb_distance: nearest neighboring distance.
    """
    def __init__(self, vector, center_element, belonging_structure_index, \
    site_index, min_nb_distance, EN_diff):
        self.vector = vector #soap vector
        self.center_element = center_element #center element
        self.belonging_structure_index = belonging_structure_index #which structure it belongs to
        self.site_index = site_index #at which site
        self.min_nb_distance = min_nb_distance #minimum neighboring distance
        self.EN_diff = EN_diff
        
class soap_analyzer:
    """
    for a given structure, get the soap descriptors for all the non-identical sites
    """
    def __init__(self, elements, struct, struct_index, soap_params):
        """
        Args:
        
        elements: (list): list of elements considered
        structure (Structure): structure to calculate soap
        struct_index (int): index of the structure
        soap_params (dict): soap parameters
        """
        self.struct = struct
        self.get_non_equi_sites_indices()
        
        periodic_soap = SOAP(
        species={i: Element(i).Z for i in elements},
        r_cut=soap_params['r_cut'],
        n_max=soap_params['n_max'],
        l_max=soap_params['l_max'],
        periodic=True,
        sparse=False,
        weighting = soap_params['weighting'],
        #compression = {"mode": "mu2", "species_weighting":{el.symbol:el.Z * soap_params['Z_scale'] for el in Element}}
        )
        self.soap_discriptors_nesites = periodic_soap.create(self.ase_struct, \
                                                             centers = self.non_equi_sites_indices)
        self.struct_index = struct_index
        
    def get_non_equi_sites_indices(self):
        """given structure, get the indices of the non-equivalent sites
        """
        analyzer = SpacegroupAnalyzer(self.struct.get_primitive_structure())
        symmetrized_structure = analyzer.get_symmetrized_structure()
        self.non_equi_sites_indices = [i[0] for i in symmetrized_structure.equivalent_indices]
        self.non_equi_sites_elements = [i[0].label for i in symmetrized_structure.equivalent_sites]
        self.struct = symmetrized_structure
        self.ase_struct = to_ase(symmetrized_structure)
    
    def extract_soap_for_searching_elements(self, cons_elements):
        """extract soap for determined elements
        
        Args:
        cons_elements (list): elements to extract their soaps
        """
        self.soap_infos = []
        for i in range(len(self.non_equi_sites_elements)):
            if self.non_equi_sites_elements[i] in cons_elements:
                this_soap = soap_info(self.soap_discriptors_nesites[i],
                                      self.non_equi_sites_elements[i],
                                      self.struct_index,
                                      self.non_equi_sites_indices[i],
                                      get_min_nb_distance(self.non_equi_sites_indices[i], self.struct),
                                      get_EN_diff_crystall(self.struct, self.non_equi_sites_indices[i]))
                self.soap_infos.append(this_soap)

def get_delta_distances(atom_index, structure, cutoff):
    neighbors = structure.get_neighbors(structure[atom_index], r=cutoff)
    if len(neighbors) > 0:
        return array([neighbor[1] for neighbor in neighbors])
    else:
        return array([cutoff])
    
def get_min_nb_distance(atom_index, structure):
    """
    get the minimum neighboring distance for certain atom in a structure
    
    Args:
    atom_index (int): atom index in the structure
    structure (Structure)
    
    Return:
    (float): nearest neighboring distance
    """
    neighbors = structure.get_neighbors(structure[atom_index], r=10)
    return min([neighbor[1] for neighbor in neighbors])

def get_EN_diff_crystall(structure, site_idx):
    cn = CrystalNN()
    center_EN = structure[site_idx].specie.X
    nb_ENs = array([i['site'].specie.X for i in cn.get_nn_shell_info(structure, site_idx, 1)])
    return sum(nb_ENs - center_EN)

def get_EN_diff_interface(interface, site_idx, r_cut):
    cn = CrystalNN()
    center_EN = interface[site_idx].specie.X
    nb_ENs = array([i['site'].specie.X for i in cn.get_nn_info(interface, site_idx) if i['site'].distance(interface[site_idx]) < r_cut])
    return sum(nb_ENs - center_EN)
