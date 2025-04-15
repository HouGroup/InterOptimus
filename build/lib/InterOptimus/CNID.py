"""
This module calculate CNID vectors for a interface
"""
from numpy import *
from numpy.linalg import *
from interfacemaster.cellcalc import DSCcalc
from interfacemaster.hetero_searching import apply_function_to_array, float_to_rational

def get_au_vector(B):
    """
    calculate the auxiliary vector of two vectors
    
    Args:
    B (array): two column vectors
    
    Return:
    (array): auxiliary normalized vector perpendicular to the input
    """
    auv = cross(B[:,0], B[:,1])
    return auv / norm(auv)

def from_2D_to_3D_transformation(B1, B2):
    """
    calculate the 3D transformation matrix of two 2D bases
    
    Args:
    B1, B2 (arrays): two bases
    
    Return:
    rotation matrix converting B2 to B1
    """
    auv_B1 = get_au_vector(B1)
    auv_B2 = get_au_vector(B2)
    C1 = column_stack((B1, auv_B1))
    C2 = column_stack((B2, auv_B2))
    return dot(C1, inv(C2))

def get_au_lattice(B):
    """
    get the auxiliary 3D lattice for a 2D basis
    
    Args:
    B (array): 2D basis
    
    Return:
    (array): 3D auxiliary lattice vectors
    """
    auv = get_au_vector(B)
    return column_stack((B,auv))
    
def triple_dot(a, b, c):
    """
    combined product
    """
    return dot(a, dot(b, c))
    
def calculate_cnid_in_supercell(interface):
    """
    calculate CNID for a interface
    
    Args:
    interface (Interface)
    
    Return:
    (array, dtype = float): CNID vectors by float
    (array, dtype = string): CNID vectors by rational numbers
    """
    props = interface.interface_properties
    transformation = from_2D_to_3D_transformation(props['substrate_sl_vectors'].T, props['film_sl_vectors'].T)
    B_substrate = array(props['substrate_vectors']).T
    B_film = array(props['film_vectors']).T
    B_substrate = get_au_lattice(B_substrate)
    B_film = get_au_lattice(B_film)
    B_film = dot(transformation, B_film)
    calc = DSCcalc()
    calc.parse_int_U(B_substrate, B_film, 200)
    calc.compute_CSL()
    calc.compute_CNID([0,0,1])
    CSL = calc.CSL
    CNID = calc.CNID
    slB = get_au_lattice(array(props['substrate_sl_vectors']).T)
    B = get_au_lattice(array(props['substrate_vectors']).T)
    CNID_sl = triple_dot(inv(slB), B, CNID)
    return CNID_sl, apply_function_to_array(CNID_sl, float_to_rational)
