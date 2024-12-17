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
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.mstats import spearmanr

def dp_vs_ndp(dp_data, ndp_data):
    dp_results = {}
    ndp_results = {}
    for i in dp_data.keys():
        for key in ['sup_Es', 'it_Es', 'bd_Es']:
            dp_results[key] = []
            ndp_results[key] = []
        for k in dp_data.keys():
            atom_num = len(Structure.from_dict(json.loads(dp_data[i]['sampled_interfaces'][0])))
            dp_results['sup_Es'] += list(np.array(dp_data[k]['DFT_results']['xyz_Es'])/atom_num)
            ndp_results['sup_Es'] += list(np.array(ndp_data[k]['DFT_results']['xyz_Es'])/atom_num)

            for ikey in ['it_Es', 'bd_Es']:
                dp_results[ikey] += dp_data[k]['DFT_results'][ikey]
                ndp_results[ikey] += ndp_data[k]['DFT_results'][ikey]
    return dp_results, ndp_results

def round_sf_np(x,significant_figure=0):
    r=np.ceil(np.log(x)/np.log(10))
    f=significant_figure
    return np.around(np.round(x*(10**(f-r)),0)*(10**(r-f)),6)

def mae_ys_xs(ys, xs):
    return sum(abs(ys - xs))/len(ys)

def mse_ys_xs(ys, xs):
    return sum((ys - xs) ** 2)/len(ys)

def dft_vs_predict_from_dict(dft_data, predicted_data):
    dft_results = {}
    predict_results = {}
    for mlip in ['grace-2l', 'chgnet', 'mace', 'orb-models', 'sevenn']:
        dft_results[mlip] = {}
        predict_results[mlip] = {}
        dft_results[mlip]['sup_Es'] = []
        dft_results[mlip]['it_Es'] = []
        dft_results[mlip]['bd_Es'] = []
        dft_results[mlip]['slab_Es'] = []
        
        predict_results[mlip]['sup_Es'] = []
        predict_results[mlip]['it_Es'] = []
        predict_results[mlip]['bd_Es'] = []
        predict_results[mlip]['slab_Es'] = []
        
        for i in predicted_data.keys():
            atom_num = len(Structure.from_dict(json.loads(predicted_data[i]['sampled_interfaces'][0])))
            dft_results[mlip]['sup_Es'] += list(np.array(dft_data[i]['DFT_results']['xyz_Es'])/atom_num)
            
            predict_results[mlip]['sup_Es'] += list(np.array(predicted_data[i]['predict'][mlip]['sup_Es'])/atom_num)
            
            dft_results[mlip]['it_Es'] += dft_data[i]['DFT_results']['it_Es']
            predict_results[mlip]['it_Es'] += predicted_data[i]['predict'][mlip]['it_Es']
    
            dft_results[mlip]['bd_Es'] += dft_data[i]['DFT_results']['bd_Es']
            predict_results[mlip]['bd_Es'] += predicted_data[i]['predict'][mlip]['bd_Es']
    return dft_results, predict_results

def draw_dft_vs_predict_energies(ax, dft_results, predict_results, e, title, drop_E_cut=-2.8):
    # Initialize lists for DFT and predicted values
    all_dfts = []
    all_predicts = []
    count = 0
    sizes = []
    ax.plot([-100, 100], [-100, 100], 'k--', lw=1, zorder=1)
    for mlip in ['grace-2l', 'chgnet', 'mace', 'orb-models', 'sevenn']:
        dft_data = np.array(dft_results[mlip][e])
        predict_data = np.array(predict_results[mlip][e])

        # Filter based on the drop_E_cut condition
        con = np.array(dft_results[mlip]['sup_Es']) < drop_E_cut
        predict_data = predict_data[con]
        dft_data = dft_data[con]

        # Remove zero values
        non_zero_ids = np.where(dft_data != 0)[0]
        non_zero_dft = dft_data[non_zero_ids]
        non_zero_predict = predict_data[non_zero_ids]

        all_dfts += list(non_zero_dft)
        all_predicts += list(non_zero_predict)

        # Scatter plot with custom markers and color
        ax.scatter(non_zero_predict, non_zero_dft, alpha=0.5, s=(5-count)*80, label=mlip,
                   marker='o', color=plt.cm.tab10(count), edgecolor='black', linewidth=0.5, zorder=2)

        count += 1
        sizes.append((5-count)*80)
        
    if e == 'sup_Es':
        unit = r'$\mathrm{eV/atom}$'
        sh = 0.1
        labelpad = 1
    else:
        unit = r'$\mathrm{J/m^2}$'
        sh = 0.5
        labelpad = 1
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # Diagonal line y=x
    ax.set_xlim(min(min(dft_data), min(predict_data)) - sh, max(max(dft_data), max(predict_data)) + sh)
    ax.set_ylim(min(min(dft_data), min(predict_data)) - sh, max(max(dft_data), max(predict_data)) + sh)
    
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    if len(yticks) > len(xticks):
        ax.set_yticks(xticks)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(yticks)
        ax.set_yticks(yticks)

    # 获取当前 xlim 和 ylim
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # 计算边界扩展量
    x_margin = (x_max - x_min) * 0.007 * np.sqrt(max(sizes))  # 扩展 5% + 考虑散点大小
    y_margin = (y_max - y_min) * 0.007 * np.sqrt(max(sizes))
    #print(x_margin)
    # 自动调整 xlim 和 ylim
    ax.set_xlim(min(all_predicts) - x_margin, max(all_predicts) + x_margin)
    ax.set_ylim(min(all_dfts) - y_margin, max(all_dfts) + y_margin)

    # Adjust axis labels and titles
    ax.set_xlabel(f'MLIP {unit}', fontsize=15, labelpad=1)
    ax.set_ylabel(f'DFT {unit}', fontsize=15, labelpad=labelpad)
    ax.set_title(title, fontsize=15)
    ax.grid(True, linestyle='--', linewidth=0.5)

def draw_dft_vs_predict_energy_stat(ax, dft_results, predict_results, e, title, drop_E_cut = -2.8):
    count = 0
    sizes = []
    mlips = ['grace-2l', 'chgnet', 'mace', 'orb-models', 'sevenn']
    xs = []
    ys = []
    for mlip in mlips:
        dft_data = np.array(dft_results[mlip][e])
        predict_data = np.array(predict_results[mlip][e])

        # Filter based on the drop_E_cut condition
        con = np.array(dft_results[mlip]['sup_Es']) < drop_E_cut
        predict_data = predict_data[con]
        dft_data = dft_data[con]

        # Remove zero values
        non_zero_ids = np.where(dft_data != 0)[0]
        non_zero_dft = dft_data[non_zero_ids]
        non_zero_predict = predict_data[non_zero_ids]
        
        sp = spearmanr(dft_data, predict_data).correlation
        mse = mean_squared_error(dft_data, predict_data)
        scatter = ax.scatter(mse, sp, color=plt.cm.tab10(count), s=(len(mlips)-count)*80, edgecolors='black', marker = 'D', label=mlip, alpha = 0.5)
        count += 1
        sizes.append((len(mlips)-count)*100)
        xs.append(mse)
        ys.append(sp)
    
    
    # 获取当前 xlim 和 ylim
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # 计算边界扩展量
    x_margin = (x_max - x_min) * 0.007 * np.sqrt(max(sizes))  # 扩展 5% + 考虑散点大小
    y_margin = (y_max - y_min) * 0.007 * np.sqrt(max(sizes))
    #print(x_margin)
    # 自动调整 xlim 和 ylim
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    ax.set_xlabel('MSE',fontsize = 15)
    ax.set_ylabel(r'$\rho$',fontsize = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(title, fontsize = 15)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
def plot_random_sampling_results(drop_E_cut = -2.8, show_legend = True, auto_E_cut =False):
    with open('global_random_sampling_predict.json', 'r') as f:
        predicted_data = json.load(f)
    with open('global_random_sampling_dft_False.json', 'r') as f:
        ndp_data = json.load(f)
    with open('global_random_sampling_dft_True.json', 'r') as f:
        dp_data = json.load(f)
    dft_results, predict_results = dft_vs_predict_from_dict(ndp_data, predicted_data)
    
    if auto_E_cut:
        all_dft_sup_Es = []
        for key in dft_results.keys():
            all_dft_sup_Es += dft_results[key]['sup_Es']
        drop_E_cut = min(all_dft_sup_Es) + 1
    
    es = ['sup_Es', 'it_Es', 'bd_Es']
    ttls = [r'$E_{sp}$', r'$E_{it}$', r'$E_{ch}$']
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i in range(3):
        draw_dft_vs_predict_energies(axes[i], dft_results, predict_results, es[i], ttls[i], drop_E_cut)
         # Add a single legend below all subplots
        if show_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=5, \
                        bbox_to_anchor=(0.5, -0.05), fontsize = 13, \
                        columnspacing=1, handletextpad=0.1)
    
    if len(axes[1].get_xticks()) > len(axes[2].get_xticks()):
        axes[2].set_xticks(axes[1].get_xticks())
        axes[2].set_yticks(axes[1].get_yticks())
    else:
        axes[1].set_xticks(axes[2].get_xticks())
        axes[1].set_yticks(axes[2].get_yticks())

    xlim1, ylim1 = axes[1].get_xlim(), axes[1].get_ylim()
    xlim2, ylim2 = axes[2].get_xlim(), axes[2].get_ylim()
    
    for id in range(1, 3):
        axes[i].set_xlim(min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
        axes[i].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    
        # Adjust layout to avoid overlap
    if show_legend:
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at the bottom
    else:
        plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    plt.savefig("random_sampling_predict_vs_dft_energies.jpg", dpi=600, format='jpg', bbox_inches='tight')

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i in range(3):
        draw_dft_vs_predict_energy_stat(axes[i], dft_results, predict_results, es[i], ttls[i], drop_E_cut)
        if show_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=5, \
                        bbox_to_anchor=(0.5, -0.05), fontsize = 13, \
                        columnspacing=1, handletextpad=0.1, borderpad=1)
    xlim1 = axes[1].get_xlim()
    xlim2 = axes[2].get_xlim()
    if max(xlim2) > max(xlim1):
        axes[1].set_xlim(xlim2)
    else:
        axes[2].set_xlim(xlim1)
    if show_legend:
        plt.tight_layout(rect=[0, 0.11, 1, 1])  # Leave space for the legend at the bottom
    else:
        plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    plt.savefig("random_sampling_predict_vs_dft_energies_stat.jpg", dpi=600, format='jpg', bbox_inches='tight')
    

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i in range(3):
        draw_dp_vs_ndp(axes[i], dp_data, ndp_data, es[i], ttls[i], drop_E_cut)
    plt.tight_layout()  # Leave space for the legend at the bottom
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    plt.savefig("dp_vs_ndp.jpg", dpi=600, format='jpg', bbox_inches='tight')

def draw_dp_vs_ndp(ax, dp_data, ndp_data, e, title, drop_E_cut = -2.8):
    dp_results, ndp_results = dp_vs_ndp(dp_data, ndp_data)
    sp_dps, sp_ndps = np.array(dp_results['sup_Es']), np.array(ndp_results['sup_Es'])
    dps = np.array(dp_results[e])
    ndps = np.array(ndp_results[e])
    con = (sp_dps < drop_E_cut) & (sp_ndps < drop_E_cut) & (dps != 0) & (ndps != 0)
    dps = dps[con]
    ndps = ndps[con]
    ax.scatter(dps, ndps, alpha = 0.2, s =200)

    if e == 'sup_Es':
        unit = 'eV/atom'
        sh = 0.05
    else:
        unit = 'J/m$^2$'
        sh = 0.5
    ax.set_xlabel(f'DPC {unit}', fontsize =15, labelpad = 0)
    ax.set_ylabel(f'NDPC {unit}', fontsize = 15, labelpad = 0)
    ax.text(0.95, 0.15, f"MAE = {round_sf_np(mae_ys_xs(np.array(dps), np.array(ndps)),2)}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    ax.set_title(title, fontsize =15,)
    ax.plot([-100, 100], [-100, 100], 'k--', lw=1)
    ax.set_xlim(min(min(dps), min(ndps))-sh, max(max(dps), max(ndps)) +sh)
    ax.set_ylim(min(min(dps), min(ndps))-sh, max(max(dps), max(ndps)) +sh)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    if len(yticks) > len(xticks):
        ax.set_yticks(xticks)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(yticks)
        ax.set_yticks(yticks)

def get_min_nb_distance(atom_index, structure, cutoff):
    """
    get the minimum neighboring distance for certain atom in a structure
    
    Args:
    atom_index (int): atom index in the structure
    structure (Structure)
    
    Return:
    (float): nearest neighboring distance
    """
    neighbors = structure.get_neighbors(structure[atom_index], r=cutoff)
    if len(neighbors) == 0:
        return np.inf
    else:
        return min([neighbor[1] for neighbor in neighbors])

def sort_list(array_to_sort, keys):
    """
    sort list by keys
    
    Args:
    array_to_sort (array): array to sort
    keys (array): sorting keys
    
    Return:
    (array): sorted array
    """
    combined_array = []
    for id, row in enumerate(array_to_sort):
        combined_array.append((keys[id], row))
    combined_array_sorted = sorted(combined_array, key = lambda x: x[0])
    keys_sorted, array_sorted = zip(*combined_array_sorted)
    return list(array_sorted)

def apply_cnid_rbt(interface, x, y, z):
    """
    apply rigid body translation to an interface.
    
    Args:
    interface (Interface): interface before translation
    x (float), y (float): fractional cnid coordinates
    z: fractional coordinates in c
    Return:
    interface (Interface): interface after translation
    """
    CNID = calculate_cnid_in_supercell(interface)[0]
    CNID_translation = TranslateSitesTransformation(interface.film_indices, x*CNID[:,0] + y*CNID[:,1] + [0, 0, z])
    return CNID_translation.apply_transformation(interface)

def existfilehere(filename):
    return os.path.isfile(os.path.join(os.getcwd(), filename))

def get_termination_indices(slab, ftol= 0.25):
    """
    get the terminating atom indices of a slab.
    
    Args:
    (Structure): slab structure.
    
    Return:
    (arrays): terminating atom indices at the top and bottom.
    """
    frac_coords = slab.frac_coords
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    h = slab.lattice.c
    # Projection of c lattice vector in
    # direction of surface normal.
    for ii, jj in combinations(list(range(n)), 2):
        if ii != jj:
            cdist = frac_coords[ii][2] - frac_coords[jj][2]
            cdist = abs(cdist - np.round(cdist)) * h
            dist_matrix[ii, jj] = cdist
            dist_matrix[jj, ii] = cdist

    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, ftol, criterion="distance")
    clustered_sites: dict[int, list[Site]] = {c: [] for c in clusters}
    for idx, cluster in enumerate(clusters):
        clustered_sites[cluster].append(slab[idx])
    plane_heights = {np.mean(np.mod([s.frac_coords[2] for s in sites], 1)): c for c, sites in clustered_sites.items()}
    term_cluster_min = min(plane_heights.items(), key=lambda x: x[0])[1]
    term_cluster_max = max(plane_heights.items(), key=lambda x: x[0])[1]
    return np.where(clusters == term_cluster_min)[0], np.where(clusters == term_cluster_max)[0]

def get_termination_indices_shell(slab, shell = 1.5):
    """
    get the terminating atom indices of a slab.
    
    Args:
    (Structure): slab structure.
    shell(float): shell size to include termination atoms
    
    Return:
    (arrays): terminating atom indices at the top and bottom.
    """
    frac_coords_z = slab.cart_coords[:,2]
    low = min(frac_coords_z)
    high = max(frac_coords_z)
    return np.where(frac_coords_z < low + shell)[0], np.where(frac_coords_z > high - shell)[0]
    
def get_it_core_indices(interface):
    """
    get the terminating atom indices of a interface.
    
    Args:
    interface (Interface).
    
    Returns:
    (arrays): film top & bottom indices; substrate top & bottom indices.
    """
    ids = np.array(interface.film_indices)
    slab = interface.film
    ids_film_min, ids_film_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    
    ids = np.array(interface.substrate_indices)
    slab = interface.substrate
    ids_substrate_min, ids_substrate_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    return ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max


def convert_value(value):
    if value.upper() == '.TRUE.' or value.upper() == 'TRUE':
        return True
    elif value.upper() == '.FALSE.' or value.upper() == 'FALSE':
        return False
    if '/' in value:
        return value
    if ',' in value:
        return np.array(value.split(','), dtype = int)
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

def read_key_item(filename):
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
                
    if 'THEORETICAL' not in data.keys():
        data['THEORETICAL'] = False
    if 'STABLE' not in data.keys():
        data['STABLE'] = True
    if 'NOELEM' not in data.keys():
        data['NOELEM'] = True
    if 'STCTMP' not in data.keys():
        data['STCTMP'] = True
    return data
    
def get_one_interface(cib, termination, slab_length, xyz, vacuum_over_film, c_periodic = False):
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
    """Find transformation matrix that will rotate and strain the film to the substrate while preserving the c-axis."""
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

def get_non_strained_film(match, it):
    f_vs = match.film_sl_vectors
    s_vs = match.substrate_sl_vectors
    R_21, s_21 = get_rot_strain(f_vs, s_vs)
    R_1it, _ = get_rot_strain(s_vs, it.lattice.matrix[:2])
    trans_f = np.dot(R_1it, R_21)
    trans_b = np.dot(np.linalg.inv(R_21), np.linalg.inv(R_1it))
    trans = triple_dot(trans_f, np.linalg.inv(s_21), trans_b)
    DST = DeformStructureTransformation(trans)
    return trans_to_bottom(DST.apply_transformation(it.film))

def trans_to_bottom(stct):
    ids = np.arange(len(stct))
    min_fc = stct.frac_coords[:,2].min()
    TST = TranslateSitesTransformation(ids, [0,0,-min_fc])
    return TST.apply_transformation(stct)

def get_film_length(match, film, it):
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

def add_sele_dyn(it):
    sub_bot_indices = get_it_core_indices(it)[2]
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(it), axis = 0)
    mobility_mtx[sub_bot_indices] = [False, False, False]
    it.add_site_property('selective_dynamics', mobility_mtx)
    return it

def add_sele_dyn_slab(slab):
    sub_bot_indices = get_termination_indices(slab)[0]
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(slab), axis = 0)
    mobility_mtx[sub_bot_indices] = [False, False, False]
    slab.add_site_property('selective_dynamics', mobility_mtx)
    return slab

def cut_vaccum(structure, c):
    lps = structure.lattice.parameters
    carts = [i.coords for i in structure]
    max_z = max(np.array(carts)[:,2])
    lps = list(lps)
    lps[2] = c + max_z
    return Structure(Lattice.from_parameters(*lps), structure.species, [i.coords for i in structure], coords_are_cartesian = True)
