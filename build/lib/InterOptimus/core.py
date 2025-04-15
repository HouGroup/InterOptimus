from InterOptimus.optimize import HPtrainer, HPoptimizer, interface_score_ranker, WorkPatcher, InputDataGenerator
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from atomate.vasp.database import VaspCalcDb
from fireworks import LaunchPad
from InterOptimus.MPsoap import MPsearch, stct_help_class, soap_data_generator
from InterOptimus.tool import read_key_item
#from scipy.stats import pearsonr
from scipy.stats.mstats import spearmanr
from chgnet.model.model import CHGNet
from InterOptimus.VaspWorkFlow import LatticeRelaxWF, readDBvasp
from InterOptimus.tool import trans_to_bottom
import numpy as np
import shutil
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from fireworks import FiretaskBase

#class generate

def RelaxCryst():
    set_data = read_key_item('INTAR')
    wf = LatticeRelaxWF('FLM.cif', 'SBS.cif', set_data['PNAME'], set_data['NCORE'], set_data['DBFILE'], set_data['VASPCMD'])
    lp = LaunchPad.auto_load()
    lp.add_wf(wf)

def get_MPdocs():
    set_data = read_key_item('INTAR')
    substrate_conv = Structure.from_file('SBS.cif')
    film_conv = Structure.from_file('FLM.cif')
    elements = list(set([i.symbol for i in film_conv.elements]).union([i.symbol for i in substrate_conv.elements]))
    if set_data['STCTMP']:
        docs = MPsearch(elements, set_data['APIKEY'], set_data['THEORETICAL'], set_data['STABLE'], set_data['NOELEM'])
    else:
        docs = [stct_help_class(film_conv).to_json(), stct_help_class(substrate_conv).to_json()]
    try:
        shutil.rmtree('docs_structures')
    except:
        print('generate searched structures')
    os.mkdir('docs_structures')
    for i in range(len(docs)):
        docs[i].structure.to_file(f'docs_structures/{i}_POSCAR')
    docs_dir = {}
    for i in range(len(docs)):
        docs_dir[i] = docs[i].structure.as_dict()
    with open('MPdocs.pkl','wb') as f:
        pickle.dump(docs_dir, f)
        
def InitProj():
    set_data = read_key_item('INTAR')
    db = VaspCalcDb.from_db_file(set_data['DBFILE'])
    film_rlxdata = db.collection.find_one({'project_name': set_data['PNAME'], 'it': 'film'})
    substrate_rlxdata = db.collection.find_one({'project_name': set_data['PNAME'], 'it': 'substrate'})
    lat_rlxsucc = False
    if film_rlxdata != None and substrate_rlxdata != None:
        if film_rlxdata['state'] == 'successful' and substrate_rlxdata['state'] == 'successful':
            lat_rlxsucc = True
    if lat_rlxsucc:
        film = Structure.from_file('lattices/film/CONTCAR')
        substrate = Structure.from_file('lattices/substrate/CONTCAR')
        film.to_file('FLM.cif')
        substrate.to_file('SBS.cif')
    else:
        print('No successfully completed lattice relaxation jobs found, using default files')
    
    substrate_conv = Structure.from_file('SBS.cif')
    film_conv = Structure.from_file('FLM.cif')
    
    IDG = InputDataGenerator()
    IDG.dump_pickle()
    ISKer = interface_score_ranker(IDG, None, substrate_conv.get_primitive_structure(), film_conv.get_primitive_structure())
    ISKer.parse_opt_params(c_periodic = set_data['CPRD'], vacuum_over_film = set_data['VCOFLM'], slab_length = set_data['SLBLTH'], \
                 termination_ftol = set_data['TFTOL'])
    ISKer.get_match_term_idx()
    try:
        shutil.rmtree('it_initial_structures')
    except:
        print('generate initial interface structures')
    os.mkdir('it_initial_structures')
    pairs = ISKer.match_term_pairs
    for i in range(len(pairs)):
        ISKer.get_interface_by_id(i, False).to_file(f'it_initial_structures/{pairs[i][0]}_{pairs[i][1]}_POSCAR')
    
def RandomSampling():
    set_data = read_key_item('INTAR')
    HPT_data = read_key_item('RDSANAR')
    match_ids = HPT_data['MTACHID']
    term_ids = HPT_data['TERMID']
    MtchTerm_tuples = []
    for i in range(len(match_ids)):
        MtchTerm_tuples.append((match_ids[i], term_ids[i]))
    lp = LaunchPad.auto_load()
    wp = WorkPatcher.from_dir('.')
    wp.param_parse(set_data['PNAME'], set_data['TFTOL'], set_data['SLBLTH'], c_periodic = set_data['CPRD'], \
                    vacuum_over_film = set_data['VCOFLM'])
    db = VaspCalcDb.from_db_file(set_data['DBFILE'])
    db.reset()
    for i in MtchTerm_tuples:
        wp.get_unique_terminations(i[0])
        wf_RGS = wp.PatchRegistrationScan(i[0], i[1], HPT_data['ANCT'], n_calls = HPT_data['NUM'], \
                                rbt_non_closer_than = HPT_data['RNCT'], NCORE = set_data['NCORE'], \
                                db_file = set_data['DBFILE'], vasp_cmd = set_data['VASPCMD'])
        lp.add_wf(wf_RGS)

def ReadRandomSamplingResults():
    set_data = read_key_item('INTAR')
    HPT_data = read_key_item('RDSANAR')
    match_ids = HPT_data['MTACHID']
    term_ids = HPT_data['TERMID']
    MtchTerm_tuples = []
    for i in range(len(match_ids)):
        MtchTerm_tuples.append((match_ids[i], term_ids[i]))
    PNAME = set_data['PNAME']
    db = VaspCalcDb.from_db_file(set_data['DBFILE'])
    DFT_results = {}
    areas = read_pickle('areas.pkl')
    for j in MtchTerm_tuples:
        xyzs = np.loadtxt(f'{j[0]}_{j[1]}_xyzs')
        xyzs_cart = np.loadtxt(f'{j[0]}_{j[1]}_xyzs_carts')
        energies = []
        binding_energies = []
        substrate_E = readDBvasp(db, {'job':f'substrate', 'project_name':f'{j[0]}_{j[1]}'})
        film_t_E = readDBvasp(db, {'job':f'film_t', 'project_name':f'{j[0]}_{j[1]}'})
        area = areas[j[0]]
        for i in range(HPT_data['NUM']):
            it_energy = readDBvasp(db, {'job':f'rg_{i}', 'project_name':f'{j[0]}_{j[1]}'})
            energies.append(it_energy)
            binding_energies.append((it_energy - substrate_E - film_t_E)/area)
        energies = np.array(energies)
        binding_energies = np.array(binding_energies)
        print(energies)
        print(binding_energies)
        xyzs = xyzs[energies != np.inf]
        xyzs_cart = xyzs_cart[energies != np.inf]
        binding_energies = binding_energies[energies != np.inf]
        energies = energies[energies != np.inf]
        DFT_results[j] = {}
        DFT_results[j]['xyzs'] = xyzs
        DFT_results[j]['xyzs_cart'] = xyzs_cart
        DFT_results[j]['energies'] = energies
        DFT_results[j]['binding_energies'] = binding_energies
    with open('DFT_results.pkl', 'wb') as f:
        pickle.dump(DFT_results, f)

def save_results(result, name):
    dict = {'xs':result.x_iters, 'ys':result.func_vals}
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(dict,f)

def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
        
def save_pickle(file, data):
    with open(file, 'wb') as f:
        return pickle.dump(data, f)

def HPtraining(n_calls = 100, training_y = 'energies'):
    with open('DFT_results.pkl', 'rb') as f:
        DFT_results = pickle.load(f)
    set_data = read_key_item('INTAR')
    substrate_conv = Structure.from_file('SBS.cif')
    film_conv = Structure.from_file('FLM.cif')
    hptrainer = HPtrainer(substrate_conv, film_conv, SubstrateAnalyzer(max_area = set_data['MAXAREA'], max_length_tol = set_data['MAXLTOL'], max_angle_tol = set_data['MAXAGTOL']), DFT_results, \
                              set_data['SLBLTH'], set_data['TFTOL'], set_data['VCOFLM'], set_data['CPRD'], set_data['STCTMP'], training_y)
    result = HPoptimizer(hptrainer, n_calls)
    save_results(result, f"HPresults_{training_y}")

def draw_xs_ys(xs, ys, i, areas, training_y):
    fig, ax = plt.subplots(figsize = (5,5))
    cor = spearmanr(xs, ys).correlation
    ax.scatter(xs, ys, alpha = 0.4, s =400)
    ax.scatter(xs[ys.argmin()], ys[ys.argmin()], alpha = 0.5, s = 400, \
    c = 'none', edgecolors='C03', linewidth = 5)
    ax.scatter(xs[xs.argmin()], ys[xs.argmin()], alpha = 0.5, s = 400, \
    c = 'none', edgecolors='C01', linewidth = 5)
    min_energy_by_score = ys[xs.argmin()]
    min_energy = ys.min()
    delta_E = (min_energy_by_score - min_energy) * 16.02176634/areas[i[0]]
    #delta_S = xs.min() - xs[ys.argmin()]
    f_rate = len(xs[xs < xs[ys.argmin()]])/len(xs)
    ax.text(0.95, 0.95, f'$r$ = {np.around(cor,2)}\n$f$ = {np.around(f_rate,2)}\n$\Delta$E = {np.around(delta_E,2)} J/m$^2$',
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('scores', fontsize = 25)
    ax.set_ylabel('E eV', fontsize = 25)
    plt.tight_layout()
    fig.savefig(f'SvE_{i[0]}_{i[1]}_{training_y}.jpg', dpi = 600, format = 'jpg')

def OutputHPtrainingResults():
    set_data = read_key_item('INTAR')
    training_y = 'energies'
    result = read_pickle(f"HPresults_{training_y}.pkl")
    xs = np.array(result['xs'])
    ys = np.array(result['ys'])
    best_params = xs[ys.argmin()]
    columns = ['$r_{cut}$', '$n_{max}$', '$l_{max}$', \
          '$r_{0}^{sw}$', '$c^{sw}$', '$d^{sw}$', '$m^{sw}$', \
          r'$\xi^{soap}$', r'$\xi^{rp}$', r'$\xi^{EG}$', r'$f_{en}$', r'$P$']
    data = pd.DataFrame(np.column_stack((xs,ys)), columns=columns)
    plt.tight_layout()
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
    for i in range(len(data.columns)):
        axes[int(i/4)][i%4].scatter(data[columns[i]], data[columns[-1]])
        axes[int(i/4)][i%4].set_xlabel(columns[i], fontsize = 20)
        axes[int(i/4)][i%4].set_ylabel(columns[-1], fontsize = 20)
    plt.savefig(f"HP_{training_y}.jpg", dpi=600, format = 'jpg')
    DFT_results = read_pickle('DFT_results.pkl')

    rcut, n_max, l_max, \
    soapWr0, soapWc, soapWd, soapWm, \
    KFsoap, KFrp, KFen, \
    en_cut = best_params
    soap_params = {'r_cut':rcut, 'n_max':int(n_max), 'l_max':int(l_max), \
               'weighting':{"function":"pow", "r0":soapWr0, "c":soapWc, "d":soapWd, "m":soapWm}}

    soap_data = soap_data_generator.from_dir()
    soap_data.calculate_soaps(soap_params)
    #scores vs. energies
    substrate_conv = Structure.from_file('SBS.cif')
    film_conv = Structure.from_file('FLM.cif')
    IDG = InputDataGenerator()
    wp = WorkPatcher(IDG.unique_matches, soap_data, IDG.film, IDG.substrate)
    wp.param_parse(project_name = set_data['PNAME'], termination_ftol = set_data['TFTOL'], slab_length = set_data['SLBLTH'], c_periodic = set_data['CPRD'], vacuum_over_film = set_data['VCOFLM'], kernel_factors = {'soap':KFsoap, 'rp':KFrp, 'en':KFen}, en_cut = en_cut)
    wp.get_all_unique_terminations()
    results_by_BPs = {}
    all_xs, all_ys = [], []
    for i in list(DFT_results.keys()):
        scores_here = wp.score_interfaces(i[0], i[1], DFT_results[i]['xyzs'])
        scores_here = np.array(scores_here)
        xs, ys = scores_here, DFT_results[i][training_y]
        results_by_BPs[i] = np.column_stack((xs, ys))
        all_xs += list(xs)
        all_ys += list(ys)
        draw_xs_ys(xs, ys, i, IDG.areas, training_y)

    results_by_BPs[(-1,-1)] = np.column_stack((all_xs, all_ys))
    draw_xs_ys(np.array(all_xs), np.array(all_ys), (-1,-1), IDG.areas, training_y)
    save_pickle('SvE_by_BP_{training_y}.pkl', results_by_BPs)

def OutputCHGnetResults():
    chgnet = CHGNet.load()
    set_data = read_key_item('INTAR')
    DFT_results = read_pickle('DFT_results.pkl')
    training_y = 'energies'
    all_xs, all_ys = [], []
    IDG = InputDataGenerator()
    wp = WorkPatcher(IDG.unique_matches, '', IDG.film, IDG.substrate)
    wp.param_parse(project_name = set_data['PNAME'], termination_ftol = set_data['TFTOL'], slab_length = set_data['SLBLTH'], c_periodic = set_data['CPRD'], vacuum_over_film = set_data['VCOFLM'])
    wp.get_all_unique_terminations()
    results_by_BPs = {}
    for i in list(DFT_results.keys()):
        xs, ys = [], DFT_results[i][training_y]
        for j in DFT_results[i]['xyzs']:
            interface_here = trans_to_bottom(wp.get_interface_by_key(i[0], i[1], j))
            xs.append(chgnet.predict_structure(interface_here)['e'] * len(interface_here))
        results_by_BPs[i] = np.column_stack((xs, ys))
        xs = np.array(xs)
        all_xs += list(xs)
        all_ys += list(ys)
        draw_xs_ys(xs, ys, i, IDG.areas, training_y)

    results_by_BPs[(-1,-1)] = np.column_stack((all_xs, all_ys))
    draw_xs_ys(np.array(all_xs), np.array(all_ys), (-1,-1), IDG.areas, training_y)
    save_pickle('SvE_by_BP_{training_y}.pkl', results_by_BPs)

def ISKerInit():
    set_data = read_key_item('INTAR')
    HPT_data = read_key_item('RDSANAR')
    training_y = 'energies'
    result = read_pickle(f"HPresults_{training_y}.pkl")
    xs = np.array(result['xs'])
    ys = np.array(result['ys'])
    best_params = xs[ys.argmin()]
    rcut, n_max, l_max, \
    soapWr0, soapWc, soapWd, soapWm, \
    KFsoap, KFrp, KFen, \
    en_cut = best_params
    soap_params = {'r_cut':rcut, 'n_max':int(n_max), 'l_max':int(l_max), \
               'weighting':{"function":"pow", "r0":soapWr0, "c":soapWc, "d":soapWd, "m":soapWm}}
    #get_MPdocs()
    IDG = InputDataGenerator()
    soap_data = soap_data_generator.from_dir()
    soap_data.calculate_soaps(soap_params)
    ISRker = interface_score_ranker(IDG, soap_data, IDG.substrate, IDG.film)
    ISRker.parse_opt_params(c_periodic = set_data['CPRD'], vacuum_over_film = set_data['VCOFLM'], slab_length = set_data['SLBLTH'],\
                            termination_ftol = set_data['TFTOL'], opt_num=HPT_data['NUM'], \
                            kernel_factors = {'soap':KFsoap, 'rp':KFrp, 'en':KFen}, en_cut = en_cut)
    
    ISRker.global_searching()
    return ISRker

def HighThrougput(delta_S = 0.1):
    ISRker = ISKerInit()
    set_data = read_key_item('INTAR')
    wf = ISRker.PatchHighThroughputWF(delta_S, 'HighTP', set_data['NCORE'], set_data['DBFILE'], vasp_cmd = set_data['VASPCMD'])
    lp = LaunchPad.auto_load()
    lp.add_wf(wf)

def PatchISRkerBenchMark():
    ISRker = ISKerInit()
    set_data = read_key_item('INTAR')
    wf = ISRker.PatchAllMatchTermOPWF('ISRkerBenchMark', set_data['NCORE'], set_data['DBFILE'], vasp_cmd = set_data['VASPCMD'])
    lp = LaunchPad.auto_load()
    lp.add_wf(wf)
    
def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
        
def ReadISRkerBenchMark():
    set_data = read_key_item('INTAR')
    db = VaspCalcDb.from_db_file(set_data['DBFILE'])
    MTdata = np.loadtxt('ISRkerBenchMark/match_term_id_score.dat')
    scores = MTdata[:,2]
    it_ids = np.array(MTdata[:,-1], dtype = int)
    match_ids = np.array(MTdata[:,0], dtype = int)
    term_ids = np.array(MTdata[:,1], dtype = int)
    areas = np.array(read_pickle('areas.pkl'))[match_ids]
    binding_Es = []
    slab_Es = []
    it_Es = []
    for i in range(len(MTdata)):
        it_id = int(MTdata[i][-1])
        substrate_E = readDBvasp(db, {'job':f'substrate_{it_id}', 'project_name':'ISRkerBenchMark'})
        film_t_E = readDBvasp(db, {'job':f'film_t_{it_id}', 'project_name':'ISRkerBenchMark'})
        it_E = readDBvasp(db, {'job':f'it_{it_id}', 'project_name':'ISRkerBenchMark'})
        slab_Es.append(film_t_E + substrate_E)
        it_Es.append(it_E)
        binding_Es.append((substrate_E+film_t_E-it_E)/areas[i]*16.02176634)
    np.savetxt('ISRkerBenchMark_results.dat', np.column_stack((it_ids, match_ids, term_ids, scores, binding_Es, slab_Es, it_Es)), fmt = '%i %i %i %f %f %f %f')
        
        
