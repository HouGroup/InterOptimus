import json
from atomate.vasp.database import VaspCalcDb

def readDBvasp(db, field):
    data = db.collection.find_one(field)
    energy = 0
    if data != None:
        print(data['state'])
        print(data['task_id'])
        if data['state'] == 'successful':
            energy = data['output']['energy']
    return energy

def readDBvasp_dp(db, field, dp = True):
    if not dp:
        cursor = list(db.collection.find(field).sort({ '_id': 1 }).limit(1))
    else:
        cursor = list(db.collection.find(field).sort({ '_id': 1 }).skip(1).limit(1))
    energy = 0
    if len(cursor) != 0:
        data = cursor[0]
        #print(data.keys())
        #print(data['state'])
        #print(data['task_id'])
        if data['state'] == 'successful':
            energy = data['output']['energy']
    return energy
    
def read_global_random_sampling(db_file, project_name, dp = True):
    db = VaspCalcDb.from_db_file(db_file = db_file)
    with open('global_random_sampling.json', 'r') as f:
        input_data = json.load(f)
    tuple_keys = []
    output_data = input_data
    for key in input_data.keys():
        kl = key.split('_')
        i, j = int(kl[0]), int(kl[1])
        output_data[key]['DFT_results'] = {}
        output_data[key]['DFT_results']['xyz_Es'] = []
        output_data[key]['DFT_results']['it_Es'] = []
        output_data[key]['DFT_results']['bd_Es'] = []
        
        fmsg_E, fmdb_E, stsg_E, stdb_E = \
                readDBvasp_dp(db, field = {'pname':project_name, 'i':i, 'j':j, 'tp':'fmsg'}, dp = dp), \
                readDBvasp_dp(db, field = {'pname':project_name, 'i':i, 'j':j, 'tp':'fmdb'}, dp = dp), \
                readDBvasp_dp(db, field = {'pname':project_name, 'i':i, 'j':j, 'tp':'stsg'}, dp = dp), \
                readDBvasp_dp(db, field = {'pname':project_name, 'i':i, 'j':j, 'tp':'stdb'}, dp = dp)
                                                    
        output_data[key]['DFT_results']['fmsg_E'], \
        output_data[key]['DFT_results']['fmdb_E'], \
        output_data[key]['DFT_results']['stsg_E'], \
        output_data[key]['DFT_results']['stdb_E'] = fmsg_E, fmdb_E, stsg_E, stdb_E
        print(fmsg_E, fmdb_E, stsg_E, stdb_E)
        if fmsg_E != 0 and fmdb_E != 0 and stsg_E != 0 and stdb_E != 0:
            output_data[key]['slabs_suc'] = True
        else:
            output_data[key]['slabs_suc'] = False
        
        for k in range(len(input_data[f'{i}_{j}']['xyzs'])):
            sup_e = readDBvasp_dp(db, field = {'pname':project_name, 'i':i, 'j':j, 'k':k, 'tp':'it'}, dp = dp)
            if output_data[key]['slabs_suc']:
                it_e = (sup_e - (fmdb_E + stdb_E) / 2) / output_data[key]['A'] * 16.02176634
                bd_e = (sup_e - (fmsg_E + stsg_E)) / output_data[key]['A'] * 16.02176634
            else:
                it_e, bd_e = 0, 0
            output_data[key]['DFT_results']['it_Es'].append(it_e)
            output_data[key]['DFT_results']['bd_Es'].append(bd_e)
            output_data[key]['DFT_results']['xyz_Es'].append(sup_e)
        print(output_data[key]['DFT_results']['xyz_Es'])

    with open(f'global_random_sampling_dft_{dp}.json', 'w') as f:
        json.dump(output_data, f)
        
db_file = '/public5/home/t6s001944/.conda/envs/general/lib/python3.12/site-packages/atomate/config/db.json'
db = VaspCalcDb.from_db_file(db_file)
project_name = 'Li_Li3PS4'

read_global_random_sampling(db_file, project_name, True)
read_global_random_sampling(db_file, project_name, False)
