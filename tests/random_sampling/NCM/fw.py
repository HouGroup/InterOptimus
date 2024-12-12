from fireworks import LaunchPad, Workflow
#from atomate.vasp.database import VaspCalcDb
import pickle

lp = LaunchPad.auto_load()
#db = VaspCalcDb.from_db_file('/public5/home/t6s001944/.conda/envs/general/lib/python3.12/site-packages/atomate/config/db.json')
with open('wf.pkl','rb') as f:
    wf = Workflow.from_dict(pickle.load(f))
lp.add_wf(wf)
print(len(wf))
