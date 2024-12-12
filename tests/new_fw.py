from fireworks import LaunchPad, Workflow
import pickle

lp = LaunchPad.auto_load()
with open('new_wf.pkl','rb') as f:
    wf = Workflow.from_dict(pickle.load(f))
lp.add_wf(wf)
print(len(wf))
