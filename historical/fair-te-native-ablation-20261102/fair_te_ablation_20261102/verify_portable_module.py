"""Independent portable inference agrees with training-model code on real inputs."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;sys.path.insert(0,str(HERE))
import experiment as ex
from models import build
def main():
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    path=WORK/'repos/riboseq_fair_evidence_20261102/physics_scorer/fair_te_ablation.py'
    spec=importlib.util.spec_from_file_location('physics_scorer.portable_fair_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=[r for r in ex.read(HERE/'train_validation.json') if r['split']=='val'][:2];features=ex.load(rows);inputs=tuple(t.cpu() for t in ex.batch(rows,features));checks=[]
    for arm in ['full','no_physics','no_graph','no_context','seq_transformer']:
        torch.manual_seed(31);reference=build(arm).cpu().eval()
        candidate=m.FairTEAblation(arm,dict(width=192,sequence_v2=True,structure_gate=True,anchored_head=False,physics=True)).cpu().eval()
        state=dict(reference.state_dict());state.update(stats_mean=torch.zeros(3),stats_scale=torch.ones(3),target_mean=torch.tensor(0.),target_scale=torch.tensor(1.));candidate.load_state_dict(state,strict=True)
        with torch.no_grad():error=float((reference(rows,features,inputs)-candidate(inputs)).abs().max())
        assert error<2e-5;checks.append(dict(arm=arm,max_error=error))
    report=dict(status='passed',checks=checks,real_inputs=2,random_weights=True,biological_metrics_computed=False,portable_source_sha256=ex.sha(path))
    ex.save(HERE/'portable_module_parity.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
