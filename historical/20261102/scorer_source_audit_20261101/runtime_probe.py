"""Frozen-checkpoint path sensitivity; synthetic length checks are not accuracy tests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
REPO=WORK/'repos/riboseq_scorer_20261031'
sys.path.insert(0,str(REPO))
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.scalar_downstream import TaskScalarModel

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main(device):
    torch.set_num_threads(2)
    if device=='cuda':assert torch.cuda.is_available()
    probe=load_module('frozen_branch_probe',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
    previous=WORK/'runs/vmae_own_full30_cpu_20261031'
    rows=json.loads((previous/'inputs_locked.json').read_text(encoding='utf-8'))[:3]
    records=[]
    for seed in [30101,30102]:
        path=WORK/f'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_{seed}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        backbone=ReliableGraphContext(**payload['metadata']['spec']['config'])
        backbone.load_state_dict(payload['state_dict'],strict=True)
        model=TaskScalarModel(backbone,'protein').to(device).eval().requires_grad_(False)
        head_path=WORK/f'runs/mrfp_branch_probe_20261031/results/heads/gated_graph_{seed}_c.pt'
        meta=torch.load(head_path,map_location='cpu',weights_only=True)
        head=probe.Probe(meta['mask'],meta['nstats']).to(device).eval().requires_grad_(False)
        head.load_state_dict(meta['state_dict'],strict=True)
        for row in rows:
            dna=row['original']['cds_dna'];uid=hashlib.sha256(dna.encode()).hexdigest()
            with np.load(previous/'features'/f'{uid}.npz',allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
            x,global_values=probe.batch([dict(dna=dna,uid=uid)],{uid:features},device)
            stats=torch.tensor(((global_values-np.asarray(meta['global_mean']))/np.asarray(meta['global_scale'])).astype(np.float32),device=device)
            def predict(xx,**kwargs):
                # representations() has no intervention kwargs; capture its hooks directly.
                try:
                    model.backbone(*xx,**kwargs)
                    valid=xx[1]
                    from physics_scorer.scalar_downstream import masked_pool
                    pooled=torch.stack([masked_pool(model._branches[k].reshape(*valid.shape,-1),valid) for k in ['physics','graph','context']],1)
                    return head(pooled,stats)[0]*meta['y_scale']+meta['y_mean']
                finally:model._branches.clear()
            with torch.no_grad():
                baseline=float(predict(x));values={'baseline':baseline}
                values['baseline_structure_gates']=[dict(mean=float(layer.last_gate_mean),count=layer.last_gate_count) for layer in backbone.graph]
                for name,kwargs in [('without_pair_messages',{'disable_pairs':True}),('diagonal_context',{'disable_global':True})]:
                    values[name]=float(predict(x,**kwargs))
                for name,channel in [('physics_zero',2),('structure_zero',3)]:
                    altered=list(x);altered[channel]=torch.zeros_like(x[channel]);values[name]=float(predict(tuple(altered)))
            grad_x=list(x);grad_x[2]=x[2].detach().requires_grad_(True)
            grad_x[3]=x[3].detach().requires_grad_(True)
            prediction=predict(tuple(grad_x));prediction.backward()
            values['physics_input_gradient_l1']=float(grad_x[2].grad.abs().sum())
            values['structure_input_gradient_l1']=float(grad_x[3].grad.abs().sum())
            records.append(dict(seed=seed,sequence_id=row['sequence_id'],n_codons=len(dna)//3,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**values))
        del model,backbone,head
    result=dict(device=device,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                interpretation='Frozen sensitivity on three previously opened historical-panel originals, not new held-out accuracy, causal attribution or sufficiency.',records=records)
    (ROOT/f'runtime_probe_{device}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status='completed',device=device,cases=len(records))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    main(parser.parse_args().device)
