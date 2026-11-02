"""Strict multi-task inference on complete CDS and optional annotated UTRs."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import torch
from Bio.Seq import Seq
import train as tr
import features as ft
from model import MultiTaskTransformer,TASKS
HERE=Path(__file__).resolve().parent
class Scorer:
    def __init__(self,arm='dynamics'):
        tr.setup();self.statistics=ft.read(HERE/'target_statistics.json');self.models=[];self.hashes=[]
        for seed in tr.SEEDS:
            root=HERE/'models'/f'{arm}_{seed}';p=root/'best.pt';complete=ft.read(root/'complete.json');assert ft.sha(p)==complete['best_sha256']
            ck=torch.load(p,map_location='cpu',weights_only=True);m=MultiTaskTransformer(**ck['config']).cuda().eval();m.load_state_dict(ck['model'],strict=True);self.models.append(m);self.hashes.append(ft.sha(p))
    def __call__(self,dna,host,utr5='',utr3=''):
        dna=dna.upper().replace('U','T');params=ft.read(HERE/(host+'_parameters.json'))
        # Validation and provenance enforced by the same event engine as training.
        v=ft.dynamics.extract(dna,host,params=params,geometry=ft.geometry(dna),seed=20261102,n_runs=2,burn_steps=8192,observe_steps=4096)
        protein=str(Seq(dna).translate(table=11 if host=='ecoli' else 1,cds=True));uid=hashlib.sha256(dna.encode()).hexdigest();r=dict(record_id='inference:'+uid,uid=uid,dna=dna,aa=protein,host=host,utr5=utr5,utr3=utr3)
        f=ft.base_features(r)
        with np.load(HERE/'input_scaler.npz',allow_pickle=False) as z:
            f.update(physics=((v['node']-z['node_mean'])/z['node_scale']).astype(np.float32),physical_global=((v['global']-z['global_mean'])/z['global_scale']).astype(np.float32),stats=((f['stats']-z['stats_mean'])/z['stats_scale']).astype(np.float32),physics_calibrated=1.)
        p=np.mean([tr.predict(m,[r],{r['record_id']:f})[0] for m in self.models],axis=0)
        result={t:float(p[j]*self.statistics[t]['scale']+self.statistics[t]['mean']) for j,t in enumerate(TASKS)}
        return dict(predictions=result,checkpoint_sha256=self.hashes,host=host,UTR_context_available=bool(utr5 and utr3),
            warnings=['Expression head calibrated on Ecoli mRFP fluorescence, not arbitrary human protein concentration',
                      'HL head predicts HEK293 log10(HL+1), not absolute hours or chemical stability',
                      'Solubility head predicts PURE assay values; can exceed0..100; not cell solubility guarantee',
                      'Crosshost/task outputs are extrapolations; no calibrated all-task universal scalar'],parameter_status=v['audit']['parameter_status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='JSON dna,host,optional utr5,utr3');p.add_argument('--output',required=True);a=p.parse_args();r=ft.read(a.input);ft.save(a.output,Scorer()(**r))
