"""Experimental human HEK293 TE score; separate from all historical mRFP search locks."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import torch
from run_pilot import ROOT,WORK,REPO,Head,ReliableGraphContext,TaskScalarModel,module,read,sha

class HumanTEScorer:
    def __init__(self,device='cpu'):
        self.device=device;self.arm=read(ROOT/'selection_locked.json')['arm'];self.models=[]
        self.source=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/source'
        self.parameters=read(self.source.parent/'data/physics_parameters_train_only.json')
        self.helper=module('human_inference_features',REPO/'historical/drivers/run_aligned_profile_ablation_20261030.py')
        self.probe=module('human_inference_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
        self.cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json');self.cache={}
        for seed in [30101,30102]:
            metadata=torch.load(WORK/f'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_{seed}.pt',map_location='cpu',weights_only=True)['metadata']
            base=ReliableGraphContext(**metadata['spec']['config'])
            if self.arm=='finetune4_properties':base.context.layers.extend([copy.deepcopy(z) for z in base.context.layers]);base.context.num_layers=4
            if self.arm=='finetune2_no_physics':base.use_physics=False
            if self.arm=='finetune2_sequence_only':base.input_fusion.register_forward_pre_hook(lambda m,args:(torch.cat((args[0][...,:192],torch.zeros_like(args[0][...,192:])),dim=-1),))
            path=ROOT/'checkpoints'/f'{self.arm}_{seed}.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
            base.load_state_dict(payload['backbone'],strict=True);head=Head();head.load_state_dict(payload['head'],strict=True)
            self.models.append((TaskScalarModel(base,'te').to(device).eval().requires_grad_(False),head.to(device).eval().requires_grad_(False),payload,sha(path)))

    @torch.no_grad()
    def __call__(self,dna):
        dna=dna.upper().replace('U','T');assert len(dna)%3==0 and set(dna)<=set('ACGT') and dna[-3:] in {'TAA','TAG','TGA'}
        dna_sha=hashlib.sha256(dna.encode()).hexdigest()
        # Native TE cohort identifiers hash RNA; keep simulator seeds/cache in that domain.
        uid=hashlib.sha256(dna.replace('T','U').encode()).hexdigest()
        if uid in self.cache:return self.cache[uid]
        path=ROOT/'features'/f'{uid}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        else:
            f=self.helper.structure_job((str(self.source),dna));seed=int(hashlib.sha256(('human_te:'+uid).encode()).hexdigest()[:8],16)
            f.update(self.helper.simulation_job((str(self.source),dna,f,self.parameters,seed)));f['dna_sha256']=uid
        assert str(f['dna_sha256'])==uid
        x,_=self.probe.batch([dict(dna=dna,uid=uid)],{uid:f},self.device)
        raw=np.array([(dna.count('G')+dna.count('C'))/len(dna),float(f['mfe'])/len(dna),np.exp(np.mean([np.log(self.cai[dna[i:i+3]]) for i in range(0,len(dna)-3,3)]))])
        values=[]
        for model,head,meta,_ in self.models:
            g=(raw-np.asarray(meta['stats_mean']))/np.asarray(meta['stats_scale'])
            if self.arm=='finetune2_no_stats':g*=0
            pooled,_=model.representations(x)
            values.append(float(head(pooled[:,2],torch.tensor(g[None],device=self.device,dtype=torch.float32))[0])*meta['target_scale']+meta['target_mean'])
        result=dict(own_score=float(np.mean(values)),seed_scores=values,task='human_HEK293_CDS_only_TE',arm=self.arm,
            gc_percent=100*float(raw[0]),mfe_per_nt=float(raw[1]),cai=float(raw[2]),mfe=float(f['mfe']),dna=dna,sequence_sha256=dna_sha,canonical_rna_sha256=uid,
            outside_pilot_length_domain=not 80<=len(dna)//3<=500,checkpoint_sha256=[z[3] for z in self.models],
            validation_scope='256train96validation, no test confirmation; mouse-proxy physics remains')
        self.cache[uid]=result;return result
