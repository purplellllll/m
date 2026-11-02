"""Portable inference for the released own multi-task checkpoints.

No original workspace or raw training data is required. Install requirements.txt
and download/extract the own-models release asset.
"""
from pathlib import Path
import json,hashlib,itertools,importlib.util,sys
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from Bio.Seq import Seq
TASKS=['expression','half_life','solubility','te'];AA='ACDEFGHIKLMNPQRSTVWY'
KMERS=[''.join(x) for k in [1,2,3] for x in itertools.product('ACGT',repeat=k)]
CODONS=[''.join(x) for x in itertools.product('ACGT',repeat=3)]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def composition(s):
    s=s.upper().replace('U','T');counts={k:Counter(s[i:i+k] for i in range(len(s)-k+1)) for k in [1,2,3]}
    return [counts[len(k)][k]/max(1,len(s)-len(k)+1) for k in KMERS]
class Scorer:
    def __init__(self,models_root,stage='profile',arm='dynamics',device='cpu'):
        if stage not in ['neutral','profile','diffusion'] or arm not in ['sequence','dynamics']:raise ValueError('Unknown stage/arm')
        self.root=Path(models_root);self.stage=stage;self.arm=arm;self.device=device
        torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
        source=Path(__file__).resolve().parent/'runtime'/stage
        self.dynamics=module('released_dynamics_'+stage,source/'dynamics_v2.py')
        if stage=='diffusion':module('base_model',source/'base_model.py')
        Model=module('released_model_'+stage,source/'model.py').MultiTaskTransformer
        if stage!='neutral':
            import RNA
            if RNA.__version__!='2.7.2':raise RuntimeError('Profile stage requires ViennaRNA2.7.2')
            self.RNA=RNA
        with np.load(self.root/stage/'input_scaler.npz',allow_pickle=False) as z:self.scale={k:z[k].copy() for k in z.files}
        self.stats=json.loads((self.root/stage/'target_statistics.json').read_text(encoding='utf8'));manifest=json.loads((self.root/'model_manifest.json').read_text(encoding='utf8'))
        self.models=[];self.hashes=[]
        for seed in [30101,30102]:
            p=self.root/stage/f'{arm}_{seed}.pt';relative=p.relative_to(self.root).as_posix();expected=next(r['sha256'] for r in manifest['models'] if r['file']==relative)
            if sha(p)!=expected:raise ValueError('Checkpoint SHA mismatch')
            raw=torch.load(p,map_location='cpu',weights_only=True);m=Model(**raw['config']).to(device).eval();m.load_state_dict(raw['model'],strict=True);self.models.append(m);self.hashes.append(expected)
    def predict(self,dna,host,utr5='',utr3=''):
        dna=dna.upper().replace('U','T');utr5=utr5.upper().replace('U','T');utr3=utr3.upper().replace('U','T')
        if host not in ['human','ecoli']:raise ValueError('host must be human or ecoli')
        params=None;geometry=None
        if self.stage!='neutral':
            params=json.loads((self.root/'profile'/f'{host}_parameters.json').read_text(encoding='utf8'));bracket,_=self.RNA.fold(dna[:150].replace('T','U'));paired=np.zeros(len(dna)//3,dtype=np.int64)
            for i in range(len(bracket)//3):paired[i]=sum(c!='.' for c in bracket[3*i:3*i+3])
            geometry=dict(paired=paired,abin=np.zeros_like(paired))
        event=self.dynamics.extract(dna,host,params=params,geometry=geometry,seed=20261102,n_runs=2,burn_steps=8192,observe_steps=4096)
        protein=str(Seq(dna).translate(table=11 if host=='ecoli' else 1,cds=True));n=len(dna)//3
        cod=np.array([CODONS.index(dna[i:i+3]) for i in range(0,len(dna),3)],dtype=np.int64)
        amino=np.array([AA.index(a) if a in AA else 20 for a in protein]+[20],dtype=np.int64);assert len(amino)==n
        stats=[]
        for region in [utr5,dna,utr3]:stats.extend(composition(region))
        stats.extend(protein.count(a)/max(1,len(protein)) for a in AA);stats.extend([np.log1p(len(dna)),np.log1p(len(utr5)),np.log1p(len(utr3)),np.log1p(len(protein))])
        stats.extend(sum(b in 'GC' for b in dna[i:i+30])/max(1,len(dna[i:i+30])) for i in [0,30,60])
        if self.stage!='neutral':
            for seq in [dna[:k] for k in [30,60,90,150]]+[utr5[-80:]+dna[:120],utr5[:150]]:stats.append(float(self.RNA.fold(seq.replace('T','U'))[1])/len(seq) if seq else 0.)
        utrs=[]
        for seq in [utr5,utr3]:
            if len(seq)>1024:seq=seq[:512]+seq[-512:]
            utrs.append(np.array([{'A':0,'C':1,'G':2,'T':3}.get(c,4) for c in seq],dtype=np.int64))
        u=max(3,*[len(a) for a in utrs]);uv=np.full((1,2,u),4,np.int64);present=np.zeros((1,2),np.float32)
        for i,a in enumerate(utrs):uv[0,i,:len(a)]=a;present[0,i]=len(a)>0
        s=self.scale
        arrays=dict(codons=cod[None],amino=amino[None],valid=np.ones((1,n),bool),utr=uv,utr_present=present,host=np.array([host=='ecoli'],np.int64),
            physics=((event['node']-s['node_mean'])/s['node_scale']).astype(np.float32)[None],physical_global=((event['global']-s['global_mean'])/s['global_scale']).astype(np.float32)[None],
            stats=((np.array(stats)-s['stats_mean'])/s['stats_scale']).astype(np.float32)[None],physics_present=np.ones(1,np.float32),physics_calibrated=np.array([self.stage!='neutral'],np.float32))
        x={k:torch.from_numpy(v).to(self.device) for k,v in arrays.items()}
        with torch.inference_mode():z=np.mean([m(x).float().cpu().numpy()[0] for m in self.models],axis=0)
        return dict(predictions={t:float(z[j]*self.stats[t]['scale']+self.stats[t]['mean']) for j,t in enumerate(TASKS)},checkpoint_sha256=self.hashes,
            scope={'expression':'Ecoli mRFP supplied fluorescence target','half_life':'HEK293 mean log10(HL+1), not hours','solubility':'PURE assay percent, unbounded/no clipping','te':'Auxiliary human TE'},
            warning='Different assay/host heads; crosshost outputs are extrapolation. No validated joint functional-expression/HL/solubility scalar.',parameter_status=event['audit']['parameter_status'])
