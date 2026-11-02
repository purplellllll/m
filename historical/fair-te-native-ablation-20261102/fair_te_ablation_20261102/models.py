"""Explicit task adaptations; upstream numerical architecture definitions are pinned."""
from pathlib import Path
import ast,sys,types,importlib.util
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
WORK=Path(__file__).resolve().parents[2]
REPO=WORK/'repos/riboseq_human_te_release_20261102'
sys.path.insert(0,str(WORK));sys.path.insert(0,str(REPO))
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.human_te_model import HumanTEHead
from physics_scorer.scalar_downstream import masked_pool
FROZEN=WORK/'runs/profile_model_comparison_20261030/completed_snapshot/frozen_sources'

def definitions(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns

class Ours(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.backbone=ReliableGraphContext(width=192,sequence_v2=True,structure_gate=True,anchored_head=False,physics=True)
        self.head=HumanTEHead(192)
    def forward(self,rows,features,batch):
        codons,valid,physics,structure,pair_edges=batch;b=self.backbone
        mask=~valid[...,None];s=structure.masked_fill(mask,0);e=b.codons(codons)
        p=b.physics(physics.masked_fill(mask,0)) if self.arm not in ['no_physics','seq_transformer'] else torch.zeros_like(e)
        if self.arm in ['no_graph','seq_transformer']:h=torch.zeros_like(e)
        else:
            h=(e+b.structure(s)).masked_fill(mask,0)
            pairs=torch.unique(torch.cat((pair_edges,pair_edges.flip(0)),1),dim=1)
            for layer in b.graph:h=layer(h,valid,s,pairs)
        z=e+.1*b.input_fusion(torch.cat((e,p,h),-1))
        from physics_scorer.model import GraphSequenceScorer
        z=z+GraphSequenceScorer._positions(z.shape[1],b.width,z.device,z.dtype)
        c=z if self.arm=='no_context' else b.context(z,src_key_padding_mask=~valid).masked_fill(mask,0)
        return self.head(masked_pool(c,valid),c.new_zeros((len(valid),3)))

class RiboGLScalar(nn.Module):
    def __init__(self):
        super().__init__()
        ns=dict(torch=torch,np=np,nn=nn,L=types.SimpleNamespace(LightningModule=nn.Module),Variable=Variable,SAGEConv=TorchSAGE,GraphNorm=TorchGraphNorm)
        names=['MaskedPearsonLoss','MaskedPearsonCorr','MaskedL1Loss','MaskedPCCL1Loss','ConvModule','GCN_LSTM']
        ns=definitions(FROZEN/'ribogl/gnn_utils.py',names,ns);lstm=nn.LSTM
        def compatible(size,*a,**kw):return lstm(int(size),*a,**kw)
        with patch.object(nn,'LSTM',compatible):self.base=ns['GCN_LSTM']([256,128,128,64],.1,18,1,.001,256,'GCN_LSTM','SAGE','None',2)
        self.head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        self.hidden=None;self.base.linear.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.hidden=args[0]
    def forward(self,rows,features,batch):
        values=[]
        for row in rows:
            f=features[row['uid']];device=batch[0].device
            graph=types.SimpleNamespace(x=torch.from_numpy(f['ribogl_ids']).to(device),edge_index=torch.from_numpy(f['ribogl_edges']).to(device),random_walk_pe=torch.from_numpy(f['ribogl_pe']).to(device))
            try:
                with torch.backends.cudnn.flags(enabled=False):self.base(graph)
                h=self.hidden.squeeze(1)
                values.append(self.head(torch.cat((h.mean(0),h.max(0).values))).squeeze(-1))
            finally:self.hidden=None
        return torch.stack(values)

class TorchSAGE(nn.Module):
    """PyG SAGE(mean, project=True) equations/state keys; avoids unrelated PyG imports."""
    def __init__(self,in_channels,out_channels,project=True):
        super().__init__();assert project
        self.lin=nn.Linear(in_channels,in_channels)
        self.lin_l=nn.Linear(in_channels,out_channels)
        self.lin_r=nn.Linear(in_channels,out_channels,bias=False)
        self.lin.reset_parameters();self.lin_l.reset_parameters();self.lin_r.reset_parameters()
    def forward(self,x,edge_index):
        projected=self.lin(x).relu();source,target=edge_index
        aggregate=torch.zeros_like(projected);aggregate.index_add_(0,target,projected[source])
        degree=torch.bincount(target,minlength=len(x)).clamp_min(1).to(x.dtype)[:,None]
        return self.lin_l(aggregate/degree)+self.lin_r(x)

class TorchGraphNorm(nn.Module):
    """Exact single-graph GraphNorm equations and parameter names."""
    def __init__(self,in_channels,eps=1e-5):
        super().__init__();self.eps=eps
        self.weight=nn.Parameter(torch.ones(in_channels));self.bias=nn.Parameter(torch.zeros(in_channels));self.mean_scale=nn.Parameter(torch.ones(in_channels))
    def forward(self,x):
        out=x-x.mean(0,keepdim=True)*self.mean_scale
        return self.weight*out/(out.square().mean(0,keepdim=True)+self.eps).sqrt()+self.bias

class Seq2Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        from latent_decoder.seq2ribo_cpu_reference import install
        backend=sys.modules.get('mamba_ssm') or install()
        ns=definitions(WORK/'tmp/seq2ribo_source_20261020/seq2ribo-main/seq2ribo/models.py',
            ['RiboPolisherMamba','TEHeadFullCounts'],dict(torch=torch,nn=nn,F=torch.nn.functional,Mamba=backend.Mamba,Mamba2=backend.Mamba2))
        self.base=ns['RiboPolisherMamba']();self.head=ns['TEHeadFullCounts'](positive_output=False)
    def forward(self,rows,features,batch):
        cod,valid=batch[:2];sim=torch.zeros_like(cod,dtype=torch.float32);angle=torch.zeros_like(cod);pair=angle.clone();bucket=angle.clone()
        for j,r in enumerate(rows):
            f=features[r['uid']];n=len(f['cod'])
            sim[j,:n]=torch.from_numpy(np.log1p(f['simulation_mean']).astype(np.float32)).to(cod.device)
            angle[j,:n]=torch.from_numpy(f['abin']).to(cod.device);pair[j,:n]=torch.from_numpy(f['paired']).to(cod.device)
            v=np.zeros(n,dtype=np.int64);v[n//3:2*n//3]=1;v[2*n//3:]=2;bucket[j,:n]=torch.from_numpy(v).to(cod.device)
        logs=self.base(cod,sim,valid,angle_bin=angle,pair_bin=pair,bucket_idx=bucket)
        return self.head(torch.expm1(logs.clamp(max=20))*valid,valid)

class RiboDecodeScalar(nn.Module):
    def __init__(self):
        super().__init__();path=WORK/'deliverables/public_riboseq_validation_20261028/source/inference.py'
        spec=importlib.util.spec_from_file_location('fair_ribodecode_native_cnn',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        import json
        config=json.loads((path.parent/'model_config.json').read_text(encoding='utf-8'))
        self.base=m.CNN_Encoder_Model(config,m.InferenceModel.input_shape)
        condition=torch.load(WORK/'runs/vmae_adjustment_gpu_20261024/transfer_assets/canonical_condition.pt',map_location='cpu',weights_only=True).reshape(1,-1)
        self.register_buffer('condition',condition)
    def forward(self,rows,features,batch):
        x=torch.zeros(len(rows),4,4500,device=batch[0].device)
        for j,r in enumerate(rows):
            ids=torch.tensor([dict(A=0,C=1,G=2,T=3)[c] for c in r['dna'][:-3]],device=x.device)
            x[j,:,:len(ids)]=torch.nn.functional.one_hot(ids,4).float().T
        return self.base(x,self.condition.expand(len(rows),-1),x.new_full((len(rows),1),4.5)).flatten()

ARMS=['full','no_physics','no_graph','no_context','seq_transformer','seq2ribo_TE_adapt','RiboGL_TE_adapt','RiboDecode_TE_adapt']
def build(arm):
    if arm=='seq2ribo_TE_adapt':return Seq2Scalar()
    if arm=='RiboGL_TE_adapt':return RiboGLScalar()
    if arm=='RiboDecode_TE_adapt':return RiboDecodeScalar()
    return Ours(arm)
