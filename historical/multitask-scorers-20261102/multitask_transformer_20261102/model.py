"""Host-aware multi-assay Transformer with explicit missing-label masks."""
import math
import torch
from torch import nn
from torch.nn import functional as F
TASKS=['expression','half_life','solubility','te']
class MultiTaskTransformer(nn.Module):
    def __init__(self,stats_dim,width=192,use_physics=True,max_tokens=512):
        super().__init__();self.width=width;self.use_physics=use_physics;self.max_tokens=max_tokens
        self.codons=nn.Embedding(65,width,padding_idx=64)
        self.amino=nn.Embedding(22,width,padding_idx=21)
        self.host=nn.Embedding(2,16)
        self.node_physics=nn.Linear(12,width)
        nn.init.zeros_(self.node_physics.weight);nn.init.zeros_(self.node_physics.bias)
        layer=nn.TransformerEncoderLayer(width,8,2*width,.1,'gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.utr_embedding=nn.Embedding(5,64,padding_idx=4)
        self.utr_conv=nn.Conv1d(64,64,9,stride=3,padding=4)
        self.stats=nn.Sequential(nn.Linear(stats_dim,128),nn.GELU(),nn.LayerNorm(128))
        self.global_physics=nn.Sequential(nn.Linear(36,64),nn.GELU(),nn.LayerNorm(64))
        dimension=2*width+256+128+16+64+3
        self.heads=nn.ModuleDict({task:nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1)) for task in TASKS})
    @staticmethod
    def positions(n,d,device,dtype):
        pos=torch.arange(n,device=device,dtype=dtype)[:,None]
        f=torch.exp(torch.arange(0,d,2,device=device,dtype=dtype)*(-math.log(10000.)/d))
        return torch.stack((torch.sin(pos*f),torch.cos(pos*f)),-1).flatten(1)
    def forward(self,batch):
        valid=batch['valid'];lengths=valid.sum(1)
        if (lengths==0).any():raise ValueError('Empty CDS')
        embedded=self.codons(batch['codons'])+.1*self.amino(batch['amino'])
        if self.use_physics:embedded=embedded+.1*self.node_physics(batch['physics'])*batch['physics_present'][:,None,None]
        embedded=embedded+self.positions(embedded.shape[1],self.width,embedded.device,embedded.dtype)
        # Keep every CDS position; compress long CDS by average pooling, not truncation.
        tokens=[]
        for i,n in enumerate(lengths.tolist()):
            x=embedded[i,:n]
            if n>self.max_tokens:x=F.adaptive_avg_pool1d(x.T[None],self.max_tokens)[0].T
            tokens.append(x)
        x=nn.utils.rnn.pad_sequence(tokens,batch_first=True)
        mask=torch.arange(x.shape[1],device=x.device)[None]<torch.tensor([len(t) for t in tokens],device=x.device)[:,None]
        x=self.context(x,src_key_padding_mask=~mask)
        mean=(x*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        maximum=x.masked_fill(~mask[...,None],-torch.inf).amax(1)
        u=batch['utr'];b,regions,n=u.shape;flat=u.reshape(b*regions,n)
        z=F.gelu(self.utr_conv(self.utr_embedding(flat).transpose(1,2))).transpose(1,2)
        counts=(flat!=4).sum(1);m=torch.arange(z.shape[1],device=z.device)[None]<((counts+2)//3)[:,None]
        um=(z*m[...,None]).sum(1)/m.sum(1).clamp_min(1)[:,None]
        ux=z.masked_fill(~m[...,None],-torch.inf).amax(1);ux=torch.where(counts[:,None]>0,ux,torch.zeros_like(ux))
        uv=torch.cat([um,ux],1).reshape(b,-1)
        ph=self.global_physics(batch['physical_global'])*batch['physics_present'][:,None] if self.use_physics else mean.new_zeros((b,64))
        availability=torch.cat([batch['utr_present'],batch['physics_calibrated'][:,None]],1)
        rep=torch.cat([mean,maximum,uv,self.stats(batch['stats']),self.host(batch['host']),ph,availability],1)
        return torch.cat([self.heads[t](rep) for t in TASKS],1)
def masked_loss(pred,target,mask):
    if pred.shape!=target.shape or target.shape!=mask.shape:raise ValueError('Inconsistent target mask')
    if not torch.isfinite(target[mask]).all() or not mask.any():raise ValueError('Invalid observed targets')
    observed=torch.where(mask,target,torch.zeros_like(target))
    losses=(pred-observed).square()*mask
    # Equal loss per observed task; missing labels never become zero supervision.
    count=mask.sum(0);per=losses.sum(0)/count.clamp_min(1)
    return per[count>0].mean()
def selfcheck():
    torch.manual_seed(20261102);torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    b,n=2,20
    x=dict(codons=torch.randint(0,64,(b,n)),amino=torch.randint(0,20,(b,n)),valid=torch.ones(b,n,dtype=torch.bool),
        physics=torch.randn(b,n,12),physics_present=torch.ones(b),physical_global=torch.randn(b,36),physics_calibrated=torch.zeros(b),
        utr=torch.randint(0,4,(b,2,48)),utr_present=torch.ones(b,2),stats=torch.randn(b,279),host=torch.tensor([0,1]))
    m=MultiTaskTransformer(279).eval();p=m(x);assert p.shape==(b,4) and torch.isfinite(p).all()
    target=torch.randn_like(p);mask=torch.tensor([[1,0,0,0],[0,1,0,1]],dtype=torch.bool)
    a=masked_loss(p,target,mask);target[~mask]=float('nan');torch.testing.assert_close(a,masked_loss(p,target,mask))
    a.backward();assert m.context.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    no=MultiTaskTransformer(279,use_physics=False).eval();p=no(x);x['physics']+=100;x['physical_global']+=100
    torch.testing.assert_close(p,no(x),atol=0,rtol=0)
    print('PASS masked-label gradients and no-physics invariance')
if __name__=='__main__':selfcheck()
