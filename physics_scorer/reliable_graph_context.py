"""Directional sequence aggregation, learned structural gates and anchored context output."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import GraphSequenceScorer


class ReliableGraphLayer(nn.Module):
    def __init__(self,width,sequence_v2=True,structure_gate=True):
        super().__init__()
        self.sequence_v2=sequence_v2;self.structure_gate=structure_gate
        self.self_proj=nn.Linear(width,width)
        self.sequence_proj=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(6)])
        self.pair_proj=nn.Linear(width,width,bias=False)
        self.edge_gate=nn.Sequential(nn.Linear(width+3,32),nn.GELU(),nn.Linear(32,1))
        nn.init.constant_(self.edge_gate[-1].bias,-2.)
        self.norm=nn.LayerNorm(width);self.drop=nn.Dropout(.1)
        self.last_gate_mean=None
        self.last_gate_count=0

    def forward(self,x,valid,structure,pairs,disable_pairs=False):
        b,length,width=x.shape;update=self.self_proj(x);den=x.new_zeros(b,length,1)
        sequence=x.new_zeros(x.shape)
        distances=[1,2,4] if self.sequence_v2 else [1]
        for j,d in enumerate(distances):
            if d>=length:continue
            mask=valid[:,:-d]&valid[:,d:]
            sequence[:,d:]+=self.sequence_proj[2*j](x[:,:-d])*mask[...,None]
            sequence[:,:-d]+=self.sequence_proj[2*j+1](x[:,d:])*mask[...,None]
            den[:,d:]+=mask[...,None];den[:,:-d]+=mask[...,None]
        update=update+sequence/den.clamp_min(1)
        flat=x.reshape(-1,width);messages=torch.zeros_like(flat);degree=x.new_zeros(b*length,1)
        if pairs.numel() and not disable_pairs:
            src,dst=pairs
            st=structure.reshape(-1,5)
            descriptor=torch.cat(((flat[src]-flat[dst]).abs(),
                torch.minimum(st[src,0:1],st[dst,0:1]),
                torch.abs(st[src,0:1]-st[dst,0:1]),
                torch.log1p((src-dst).abs().to(x.dtype))[:,None]/8),-1)
            gate=self.edge_gate(descriptor).sigmoid() if self.structure_gate else x.new_ones(len(src),1)
            messages.index_add_(0,dst,self.pair_proj(flat[src])*gate)
            # Divide by count rather than sum(gate), so absolute reliability is retained.
            degree.index_add_(0,dst,x.new_ones(len(src),1))
            update=update+(messages/degree.clamp_min(1)).reshape_as(x)
            self.last_gate_mean=gate.detach().mean()
            self.last_gate_count=len(src)
        else:
            self.last_gate_mean=x.new_tensor(0.)
            self.last_gate_count=0
        return self.norm(x+self.drop(F.gelu(update))).masked_fill(~valid[...,None],0)


class ReliableGraphContext(nn.Module):
    def __init__(self,width=192,sequence_v2=True,structure_gate=True,anchored_head=True,physics=True):
        super().__init__()
        self.width=width;self.use_physics=physics;self.anchored_head=anchored_head
        self.codons=nn.Embedding(65,width,padding_idx=64)
        self.physics=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.LayerNorm(width))
        self.structure=nn.Linear(5,width)
        self.graph=nn.ModuleList([ReliableGraphLayer(width,sequence_v2,structure_gate) for _ in range(2)])
        self.input_fusion=nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.LayerNorm(width))
        layer=nn.TransformerEncoderLayer(width,8,2*width,.1,'gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.physics_head=nn.Linear(width,1);self.graph_head=nn.Linear(width,1);self.context_head=nn.Linear(width,1)
        self.correction=nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)

    def forward(self,codons,valid,physics,structure,pair_edges,disable_pairs=False,disable_global=False):
        b,length=codons.shape
        if codons.dtype!=torch.long or valid.dtype!=torch.bool or valid.shape!=codons.shape:
            raise ValueError('Invalid codons or mask')
        n=valid.sum(1)
        if (n==0).any() or not torch.equal(valid,torch.arange(length,device=codons.device)[None]<n[:,None]):
            raise ValueError('Expected nonempty right-padded prefixes')
        if physics.shape!=(*codons.shape,2) or structure.shape!=(*codons.shape,5):raise ValueError('Misaligned features')
        if ((codons[valid]<0)|(codons[valid]>=64)).any() or (codons[~valid]!=64).any():raise ValueError('Invalid token IDs')
        if not torch.isfinite(physics[valid]).all() or not torch.isfinite(structure[valid]).all():raise ValueError('Nonfinite features')
        if pair_edges.dtype!=torch.long or pair_edges.ndim!=2 or pair_edges.shape[0]!=2:raise ValueError('Invalid edges')
        if pair_edges.numel():
            if (pair_edges<0).any() or (pair_edges>=b*length).any():raise ValueError('Edge out of range')
            if not valid.flatten()[pair_edges].all() or (pair_edges[0]//length!=pair_edges[1]//length).any():raise ValueError('Cross-sample or padding edges')
        mask=~valid[...,None];s=structure.masked_fill(mask,0)
        e=self.codons(codons);p=self.physics(physics.masked_fill(mask,0))
        if not self.use_physics:p=torch.zeros_like(p)
        h=(e+self.structure(s)).masked_fill(mask,0)
        pairs=torch.unique(torch.cat((pair_edges,pair_edges.flip(0)),1),dim=1)
        for layer in self.graph:h=layer(h,valid,s,pairs,disable_pairs)
        z=e+.1*self.input_fusion(torch.cat((e,p,h),-1))
        z=z+GraphSequenceScorer._positions(length,self.width,z.device,z.dtype)
        attention_mask=~torch.eye(length,device=z.device,dtype=torch.bool) if disable_global else None
        c=self.context(z,mask=attention_mask,src_key_padding_mask=~valid).masked_fill(mask,0)
        raw_context=self.context_head(c).squeeze(-1)
        correction=self.correction(torch.cat((p,h,c),-1)).squeeze(-1)
        raw=raw_context+.1*correction if self.anchored_head else correction
        q=F.softplus(raw).masked_fill(~valid,0)
        aux=dict(physics=F.softplus(self.physics_head(p).squeeze(-1)).masked_fill(~valid,0),
            gnn=F.softplus(self.graph_head(h).squeeze(-1)).masked_fill(~valid,0),
            context=F.softplus(raw_context).masked_fill(~valid,0))
        return q,aux
