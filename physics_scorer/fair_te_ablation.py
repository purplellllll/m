"""Portable scalar inference for the controlled human-TE branch ablations."""
import torch
from torch import nn
from .reliable_graph_context import ReliableGraphContext
from .human_te_model import HumanTEHead
from .scalar_downstream import masked_pool
from .model import GraphSequenceScorer

class FairTEAblation(nn.Module):
    def __init__(self,arm,backbone_config):
        super().__init__()
        if arm not in ['full','no_physics','no_graph','no_context','seq_transformer']:raise ValueError('Unknown TE ablation')
        self.arm=arm;self.backbone=ReliableGraphContext(**backbone_config);self.head=HumanTEHead(self.backbone.width)
        self.register_buffer('stats_mean',torch.zeros(3));self.register_buffer('stats_scale',torch.ones(3))
        self.register_buffer('target_mean',torch.tensor(0.));self.register_buffer('target_scale',torch.tensor(1.))
    def forward(self,inputs,properties=None):
        codons,valid,physics,structure,pair_edges=inputs;b=self.backbone
        if codons.dtype!=torch.long or valid.dtype!=torch.bool or valid.shape!=codons.shape:raise ValueError('Invalid tokens or mask')
        if physics.shape!=(*codons.shape,2) or structure.shape!=(*codons.shape,5):raise ValueError('Misaligned features')
        if (valid.sum(1)==0).any():raise ValueError('Empty sequence')
        length=codons.shape[1]
        if not torch.equal(valid,torch.arange(length,device=valid.device)[None]<valid.sum(1)[:,None]):raise ValueError('Expected right-padded prefixes')
        if ((codons[valid]<0)|(codons[valid]>=64)).any() or (codons[~valid]!=64).any():raise ValueError('Invalid token IDs')
        if pair_edges.dtype!=torch.long or pair_edges.ndim!=2 or pair_edges.shape[0]!=2:raise ValueError('Invalid edges')
        if pair_edges.numel():
            if (pair_edges<0).any() or (pair_edges>=codons.numel()).any():raise ValueError('Edge out of range')
            if not valid.flatten()[pair_edges].all() or (pair_edges[0]//length!=pair_edges[1]//length).any():raise ValueError('Cross-sample or padding edges')
        if not torch.isfinite(physics[valid]).all() or not torch.isfinite(structure[valid]).all():raise ValueError('Nonfinite features')
        mask=~valid[...,None];s=structure.masked_fill(mask,0);e=b.codons(codons)
        p=b.physics(physics.masked_fill(mask,0)) if self.arm not in ['no_physics','seq_transformer'] else torch.zeros_like(e)
        if self.arm in ['no_graph','seq_transformer']:h=torch.zeros_like(e)
        else:
            h=(e+b.structure(s)).masked_fill(mask,0);pairs=torch.unique(torch.cat((pair_edges,pair_edges.flip(0)),1),dim=1)
            for layer in b.graph:h=layer(h,valid,s,pairs)
        z=e+.1*b.input_fusion(torch.cat((e,p,h),-1))
        z=z+GraphSequenceScorer._positions(z.shape[1],b.width,z.device,z.dtype)
        c=z if self.arm=='no_context' else b.context(z,src_key_padding_mask=~valid).masked_fill(mask,0)
        return self.head(masked_pool(c,valid),c.new_zeros((len(valid),3)))*self.target_scale+self.target_mean
