"""Portable human TE inference: strictly load both the backbone and scalar head."""
import copy
import torch
from torch import nn
from .reliable_graph_context import ReliableGraphContext
from .scalar_downstream import masked_pool

class HumanTEHead(nn.Module):
    def __init__(self,width=192):
        super().__init__();self.norm=nn.LayerNorm(2*width)
        self.net=nn.Sequential(nn.Linear(2*width+3,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,pooled,properties):return self.net(torch.cat((self.norm(pooled),properties),-1)).squeeze(-1)

class HumanTEScalar(nn.Module):
    """Accept original aligned inputs plus raw [GC fraction, MFE/nt, CAI]."""
    def __init__(self,backbone_config,context_layers=2,use_properties=False,sequence_only=False):
        super().__init__();self.backbone=ReliableGraphContext(**backbone_config)
        if context_layers not in [2,4]:raise ValueError('Only published2/4-layer contexts are supported')
        if context_layers==4:
            self.backbone.context.layers.extend([copy.deepcopy(layer) for layer in self.backbone.context.layers])
            self.backbone.context.num_layers=4
        self.head=HumanTEHead(backbone_config.get('width',192));self.use_properties=use_properties
        self.register_buffer('stats_mean',torch.zeros(3));self.register_buffer('stats_scale',torch.ones(3))
        self.register_buffer('target_mean',torch.tensor(0.));self.register_buffer('target_scale',torch.tensor(1.))
        self._context=None;self.backbone.context.register_forward_hook(self._capture)
        if sequence_only:self.backbone.input_fusion.register_forward_pre_hook(self._sequence_only)
    def _capture(self,module,args,value):self._context=value
    def _sequence_only(self,module,args):
        width=self.backbone.width;x=args[0]
        return (torch.cat((x[...,:width],torch.zeros_like(x[...,width:])),dim=-1),)
    def forward(self,inputs,properties=None):
        valid=inputs[1]
        if self.use_properties:
            if properties is None or properties.shape!=(len(valid),3):raise ValueError('Expected raw [GC fraction,MFE/nt,CAI]')
            if not torch.isfinite(properties).all():raise ValueError('Nonfinite properties')
            stats=(properties-self.stats_mean)/self.stats_scale
        else:stats=self.stats_mean.new_zeros((len(valid),3))
        try:
            self.backbone(*inputs);pooled=masked_pool(self._context,valid)
            return self.head(pooled,stats)*self.target_scale+self.target_mean
        finally:self._context=None
