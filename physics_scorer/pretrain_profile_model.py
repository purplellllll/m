"""Masked physical-summary reconstruction from graph representations."""
import torch
from torch import nn
from torch.nn import functional as F
from .physics_routing_model import PhysicsRoutingProfileModel


class PretrainProfileModel(PhysicsRoutingProfileModel):
    def __init__(self,width=424,graph_physics='add',late_physics=False):
        super().__init__(width=width,graph_physics=graph_physics,late_physics=late_physics)
        self.reconstruction_head=nn.Linear(width,2)
        self._capture_reconstruction=False;self._reconstruction_hidden=None
        self.graph[-1].register_forward_hook(self._capture_hidden)

    def _capture_hidden(self,module,args,output):
        if self._capture_reconstruction:self._reconstruction_hidden=output

    def forward_with_reconstruction(self,*args,**kwargs):
        try:
            self._capture_reconstruction=True
            prediction,aux=self(*args,**kwargs)
            valid=args[1];hidden=self._reconstruction_hidden.reshape(*valid.shape,self.width)
            reconstruction=self.reconstruction_head(hidden).masked_fill(~valid[...,None],0)
            return prediction,aux,reconstruction
        finally:self._capture_reconstruction=False;self._reconstruction_hidden=None


def masked_reconstruction_loss(prediction,target,mask):
    if prediction.shape!=target.shape or mask.shape!=target.shape[:2] or mask.dtype!=torch.bool:
        raise ValueError('Misaligned reconstruction inputs')
    if not mask.any(1).all():raise ValueError('Each transcript needs masked positions')
    target=target.detach().masked_fill(~mask[...,None],0)
    error=F.smooth_l1_loss(prediction.masked_fill(~mask[...,None],0),target,reduction='none').mean(-1)
    return ((error*mask).sum(1)/mask.sum(1)).mean()
