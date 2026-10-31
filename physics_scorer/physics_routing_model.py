"""Isolate physical input to the graph while retaining the direct sequence path."""
import torch
from torch import nn
from .profile_model import AlignedProfileModel


class PhysicsRoutingProfileModel(AlignedProfileModel):
    def __init__(self, graph_physics='none', late_physics=True, width=424):
        super().__init__(physics=True, graph=True, context=True, encoder='bilstm', width=width)
        if graph_physics not in {'none','add','gated'}:raise ValueError('Unknown physical route')
        self.graph_physics=graph_physics;self.late_physics=late_physics;self.width=width
        # Present in every control. A zero gate preserves the original starting function.
        self.physical_gate=nn.Parameter(torch.zeros(()))
        self._active=False;self._physical=None;self._override=None;self._valid=None;self._disable=False
        self.physics.register_forward_hook(self._capture_physics)
        self.graph[0].register_forward_pre_hook(self._inject_graph)
        self.fusion[0].register_forward_pre_hook(self._mask_late_physics)

    def _capture_physics(self,module,args,output):
        if self._active:
            self._physical=(self._override if self._override is not None else output).masked_fill(~self._valid[...,None],0)

    def _inject_graph(self,module,args):
        if self.graph_physics=='none' or self._disable:return args
        gain=1. if self.graph_physics=='add' else self.physical_gate.tanh()
        return (args[0]+gain*self._physical.reshape(args[0].shape),*args[1:])

    def _mask_late_physics(self,module,args):
        if self.late_physics:return args
        x=args[0]
        return (torch.cat((x[...,:self.width]*0,x[...,self.width:]),-1),*args[1:])

    def forward(self,codons,valid,physics,structure,pair_edges,graph_physics=None,disable_graph_physics=False):
        try:
            if graph_physics is not None:
                if graph_physics.shape!=physics.shape or not torch.isfinite(graph_physics[valid]).all():
                    raise ValueError('Misaligned graph-only physical input')
                self._override=self.physics(graph_physics.masked_fill(~valid[...,None],0))
            self._valid=valid;self._active=True;self._disable=disable_graph_physics
            return super().forward(codons,valid,physics,structure,pair_edges)
        finally:
            self._active=False;self._physical=None;self._override=None;self._valid=None;self._disable=False
