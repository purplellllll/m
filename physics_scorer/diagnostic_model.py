"""Parameter-preserving routing interventions on the frozen parallel architecture."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .profile_model import AlignedProfileModel


class RoutingProfileModel(AlignedProfileModel):
    def __init__(self, physics_into_graph=False, graph_into_context=False, **kwargs):
        super().__init__(**kwargs)
        if physics_into_graph and not (self.enabled[0] and self.enabled[1]):
            raise ValueError('Physical routing requires both branches')
        if graph_into_context and not (self.enabled[1] and self.enabled[2] and self.encoder_kind=='bilstm'):
            raise ValueError('Graph routing requires GNN and BiLSTM')
        self.physics_into_graph=physics_into_graph
        self.graph_into_context=graph_into_context
        self._routing_mask=None;self._captured_physics=None;self._captured_graph=None
        if physics_into_graph:
            self.physics.register_forward_hook(self._capture_physics)
            self.graph[0].register_forward_pre_hook(self._inject_physics)
        if graph_into_context:
            self.graph[-1].register_forward_hook(self._capture_graph)
            self.context.register_forward_pre_hook(self._inject_graph)

    def _capture_physics(self,module,args,output):
        self._captured_physics=output.masked_fill(~self._routing_mask[...,None],0)

    def _inject_physics(self,module,args):
        physical=self._captured_physics.reshape(args[0].shape)
        return (args[0]+physical,*args[1:])

    def _capture_graph(self,module,args,output):self._captured_graph=output

    def _inject_graph(self,module,args):
        b,length=self._routing_mask.shape
        hidden=self._captured_graph.reshape(b,length,-1)
        packed=pack_padded_sequence(hidden,self._routing_mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        return (packed,*args[1:])

    def forward(self,codons,valid,physics,structure,pair_edges):
        self._routing_mask=valid
        try:return super().forward(codons,valid,physics,structure,pair_edges)
        finally:
            self._routing_mask=None;self._captured_physics=None;self._captured_graph=None


class PhysicsEmbedding(nn.Module):
    """Zero-initialized physical residual, preserving the native initial backbone."""
    def __init__(self,embedding):
        super().__init__();self.source=embedding
        self.physical_weight=nn.Parameter(torch.zeros(2,embedding.embedding_dim))
        self.current_physics=None

    def forward(self,ids):
        if self.current_physics is None:raise ValueError('Missing aligned physical input')
        return self.source(ids)+self.current_physics@self.physical_weight


class RiboGLPhysicalResidual(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=native
        self.native.embedding=PhysicsEmbedding(self.native.embedding)

    def forward(self,data):
        if data.physics.shape!=(data.x.numel(),2):raise ValueError('Misaligned physical features')
        self.native.embedding.current_physics=data.physics
        try:return self.native(data)
        finally:self.native.embedding.current_physics=None

    def _get_loss(self,data):
        prediction=self(data);mask=torch.isfinite(data.y)
        return self.native.loss(prediction,data.y,mask)


def native_profile_loss(prediction,target,observed):
    """The native PCC + sqrt(mean absolute error), after native target normalization."""
    if not observed.any(1).all():raise ValueError('Every transcript requires observed targets')
    target=target.masked_fill(~observed,0)
    target=target/target.sum(1,keepdim=True).clamp_min(1e-12)
    prediction=prediction/prediction.sum(1,keepdim=True).clamp_min(1e-12)
    losses=[]
    for p,y,mask in zip(prediction,target,observed):
        p=p[mask];y=y[mask]
        pcc=torch.nn.functional.cosine_similarity(p-p.mean(),y-y.mean(),dim=0,eps=1e-6)
        losses.append(1-pcc+torch.sqrt(torch.mean(torch.abs(p-y))))
    return torch.stack(losses).mean()
