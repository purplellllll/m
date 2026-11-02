"""Task-supervised scalar outputs from the existing three-branch profile encoder.

TE/protein are independent supervised labels; the normalized profile is never
summed and relabelled as either quantity. Wrapping leaves old checkpoints intact.
"""
import torch
from torch import nn


def masked_pool(value, valid):
    if value.shape[:2] != valid.shape or not valid.any(1).all():
        raise ValueError('Nonempty aligned sequence required')
    mean=value.masked_fill(~valid[...,None],0).sum(1)/valid.sum(1,keepdim=True)
    maximum=value.masked_fill(~valid[...,None],-torch.inf).amax(1)
    return torch.cat((mean,maximum),-1)


class ScalarHeads(nn.Module):
    """Each output is one scalar per sequence, with explicit task inputs."""
    def __init__(self,width,task,n_outputs=1,environment_dim=10552,use_utr=False):
        super().__init__()
        if task not in {'rpf','te','protein'} or n_outputs<1:
            raise ValueError('Unknown scalar task')
        if task!='te' and use_utr:raise ValueError('UTRs belong to TE input variant')
        self.task=task;self.use_utr=use_utr;self.width=width
        self.branch_logits=nn.Parameter(torch.zeros(3))
        self.branch_norm=nn.ModuleList([nn.LayerNorm(2*width) for _ in range(3)])
        extra=6
        if task=='rpf':
            self.environment=nn.Sequential(nn.Linear(environment_dim,64),nn.GELU(),nn.LayerNorm(64))
            extra+=65
        if use_utr:
            self.nucleotides=nn.Embedding(5,32,padding_idx=4)
            self.utr_conv=nn.Conv1d(32,64,9,padding=4)
            extra+=128
        self.readout=nn.Sequential(nn.Linear(6*width+extra,256),nn.GELU(),nn.Dropout(.1),
                                   nn.Linear(256,64),nn.GELU(),nn.Linear(64,n_outputs))

    def forward(self,pooled,global_features,log_rna=None,environment=None,utrs=None,environment_embedding=None,
                physics_enabled=True):
        if pooled.ndim!=3 or pooled.shape[1:]!=(3,2*self.width):raise ValueError('Expected three pooled branches')
        if global_features.shape!=(len(pooled),6):raise ValueError('Six sequence/simulation statistics required')
        logits=self.branch_logits
        if not physics_enabled:
            # Cached representations and simulation summary statistics are separate physics entrances.
            pooled=torch.cat((torch.zeros_like(pooled[:,:1]),pooled[:,1:]),1)
            global_features=torch.cat((global_features[:,:3],torch.zeros_like(global_features[:,3:])),1)
            logits=logits.masked_fill(torch.tensor([True,False,False],device=logits.device),-torch.inf)
        gains=(3 if physics_enabled else 2)*logits.softmax(0)
        branches=[norm(pooled[:,j])*gains[j] for j,norm in enumerate(self.branch_norm)]
        parts=branches+[global_features]
        if self.task=='rpf':
            if log_rna is None or (environment is None and environment_embedding is None):raise ValueError('RPF requires actual RNA and sample context')
            if log_rna.shape!=(len(pooled),1):raise ValueError('One RNA scalar per record')
            encoded=self.environment(environment) if environment_embedding is None else environment_embedding
            if encoded.shape!=(len(pooled),64):raise ValueError('Invalid RNA-context representation')
            parts += [log_rna,encoded]
        if self.use_utr:
            if utrs is None or len(utrs)!=2:raise ValueError('Both UTR inputs required')
            for ids,mask in utrs:
                if ids.shape!=mask.shape or ids.dtype!=torch.long:raise ValueError('Invalid UTR tensors')
                # Mask before and after convolution, including fully absent UTRs.
                value=self.nucleotides(ids).masked_fill(~mask[...,None],0)
                value=torch.nn.functional.gelu(self.utr_conv(value.transpose(1,2))).transpose(1,2)
                parts.append(value.masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True).clamp_min(1))
        prediction=self.readout(torch.cat(parts,-1))
        return prediction+log_rna if self.task=='rpf' else prediction


class TaskScalarModel(nn.Module):
    def __init__(self,backbone,task,n_outputs=1,use_utr=False,environment_dim=10552):
        super().__init__()
        self.backbone=backbone
        width=getattr(backbone,'width',backbone.codons.embedding_dim)
        self.heads=ScalarHeads(width,task,n_outputs,environment_dim,use_utr)
        self._branches={}
        backbone.physics.register_forward_hook(self._capture_physics)
        backbone.graph[-1].register_forward_hook(self._capture_graph)
        backbone.context.register_forward_hook(self._capture_context)

    def _capture_physics(self,module,args,value):self._branches['physics']=value
    def _capture_graph(self,module,args,value):self._branches['graph']=value
    def _capture_context(self,module,args,value):
        if isinstance(value,tuple):raise ValueError('Scalar adapter currently requires Transformer context')
        self._branches['context']=value

    def representations(self,inputs):
        try:
            profile,_=self.backbone(*inputs)
            valid=inputs[1]
            if not getattr(self.backbone,'use_physics',True):
                self._branches['physics']=torch.zeros_like(self._branches['physics'])
            pooled=torch.stack([masked_pool(self._branches[k].reshape(*valid.shape,-1),valid)
                                for k in ['physics','graph','context']],1)
            return pooled,profile
        finally:self._branches.clear()

    def forward(self,inputs,global_features,**task_inputs):
        pooled,_=self.representations(inputs)
        return self.heads(pooled,global_features,physics_enabled=getattr(self.backbone,'use_physics',True),**task_inputs)
