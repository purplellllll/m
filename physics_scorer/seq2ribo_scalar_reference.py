"""Author's scalar TE architecture, retrained on the locked common split.

Uses the same cached four-trajectory mean as our model, rather than claiming
reproduction of the unavailable native seq2ribo training protocol.
"""
import torch
from torch import nn


class Seq2RiboScalarReference(nn.Module):
    def __init__(self,base,use_utr,use_log1p,normalization):
        super().__init__()
        from seq2ribo.models import TEHeadFullCounts,MambaUTRTEHead
        self.backbone=base;self.use_utr=use_utr
        self.heads=MambaUTRTEHead(use_log1p=False) if use_utr else TEHeadFullCounts(hidden=256,use_log1p=use_log1p)
        for name,value in normalization.items():self.register_buffer(name,torch.tensor(value,dtype=torch.float32).reshape(1,1))

    def forward(self,inputs,global_features,utrs=None):
        cod,valid,physics,structure,_=inputs
        # Invert the current feature normalization; both methods receive the
        # same simulation mean, geometry and entire CDS.
        raw_mean=torch.expm1(physics[...,0])*torch.expm1(8*global_features[:,3:4])
        positions=torch.arange(cod.shape[1],device=cod.device)[None]
        length=valid.sum(1,keepdim=True)
        bucket=torch.where(positions<length//3,0,torch.where(positions<2*length//3,1,2))
        logits=self.backbone(cod,torch.log1p(raw_mean),valid,
            angle_bin=(3*structure[...,1]).round().long(),pair_bin=(3*structure[...,0]).round().long(),bucket_idx=bucket)
        counts=torch.expm1(logits.clamp(max=20))*valid
        if self.use_utr:
            if utrs is None:raise ValueError('Reference UTR model needs both full UTRs')
            # Our alphabet ACGT; author's alphabet AUGC. Padding remains4.
            mapping=torch.tensor([0,3,2,1,4],device=cod.device)
            (u5,m5),(u3,m3)=utrs
            scaled=self.heads(mapping[u5],mapping[u3],counts,m5,valid,m3)
        else:scaled=self.heads(counts,valid)
        actual=scaled[:,None]*(self.target_high-self.target_low)+self.target_low
        return (actual-self.target_mean)/self.target_scale
