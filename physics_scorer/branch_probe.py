"""Frozen-embedding mRFP probe architecture; output remains training-standardized."""
import torch
from torch import nn

class BranchProbe(nn.Module):
    def __init__(self,mask,nstats,width=192):
        super().__init__();self.mask=mask;self.nstats=nstats
        self.norms=nn.ModuleList([nn.LayerNorm(2*width) for _ in range(3)])
        self.readout=nn.Sequential(nn.Linear(6*width+6,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,pooled,globals_standardized):
        stats=torch.cat((globals_standardized[:,:self.nstats],torch.zeros_like(globals_standardized[:,self.nstats:])),1)
        return self.readout(torch.cat([norm(pooled[:,i])*self.mask[i] for i,norm in enumerate(self.norms)]+[stats],1)).squeeze(1)
