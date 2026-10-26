"""Length-invariant RMS direction for a frozen, position-wise latent search."""
import torch


def rms_direction(gradient,active):
    if gradient.ndim!=3 or active.shape!=gradient.shape[:2] or active.dtype!=torch.bool:
        raise ValueError('Expected gradient[B,L,D] and boolean active[B,L]')
    if not active.any(-1).all() or not torch.isfinite(gradient).all():
        raise ValueError('Every sequence needs mutable positions and finite gradients')
    value=gradient*active[...,None]
    rms=(value.double().square().sum((1,2))/(active.sum(-1)*gradient.shape[-1])).sqrt()
    return torch.where(rms[:,None,None]>0,value/rms.to(value.dtype).clamp_min(1e-30)[:,None,None],torch.zeros_like(value))
