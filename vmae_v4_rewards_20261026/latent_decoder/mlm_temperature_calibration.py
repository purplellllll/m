"""Convex scalar inverse-temperature fit on fixed synonymous logits."""
import torch


def inverse_temperature(logits,targets,weights):
    if logits.ndim!=2 or targets.shape!=(len(logits),) or weights.shape!=targets.shape:
        raise ValueError('Expected logits[N,C], targets[N], weights[N]')
    if not len(logits) or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Nonempty positive total finite calibration weights required')
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits.gather(1,targets[:,None])).all():
        raise ValueError('Targets need finite supported logits; invalid classes may be -inf')
    values=logits.double();safe=values.masked_fill(~torch.isfinite(values),0.)
    chosen=values.gather(1,targets[:,None]).squeeze(1);weights=weights.double()/weights.sum()
    def derivative(inverse):
        probability=torch.softmax(values*inverse,-1)
        return float((((probability*safe).sum(-1)-chosen)*weights).sum())
    low,high=.25,4.
    if derivative(low)>=0:return low
    if derivative(high)<=0:return high
    for _ in range(32):
        midpoint=(low+high)/2
        if derivative(midpoint)>0:high=midpoint
        else:low=midpoint
    return (low+high)/2
