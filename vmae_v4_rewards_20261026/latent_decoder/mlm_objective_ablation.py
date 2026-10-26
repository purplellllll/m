"""Remove visible reconstruction without changing the original masked supervision weight."""
from .model import PAD,vae_loss


def objective(logits,tokens,mu,logvar,selected,beta,visible_weight):
    if visible_weight not in (0.,1.) or beta<0:
        raise ValueError('This paired ablation supports visible weight 0/1 and nonnegative beta')
    valid=tokens<PAD
    fraction=((selected&valid).sum()/valid.sum()).item()
    return vae_loss(logits,tokens,mu,logvar,selected,beta,
        full_weight=visible_weight,masked_weight=1+(1-visible_weight)*fraction)
