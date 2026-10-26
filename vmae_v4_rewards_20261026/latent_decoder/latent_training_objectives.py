"""Controlled beta and mean-reconstruction weight; evaluation stays fixed at beta=.001."""
import math

import torch

from .model import PAD, vae_loss


def reconstruction_objective(model, tokens, inputs, selected, recipe, noise_seed, amp):
    beta = recipe['training_beta']
    assert math.isfinite(beta) and beta >= 0 and recipe['sigma'] is None
    with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16, enabled=amp):
        mu, lv, middle = model.encode_features(inputs, tokens)
        mu, lv = mu.float(), lv.float()
        rng = torch.Generator(device=tokens.device).manual_seed(noise_seed)
        z = mu + torch.randn(mu.shape, device=mu.device, generator=rng)*(.5*lv).exp()
        logits = model.decode(z, tokens)
    rec, terms = vae_loss(logits.float(), tokens, mu, lv, selected, beta=0.)
    if recipe['mean_weight']:
        with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16, enabled=amp):
            mean_logits = model.decode(mu.to(logits.dtype), tokens)
        mean_rec, _ = vae_loss(mean_logits.float(), tokens, mu, lv, selected, beta=0.)
        rec = (rec+recipe['mean_weight']*mean_rec)/(1+recipe['mean_weight'])
    kl = .5*(mu.square()+lv.exp()-1-lv)
    penalty = kl[tokens<PAD].mean(0).clamp_min(recipe['free_bits']).sum()
    return rec+beta*penalty, terms, mu, middle
