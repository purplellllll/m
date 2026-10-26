"""Search in standardized latent displacement u and actual residual logits e."""
import hashlib
import math
import time

import torch
from torch.nn import functional as F

from .budget_comparison import native_generator, project_budget
from .dual_space_ablation import center_direct, masks, mean_square, regularizers
from .js_constraint import JSConstraint, mutable_js
from .latent_optimization import design_probabilities
from .length_diagnostics import encoder_row, gradient_summary
from .model import CODONS

BASE_U_LR = .5*.1/math.sqrt(2)*.25
BASE_E_LR = 1.251784384*.1/math.sqrt(2)*.25*1.6014172
REG = dict(latent_l2=.4, direct_l2=.1/1.251784384**2, decoder_kl=.1)


def search(model, scorer, tokens, identity, seed, percent, coefficients=(1., 1.), js_limit=.05, steps=80):
    active, support = masks(tokens)
    with torch.no_grad():
        mu, logvar = model.encode(tokens, tokens)
        sigma = (.5*logvar).exp()
        reference = design_probabilities(model.decode(mu, tokens), tokens)
        baseline = float(scorer(F.one_hot(tokens, 64).float(), tokens))
    key = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big') % (2**63-1)
    noise = torch.randn(mu.shape, generator=torch.Generator().manual_seed(key)).to(tokens.device)*active[..., None]
    u = (noise/mean_square(noise, active[..., None]).sqrt()[:, None, None]*.025).requires_grad_()
    e = torch.zeros((*tokens.shape, 64), device=tokens.device, requires_grad=True)
    initial = u.detach().clone()
    optimizer = torch.optim.Adam([dict(params=[u], lr=BASE_U_LR*coefficients[0]), dict(params=[e], lr=BASE_E_LR*coefficients[1])])
    constraint = JSConstraint(reference, active, js_limit, 1.)
    limit = tokens.shape[1]*percent//100
    best = raw = dict(hard_score=baseline, step=0, soft_js=0., changed_codons=0,
                      cds_dna=''.join(CODONS[t] for t in tokens[0].tolist()), original_fallback=True)
    trace, feasible, before = [], 0, scorer.forward_sequences
    started = time.monotonic()
    def probabilities():
        decoder = model.decode(mu+sigma*u, tokens)
        p = design_probabilities(decoder+e*support, tokens)
        terms = regularizers(p, design_probabilities(decoder, tokens), u, e, active, support)
        return p, sum(REG[k]*v.mean() for k, v in terms.items())
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        p, penalty = probabilities()
        hard = project_budget(p, tokens, limit)
        score = scorer(hard+p-p.detach(), tokens).mean()
        if step == 1:
            gradient = torch.autograd.grad(score, u, retain_graph=True)[0]
            score_gradient = gradient_summary(gradient[active])
        loss = -score+penalty+constraint.penalty(p).mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([u, e], 1., error_if_nonfinite=True)
        previous = u.detach().clone()
        optimizer.step()
        with torch.no_grad():
            u.mul_(active[..., None])
            u.mul_((8./mean_square(u, active[..., None]).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
            e.copy_(center_direct(e, support))
            e.mul_((16*1.251784384/mean_square(e, support).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
            p, _ = probabilities()
            hard = project_budget(p, tokens, limit)
            value, js = float(scorer(hard, tokens)), float(mutable_js(p, reference, active))
            constraint.update(p)
            selected = hard.argmax(-1)
            candidate = dict(hard_score=value, step=step, soft_js=js, changed_codons=int((selected!=tokens).sum()),
                             cds_dna=''.join(CODONS[t] for t in selected[0].tolist()))
            if value > raw['hard_score']:
                raw = candidate
            if js <= js_limit+1e-7:
                feasible += 1
                if value > best['hard_score']:
                    best = candidate
            trace.append(dict(step=step, score=value, js=js, gradient_norm=float(norm),
                              delta_z_rms=float(mean_square(sigma*(u-previous), active[..., None]).sqrt())))
    assert scorer.forward_sequences-before == 2*steps
    assert best['changed_codons'] <= limit and best['soft_js'] <= js_limit+1e-7
    return dict(sequence_id=identity, seed=seed, budget_percent=percent, coefficients=list(coefficients), js_limit=js_limit,
                baseline=baseline, best=best, best_budget_only=raw, gain=best['hard_score']-baseline,
                filter_gap=raw['hard_score']-best['hard_score'], feasible_fraction=feasible/steps,
                encoder=encoder_row(mu, logvar, tokens), score_gradient=score_gradient,
                displacement=gradient_summary((u.detach()-initial)[active]), trace=trace,
                scorer_queries=2*steps, setup_queries=1, seconds=time.monotonic()-started)


def native(scorer, tokens, identity, seed, percent, steps=80):
    torch.manual_seed(seed)
    active, support = masks(tokens)
    n, limit = tokens.shape[1], tokens.shape[1]*percent//100
    generator, order = native_generator(n, tokens.device)
    mask = torch.zeros((1, n, 65), device=tokens.device)
    mask[:, :, order] = torch.where(active[..., None], support, F.one_hot(tokens, 64).bool()).float()
    mask[:, 0] = 0
    mask[:, 0, order[tokens[0, 0]]] = 1
    optimizer = torch.optim.AdamW(generator.parameters(), lr=.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 70, 85])
    with torch.no_grad():
        baseline = float(scorer(F.one_hot(tokens, 64).float(), tokens))
    best = dict(step=0, hard_score=baseline, changed_codons=0, cds_dna=''.join(CODONS[t] for t in tokens[0].tolist()))
    before = scorer.forward_sequences
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        p = generator.generate(mask).transpose(1, 2)[:, :, order]
        hard = project_budget(p, tokens, limit)
        score = scorer(hard+p-p.detach(), tokens).mean()
        loss = (100-.2*score.expm1()).abs()/100
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            p = generator.generate(mask).transpose(1, 2)[:, :, order]
            hard = project_budget(p, tokens, limit)
            value = float(scorer(hard, tokens))
            if value > best['hard_score']:
                selected = hard.argmax(-1)
                best = dict(step=step, hard_score=value, changed_codons=int((selected!=tokens).sum()),
                            cds_dna=''.join(CODONS[t] for t in selected[0].tolist()))
    assert scorer.forward_sequences-before == 2*steps
    return dict(sequence_id=identity, seed=seed, budget_percent=percent, baseline=baseline, best=best,
                gain=best['hard_score']-baseline, scorer_queries=2*steps, setup_queries=1)
