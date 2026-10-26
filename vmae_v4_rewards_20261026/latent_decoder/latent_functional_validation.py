"""Protein-fixed predictive interventions and conditional signal/noise geometry."""
import hashlib
import statistics

import torch

from .evaluate_prior import shuffle_latents
from .latent_collapse_cpu import GROUPS, Pool, ce_metrics
from .latent_correlated_projection import ProjectedVMAE
from .latent_information_probe import recode_visible
from .latent_rank_cpu import within_geometry
from .latent_recovery_gpu import batch_mask
from .latent_signal_geometry import pool_signal_geometry
from .length_diagnostics import length_group
from .model import MASK, PAD, mutable_positions


@torch.no_grad()
def functional_validation(model, records, counts, repeats=2):
    assert repeats > 0
    device = next(model.parameters()).device
    modes = ('clean', 'masked', 'hidden')
    pools = {(g, m): Pool(model.latent_dim, covariance=True) for g in GROUPS for m in modes}
    noise = {key: torch.zeros(model.latent_dim, model.latent_dim, dtype=torch.float64) for key in pools}
    rows = []
    projected = isinstance(model, ProjectedVMAE)
    def distribution(inputs, tokens):
        return model.distribution(inputs, tokens) if projected else model.encode(inputs, tokens)
    def add(group, mode, mu, lv, tokens, selected):
        key = group, mode
        pools[key].add(mu.cpu(), tokens.masked_fill(~selected, PAD).cpu())
        variances = lv[selected].double().exp().sum(0).cpu()
        if projected:
            basis = model.noise_projection.cpu().double()
            noise[key] += (basis.T*variances)@basis
        else:
            noise[key] += torch.diag(variances)
    for record in records:
        tokens, inputs, selected = batch_mask([record], 175, device)
        assert selected.any()
        group = length_group(record['n_codons']-1)
        valid = mutable_positions(tokens)
        mu, lv = distribution(inputs, tokens)
        clean_mu, clean_lv = distribution(tokens, tokens)
        add(group, 'clean', clean_mu, clean_lv, tokens, valid)
        add(group, 'masked', mu, lv, tokens, valid)
        add(group, 'hidden', mu, lv, tokens, selected)
        zero = torch.zeros_like(mu)
        def ce(z):
            return ce_metrics(model.decode(z, tokens), tokens, zero, zero, selected)['ce']
        baseline = ce(mu)
        shuffled = shuffle_latents(mu, tokens, selected, torch.Generator(device=device).manual_seed(9473))
        rows.append(dict(sequence_id=record['sequence_id'], group=group, kind='latent_zero',
            baseline_ce=baseline, delta_ce=ce(zero)-baseline, target_count=int(selected.sum())))
        rows.append(dict(sequence_id=record['sequence_id'], group=group, kind='latent_shuffle',
            baseline_ce=baseline, delta_ce=ce(shuffled)-baseline, target_count=int(selected.sum())))
        for kind in ('permutation', 'train_frequency'):
            for repeat in range(repeats):
                seed = int(hashlib.sha256(f'{record["sequence_id"]}:{kind}:{repeat}'.encode()).hexdigest()[:15], 16)
                changed = recode_visible(tokens.cpu(), selected.cpu(), kind, seed, counts)
                changed_inputs = changed.masked_fill(selected.cpu(), MASK).to(device)
                alternative, _ = distribution(changed_inputs, tokens)
                delta = alternative-mu
                projected_delta = delta@model.from_latent.weight.T
                rows.append(dict(sequence_id=record['sequence_id'], group=group, kind=kind, repeat=repeat,
                    baseline_ce=baseline, delta_ce=ce(alternative)-baseline, target_count=int(selected.sum()),
                    latent_response_energy_per_hidden_token=float(delta[selected].square().sum(-1).mean()),
                    decoder_response_energy_per_hidden_token=float(projected_delta[selected].square().sum(-1).mean()),
                    visible_change_fraction=float((changed!=tokens.cpu()).sum()/(valid&~selected).sum().clamp_min(1)),
                    same_complete_protein=True, hidden_targets_unchanged=True))
    summary, geometry = {}, {}
    for group in GROUPS:
        chosen = [r for r in rows if r['group'] == group]
        if not chosen:
            continue
        summary[group] = {}
        for kind in ('latent_zero', 'latent_shuffle', 'permutation', 'train_frequency'):
            subset = [r for r in chosen if r['kind'] == kind]
            summary[group][kind] = dict(mean_delta_ce=statistics.mean(r['delta_ce'] for r in subset),
                records=len({r['sequence_id'] for r in subset}))
        geometry[group] = {}
        for mode in modes:
            pool = pools[group, mode]
            geometry[group][mode] = dict(**pool.summary(), within_aa_geometry=within_geometry(pool),
                signal_noise=pool_signal_geometry(pool, noise[group, mode]/pool.n),
                average_noise_diagonal=(noise[group, mode].diag()/pool.n).tolist())
    return dict(rows=rows, by_length=summary, geometry=geometry,
        interpretation='Fixed hidden labels and complete protein for synonymous visible recoding. Zeroing can be out of distribution; response and SNR modes are not mutual-information bits or causal active-dimension counts.')
