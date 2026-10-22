"""B0-B5 paired, query-matched frozen-model dual-space ablation."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .diagnose_mask_plateau import choose
from .latent_optimization import (TranslationObjective, design_probabilities, snapshot,
                                  state_hash, write_rows)
from .model import PAD, SYNONYMS, CodonVMAE, collate_records, mutable_positions
from .train import codon_frequency, load_data, sha256

ARMS = {'B0': 'Original CDS', 'B1': 'Latent-only', 'B2': 'Direct-only',
        'B3': 'Latent + Direct', 'B4': 'Latent -> Direct',
        'B5': 'Latent + Direct + Regularization'}


def masks(tokens):
    active = mutable_positions(tokens)
    support = SYNONYMS.to(tokens.device)[tokens] & active[..., None]
    return active, support


def mean_square(value, support):
    weights = support.expand_as(value)
    return (value.square() * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)


def center_direct(value, support):
    masked = value * support
    return (masked - masked.sum(-1, keepdim=True) / support.sum(-1, keepdim=True).clamp_min(1)) * support


@torch.no_grad()
def project_joint(latent, direct, active, support, component_radius=None, frozen_latent=False):
    latent.mul_(active[..., None])
    direct.copy_(center_direct(direct, support))
    if component_radius is not None:
        if not frozen_latent:
            latent.mul_((component_radius / mean_square(latent, active[..., None]).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
        direct.mul_((component_radius / mean_square(direct, support).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
        return
    energy = mean_square(latent, active[..., None]) + mean_square(direct, support)
    factor = energy.clamp_min(1e-12).rsqrt().clamp_max(1)
    latent.mul_(factor[:, None, None])
    direct.mul_(factor[:, None, None])


def regularizers(probabilities, decoder_probabilities, latent, direct, active, support):
    kl = (probabilities * (probabilities.clamp_min(1e-12).log()
                          - decoder_probabilities.clamp_min(1e-12).log())).sum(-1)
    return dict(latent_l2=mean_square(latent, active[..., None]),
                direct_l2=mean_square(direct, support),
                decoder_kl=(kl * active).sum(-1) / active.sum(-1))


def phase(arm, step, steps):
    if arm in ('B7', 'B8'):
        return 'direct' if step <= steps // 2 else 'latent'
    if arm in ('B4', 'B6'):
        return 'latent' if step <= steps // 2 else 'direct'
    return 'latent' if arm == 'B1' else 'direct' if arm == 'B2' else 'joint'


def hard_only_snapshot(probabilities, tokens, scorer, counts, reference):
    # Reallocate optional soft diagnostics to the handoff without losing any hard candidates.
    cached = []
    def hard_scorer(p, t):
        if not cached: cached.append(scorer(p, t))
        return cached[0]
    rows = snapshot(probabilities, tokens, hard_scorer, counts, reference)
    for row in rows:
        row['soft_score'] = None
        row['soft_hard_gap'] = None
    return rows


def project_to_latent(model, target, tokens, mu, sigma, latent, config):
    """Fit a detached distribution without consulting the translation scorer."""
    settings = config['handoff']
    active, _ = masks(tokens)
    temperature = settings['temperature']
    if temperature != 1.:
        raise ValueError('Current frozen search uses temperature 1; handoff must match')
    target = target.detach()
    if settings['method'] == 'reencode':
        with torch.no_grad():
            hard = torch.where(tokens < PAD, target.argmax(-1), tokens)
            encoded, _ = model.encode(hard, tokens)
            proposal = (encoded - mu) / (sigma * config['latent_scale'])
            proposal *= active[..., None]
    else:
        assert settings['method'] == 'kl'
        proposal = latent.detach().clone()
    with torch.no_grad():
        proposal.mul_((16. / mean_square(proposal, active[..., None]).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
    proposal.requires_grad_(True)
    optimizer = torch.optim.Adam([proposal], lr=settings['lr'])
    def objective():
        probabilities = design_probabilities(model.decode(mu + sigma * config['latent_scale'] * proposal, tokens), tokens)
        kl = (target * (target.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())).sum(-1)
        kl = (kl * active).sum(-1) / active.sum(-1)
        return kl + settings['eta'] * mean_square(proposal, active[..., None]), kl
    best_value = torch.full((len(tokens),), float('inf'), device=tokens.device)
    best = proposal.detach().clone()
    trace = []
    updates = settings['steps'] if settings['method'] == 'kl' else 0
    for step in range(updates + 1):
        loss, kl = objective()
        if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite projection objective')
        improved = loss.detach() < best_value
        best[improved] = proposal.detach()[improved]
        best_value = torch.minimum(best_value, loss.detach())
        trace.append(dict(step=step, objective=loss.detach().tolist(), kl=kl.detach().tolist()))
        if step == updates: break
        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_([proposal], 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            proposal.mul_(active[..., None])
            proposal.mul_((16. / mean_square(proposal, active[..., None]).clamp_min(1e-12).sqrt()).clamp_max(1)[:, None, None])
    return best, trace


def feasible(row, initial, config):
    return (row['mutable_hamming'] <= config['max_mutable_hamming'] + 1e-7
            and abs(row['gc_fraction'] - initial['gc_fraction']) <= config['max_gc_change'] + 1e-7
            and row['js_from_initial'] <= config['max_js'] + 1e-7)


class CountedScorer(torch.nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.forward_sequences = 0

    def forward(self, probabilities, tokens):
        self.forward_sequences += len(tokens)
        return self.scorer(probabilities, tokens)


def calibrate(model, scorer, records, config):
    """Match the first Adam step's synonymous-centered output-logit RMS on train CDS."""
    measurements = []
    for begin in range(0, len(records), config['batch_size']):
        group = records[begin:begin + config['batch_size']]
        tokens = collate_records(group).to(next(model.parameters()).device)
        active, support = masks(tokens)
        with torch.no_grad():
            mu, logvar = model.encode(tokens, tokens)
            sigma = (.5 * logvar).exp()
            base = model.decode(mu, tokens)
        latent = torch.zeros_like(mu, requires_grad=True)
        direct = torch.zeros_like(base, requires_grad=True)
        raw = model.decode(mu + sigma * config['latent_scale'] * latent * active[..., None], tokens) + direct * support
        p = design_probabilities(raw, tokens)
        hard = F.one_hot(p.argmax(-1), 64).float() * (tokens < PAD)[..., None]
        loss = -scorer(hard + p - p.detach(), tokens).mean()
        gradients = torch.autograd.grad(loss, [latent, direct])
        proposals = [-config['lr'] * g / (g.abs() + 1e-8) for g in gradients]
        with torch.no_grad():
            z_raw = model.decode(mu + sigma * config['latent_scale'] * proposals[0] * active[..., None], tokens)
            z_shift = center_direct(z_raw - base, support)
            d_shift = center_direct(proposals[1], support)
            z_rms = mean_square(z_shift, support).sqrt()
            d_rms = mean_square(d_shift, support).sqrt()
            z_direction = proposals[0] / mean_square(proposals[0], active[..., None]).clamp_min(1e-12).sqrt()[:, None, None]
            z_full = model.decode(mu + sigma * config['latent_scale'] * z_direction * active[..., None], tokens)
            full_rms = mean_square(center_direct(z_full - base, support), support).sqrt()
        for identity, z_value, d_value, full_value in zip(group, z_rms.tolist(), d_rms.tolist(), full_rms.tolist()):
            if min(z_value, d_value) <= 1e-8 or not math.isfinite(z_value / d_value):
                raise ValueError('Degenerate calibration gradient')
            measurements.append(dict(sequence_id=identity['sequence_id'], latent_step_logit_rms=z_value,
                                     direct_unit_step_logit_rms=d_value, first_step_ratio=z_value / d_value,
                                     latent_full_radius_logit_rms=full_value))
    scale = float(np.median([r['latent_full_radius_logit_rms'] for r in measurements]))
    lr_factor = float(np.median([r['first_step_ratio'] for r in measurements])) / scale
    if not 1e-4 < scale < 100:
        raise ValueError('Implausible direct-space calibration scale')
    if not 1e-4 < lr_factor < 100: raise ValueError('Implausible calibrated learning-rate ratio')
    return dict(direct_scale=scale, direct_lr_factor=lr_factor, records=measurements, split='train',
                rule='match full-radius output-logit RMS; then match first Adam-step RMS via direct LR; train-only, local approximate calibration')


def optimize_batch(model, scorer, records, counts, config, output, run_start):
    if any(arm in ('B7', 'B8') for arm in config.get('run_arms', [])):
        if 'handoff' not in config:
            raise ValueError('B7/B8 require explicit projection or reencode handoff settings')
    tokens = collate_records(records).to(next(model.parameters()).device)
    active, support = masks(tokens)
    identities = [r['sequence_id'] for r in records]
    with torch.no_grad():
        mu, logvar = model.encode(tokens, tokens)
        sigma = (.5 * logvar).exp()
        base = model.decode(mu, tokens)
        reference = design_probabilities(base, tokens)
        originals = F.one_hot(tokens.clamp_max(63), 64).float() * (tokens < PAD)[..., None]
    initial = snapshot(reference, tokens, scorer, counts, reference)
    if not config.get('tuning', False):
        assert all(r['cds_dna'] == source['cds_dna'] for r, source in zip(initial, records)), 'B0 and decoded initial CDS must match'
    original_values = snapshot(originals, tokens, scorer, counts, reference)
    if not config.get('tuning', False):
        assert all(abs(a['hard_score'] - b['hard_score']) < 1e-6 for a, b in zip(initial, original_values))
    write_rows(output / 'initial.jsonl', [dict(sequence_id=i, n_codons=r['n_codons'], source_cds_dna=r['cds_dna'], **v)
                                       for i, r, v in zip(identities, records, initial)])
    results = [dict(sequence_id=i, arm='B0', initial=v, best=dict(step=0, feasible=True, **v),
                    final=dict(step=0, feasible=True, **v), best_hard_gain=0., final_hard_gain=0.,
                    feasible_steps=0, unique_candidates=1, elapsed_seconds=0., scorer_forward_sequences=0,
                    backward_sequence_evaluations=0, decoder_sequence_evaluations=0)
               for i, v in zip(identities, original_values)]
    write_rows(output / 'results.jsonl', results)
    shared_latent = None
    shared_start = None
    if 'random_start_seed' in config:
        assert set(config['run_arms']) <= {'B4', 'B5', 'B6', 'B7', 'B8'}
        assert config['random_start_rms'] > 0 and bool((active.sum(-1) > 0).all())
        shared_latent = torch.zeros_like(mu)
        for index, identity in enumerate(identities):
            key = '{}:{}'.format(config['random_start_seed'], identity)
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)
            noise = torch.randn(mu[index].shape, generator=torch.Generator().manual_seed(seed)).to(mu.device)
            noise *= active[index, :, None]
            rms = (noise.square().sum() / (active[index].sum() * noise.shape[-1])).sqrt()
            shared_latent[index] = noise / rms.clamp_min(1e-12) * config['random_start_rms']
        with torch.no_grad():
            start_logits = model.decode(mu + sigma * config['latent_scale'] * shared_latent, tokens)
            shared_start = snapshot(design_probabilities(start_logits, tokens), tokens, scorer, counts, reference)
        write_rows(output / 'random_initial.jsonl', [dict(sequence_id=identity,
            latent_sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
            latent_l2=float(mean_square(shared_latent, active[..., None])[i]), direct_l2=0.,
            feasible=feasible(row, initial[i], config), **row)
            for i, (identity, value, row) in enumerate(zip(identities, shared_latent, shared_start))])
    for arm in config.get('run_arms', list(ARMS)[1:]):
        start = time.monotonic()
        score_start = scorer.forward_sequences
        latent = torch.zeros_like(mu, requires_grad=arm != 'B2')
        if shared_latent is not None:
            with torch.no_grad(): latent.copy_(shared_latent)
        direct = torch.zeros_like(base, requires_grad=arm != 'B1')
        dual_rate = arm in ('B3', 'B5', 'B6', 'B7', 'B8') or (arm == 'B4' and config.get('output_space_budget', False))
        lr = config['lr'] / math.sqrt(2) if dual_rate else config['lr']
        lr *= config.get('arm_lr_multipliers', {}).get(arm, config.get('lr_multiplier', 1.))
        opt_z = torch.optim.Adam([latent], lr=lr) if latent.requires_grad else None
        opt_d = torch.optim.Adam([direct], lr=lr * config['direct_lr_factor']) if direct.requires_grad else None
        best = [dict(step=0, feasible=True, latent_l2=0., direct_l2=0., decoder_kl=0., **r) for r in initial]
        if shared_start is not None:
            for i, row in enumerate(shared_start):
                if feasible(row, initial[i], config) and row['hard_score'] > best[i]['hard_score']:
                    best[i] = dict(step=0, feasible=True, latent_l2=config['random_start_rms']**2,
                                   direct_l2=0., decoder_kl=0., **row)
        unique = [{r['cds_dna']} for r in initial]
        best_distribution = reference.detach().clone()
        if shared_start is not None:
            for i, row in enumerate(shared_start):
                if best[i]['cds_dna'] == row['cds_dna'] and best[i]['step'] == 0 and row['hard_score'] > initial[i]['hard_score']:
                    best_distribution[i] = design_probabilities(start_logits, tokens)[i].detach()
        diagnostics = None
        if config.get('length_diagnostics', False):
            from .length_diagnostics import SearchDiagnostics
            diagnostics = SearchDiagnostics(output, records, tokens, active, support, sigma, config, initial, shared_start)
        js_constraint = None
        if config.get('js_constraint'):
            if arm != 'B5': raise ValueError('JS optimization constraint is currently validated for B5 only')
            from .js_constraint import JSConstraint
            js_constraint = JSConstraint(reference, active, config['max_js'], config['js_constraint']['rho'])
        handoff_forwards = 0
        projection_updates = 0
        projection_seconds = 0.
        omitted_soft_queries = 0
        accepted = [0] * len(tokens)
        decoder_evaluations = 0
        frozen_stage_base = base
        if arm in ('B7', 'B8') and shared_latent is not None:
            with torch.no_grad():
                frozen_stage_base = model.decode(mu + sigma * config['latent_scale'] * latent * active[..., None], tokens)
            decoder_evaluations += len(tokens)
        last_phase = None
        for step in range(1, config['steps'] + 1):
            if time.monotonic() - run_start > config['timeout_seconds']:
                raise TimeoutError('Dual-space experiment exceeded declared timeout')
            current_phase = phase(arm, step, config['steps'])
            update_z = current_phase in ('latent', 'joint')
            update_d = current_phase in ('direct', 'joint')
            if arm in ('B7', 'B8') and current_phase == 'latent' and last_phase == 'direct':
                before = [dict(row) for row in best]
                if tokens.is_cuda: torch.cuda.synchronize(tokens.device)
                projection_start = time.monotonic()
                projected, trace = project_to_latent(model, best_distribution, tokens, mu, sigma, latent, config)
                with torch.no_grad():
                    latent.copy_(projected)
                    direct.zero_()
                    handoff_p = design_probabilities(model.decode(mu + sigma * config['latent_scale'] * latent, tokens), tokens)
                    after = snapshot(handoff_p, tokens, scorer, counts, reference)
                    handoff_kl = (best_distribution * (best_distribution.clamp_min(1e-12).log() - handoff_p.clamp_min(1e-12).log())).sum(-1)
                    handoff_kl = (handoff_kl * active).sum(-1) / active.sum(-1)
                    handoff_rz = mean_square(latent, active[..., None])
                if tokens.is_cuda: torch.cuda.synchronize(tokens.device)
                projection_seconds = time.monotonic() - projection_start
                handoff_forwards += 2 * len(tokens)
                projection_updates = config['handoff']['steps'] if config['handoff']['method'] == 'kl' else 0
                decoder_evaluations += (projection_updates + 2) * len(tokens)
                opt_z = torch.optim.Adam([latent], lr=lr)
                for i, row in enumerate(after):
                    legal = feasible(row, initial[i], config)
                    if legal and row['hard_score'] > best[i]['hard_score']:
                        best[i] = dict(step=step-1, phase='projection', feasible=True,
                                       latent_l2=float(handoff_rz[i]), direct_l2=0., decoder_kl=0., **row)
                    write_rows(output / 'handoff.jsonl', [dict(sequence_id=identities[i], arm=arm,
                        method=config['handoff']['method'], before=before[i], after=dict(feasible=legal, **row),
                        delta_projection=row['hard_score']-before[i]['hard_score'], direct_l2_after=0.,
                        projection_kl=float(handoff_kl[i]), latent_l2_after=float(handoff_rz[i]),
                        elapsed_seconds=projection_seconds / len(tokens), scorer_forward_sequences=2,
                        decoder_forward_evaluations=projection_updates+2, decoder_backward_evaluations=projection_updates,
                        encoder_forward_evaluations=int(config['handoff']['method']=='reencode'),
                        trace=[dict(step=t['step'], objective=t['objective'][i], kl=t['kl'][i]) for t in trace])])
            if arm in ('B4', 'B6') and current_phase == 'direct' and last_phase == 'latent':
                with torch.no_grad():
                    frozen_stage_base = model.decode(mu + sigma * config['latent_scale'] * latent * active[..., None], tokens)
                decoder_evaluations += len(tokens)
            if opt_z: opt_z.zero_grad(set_to_none=True)
            if opt_d: opt_d.zero_grad(set_to_none=True)
            if update_z:
                decoder_logits = model.decode(mu + sigma * config['latent_scale'] * latent * active[..., None], tokens)
                decoder_evaluations += len(tokens)
            else:
                decoder_logits = frozen_stage_base
            delta = config['direct_scale'] * direct * support
            p = design_probabilities(decoder_logits + delta, tokens)
            decoder_p = design_probabilities(decoder_logits, tokens)
            terms = regularizers(p, decoder_p, latent, direct, active, support)
            hard = F.one_hot(p.argmax(-1), 64).float() * (tokens < PAD)[..., None]
            loss = -scorer(hard + p - p.detach(), tokens).mean()
            if arm in ('B5', 'B6', 'B8'):
                keys = (['direct_l2', 'decoder_kl'] if update_d else ['latent_l2']) if arm == 'B8' else terms.keys()
                loss = loss + sum(config['regularization'][key] * terms[key].mean() for key in keys)
            if js_constraint: loss = loss + js_constraint.penalty(p).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
            if diagnostics: diagnostics.before_update(latent, direct, decoder_logits)
            loss.backward()
            if diagnostics: diagnostics.after_backward(latent, direct)
            parameters = ([latent] if update_z else []) + ([direct] if update_d else [])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            if update_z: opt_z.step()
            if update_d: opt_d.step()
            fixed_direct = direct.detach().clone() if arm in ('B7', 'B8') and not update_d else None
            project_joint(latent, direct, active, support,
                          component_radius=16. if config.get('output_space_budget', False) else math.sqrt(.5) if arm in ('B3', 'B4', 'B5') else None,
                          frozen_latent=arm in ('B4', 'B6') and current_phase == 'direct')
            if fixed_direct is not None:
                with torch.no_grad(): direct.copy_(fixed_direct)
            with torch.no_grad():
                if update_z:
                    decoder_logits = model.decode(mu + sigma * config['latent_scale'] * latent * active[..., None], tokens)
                    decoder_evaluations += len(tokens)
                p = design_probabilities(decoder_logits + config['direct_scale'] * direct * support, tokens)
                decoder_p = design_probabilities(decoder_logits, tokens)
                terms = regularizers(p, decoder_p, latent, direct, active, support)
                if arm in ('B7', 'B8') and step in (config['steps']//2, config['steps']):
                    values = hard_only_snapshot(p, tokens, scorer, counts, reference)
                    omitted_soft_queries += len(tokens)
                else:
                    values = snapshot(p, tokens, scorer, counts, reference)
                joint_energy = mean_square(latent, active[..., None]) + mean_square(direct, support)
                assert float(joint_energy.max()) < (512.0001 if config.get('output_space_budget', False) else 1.00001)
                if js_constraint: js_constraint.update(p)
            rows = []
            for index, value in enumerate(values):
                value.update(step=step, feasible=feasible(value, initial[index], config),
                             **{key: float(term[index]) for key, term in terms.items()})
                if shared_latent is not None:
                    value.update(R_z=value['latent_l2'], R_delta=value['direct_l2'],
                                 actual_residual_rms=config['direct_scale'] * math.sqrt(value['direct_l2']))
                if js_constraint: value.update(js_constraint.metrics(index))
                accepted[index] += int(value['feasible'])
                unique[index].add(value['cds_dna'])
                if value['feasible'] and value['hard_score'] > best[index]['hard_score']:
                    best[index] = dict(value)
                    best_distribution[index] = p[index].detach()
                rows.append(dict(sequence_id=identities[index], arm=arm, phase=current_phase,
                                 grad_norm=grad_norm, joint_parameter_rms=float(joint_energy[index].sqrt()),
                                 **{k: v for k, v in value.items() if k != 'cds_dna'}))
            write_rows(output / 'steps.jsonl', rows)
            if diagnostics: diagnostics.record(step, latent, direct, decoder_logits, p, values[0])
            last_phase = current_phase
            if step % 20 == 0:
                print(json.dumps(dict(event='progress', arm=arm, step=step, records=len(tokens),
                                      tuning=config.get('tuning', False), lr_multiplier=config.get('arm_lr_multipliers', {}).get(arm, config.get('lr_multiplier', 1.)),
                                      best_gains=[b['hard_score'] - v['hard_score'] for b, v in zip(best, initial)])), flush=True)
        assert scorer.forward_sequences - score_start == len(tokens) * config['steps'] * 3 + handoff_forwards - omitted_soft_queries
        if arm in ('B7', 'B8'): assert handoff_forwards == omitted_soft_queries
        if diagnostics: diagnostics.finish(best)
        elapsed = time.monotonic() - start
        for index, identity in enumerate(identities):
            results.append(dict(sequence_id=identity, arm=arm, initial=initial[index], best=best[index], final=values[index],
                best_hard_gain=best[index]['hard_score'] - initial[index]['hard_score'],
                final_hard_gain=values[index]['hard_score'] - initial[index]['hard_score'],
                feasible_steps=accepted[index], unique_candidates=len(unique[index]),
                elapsed_seconds=elapsed / len(tokens), scorer_forward_sequences=3 * config['steps'] + (handoff_forwards-omitted_soft_queries) // len(tokens),
                projection_elapsed_seconds=projection_seconds / len(tokens),
                projection_backward_evaluations=projection_updates,
                projection_encoder_evaluations=int(arm in ('B7', 'B8') and config['handoff']['method'] == 'reencode'),
                backward_sequence_evaluations=config['steps'],
                decoder_sequence_evaluations=decoder_evaluations // len(tokens)))
        write_rows(output / 'results.jsonl', results[-len(tokens):])
    return results


def tune_methods(model, scorer, records, counts, config, output, start):
    tuning_root = output / 'train_tuning'
    tuning_root.mkdir()
    results = {}
    for multiplier in [.25, .5, 1.]:
        directory = tuning_root / ('lr_' + str(multiplier))
        directory.mkdir()
        settings = dict(config, lr_multiplier=multiplier, tuning=True)
        rows = []
        for begin in range(0, len(records), config['batch_size']):
            rows += optimize_batch(model, scorer, records[begin:begin + config['batch_size']], counts, settings, directory, start)
        results[str(multiplier)] = {arm: float(np.mean([r['best_hard_gain'] for r in rows if r['arm'] == arm])) for arm in list(ARMS)[1:]}
    selected = {arm: float(max(results, key=lambda key: (results[key][arm], -float(key)))) for arm in list(ARMS)[1:]}
    independent_selection = dict(selected)
    selected['B4'] = selected['B3']
    selected['B5'] = selected['B3']
    receipt = dict(grid=[.25, .5, 1.], selected=selected, mean_best_feasible_gains=results,
                   training_ids=[r['sequence_id'] for r in records], steps_per_trial=80,
                   scorer_forward_sequences_per_arm=8*3*80*3,
                   backward_sequence_evaluations_per_arm=8*3*80,
                   criterion='mean best feasible hard-score gain from shared decoded train starting CDS; ties choose smaller multiplier',
                   evaluation_data_used=False)
    receipt['independent_grid_maxima'] = independent_selection
    receipt['shared_dual_rate_rule'] = 'B4 and B5 use B3 selected LR to isolate scheduling and regularization; all train trials retained'
    (output / 'tuning.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(event='train_tuning_complete', **receipt)), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--scorer', default='score_model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output-space-budget', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    config = dict(steps=80, batch_size=4, lr=.1, latent_scale=.5,
                  max_mutable_hamming=.20, max_gc_change=.05, max_js=.05,
                  regularization=dict(latent_l2=.1, direct_l2=.1, decoder_kl=.1),
                  timeout_seconds=5400, device=args.device, torch_version=torch.__version__, arms=ARMS,
                  checkpoint_sha256=sha256(args.checkpoint), scorer_sha256=sha256(Path(args.scorer) / 'best_model.p'),
                  dataset_manifest_sha256=sha256(Path(args.data) / 'manifest.json'),
                  tokenizer_sha256=sha256(Path(args.data) / 'tokenizer.json'),
                  selection='highest feasible hard score including initial; same rule for B1-B5',
                  feasibility='post-step candidate filter; optimization trajectory may be infeasible',
                  phase_split='B4: 40 latent updates, then freeze z and 40 direct updates; retain last phase endpoint',
                  regularizer_scope='B5 only: anchor z to original posterior; residual L2; KL(p_final || p_decoder(z))',
                  parameter_radius='single space energy <=1; dual/staged each component energy <=0.5; total <=1',
                  fairness='matched scorer queries/backward counts, train-calibrated coordinate scales, shared joint radius and output constraints; FLOPs differ')
    config['output_space_budget'] = args.output_space_budget
    if args.output_space_budget:
        config['parameter_radius'] = 'per-component RMS<=16 numerical guard only; common feasibility limits are in output space'
        config['fairness'] = 'shared output constraints, equal 3-trial train-only LR selection per B1-B5, matched evaluation scorer/backward counts; FLOPs differ'
        config['revision'] = 'v2: v1 Direct-only was blocked by parameter radius; preserve v1, relax coordinate guards without changing output feasibility'
    (output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    scorer = CountedScorer(TranslationObjective(args.scorer)).eval().requires_grad_(False).to(args.device)
    frozen_hashes = (state_hash(model), state_hash(scorer))
    splits, _ = load_data(args.data, allow_development=True)
    evaluation = choose(splits['val'], 16, short=True)
    calibration_records = choose(splits['train'], 8, short=True)
    assert set(r['sequence_id'] for r in evaluation).isdisjoint(r['sequence_id'] for r in calibration_records)
    calibration = calibrate(model, scorer, calibration_records, config)
    config['direct_scale'] = calibration['direct_scale']
    config['direct_lr_factor'] = calibration['direct_lr_factor']
    config['evaluation_ids'] = [r['sequence_id'] for r in evaluation]
    config['calibration_ids'] = [r['sequence_id'] for r in calibration_records]
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n', encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    counts = codon_frequency(splits['train']).to(args.device)
    if args.output_space_budget and not args.preflight_only:
        print(json.dumps(dict(event='train_tuning_start', grid=[.25, .5, 1.], records=8, steps=80)), flush=True)
        config['arm_lr_multipliers'] = tune_methods(model, scorer, calibration_records, counts, config, output, start)
    if args.preflight_only:
        config = dict(config, steps=2)
        evaluation = evaluation[:1]
        config['evaluation_ids'] = [r['sequence_id'] for r in evaluation]
    config['preflight'] = args.preflight_only
    (output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(event='start', records=len(evaluation), steps=config['steps'],
                          direct_scale=config['direct_scale'], preflight=args.preflight_only)), flush=True)
    results = []
    for begin in range(0, len(evaluation), config['batch_size']):
        results += optimize_batch(model, scorer, evaluation[begin:begin + config['batch_size']], counts, config, output, start)
    assert frozen_hashes == (state_hash(model), state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters()) + list(scorer.parameters()))
    assert config['checkpoint_sha256'] == sha256(args.checkpoint)
    assert config['scorer_sha256'] == sha256(Path(args.scorer) / 'best_model.p')
    summary = {}
    for arm in ARMS:
        rows = [r for r in results if r['arm'] == arm]
        summary[arm] = dict(name=ARMS[arm], records=len(rows),
            mean_best_gain=float(np.mean([r['best_hard_gain'] for r in rows])),
            mean_final_gain=float(np.mean([r['final_hard_gain'] for r in rows])),
            improved=sum(r['best_hard_gain'] > 1e-5 for r in rows),
            final_feasible=sum(r['final']['feasible'] for r in rows),
            mean_hamming=float(np.mean([r['best']['mutable_hamming'] for r in rows])),
            mean_gc_change=float(np.mean([r['best']['gc_fraction'] - r['initial']['gc_fraction'] for r in rows])),
            feasible_candidate_fraction=float(np.mean([r['feasible_steps'] / config['steps'] for r in rows])) if arm != 'B0' else 1.,
            scorer_forward_sequences=sum(r['scorer_forward_sequences'] for r in rows),
            backward_sequence_evaluations=sum(r['backward_sequence_evaluations'] for r in rows),
            decoder_sequence_evaluations=sum(r['decoder_sequence_evaluations'] for r in rows),
            elapsed_seconds=sum(r['elapsed_seconds'] for r in rows))
    result = dict(arms=summary, model_and_scorer_unchanged=True, elapsed_seconds=time.monotonic() - start,
                  records=len(evaluation), steps=config['steps'], preflight=args.preflight_only,
                  scorer_forward_sequences_including_shared_setup=scorer.forward_sequences)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
