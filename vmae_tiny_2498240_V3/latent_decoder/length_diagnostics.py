"""Read-only measurements for frozen B5 trajectories and encoder outputs."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .latent_optimization import design_probabilities, write_rows
from .model import AA_IDS, PAD, collate_records, mutable_positions


def length_group(n_aa):
    return 'short' if n_aa <= 300 else 'medium' if n_aa < 700 else 'long'


def select_records(records, per_group=8):
    selected = []
    used = set()
    for group in ('short', 'medium', 'long'):
        candidates = sorted((r for r in records if length_group(r['n_codons'] - 1) == group),
                            key=lambda r: hashlib.sha256(('length-diagnostic-20261021:' + r['sequence_id']).encode()).hexdigest())
        picked = []
        for record in candidates:
            if record['cluster_id'] not in used:
                picked.append(record)
                used.add(record['cluster_id'])
            if len(picked) == per_group:
                break
        if len(picked) != per_group:
            raise ValueError('Not enough distinct components in ' + group)
        selected.extend(picked)
    return selected


def variance_summary(variance):
    variance = variance.clamp_min(0)
    energy = variance.sum()
    return dict(variance_by_dim=variance.tolist(),
                active_dimensions={str(t): int((variance > t).sum()) for t in (.001, .01, .1)},
                variance_energy_participation=float(energy.square() / variance.square().sum().clamp_min(1e-30)))


def gradient_summary(gradient):
    energy = gradient.double().square().sum(0)
    total = energy.sum()
    if total == 0:
        return dict(energy_by_dim=energy.tolist(), dims_for_99pct_gradient_energy=0, energy_participation=0.)
    cumulative = energy.sort(descending=True).values.cumsum(0)
    return dict(energy_by_dim=energy.tolist(), dims_for_99pct_gradient_energy=int((cumulative < total * .99).sum()) + 1,
                energy_participation=float(total.square() / energy.square().sum()))


@torch.no_grad()
def encoder_row(mu, logvar, tokens):
    valid = tokens < PAD
    active = mutable_positions(tokens)
    values = mu[valid].double()
    sigma = (.5 * logvar[valid].double()).exp()
    return dict(mu_mean=float(values.mean()), mu_mean_abs=float(values.abs().mean()), mu_rms=float(values.square().mean().sqrt()),
                sigma_mean=float(sigma.mean()), sigma_rms=float(sigma.square().mean().sqrt()),
                mutable_sigma_mean=float((.5 * logvar[active].double()).exp().mean()),
                **variance_summary(values.var(0, correction=0)))


class EncoderPool:
    def __init__(self, dimensions):
        self.count = 0
        self.total = torch.zeros(dimensions, dtype=torch.float64)
        self.square = torch.zeros_like(self.total)
        self.aa_counts = torch.zeros(21, dtype=torch.float64)
        self.aa_total = torch.zeros(21, dimensions, dtype=torch.float64)
        self.aa_square = torch.zeros_like(self.aa_total)
        self.sequence_rows = []

    def add(self, mu, logvar, tokens):
        valid = tokens < PAD
        values = mu[valid].detach().double().cpu()
        labels = AA_IDS[tokens[valid].cpu()]
        self.count += len(values)
        self.total += values.sum(0)
        self.square += values.square().sum(0)
        self.aa_counts += torch.bincount(labels, minlength=21)
        self.aa_total.index_add_(0, labels, values)
        self.aa_square.index_add_(0, labels, values.square())
        self.sequence_rows.append(encoder_row(mu, logvar, tokens))

    def summary(self):
        variance = self.square / self.count - (self.total / self.count).square()
        within_aa = (self.aa_square - self.aa_total.square() / self.aa_counts.clamp_min(1)[:, None]).sum(0) / self.count
        return dict(sequences=len(self.sequence_rows), valid_tokens=self.count,
            sequence_mean={k: sum(r[k] for r in self.sequence_rows) / len(self.sequence_rows)
                           for k in ('mu_mean', 'mu_mean_abs', 'mu_rms', 'sigma_mean', 'sigma_rms', 'mutable_sigma_mean')},
            mean_per_sequence_active_dimensions=sum(r['active_dimensions']['0.01'] for r in self.sequence_rows) / len(self.sequence_rows),
            pooled=variance_summary(variance), within_amino_acid=variance_summary(within_aa))


def functional_probe(model, scorer, record, device):
    tokens = collate_records([record]).to(device)
    active = mutable_positions(tokens)
    with torch.no_grad():
        mu, logvar = model.encode(tokens, tokens)
    z = mu.detach().clone().requires_grad_()
    p = design_probabilities(model.decode(z, tokens), tokens)
    hard = F.one_hot(p.argmax(-1), 64).float()
    score = scorer(hard + p - p.detach(), tokens).mean()
    score_gradient = torch.autograd.grad(score, z, retain_graph=True)[0]
    nll = -p.gather(-1, tokens[..., None]).squeeze(-1).clamp_min(1e-30).log()[active].mean()
    reconstruction_gradient = torch.autograd.grad(nll, z)[0]
    row = dict(sequence_id=record['sequence_id'], group=length_group(record['n_codons']-1),
               score_gradient=gradient_summary(score_gradient[active]), reconstruction_gradient=gradient_summary(reconstruction_gradient[active]))
    with torch.no_grad():
        high_variance = mu[tokens < PAD].var(0, correction=0) > .01
        interventions = {}
        for name, channels in [('zero_high_variance_dims', high_variance), ('zero_low_variance_dims', ~high_variance)]:
            altered = mu.clone()
            altered[..., channels] = 0
            q = design_probabilities(model.decode(altered, tokens), tokens)
            interventions[name] = dict(channels=int(channels.sum()),
                mutable_hamming=float((q.argmax(-1)[active] != tokens[active]).float().mean()),
                original_codon_nll=float(-q.gather(-1, tokens[..., None]).squeeze(-1).clamp_min(1e-30).log()[active].mean()),
                kl_from_clean=float((p.detach() * (p.detach().clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1)[active].mean()))
        row.update(clean_codon_nll=float(nll), per_sequence_high_variance_dims=int(high_variance.sum()), interventions=interventions)
    return row


class SearchDiagnostics:
    def __init__(self, output, records, tokens, active, support, sigma, config, initial, shared_start):
        if len(records) != 1 or config['run_arms'] != ['B5']:
            raise ValueError('Length diagnostics require single-CDS B5 runs')
        self.output = output
        self.identity = records[0]['sequence_id']
        self.group = length_group(records[0]['n_codons'] - 1)
        self.tokens, self.active, self.support, self.sigma, self.config = tokens, active, support, sigma, config
        self.initial = initial[0]
        self.best_raw = dict(step=0, **self.initial)
        if shared_start is not None and shared_start[0]['hard_score'] > self.best_raw['hard_score']:
            self.best_raw = dict(step=0, **shared_start[0])
        self.energy = torch.zeros(sigma.shape[-1], dtype=torch.float64)
        self.candidates = set()

    def before_update(self, latent, direct, decoder):
        self.old_latent, self.old_direct, self.old_decoder = (x.detach().clone() for x in (latent, direct, decoder))

    def after_backward(self, latent, direct):
        self.gradient = latent.grad.detach().clone()
        self.direct_gradient = direct.grad.detach().clone()

    @torch.no_grad()
    def record(self, step, latent, direct, decoder, probabilities, value):
        c = self.config
        active, support = self.active, self.support
        z_update = self.sigma * c['latent_scale'] * (latent - self.old_latent)
        z_displacement = self.sigma * c['latent_scale'] * latent
        decoded_shift = decoder - self.old_decoder
        decoded_shift = decoded_shift - (decoded_shift * support).sum(-1, keepdim=True) / support.sum(-1, keepdim=True).clamp_min(1)
        total_shift = (decoder + c['direct_scale'] * direct * support)
        original = total_shift.gather(-1, self.tokens[..., None]).squeeze(-1)
        alternatives = total_shift.masked_fill(~support, -torch.inf).scatter(-1, self.tokens[..., None], -torch.inf).max(-1).values
        margin = original - alternatives
        margin = margin[active]
        self.energy += self.gradient[active].double().square().sum(0).cpu()
        rms = lambda x: float(x.square().mean().sqrt())
        reasons = []
        if value['mutable_hamming'] > c['max_mutable_hamming'] + 1e-7: reasons.append('edit')
        if abs(value['gc_fraction'] - self.initial['gc_fraction']) > c['max_gc_change'] + 1e-7: reasons.append('gc')
        if value['js_from_initial'] > c['max_js'] + 1e-7: reasons.append('js')
        if value['hard_score'] > self.best_raw['hard_score']:
            self.best_raw = dict(value)
        self.candidates.add(value['cds_dna'])
        row = dict(sequence_id=self.identity, group=self.group, step=step,
            latent_gradient_rms=rms(self.gradient[active]), latent_gradient_norm=float(self.gradient.norm()),
            direct_gradient_rms=rms(self.direct_gradient[support]),
            gradient_clip_factor=min(1., 1. / (float(torch.cat([self.gradient.flatten(), self.direct_gradient.flatten()]).norm()) + 1e-6)),
            gradient_abs_le_1e_minus8_fraction=float((self.gradient[active].abs() <= 1e-8).float().mean()),
            standardized_update_rms=rms((latent-self.old_latent)[active]), actual_z_update_rms=rms(z_update[active]),
            actual_z_update_relative_mu_rms=rms(z_update[active]) / max(c['diagnostic_mu_rms'], 1e-30),
            actual_z_displacement_rms=rms(z_displacement[active]), decoder_logit_update_rms=rms(decoded_shift[support]),
            actual_direct_update_rms=rms((c['direct_scale']*(direct-self.old_direct))[support]),
            margin_mean=float(margin.mean()), positive_original_margin_fraction=float((margin > 0).float().mean()),
            hard_score=value['hard_score'], gain=value['hard_score']-self.initial['hard_score'], soft_score=value['soft_score'],
            mutable_hamming=value['mutable_hamming'], gc_change=value['gc_fraction']-self.initial['gc_fraction'],
            js=value['js_from_initial'], feasible=not reasons, rejected_by=reasons,
            **{k:value[k] for k in ('latent_l2','direct_l2','decoder_kl')}, cds_dna=value['cds_dna'])
        write_rows(self.output / 'length_search_steps.jsonl', [row])

    def finish(self, best):
        energy = self.energy
        total = energy.sum()
        count = int((energy.sort(descending=True).values.cumsum(0) < .99*total).sum()) + 1 if total > 0 else 0
        write_rows(self.output / 'length_search_summary.jsonl', [dict(sequence_id=self.identity, group=self.group,
            seed=self.config['random_start_seed'], baseline=self.initial['hard_score'],
            best_before_filter=self.best_raw, best_after_filter=best[0],
            filter_loss=self.best_raw['hard_score']-best[0]['hard_score'], unique_candidates=len(self.candidates),
            dims_for_99pct_total_loss_gradient_energy=count, total_loss_gradient_energy_by_dim=energy.tolist())])
