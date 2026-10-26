"""CPU-only stage tracing and bounded posterior-head counterfactual diagnostics."""
import argparse
import copy
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time

import torch
from torch.nn import functional as F

from .budget_js_extension import save_json
from .evaluate_prior import shuffle_latents
from .length_diagnostics import gradient_summary, length_group, variance_summary
from .model import AA_IDS, CodonVMAE, PAD, collate_records, corrupt, synonymous_logits, vae_loss
from .train import sha256
from .v2_data import pick

GROUPS = ('short', 'medium', 'long')
SEEDS = (173, 174)
ARMS = dict(joint=dict(beta=.001, decay=.01, reconstruction=True, sample=True),
    no_kl=dict(beta=0., decay=.01, reconstruction=True, sample=True),
    no_decay=dict(beta=.001, decay=0., reconstruction=True, sample=True),
    kl_only=dict(beta=.001, decay=0., reconstruction=False, sample=False),
    decay_only=dict(beta=0., decay=.01, reconstruction=False, sample=False),
    no_sampling=dict(beta=.001, decay=.01, reconstruction=True, sample=False))


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs)), flush=True)


def choose_records(data, per_group=4):
    manifest = json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['corpus_sha256']
    split = manifest['splits']['0']
    membership = {identity: panel for panel in ('train', 'val') for identity in split[panel]}
    metadata = []
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['sequence_id'] in membership:
                metadata.append(dict(sequence_id=r['sequence_id'], n_codons=r['n_codons'],
                    v2_group=manifest['assignments'][r['sequence_id']]['group'], panel=membership[r['sequence_id']]))
    chosen = {r['sequence_id']: r for panel in ('train', 'val')
              for r in pick([x for x in metadata if x['panel'] == panel], per_group, 'cpu-collapse-20261023')}
    records = []
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['sequence_id'] in chosen:
                records.append(dict(r, **{k:v for k,v in chosen[r['sequence_id']].items() if k != 'sequence_id'}))
    assert len(records) == per_group*6
    assert not {r['v2_group'] for r in records if r['panel']=='train'} & {
        r['v2_group'] for r in records if r['panel']=='val'}
    return records


def masked(tokens, identity, seed):
    number = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:15], 16)
    return corrupt(tokens, .15, torch.Generator().manual_seed(number))


class Pool:
    def __init__(self, dimensions, covariance=False):
        self.n = 0
        self.total = torch.zeros(dimensions, dtype=torch.float64)
        self.square = self.total.clone()
        self.aa_n = torch.zeros(21, dtype=torch.float64)
        self.aa_total = torch.zeros(21, dimensions, dtype=torch.float64)
        self.aa_square = self.aa_total.clone()
        self.cross = torch.zeros(dimensions, dimensions, dtype=torch.float64) if covariance else None

    def add(self, values, tokens):
        valid = tokens < PAD
        x, aa = values[valid].detach().double(), AA_IDS[tokens[valid]]
        self.n += len(x)
        self.total += x.sum(0)
        self.square += x.square().sum(0)
        self.aa_n += torch.bincount(aa, minlength=21)
        self.aa_total.index_add_(0, aa, x)
        self.aa_square.index_add_(0, aa, x.square())
        if self.cross is not None:
            self.cross += x.T@x

    def variance(self):
        return (self.square/self.n-(self.total/self.n).square()).clamp_min(0)

    def summary(self):
        var = self.variance()
        within = ((self.aa_square-self.aa_total.square()/self.aa_n.clamp_min(1)[:, None]).sum(0)/self.n).clamp_min(0)
        result = dict(channels=len(var), tokens=self.n, rms=float((self.square.sum()/self.n/len(var)).sqrt()),
                      pooled=variance_summary(var), within_aa=variance_summary(within))
        if self.cross is not None:
            cov = self.cross/self.n-torch.outer(self.total/self.n, self.total/self.n)
            scale = var.sqrt().clamp_min(1e-12)
            corr = cov/scale[:, None]/scale[None, :]
            corr[var<=1e-12, :] = 0
            corr[:, var<=1e-12] = 0
            eigen = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
            result.update(covariance_participation=float(var.sum().square()/cov.square().sum().clamp_min(1e-30)),
                correlation_participation=float(corr.diag().sum().square()/corr.square().sum().clamp_min(1e-30)),
                covariance_dimensions_95pct=int((eigen.cumsum(0) < .95*eigen.sum()).sum())+int(eigen.sum()>0))
        return result


def trace_cache(model, records, output):
    pools, cache, captures = {}, [], {}
    handles = []
    def capture(name):
        def hook(module, args, result):
            captures[name] = result.detach()
        return hook
    def before(name):
        def hook(module, args):
            captures[name] = args[0].detach()
        return hook
    handles.append(model.encoder.register_forward_pre_hook(before('encoder_input')))
    for name, module in [('encoder_norm', model.encoder.norm), ('mu_head', model.mu),
                         ('decoder_projection', model.from_latent), ('decoder_norm', model.decoder.norm)]:
        handles.append(module.register_forward_hook(capture(name)))
    handles.append(model.decoder.register_forward_pre_hook(before('decoder_input_with_aa')))
    for side in ('encoder', 'decoder'):
        for i, layer in enumerate(getattr(model, side).layers):
            handles.append(layer.register_forward_hook(capture(f'{side}_block_{i+1}')))
    try:
        for i, record in enumerate(records):
            tokens = collate_records([record])
            group = length_group(record['n_codons']-1)
            for mode, seed in [('clean', None), ('mask173', 173), ('mask174', 174)]:
                inputs, selected = (tokens, torch.zeros_like(tokens, dtype=torch.bool)) if seed is None else masked(tokens, record['sequence_id'], seed)
                captures.clear()
                with torch.no_grad():
                    logits, mu, logvar = model(inputs, tokens, sample=False)
                    captures['mu'] = mu
                    captures['posterior_sigma'] = (.5*logvar).exp()
                    rng = torch.Generator().manual_seed(20261023+(seed or 0))
                    captures['sampled_z'] = mu+torch.randn(mu.shape, generator=rng)*(.5*logvar).exp()
                    for stage, x in captures.items():
                        key = '/'.join((record['panel'], group, mode, stage))
                        if key not in pools:
                            pools[key] = Pool(x.shape[-1], covariance=stage=='mu')
                        pools[key].add(x, tokens)
                cache.append(dict(record=record, tokens=tokens, selected=selected, mode=mode,
                    h=captures['encoder_norm'].clone(), mu=mu.clone(), logvar=logvar.clone(),
                    baseline=ce_metrics(logits, tokens, mu, logvar, selected)))
            emit('trace_sequence', index=i+1, total=len(records), sequence_id=record['sequence_id'], group=group)
    finally:
        for handle in handles:
            handle.remove()
    save_json(output/'stage_trace.json', {key:p.summary() for key,p in pools.items()})
    return cache, pools


def ce_metrics(logits, tokens, mu, logvar, selected):
    ce = F.cross_entropy(synonymous_logits(logits, tokens).transpose(1, 2), tokens.clamp_max(63), reduction='none')
    valid = tokens < PAD
    kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1)[valid].mean()
    return dict(full_ce=float(ce[valid].mean()), ce=float(ce[selected].mean()) if selected.any() else None,
                kl=float(kl), fixed_loss=float(ce[valid].mean()+ce[selected].mean()+.001*kl) if selected.any() else None)


def gradient_probe(model, item):
    mu, logvar, h = item['mu'], item['logvar'], item['h']
    tokens, selected = item['tokens'], item['selected']
    valid = tokens < PAD
    rng = torch.Generator().manual_seed(20261023)
    epsilon = torch.randn(mu.shape, generator=rng)
    z = (mu+epsilon*(.5*logvar).exp()).detach().requires_grad_()
    rec, _ = vae_loss(model.decode(z, tokens), tokens, mu, logvar, selected, beta=0.)
    g_rec, = torch.autograd.grad(rec, z)
    g_kl = .001*mu/valid.sum()
    centered = mu[valid]-mu[valid].mean(0)
    radial_rec = (centered*g_rec[valid]).sum(0)
    radial_kl = (centered*g_kl[valid]).sum(0)
    low = mu[valid].var(0, unbiased=False) <= .01
    gw_rec = g_rec[valid].T@h[valid]
    gw_kl = g_kl[valid].T@h[valid]
    return dict(sequence_id=item['record']['sequence_id'], panel=item['record']['panel'], mode=item['mode'],
        group=length_group(item['record']['n_codons']-1), low_variance_channels=int(low.sum()),
        kl_to_reconstruction_mu_grad_norm=float(g_kl.norm()/g_rec.norm().clamp_min(1e-30)),
        kl_to_reconstruction_head_grad_norm=float(gw_kl.norm()/gw_rec.norm().clamp_min(1e-30)),
        gradient_cosine=float(F.cosine_similarity(g_kl.flatten(), g_rec.flatten(), dim=0)),
        contraction_reconstruction=float(radial_rec.sum()), contraction_kl=float(radial_kl.sum()),
        low_dimensions_with_net_contracting_gradient=int(((radial_rec+radial_kl > 0)&low).sum()),
        reconstruction_gradient=gradient_summary(g_rec[valid]),
        contraction_by_dim=dict(reconstruction=radial_rec.tolist(), kl=radial_kl.tolist()))


@torch.no_grad()
def interventions(model, item, high):
    mu, tokens, selected = item['mu'], item['tokens'], item['selected']
    baseline = item['baseline']['ce']
    if baseline is None:
        return None
    rng = torch.Generator().manual_seed(9473)
    zs = dict(zero=torch.zeros_like(mu), shuffled=shuffle_latents(mu, tokens, selected, rng),
              zero_low=mu.masked_fill((~high)[None, None, :], 0.), zero_high=mu.masked_fill(high[None, None, :], 0.))
    result = dict(sequence_id=item['record']['sequence_id'], group=length_group(item['record']['n_codons']-1),
                  mode=item['mode'], baseline_ce=baseline, high_channels=int(high.sum()))
    for name, z in zs.items():
        result[name+'_ce_delta'] = ce_metrics(model.decode(z, tokens), tokens, mu, item['logvar'], selected)['ce']-baseline
    # A coordinate change can inflate AU without changing the decoded mean or adding information.
    rescaled_projection = F.linear(mu*2., model.from_latent.weight/2., model.from_latent.bias)
    projection_error = (rescaled_projection-model.from_latent(mu)).abs().max()
    var = mu[0].var(0, unbiased=False)
    result.update(rescale2_au_before=int((var>.01).sum()), rescale2_au_after=int((4*var>.01).sum()),
                  rescale2_projection_max_error=float(projection_error))
    assert torch.allclose(rescaled_projection, model.from_latent(mu), atol=2e-6)
    return result


@torch.no_grad()
def head_state(model, cache, evaluate_ce=False):
    pools, ce = {}, {g: [] for g in GROUPS}
    for item in cache:
        r = item['record']
        if item['mode'] != 'mask173':
            continue
        mu = model.mu(item['h'])
        key = r['panel']+'/'+length_group(r['n_codons']-1)
        if key not in pools:
            pools[key] = Pool(model.latent_dim, covariance=False)
        pools[key].add(mu, item['tokens'])
        if evaluate_ce and r['panel']=='val':
            ce[length_group(r['n_codons']-1)].append(ce_metrics(model.decode(mu, item['tokens']), item['tokens'],
                                                       mu, item['logvar'], item['selected']))
    return dict(latent={key:p.summary() for key,p in pools.items()}, validation=ce if evaluate_ce else None)


def replay_heads(model, cache, output, steps):
    initial = copy.deepcopy(model.mu.state_dict())
    train = sorted([item for item in cache if item['record']['panel']=='train' and item['mode']=='mask173'],
                   key=lambda x:(x['record']['n_codons'],x['record']['sequence_id']))
    # Interleave lengths; every arm sees identical CDS, noise and masks in the same order.
    ordered = [x for triple in zip(*( [x for x in train if length_group(x['record']['n_codons']-1)==g] for g in GROUPS)) for x in triple]
    results = {}
    for arm, settings in ARMS.items():
        target = output/f'head_replay_{arm}.json'
        if target.exists():
            results[arm] = json.loads(target.read_text())
            continue
        model.mu.load_state_dict(initial)
        model.mu.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.mu.parameters(), lr=1e-4, weight_decay=settings['decay'])
        trajectory = [dict(step=0, **head_state(model, cache))]
        started = time.monotonic()
        for step in range(steps):
            item = ordered[step%len(ordered)]
            optimizer.zero_grad(set_to_none=True)
            mu = model.mu(item['h'])
            logvar, tokens, selected = item['logvar'], item['tokens'], item['selected']
            rng = torch.Generator().manual_seed(20261023+step)
            z = mu+torch.randn(mu.shape, generator=rng)*(.5*logvar).exp() if settings['sample'] else mu
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1)[tokens<PAD].mean()
            loss = settings['beta']*kl
            if settings['reconstruction']:
                rec, _ = vae_loss(model.decode(z, tokens), tokens, mu, logvar, selected, beta=0.)
                loss = loss+rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mu.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%len(ordered)==0 or step+1==steps:
                trajectory.append(dict(step=step+1, **head_state(model, cache)))
                emit('head_replay', arm=arm, step=step+1, steps=steps, loss=float(loss.detach()), seconds=time.monotonic()-started)
                save_json(output/'replay_progress.json', dict(arm=arm, step=step+1, steps=steps))
        result = dict(settings=settings, steps=steps, lr=1e-4, seconds=time.monotonic()-started,
            trajectory=trajectory, final=head_state(model, cache, evaluate_ce=True),
            scope='Only mu head updated; fixed encoder/decoder/logvar, dropout off, fresh AdamW. Local mechanism probe, not full-training replication.')
        save_json(target, result)
        results[arm] = result
        del optimizer
        model.mu.requires_grad_(False)
    model.mu.load_state_dict(initial)
    model.mu.requires_grad_(False)
    return results


def timeline(archive, output):
    rows = []
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if '/epoch_' not in member.name or not member.name.endswith('.json'):
                continue
            d = json.load(t.extractfile(member))
            fold, profile, _ = member.name.split('/')
            for group, v in d['validation'].items():
                rows.append(dict(fold=fold, profile=profile, epoch=d['epoch'], group=group,
                    au=v['active_latent_dimensions'], ce=v['masked_mutable_ce'], kl=v['masked_input_kl_per_token'],
                    loss=v['masked_mutable_ce']+v['masked_input_full_ce']+.001*v['masked_input_kl_per_token'],
                    beta_end=.001*min(1,d['epoch']/10)))
    with (output/'training_timeline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    records = choose_records(args.data)
    plan = dict(source_sha256=sha256(Path(__file__)), device='cpu', torch_version=torch.__version__,
        dependency_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('model.py','length_diagnostics.py','evaluate_prior.py')},
        threads=args.threads, seeds=list(SEEDS), arms=ARMS, replay_steps=args.steps,
        data_sha256=sha256(args.data/'folds.json'),
        selected=[{k:r[k] for k in ('sequence_id','panel','n_codons','v2_group')} for r in records],
        checkpoints={name:sha256(path) for name,path in args.checkpoints},
        scope='Diagnostic train and inner val of fold0 only; outer tests are not read for diagnostics.')
    if (args.output/'plan.json').exists():
        assert json.loads((args.output/'plan.json').read_text()) == plan, 'Different diagnostic plan'
    else:
        save_json(args.output/'plan.json',plan)
    timeline(args.timeline, args.output)
    for name, path in args.checkpoints:
        output = args.output/name
        output.mkdir(exist_ok=True)
        if (output/'complete.json').exists():
            continue
        emit('load_checkpoint', name=name)
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        model = CodonVMAE(**state['config']['model']).eval().requires_grad_(False)
        model.load_state_dict(state['model'])
        del state
        gc.collect()
        started = time.monotonic()
        cache, pools = trace_cache(model, records, output)
        with torch.no_grad():
            save_json(output/'projection_geometry.json',dict(
                mu_weight_row_norm=model.mu.weight.norm(dim=1).tolist(),
                decoder_weight_column_norm=model.from_latent.weight.norm(dim=0).tolist(),
                clean_variance_by_length={g:pools['train/'+g+'/clean/mu'].variance().tolist() for g in GROUPS}))
        probes, causal = [], []
        for i, item in enumerate(cache):
            if item['mode']=='clean':
                continue
            causal.append(gradient_probe(model, item))
            if item['record']['panel']=='val' and item['mode']=='mask173':
                group = length_group(item['record']['n_codons']-1)
                high = pools['train/'+group+'/clean/mu'].variance()>.01
                probes.append(interventions(model, item, high))
            emit('gradient_sequence', checkpoint=name, index=i+1, total=len(cache))
            save_json(output/'gradients.json',causal)
        save_json(output/'interventions.json',probes)
        if name.startswith('V2'):
            replay_heads(model, cache, output, args.steps)
        save_json(output/'complete.json',dict(seconds=time.monotonic()-started,checkpoint_sha256=sha256(path),
                                             sequences=len(records),masked_observations=len(causal)))
        del model, cache, pools
        gc.collect()
    save_json(args.output/'complete.json',dict(checkpoints=list(plan['checkpoints']),device='cpu'))
    emit('completed', output=str(args.output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeline',type=Path,required=True)
    parser.add_argument('--checkpoint',action='append',required=True,help='NAME=PATH')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--steps',type=int,default=48)
    args=parser.parse_args()
    args.checkpoints=[(s.split('=',1)[0],Path(s.split('=',1)[1])) for s in args.checkpoint]
    if any(name not in ('V1_epoch50','V2_epoch26') for name,_ in args.checkpoints):
        raise ValueError('Use explicit frozen checkpoint labels')
    run(args)


if __name__=='__main__':
    main()
