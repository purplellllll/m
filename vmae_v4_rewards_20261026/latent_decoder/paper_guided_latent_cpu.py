"""Bounded paper-guided warm-restart pilot on CPU or CUDA; not a reproduction."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .budget_js_extension import save_json
from .evaluate_prior import shuffle_latents
from .latent_collapse_cpu import GROUPS, Pool, ce_metrics, emit, masked
from .latent_rank_cpu import within_geometry
from .length_diagnostics import length_group
from .model import AA_IDS, PAD, CodonVMAE, collate_records, mutable_positions, vae_loss
from .train import sha256
from .v2_data import pick

ARMS = ('control', 'variance', 'vicreg_weak', 'vicreg_strong', 'anneal_fast',
        'anneal_slow', 'fixed_sigma1', 'fixed_sigma_half', 'condition_dropout', 'active64')
SEEDS = (20261023, 20261024)


def load_checkpoint(path):
    try:
        return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except TypeError as error:
        if 'mmap' not in str(error):
            raise
        return torch.load(path,map_location='cpu',weights_only=False)


def choose(data):
    manifest = json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['corpus_sha256']
    membership = {identity: panel for panel in ('train', 'val')
                  for identity in manifest['splits']['0'][panel]}
    metadata, bad = [], set()
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            g = manifest['assignments'][r['sequence_id']]['group']
            if r['split'] != 'test':
                bad.add(g)
            if r['sequence_id'] in membership:
                metadata.append(dict(sequence_id=r['sequence_id'], n_codons=r['n_codons'],
                    v2_group=g, old_split=r['split'], panel=membership[r['sequence_id']]))
    train = pick([r for r in metadata if r['panel']=='train'], 8, 'papers-train-20261023')
    audit = pick([r for r in metadata if r['panel']=='val' and r['old_split']=='test'
                  and r['v2_group'] not in bad], 4, 'papers-audit-20261023')
    audit_groups = {r['v2_group'] for r in audit}
    tune = pick([r for r in metadata if r['panel']=='val' and r['v2_group'] not in audit_groups],
                4, 'papers-tune-20261023')
    selected = {r['sequence_id']: dict(r, panel=panel)
                for panel, rows in [('train', train), ('tune', tune), ('audit', audit)] for r in rows}
    records = []
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['sequence_id'] in selected:
                records.append(dict(r, **selected[r['sequence_id']]))
    groups = [{r['v2_group'] for r in records if r['panel']==p} for p in ('train','tune','audit')]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert len(records)==48
    return records


def centered_aa(values, tokens):
    aa = AA_IDS.to(tokens.device)[tokens]
    counts = torch.bincount(aa, minlength=21).to(values).clamp_min(1)
    totals = values.new_zeros(21, values.shape[-1]).index_add(0, aa, values)
    return values-totals[aa]/counts[aa, None]


def vicreg(a, b, tokens, variance_only=False):
    # Adaptation: AA-residual token features, gamma=.15, no expander; not image VICReg.
    a, b = centered_aa(a, tokens)/.15, centered_aa(b, tokens)/.15
    invariance = F.mse_loss(a, b)
    variance = sum(F.relu(1-(x.var(0, unbiased=True)+1e-4).sqrt()).mean() for x in (a,b))/2
    covariance = a.new_zeros(())
    for x in (a,b):
        x = x-x.mean(0)
        cov = x.T@x/(len(x)-1)
        covariance = covariance+(cov.square().sum()-cov.diag().square().sum())/x.shape[1]
    return 25*variance if variance_only else 25*invariance+25*variance+covariance


def beta_for(arm, step, steps):
    if arm not in ('anneal_fast', 'anneal_slow'):
        return .001
    rate = .25 if arm=='anneal_fast' else .0625
    return .001*math.tanh(rate*step)/math.tanh(rate*steps)


def posterior(mu, logvar, arm, active):
    if arm in ('fixed_sigma1', 'fixed_sigma_half'):
        logvar = torch.full_like(logvar, 0. if arm=='fixed_sigma1' else math.log(.25))
    if arm=='active64':
        mu, logvar = mu*active, logvar*active
    return mu, logvar


def decode(model, z, tokens, arm, active, training=False):
    if arm=='active64':
        z = z*active
    aa = model.aa_embedding(model.aa_ids[tokens])
    if arm=='condition_dropout' and training:
        aa = F.dropout(aa, .2, training=True)
    return model.output(model.decoder(model.from_latent(z)+aa, tokens==PAD))


@torch.no_grad()
def cache_prefix(model, records):
    cache = []
    device = next(model.parameters()).device
    for i, record in enumerate(records):
        cpu_tokens = collate_records([record])
        tokens = cpu_tokens.to(device)
        seeds = (173,174) if record['panel']=='train' else (175,) if record['panel']=='tune' else (176,)
        views = []
        for seed in seeds:
            inputs, selected = masked(cpu_tokens, record['sequence_id'], seed)
            inputs, selected = inputs.to(device), selected.to(device)
            assert selected.any()
            h = model.embedding(inputs)+model.aa_embedding(model.aa_ids[tokens])
            for block in model.encoder.layers[:-1]:
                h = block(h, tokens==PAD)
            views.append(dict(h=h.detach(), selected=selected))
        cache.append(dict(record=record, tokens=tokens, views=views))
        emit('cache_sequence', index=i+1, total=len(records), panel=record['panel'])
    return cache


def encode_tail(model, h, tokens, arm, active):
    h = model.encoder.norm(model.encoder.layers[-1](h, tokens==PAD))
    mu, logvar = model.mu(h), model.logvar(h).clamp(-8,8)
    return posterior(mu, logvar, arm, active)


def aggregate(rows):
    result = {}
    for group in GROUPS:
        chosen = [r for r in rows if r['group']==group]
        assert chosen
        result[group] = {key:sum(r[key] for r in chosen)/len(chosen)
                         for key in ('ce','full_ce','kl','fixed_loss')}
    result['macro'] = {key:sum(result[g][key] for g in GROUPS)/3 for key in result[GROUPS[0]]}
    return result


@torch.no_grad()
def evaluate(model, cache, arm, active, detailed=False):
    rows, pools, kl_samples = [], {}, {g:[] for g in GROUPS}
    for item in cache:
        tokens, view = item['tokens'], item['views'][0]
        mu, lv = encode_tail(model, view['h'], tokens, arm, active)
        logits = decode(model, mu, tokens, arm, active)
        base = ce_metrics(logits, tokens, mu, lv, view['selected'])
        group = length_group(item['record']['n_codons']-1)
        row = dict(sequence_id=item['record']['sequence_id'], group=group, **base)
        if detailed:
            stats_tokens = tokens.masked_fill(~mutable_positions(tokens), PAD)
            if group not in pools:
                pools[group] = Pool(mu.shape[-1], covariance=True)
            pools[group].add(mu.cpu(), stats_tokens.cpu())
            per_dim = .5*(mu.square()+lv.exp()-1-lv)
            kl_samples[group].append(per_dim[mutable_positions(tokens)].cpu())
            interventions = dict(zero_all=torch.zeros_like(mu),
                shuffle=shuffle_latents(mu,tokens,view['selected'],torch.Generator(device=mu.device).manual_seed(9473)),
                zero_weak192=mu*active)
            for name, z in interventions.items():
                ce = ce_metrics(decode(model,z,tokens,arm,active),tokens,mu,lv,view['selected'])['ce']
                row[name+'_delta_ce'] = ce-base['ce']
        rows.append(row)
    result = dict(metrics=aggregate(rows), rows=rows)
    if detailed:
        result['geometry'] = {}
        for g,pool in pools.items():
            kl = torch.cat(kl_samples[g])
            result['geometry'][g] = dict(**pool.summary(), within_aa_geometry=within_geometry(pool),
                kl_collapsed_delta05={str(eps):int(((kl<eps).float().mean(0)>=.95).sum())
                                      for eps in (1e-4,.001,.01,.1)})
        result['intervention'] = {g:{name:sum(r[name] for r in rows if r['group']==g)/sum(r['group']==g for r in rows)
            for name in ('zero_all_delta_ce','shuffle_delta_ce','zero_weak192_delta_ce')} for g in GROUPS}
    return result


def train_arm(model, cache, tune, initial, arm, seed, active, steps, output):
    start = time.monotonic()
    params = dict(model.named_parameters())
    for name, value in initial.items():
        params[name].data.copy_(value)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=.01)
    initial_eval = evaluate(model,tune,arm,active)
    best, best_step = copy.deepcopy(initial), 0
    best_loss = initial_eval['metrics']['macro']['fixed_loss']
    history = [dict(step=0,**initial_eval['metrics']['macro'])]
    by_group = {g:[c for c in cache if length_group(c['record']['n_codons']-1)==g] for g in GROUPS}
    permutations = {g:torch.randperm(len(rows),generator=torch.Generator().manual_seed(seed)).tolist()
                    for g,rows in by_group.items()}
    uses_reg = arm in ('variance','vicreg_weak','vicreg_strong')
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        a_values, b_values, targets = [], [], []
        for g in GROUPS:
            item = by_group[g][permutations[g][(step-1)%len(by_group[g])]]
            tokens, (a,b) = item['tokens'], item['views']
            mu, lv = encode_tail(model,a['h'],tokens,arm,active)
            # Identical posterior noise across paired arms, even when dropout consumes RNG.
            noise_rng = torch.Generator(device=mu.device).manual_seed(seed+step*100+GROUPS.index(g))
            z = mu+torch.randn(mu.shape,device=mu.device,generator=noise_rng)*(.5*lv).exp()
            loss,_ = vae_loss(decode(model,z,tokens,arm,active,training=True),tokens,mu,lv,
                              a['selected'],beta_for(arm,step,steps))
            assert torch.isfinite(loss), (arm,seed,step,g)
            (loss/3).backward(retain_graph=uses_reg)
            if uses_reg:
                other,_ = encode_tail(model,b['h'],tokens,arm,active)
                common = mutable_positions(tokens)&~a['selected']&~b['selected']
                a_values.append(mu[common]); b_values.append(other[common]); targets.append(tokens[common])
            del loss,z
        if uses_reg:
            strength = .01 if arm=='vicreg_strong' else .001
            aux = vicreg(torch.cat(a_values),torch.cat(b_values),torch.cat(targets),arm=='variance')
            (strength*aux).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step in (steps//2, steps):
            validation = evaluate(model,tune,arm,active)
            score = validation['metrics']['macro']
            history.append(dict(step=step,**score))
            if score['fixed_loss']<best_loss:
                best = {n:p.detach().clone() for n,p in params.items() if p.requires_grad}
                best_loss, best_step = score['fixed_loss'], step
            emit('tune',arm=arm,seed=seed,step=step,seconds=round(time.monotonic()-start,1),**score)
    torch.save(dict(parameters=best,arm=arm,seed=seed,step=best_step,active64=active),output/'adapter.pt')
    result = dict(arm=arm,seed=seed,steps=steps,selected_step=best_step,history=history,
                  seconds=time.monotonic()-start,selected_tune_fixed_loss=best_loss)
    save_json(output/'training.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--v1',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    assert args.steps>=2 and args.steps%2==0
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if args.device=='cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha, v1_sha = sha256(args.checkpoint),sha256(args.v1)
    records=choose(args.data)
    plan=dict(device=args.device,threads=4,torch_version=torch.__version__,steps=args.steps,seeds=SEEDS,arms=ARMS,
        checkpoint_sha256=checkpoint_sha,v1_sha256=v1_sha,source_sha256=sha256(Path(__file__)),
        corpus_sha256=sha256(args.data/'corpus.jsonl.gz'),folds_sha256=sha256(args.data/'folds.json'),
        selected=[{k:r[k] for k in ('sequence_id','panel','v2_group','n_codons','old_split')} for r in records],
        scope='Local warm restart: last encoder block, final norm, posterior heads, decoder input projection only; dropout off except intervention',
        learning_rate=1e-5,weight_decay=.01,clip=1.,beta=.001,
        evaluation='Deterministic mu; full CE + masked mutable CE + .001 raw KL; equal length macros',
        masks=dict(train=[173,174],tune=[175],audit=[176]),
        audit='New subset of fold0 internal validation; old-test-only homology groups; no outer test; pilot, not independent V2 final test')
    target=args.output/'plan.json'
    if target.exists():
        assert json.loads(target.read_text())==json.loads(json.dumps(plan)), 'Changed experiment on resume'
    else:
        save_json(target,plan)
    state=load_checkpoint(args.checkpoint)
    model=CodonVMAE(**state['config']['model']).eval().requires_grad_(False).to(args.device)
    model.load_state_dict(state['model']); del state
    cache=cache_prefix(model,records)
    panels={p:[c for c in cache if c['record']['panel']==p] for p in ('train','tune','audit')}
    names=('encoder.layers.5.','encoder.norm.','mu.','logvar.','from_latent.')
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(names))
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    pool=Pool(256)
    with torch.no_grad():
        for item in panels['train']:
            mu,_=encode_tail(model,item['views'][0]['h'],item['tokens'],'control',None)
            pool.add(mu.cpu(),item['tokens'].masked_fill(~mutable_positions(item['tokens']),PAD).cpu())
    within=(pool.aa_square-pool.aa_total.square()/pool.aa_n.clamp_min(1)[:,None]).sum(0)/pool.n
    active=torch.zeros(256); active[within.topk(64).indices]=1
    save_json(args.output/'active64_train_only.json',dict(indices=active.nonzero().flatten().tolist(),
              criterion='Largest AA-centered training masked-view variance, no tune/audit access'))
    active=active.to(args.device)
    trials=[]
    for seed in SEEDS:
        for arm in ARMS:
            output=args.output/f'{arm}_seed{seed}'; output.mkdir(exist_ok=True)
            if (output/'training.json').exists():
                result=json.loads((output/'training.json').read_text())
            else:
                emit('train_start',arm=arm,seed=seed)
                result=train_arm(model,panels['train'],panels['tune'],initial,arm,seed,active,args.steps,output)
            trials.append(result)
    # Lock every candidate/step before any audit metrics are computed.
    save_json(args.output/'selection_locked.json',dict(trials=trials,selection='Min tune fixed loss among steps 0,12,24 (24-step run); no audit tuning'))
    parameters=dict(model.named_parameters())
    for trial in trials:
        arm,seed=trial['arm'],trial['seed']
        output=args.output/f'{arm}_seed{seed}'
        if (output/'audit.json').exists():
            continue
        adapter=torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
        for name,value in adapter['parameters'].items():
            parameters[name].data.copy_(value)
        result=evaluate(model,panels['audit'],arm,active,detailed=True)
        save_json(output/'audit.json',result)
        emit('audit',arm=arm,seed=seed,**result['metrics']['macro'])
    for name,value in initial.items():
        parameters[name].data.copy_(value)
    save_json(args.output/'V2_initial_audit.json',evaluate(model,panels['audit'],'control',active,True))
    del cache,panels
    state=load_checkpoint(args.v1)
    model.load_state_dict(state['model']); del state
    v1cache=cache_prefix(model,[r for r in records if r['panel']=='audit'])
    save_json(args.output/'V1_audit.json',evaluate(model,v1cache,'control',active,True))
    assert checkpoint_sha==sha256(args.checkpoint) and v1_sha==sha256(args.v1)
    save_json(args.output/'complete.json',dict(device=args.device,trials=len(trials),baseline_unchanged=True,
                source_sha256=sha256(Path(__file__))))
    emit('completed',trials=len(trials))


if __name__=='__main__':
    main()
