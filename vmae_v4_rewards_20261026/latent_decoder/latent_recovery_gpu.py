"""Full-network latent recovery: inner-fold screening before any outer-test evaluation."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from .budget_js_extension import save_json
from .evaluate_prior import shuffle_latents
from .latent_collapse_cpu import GROUPS, Pool, ce_metrics, emit, masked
from .latent_rank_cpu import within_geometry
from .length_diagnostics import gradient_summary, length_group
from .model import AA_IDS, PAD, MASK, CodonVMAE, collate_records, mutable_positions, vae_loss, validate_targets
from .paper_guided_latent_cpu import load_checkpoint, vicreg
from .train import sha256
from .v2_data import pick, read_corpus
from .v2_train import atomic_torch

BASE=dict(sigma=None,encoder_lr=5e-5,decoder_lr=5e-5,gain=1.,mu_reg=0.,mid_reg=0.,
          free_bits=0.,mean_weight=0.,profile=(1.,1.,1.),skip_layer=0,deterministic=False,noise_warmup=0.)
RECIPES={name:dict(BASE,**changes) for name,changes in dict(
    control={},sigma1=dict(sigma=1.),sigma1_lrhalf=dict(sigma=1.,encoder_lr=2.5e-5,decoder_lr=2.5e-5),
    sigma1_encfast=dict(sigma=1.,encoder_lr=1e-4,decoder_lr=2.5e-5),
    sigma1_gainhalf=dict(sigma=1.,gain=.5),sigma1_gaindouble=dict(sigma=1.,gain=2.),
    sigma1_mu=dict(sigma=1.,mu_reg=.001),
    sigma1_mid_mu=dict(sigma=1.,mu_reg=.001,mid_reg=.0001),
    sigma1_mid_mu_freebits=dict(sigma=1.,mu_reg=.001,mid_reg=.0001,free_bits=.02),
    sigma1_mean=dict(sigma=1.,mean_weight=.5),
    sigma1_mid_mu_mean=dict(sigma=1.,mu_reg=.001,mid_reg=.0001,mean_weight=.5),
    sigma1_mid_mu_encfast=dict(sigma=1.,mu_reg=.001,mid_reg=.0001,encoder_lr=1e-4,decoder_lr=2.5e-5),
    learned_mid_mu=dict(mu_reg=.001,mid_reg=.0001),
    sigma1_shortfast=dict(sigma=1.,profile=(2.,1.,.5)),
    sigma1_longfast=dict(sigma=1.,profile=(.5,1.,2.)),
    learned_mean=dict(mean_weight=.5),learned_skip1=dict(skip_layer=1),learned_skip3=dict(skip_layer=3),
    learned_skip3_mu=dict(skip_layer=3,mu_reg=.001),
    learned_skip3_midmu=dict(skip_layer=3,mu_reg=.001,mid_reg=.0001),
    learned_skip3_midmu_strong=dict(skip_layer=3,mu_reg=.01,mid_reg=.001),
    learned_skip3_midmu_mean=dict(skip_layer=3,mu_reg=.001,mid_reg=.0001,mean_weight=.5),
    learned_skip3_midmu_freebits=dict(skip_layer=3,mu_reg=.001,mid_reg=.0001,free_bits=.02),
    control_lrslow=dict(encoder_lr=1e-5,decoder_lr=1e-5),
    learned_skip3_lrslow=dict(skip_layer=3,encoder_lr=1e-5,decoder_lr=1e-5),
    learned_noise=dict(noise_warmup=.5),
    learned_skip3_midmu_noise=dict(skip_layer=3,mu_reg=.001,mid_reg=.0001,noise_warmup=.5),
    deterministic_sigma1=dict(sigma=1.,deterministic=True),
    deterministic_skip3_midmu=dict(sigma=1.,deterministic=True,skip_layer=3,mu_reg=.001,mid_reg=.0001)).items()}


class RecoveryVMAE(CodonVMAE):
    def __init__(self,recipe,**kwargs):
        super().__init__(**kwargs)
        self.recipe=recipe
        skip=recipe.get('skip_layer',0)
        assert 0<=skip<=len(self.encoder.layers)
        if skip:
            self.latent_skip=torch.nn.Linear(self.mu.in_features,self.latent_dim,bias=False)
            torch.nn.init.zeros_(self.latent_skip.weight)
            self.register_buffer('skip_aa_mean',torch.zeros(22,self.mu.in_features))
            self.register_buffer('skip_scale',torch.ones(()))

    def encode_features(self,inputs,tokens):
        validate_targets(tokens)
        if inputs.shape!=tokens.shape or tokens.size(1)>self.max_codons:
            raise ValueError('Aligned inputs within max_codons required; no truncation')
        pad=tokens==PAD
        if inputs.dtype!=torch.long or ((inputs<0)|(inputs>MASK)).any() or ((inputs==PAD)!=pad).any():
            raise ValueError('Encoder inputs must be codons/MASK with aligned PAD')
        h=self.embedding(inputs)+self.aa_embedding(self.aa_ids[tokens])
        middle,skip_features=None,None
        for index,layer in enumerate(self.encoder.layers):
            h=layer(h,pad)
            if index==len(self.encoder.layers)//2-1:
                middle=h
            if index==self.recipe.get('skip_layer',0)-1:
                skip_features=h
        h=self.encoder.norm(h)
        mu=self.mu(h)
        if skip_features is not None:
            residual=(skip_features-self.skip_aa_mean[self.aa_ids[tokens]])/self.skip_scale
            mu=mu+self.latent_skip(residual)
        mu=mu*(~pad)[...,None]
        lv=self.logvar(h).clamp(-8,8)*(~pad)[...,None]
        if self.recipe['sigma'] is not None:
            lv=torch.full_like(lv,2*math.log(self.recipe['sigma']))*(~pad)[...,None]
        return mu,lv,middle

    def encode(self,inputs,tokens):
        return self.encode_features(inputs,tokens)[:2]

    def decode(self,z,tokens):
        return super().decode(z*self.recipe['gain'],tokens)


def initialize(path,recipe,device='cuda'):
    state=load_checkpoint(path)
    config=state['config']
    model=RecoveryVMAE(recipe,**config['model']).to(device)
    missing,extra=model.load_state_dict(state['model'],strict=False)
    expected={'latent_skip.weight','skip_aa_mean','skip_scale'} if recipe.get('skip_layer',0) else set()
    assert set(missing)==expected and not extra,(missing,extra)
    with torch.no_grad():
        model.from_latent.weight.div_(recipe['gain'])
    return model,config


@torch.no_grad()
def calibrate_skip(model,records):
    if not model.recipe.get('skip_layer',0):
        return
    pool=Pool(model.mu.in_features)
    device=next(model.parameters()).device
    for record in pick(records,8,'recovery-probe-20261023'):
        tokens,inputs,_=batch_mask([record],175,device)
        h=model.embedding(inputs)+model.aa_embedding(model.aa_ids[tokens])
        for layer in model.encoder.layers[:model.recipe['skip_layer']]:
            h=layer(h,tokens==PAD)
        pool.add(h.cpu(),tokens.masked_fill(~mutable_positions(tokens),PAD).cpu())
    means=pool.aa_total/pool.aa_n.clamp_min(1)[:,None]
    within=(pool.aa_square-pool.aa_total.square()/pool.aa_n.clamp_min(1)[:,None]).sum(0)/pool.n
    model.skip_aa_mean[:21].copy_(means.to(device).float())
    model.skip_scale.fill_(float(within.clamp_min(0).sqrt().median().clamp_min(.01)))


def panels(data,fold,train_per=512,val_per=64):
    manifest=json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz')==manifest['corpus_sha256']
    ids=manifest['splits'][str(fold)]
    membership={identity:p for p in ('train','val') for identity in ids[p]}
    # Only train and inner-val sequence contents enter screening.
    rows=[dict(r,v2_group=manifest['assignments'][r['sequence_id']]['group'],panel=membership[r['sequence_id']])
          for r in read_corpus(data/'corpus.jsonl.gz') if r['sequence_id'] in membership]
    chosen={p:pick([r for r in rows if r['panel']==p],n,'recovery-'+p+'-20261023')
            for p,n in [('train',train_per),('val',val_per)]}
    assert not {r['v2_group'] for r in chosen['train']}&{r['v2_group'] for r in chosen['val']}
    return chosen


def training_batches(rows,seed,steps,microbatch=2):
    by_group={g:[r for r in rows if length_group(r['n_codons']-1)==g] for g in GROUPS}
    for group_rows in by_group.values():
        group_rows.sort(key=lambda r:(r['n_codons'],r['sequence_id']))
    result=[]
    cycle=0
    while len(result)<steps:
        for group in GROUPS:
            chunks=[by_group[group][i:i+microbatch] for i in range(0,len(by_group[group]),microbatch)]
            order=torch.randperm(len(chunks),generator=torch.Generator().manual_seed(seed+cycle)).tolist()
            by_group[group+'_chunks']=[chunks[i] for i in order]
        for index in range(max(len(by_group[g+'_chunks']) for g in GROUPS)):
            for group in GROUPS:
                chunks=by_group[group+'_chunks']
                result.append((group,chunks[index%len(chunks)]))
                if len(result)==steps:
                    return result
        cycle+=1


def batch_mask(rows,seed,device):
    tokens=collate_records(rows)
    inputs,selected=tokens.clone(),torch.zeros_like(tokens,dtype=torch.bool)
    for i,r in enumerate(rows):
        length=r['n_codons']
        a,b=masked(tokens[i:i+1,:length],r['sequence_id'],seed)
        inputs[i,:length],selected[i,:length]=a[0],b[0]
    return tokens.to(device),inputs.to(device),selected.to(device)


@torch.no_grad()
def geometry(model,records,seed=175,limit=8,train_basis=False):
    device=next(model.parameters()).device
    pools={g:Pool(model.latent_dim,covariance=True) for g in GROUPS}
    hidden={g:Pool(model.latent_dim,covariance=True) for g in GROUPS}
    clean={g:Pool(model.latent_dim,covariance=True) for g in GROUPS}
    noise={g:torch.zeros(model.latent_dim,dtype=torch.float64) for g in GROUPS}
    mids=Pool(model.mu.in_features)
    for r in pick(records,limit,'recovery-probe-20261023'):
        tokens,inputs,selected=batch_mask([r],seed,device)
        mu,lv,mid=model.encode_features(inputs,tokens)
        mutable=mutable_positions(tokens)
        group=length_group(r['n_codons']-1)
        pools[group].add(mu.cpu(),tokens.masked_fill(~mutable,PAD).cpu())
        noise[group]+=lv[mutable].double().exp().sum(0).cpu()
        hidden[group].add(mu.cpu(),tokens.masked_fill(~(mutable&selected),PAD).cpu())
        clean_mu,_=model.encode(tokens,tokens)
        clean[group].add(clean_mu.cpu(),tokens.masked_fill(~mutable,PAD).cpu())
        mids.add(mid.cpu(),tokens.masked_fill(~mutable,PAD).cpu())
    result={}
    for g,p in pools.items():
        sigma2=noise[g]/p.n
        within=(p.aa_square-p.aa_total.square()/p.aa_n.clamp_min(1)[:,None]).sum(0)/p.n
        snr=within.clamp_min(0)/sigma2.clamp_min(1e-12)
        active=p.variance()>.01
        result[g]=dict(**p.summary(),within_aa_geometry=within_geometry(p),
            masked_only=dict(**hidden[g].summary(),within_aa_geometry=within_geometry(hidden[g])),
            clean=dict(**clean[g].summary(),within_aa_geometry=within_geometry(clean[g])),
            posterior_noise=dict(mean_variance_by_dim=sigma2.tolist(),
                active_coordinate_mean_variance=float(sigma2[active].mean()) if active.any() else None,
                inactive_coordinate_mean_variance=float(sigma2[~active].mean()) if (~active).any() else None,
                within_aa_snr_by_dim=snr.tolist(),
                coordinates_snr_above={str(t):int((snr>t).sum()) for t in (.001,.01,.1,1.)}))
    calibration=None
    if train_basis:
        combined=Pool(model.latent_dim,covariance=True)
        for p in pools.values():
            combined.n+=p.n;combined.total+=p.total;combined.square+=p.square
            combined.aa_n+=p.aa_n;combined.aa_total+=p.aa_total;combined.aa_square+=p.aa_square
            combined.cross+=p.cross
        within=(combined.aa_square-combined.aa_total.square()/combined.aa_n.clamp_min(1)[:,None]).sum(0)/combined.n
        active=torch.zeros(model.latent_dim);active[within.topk(min(64,model.latent_dim)).indices]=1
        mid_within=(mids.aa_square-mids.aa_total.square()/mids.aa_n.clamp_min(1)[:,None]).sum(0)/mids.n
        calibration=dict(active64=active,mid_gamma=float(mid_within.clamp_min(0).sqrt().median().clamp_min(.01)))
    return result,calibration


@torch.no_grad()
def evaluate(model,records,seeds=(175,176),probes=False,active64=None):
    model.eval()
    device=next(model.parameters()).device
    rows=[]
    for r in records:
        for seed in seeds:
            tokens,inputs,selected=batch_mask([r],seed,device)
            assert selected.any(),r['sequence_id']
            mu,lv=model.encode(inputs,tokens)
            logits=model.decode(mu,tokens)
            scores=ce_metrics(logits,tokens,mu,lv,selected)
            rows.append(dict(sequence_id=r['sequence_id'],group=length_group(r['n_codons']-1),mask_seed=seed,**scores))
    by_length={g:{k:sum(r[k] for r in rows if r['group']==g)/sum(r['group']==g for r in rows)
                  for k in ('ce','full_ce','kl','fixed_loss')} for g in GROUPS}
    result=dict(by_length=by_length,macro={k:sum(by_length[g][k] for g in GROUPS)/3 for k in by_length[GROUPS[0]]},rows=rows)
    if probes:
        result['geometry']=geometry(model,records,limit=8)[0]
        probe_rows=[]
        for r in pick(records,8,'recovery-probe-20261023'):
            tokens,inputs,selected=batch_mask([r],seeds[0],device)
            mu,lv=model.encode(inputs,tokens)
            ce=ce_metrics(model.decode(mu,tokens),tokens,mu,lv,selected)['ce']
            changes=dict(zero=torch.zeros_like(mu),
                shuffle=shuffle_latents(mu,tokens,selected,torch.Generator(device=device).manual_seed(9473)))
            if active64 is not None:
                changes['zero_weak192']=mu*active64.to(device)
            row=dict(sequence_id=r['sequence_id'],group=length_group(r['n_codons']-1))
            for name,z in changes.items():
                row[name+'_delta_ce']=ce_metrics(model.decode(z,tokens),tokens,mu,lv,selected)['ce']-ce
            with torch.enable_grad():
                z=mu.detach().requires_grad_()
                loss,_=vae_loss(model.decode(z,tokens),tokens,mu,lv,selected,beta=0.)
                grad,=torch.autograd.grad(loss,z)
            row['gradient']=gradient_summary(grad[mutable_positions(tokens)])
            probe_rows.append(row)
        result['interventions']=probe_rows
    return result


def reconstruction_objective(model,tokens,inputs,selected,recipe,noise_seed,amp,step_fraction=1.):
    with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=amp):
        mu,lv,mid=model.encode_features(inputs,tokens)
        mu,lv=mu.float(),lv.float()
        rng=torch.Generator(device=tokens.device).manual_seed(noise_seed)
        noise_scale=(.1+.9*min(1.,step_fraction/recipe['noise_warmup'])) if recipe.get('noise_warmup',0.) else 1.
        z=mu if recipe.get('deterministic',False) else mu+noise_scale*torch.randn(mu.shape,device=mu.device,generator=rng)*(.5*lv).exp()
        logits=model.decode(z,tokens)
    mu,lv=mu.float(),lv.float()
    rec,terms=vae_loss(logits.float(),tokens,mu,lv,selected,beta=0.)
    if recipe['mean_weight']:
        with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=amp):
            mean_logits=model.decode(mu.to(logits.dtype),tokens)
        mean_rec,_=vae_loss(mean_logits.float(),tokens,mu,lv,selected,beta=0.)
        rec=(rec+recipe['mean_weight']*mean_rec)/(1+recipe['mean_weight'])
    kl=.5*(mu.square()+lv.exp()-1-lv)
    regularizer=kl[tokens<PAD].mean(0).clamp_min(recipe['free_bits']).sum()
    return rec+.001*regularizer,terms,mu,mid


def eligible_for_selection(recipe,step,steps):
    return step==0 or not recipe.get('noise_warmup',0.) or step/steps>=max(.75,recipe['noise_warmup'])


def train_trial(data,initial,output,fold,arm,seed,steps=576,train_per=512,val_per=64):
    recipe=RECIPES[arm]
    output.mkdir(parents=True,exist_ok=True)
    chosen=panels(data,fold,train_per,val_per)
    plan=dict(recipe=recipe,arm=arm,seed=seed,fold=fold,steps=steps,train_per_group=train_per,val_per_group=val_per,
        initial_sha256=sha256(initial),source_sha256=sha256(Path(__file__)),
        dependencies={n:sha256(Path(__file__).with_name(n)) for n in ('model.py','paper_guided_latent_cpu.py','latent_collapse_cpu.py','latent_rank_cpu.py')},
        folds_sha256=sha256(data/'folds.json'),device='cuda',precision='BF16 forward, FP32 parameters/loss/evaluation; TF32 disabled',
        train_scope='All encoder and decoder parameters; dropout disabled in every arm to isolate recovery',
        microbatch=2,mask_probability=.15,evaluation_masks=[175,176],
        selected={p:[r['sequence_id'] for r in rows] for p,rows in chosen.items()},
        selection='Min fixed inner-val loss at step0 and every144; noise-warmup candidates eligible from 75pct updates; outer test never used',
        fixed_loss='full_ce + masked_mutable_ce + .001 raw_KL/token',
        batch_objective='Token-normalized full and masked CE inside homogeneous batches; equal cyclic length exposure; evaluation first averages each CDS')
    target=output/'plan.json'
    if target.exists():
        assert json.loads(target.read_text())==json.loads(json.dumps(plan)),'Trial changed on resume'
    else:
        save_json(target,plan)
    if (output/'complete.json').exists():
        return json.loads((output/'complete.json').read_text())
    model,config=initialize(initial,recipe)
    # All parameters remain trainable. eval() disables feature dropout, never gradients.
    model.eval().requires_grad_(True)
    calibrate_skip(model,chosen['train'])
    encoder=[p for n,p in model.named_parameters() if not n.startswith(('decoder.','from_latent.','output.'))]
    decoder=[p for n,p in model.named_parameters() if n.startswith(('decoder.','from_latent.','output.'))]
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=recipe['encoder_lr']),
                                 dict(params=decoder,lr=recipe['decoder_lr'])],weight_decay=.01)
    calibration_path=output/'calibration.pt'
    if calibration_path.exists():
        calibration=torch.load(calibration_path,map_location='cpu',weights_only=False)
    else:
        _,calibration=geometry(model,chosen['train'],train_basis=True)
        atomic_torch(calibration_path,calibration)
    start_step,best,history=0,float('inf'),[]
    if (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_step,best,history=state['step'],state['best'],state['history']
        for optstate in optimizer.state.values():
            for k,v in optstate.items():
                if isinstance(v,torch.Tensor):optstate[k]=v.to('cuda')
        del state
    else:
        metrics=evaluate(model,chosen['val'],probes=True,active64=calibration['active64'])
        save_json(output/'initial_inner.json',metrics)
        best=metrics['macro']['fixed_loss']
        history=[dict(step=0,**metrics['macro'])]
        atomic_torch(output/'best.pt',dict(model=model.state_dict(),config=config,recipe=recipe,step=0))
    batches=training_batches(chosen['train'],seed,steps)
    started=time.monotonic()
    moving={k:0. for k in ('loss','ce','full_ce','kl')}
    for index in range(start_step,steps):
        group,rows=batches[index]
        step=index+1
        torch.manual_seed(seed+step*1000)
        tokens,inputs,selected=batch_mask(rows,seed+step,'cuda')
        scale=recipe['profile'][GROUPS.index(group)]
        warm=min(1.,step/24)
        decay=.25+.75*(1+math.cos(math.pi*index/max(1,steps-1)))/2
        for optgroup,base_lr in zip(optimizer.param_groups,(recipe['encoder_lr'],recipe['decoder_lr'])):
            optgroup['lr']=base_lr*scale*warm*decay
        optimizer.zero_grad(set_to_none=True)
        loss,terms,mu,mid=reconstruction_objective(model,tokens,inputs,selected,recipe,seed+step*1000+7,True,step/steps)
        if recipe['mu_reg'] or recipe['mid_reg']:
            _,other_inputs,other_selected=batch_mask(rows,seed+step+10_000_000,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                other_mu,_,other_mid=model.encode_features(other_inputs,tokens)
            common=mutable_positions(tokens)&~selected&~other_selected
            assert int(common.sum())>1
            if recipe['mu_reg']:
                loss=loss+recipe['mu_reg']*vicreg(mu[common],other_mu.float()[common],tokens[common])
            if recipe['mid_reg']:
                rescale=.15/calibration['mid_gamma']
                loss=loss+recipe['mid_reg']*vicreg(mid.float()[common]*rescale,other_mid.float()[common]*rescale,tokens[common])
        assert torch.isfinite(loss),(arm,step)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        for k,v in dict(loss=float(loss),ce=float(terms['masked_ce']),full_ce=float(terms['reconstruction_ce']),kl=float(terms['kl_per_token'])).items():
            moving[k]=.95*moving[k]+.05*v if index>start_step else v
        if step%24==0:
            emit('recovery_train',fold=fold,arm=arm,seed=seed,step=step,steps=steps,group=group,
                 encoder_lr=optimizer.param_groups[0]['lr'],decoder_lr=optimizer.param_groups[1]['lr'],
                 seconds=round(time.monotonic()-started,1),gradient_norm=float(norm),**moving)
        if step%144==0 or step==steps:
            metrics=evaluate(model,chosen['val'])
            metrics['geometry']=geometry(model,chosen['val'],limit=8)[0]
            latent={g:dict(au=metrics['geometry'][g]['pooled']['active_dimensions']['0.01'],
                clean_au=metrics['geometry'][g]['clean']['pooled']['active_dimensions']['0.01'],
                masked_only_au=metrics['geometry'][g]['masked_only']['pooled']['active_dimensions']['0.01'],
                within_pr=metrics['geometry'][g]['within_aa_geometry']['covariance_participation']) for g in GROUPS}
            eligible=eligible_for_selection(recipe,step,steps)
            history.append(dict(step=step,eligible=eligible,**metrics['macro']))
            save_json(output/f'inner_step{step:05d}.json',metrics)
            if eligible and metrics['macro']['fixed_loss']<best:
                best=metrics['macro']['fixed_loss']
                atomic_torch(output/'best.pt',dict(model=model.state_dict(),config=config,recipe=recipe,step=step))
            atomic_torch(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                         step=step,best=best,history=history))
            emit('recovery_inner',fold=fold,arm=arm,seed=seed,step=step,eligible=eligible,seconds=round(time.monotonic()-started,1),latent=latent,**metrics['macro'])
    state=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);selected_step=state['step'];del state
    metrics=evaluate(model,chosen['val'],probes=True,active64=calibration['active64'])
    save_json(output/'selected_inner.json',metrics)
    result=dict(arm=arm,fold=fold,seed=seed,selected_step=selected_step,best_sha256=sha256(output/'best.pt'),
                metrics=metrics['macro'],history=history,seconds=time.monotonic()-started,
                initial_sha256=plan['initial_sha256'],all_parameters_trainable=True,outer_test_used=False)
    result['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
    save_json(output/'complete.json',result)
    # Best checkpoint and complete logs suffice after successful trial; no resume needed.
    (output/'last.pt').unlink(missing_ok=True)
    emit('recovery_completed',**result)
    return result


def screen(args):
    args.output.mkdir(parents=True,exist_ok=True)
    arms=args.arms or list(RECIPES)
    assert 'control' in arms and len(arms)==len(set(arms))
    plan=dict(arms=arms,fold=args.fold,seed=20261030,steps=args.steps,
              source_sha256=sha256(Path(__file__)),initial_sha256=sha256(args.initial),
              train_per_group=args.train_per,val_per_group=args.val_per,
              selection='Lock all inner results before follow-up; no outer test in this stage')
    target=args.output/'screen_plan.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:save_json(target,plan)
    for arm in arms:
        out=args.output/arm
        command=[sys.executable,'-u','-m','latent_decoder.latent_recovery_gpu','worker',
                 '--data',str(args.data),'--initial',str(args.initial),'--output',str(out),
                 '--arm',arm,'--seed','20261030','--fold',str(args.fold),'--steps',str(args.steps),
                 '--train-per',str(args.train_per),'--val-per',str(args.val_per)]
        save_json(args.output/'progress.json',dict(status='running',arm=arm,completed=sum((args.output/a/'complete.json').exists() for a in arms)))
        subprocess.run(command,check=True)
    results={a:json.loads((args.output/a/'selected_inner.json').read_text()) for a in arms}
    control=results['control']['macro']
    candidates=[a for a in arms if a!='control' and results[a]['macro']['ce']<control['ce']
                and results[a]['macro']['fixed_loss']<control['fixed_loss']]
    ranked=sorted(candidates,key=lambda a:results[a]['macro']['fixed_loss'])
    selected=ranked[:1]
    def pr(arm):return sum(results[arm]['geometry'][g]['within_aa_geometry']['covariance_participation'] for g in GROUPS)/3
    alternatives=[a for a in ranked if a not in selected]
    if alternatives:selected.append(max(alternatives,key=pr))
    save_json(args.output/'selection_locked.json',dict(selected=selected,ranked_joint_ce_loss_improvements=ranked,
        rule='Lowest fixed loss plus highest AA-residual covariance PR among remaining joint CE/loss improvements',
        results={a:dict(metrics=results[a]['macro'],within_aa_pr=pr(a)) for a in arms},outer_test_used=False))
    save_json(args.output/'progress.json',dict(status='completed',completed=len(arms),selected=selected))
    emit('screen_completed',selected=selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('screen','worker'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=RECIPES,default='control')
    parser.add_argument('--arms',choices=RECIPES,nargs='+')
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--steps',type=int,default=576)
    parser.add_argument('--train-per',type=int,default=512)
    parser.add_argument('--val-per',type=int,default=64)
    args=parser.parse_args()
    assert args.steps>0 and args.train_per>=8 and args.val_per>=8
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='screen':screen(args)
    else:train_trial(args.data,args.initial,args.output,args.fold,args.arm,args.seed,args.steps,args.train_per,args.val_per)


if __name__=='__main__':
    main()
