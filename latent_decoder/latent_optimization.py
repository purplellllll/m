"""Bounded frozen VMAE/scorer optimization audit, not an expression efficacy claim."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .diagnose_mask_plateau import choose
from .model import (AA_IDS, CODONS, PAD, SYNONYMS, CodonVMAE, codon_probabilities,
                    collate_records, mutable_positions, nucleotide_probabilities)
from .train import codon_frequency, load_data, sha256


def design_probabilities(logits,tokens):
    probabilities=codon_probabilities(logits,tokens)
    fixed=F.one_hot(tokens.clamp_max(63),64).float()*(tokens<PAD)[...,None]
    return torch.where(mutable_positions(tokens)[...,None],probabilities,fixed)


def predictor_input(probs,tokens):
    # Official source CDS omit the stop; preserve it in design output but exclude it from scoring.
    lengths=(tokens<PAD).sum(-1)
    sense=torch.arange(tokens.shape[1],device=tokens.device)[None,:]<(lengths-1)[:,None]
    nts=nucleotide_probabilities(probs*sense[...,None]).transpose(1,2)
    if nts.shape[-1]>4500:
        raise ValueError('Scorer length exceeds 4500; truncation is prohibited')
    return F.pad(nts,(0,4500-nts.shape[-1]))


class TranslationObjective(torch.nn.Module):
    def __init__(self,directory):
        super().__init__()
        from score_model.inference import CNN_Encoder_Model, InferenceModel
        directory=Path(directory)
        config=json.loads((directory/'model_config.json').read_text())
        self.model=CNN_Encoder_Model(config,InferenceModel.input_shape)
        checkpoint=torch.load(directory/'best_model.p',map_location='cpu',weights_only=False)
        state=checkpoint['model_state_dict']
        required=set(self.model.state_dict())
        extra=set(state)-required
        # The official inference wrapper drops these unused training heads with strict=False.
        unused_prefixes=('reg_detector_list.','reg_filter.','reg_encoder.','reg_fc.',
                         'RPF_fc_count.','TE_fc.','TE_fc_count.','RPF_layer.')
        if required-set(state) or any(not key.startswith(unused_prefixes) for key in extra):
            raise ValueError('Unexpected scorer checkpoint schema')
        self.model.load_state_dict({key:state[key] for key in required},strict=True)
        self.unused_checkpoint_keys=sorted(extra)
        condition=np.load(directory/'conditions/HEK293T_normal_10552_RPKM_HEK293T.npz')['mRNA']
        self.register_buffer('condition',torch.from_numpy(np.log1p(condition.astype(np.float32)*5))[None,:10552])
        self.eval().requires_grad_(False)

    def forward(self,probs,tokens):
        batch=len(tokens)
        return self.model(predictor_input(probs,tokens),self.condition.expand(batch,-1),
                          probs.new_full((batch,1),4.5)).flatten()


def standardized_rms(variable,active):
    return ((variable.square()*active[...,None]).sum((1,2))/(active.sum(-1)*variable.shape[-1])).sqrt()


@torch.no_grad()
def project_radius(variable,active,radius):
    variable.mul_(active[...,None])
    factor=(radius/standardized_rms(variable,active).clamp_min(1e-12)).clamp_max(1.)
    variable.mul_(factor[:,None,None])


def state_hash(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def snapshot(probs,tokens,scorer,counts,reference_probs):
    predicted=probs.argmax(-1)
    valid=tokens<PAD
    active=mutable_positions(tokens)
    aa_ids=AA_IDS.to(tokens.device)
    assert torch.equal(aa_ids[predicted][valid],aa_ids[tokens][valid])
    assert torch.equal(predicted[valid & ~active],tokens[valid & ~active])
    hard=F.one_hot(predicted,64).float()*valid[...,None]
    hard_score=scorer(hard,tokens)
    soft_score=scorer(probs,tokens)
    mean=.5*(probs+reference_probs)
    js=.5*((probs*(probs.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
           +(reference_probs*(reference_probs.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1))
    prior=counts.to(tokens.device)[None,None,:]*SYNONYMS.to(tokens.device)[tokens]
    prior=prior/prior.sum(-1,keepdim=True)
    prior_nll=-prior.gather(-1,predicted[...,None]).squeeze(-1).clamp_min(1e-12).log()
    rows=[]
    for i in range(len(tokens)):
        ids=predicted[i,valid[i]].tolist()
        sequence=''.join(CODONS[x] for x in ids)
        rows.append(dict(soft_score=float(soft_score[i]),hard_score=float(hard_score[i]),
            soft_hard_gap=float(soft_score[i]-hard_score[i]),cds_dna=sequence,
            mutable_hamming=float(((predicted[i]!=tokens[i])&active[i]).sum()/active[i].sum()),
            js_from_initial=float(js[i,active[i]].mean()),gc_fraction=(sequence.count('G')+sequence.count('C'))/len(sequence),
            frequency_nll=float(prior_nll[i,active[i]].mean()),protein_preserved=True))
    return rows


def write_rows(path,rows):
    with path.open('a',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')


@torch.no_grad()
def geometry(model,tokens,mu,sigma,scorer,counts,reference,ids,out):
    active=mutable_positions(tokens)
    rng=torch.Generator().manual_seed(73119)
    for radius in [.01,.05,.1,.25,.5]:
        for draw in range(3):
            noise=torch.randn(mu.shape,generator=rng)
            noise=noise*active[...,None]
            noise=noise/standardized_rms(noise,active)[:,None,None]*radius
            probabilities=design_probabilities(model.decode(mu+sigma*noise,tokens),tokens)
            rows=snapshot(probabilities,tokens,scorer,counts,reference)
            write_rows(out/'geometry.jsonl',[dict(sequence_id=identity,kind='perturbation',radius=radius,draw=draw,**r)
                       for identity,r in zip(ids,rows)])
    donor=tokens.clone()
    prior=counts[None,None,:]*SYNONYMS[tokens]
    sampled=torch.multinomial(prior.reshape(-1,64),1,generator=rng).reshape(tokens.shape)
    donor[active]=sampled[active]
    donor_mu,_=model.encode(donor,tokens)
    for alpha in [0.,.25,.5,.75,1.]:
        probabilities=design_probabilities(model.decode((1-alpha)*mu+alpha*donor_mu,tokens),tokens)
        rows=snapshot(probabilities,tokens,scorer,counts,reference)
        write_rows(out/'geometry.jsonl',[dict(sequence_id=identity,kind='interpolation',alpha=alpha,**r)
                   for identity,r in zip(ids,rows)])


def optimize_batch(model,scorer,records,counts,out,start,steps):
    tokens=collate_records(records)
    ids=[r['sequence_id'] for r in records]
    active=mutable_positions(tokens)
    with torch.no_grad():
        mu,logvar=model.encode(tokens,tokens)
        sigma=torch.exp(.5*logvar)
        base_logits=model.decode(mu,tokens)
        reference=design_probabilities(base_logits,tokens)
        original=F.one_hot(tokens.clamp_max(63),64).float()*(tokens<PAD)[...,None]
        original_scores=scorer(original,tokens).tolist()
    initial=snapshot(reference,tokens,scorer,counts,reference)
    write_rows(out/'initial.jsonl',[dict(sequence_id=i,n_codons=r['n_codons'],original_cds=r['cds_dna'],
                                        original_hard_score=s,**v) for i,r,s,v in zip(ids,records,original_scores,initial)])
    geometry(model,tokens,mu,sigma,scorer,counts,reference,ids,out)
    summary=[]
    radius_trace=[]
    for name,lr in [('latent_lr_0.01',.01),('latent_lr_0.05',.05),('direct_logit_control',.05),('random_latent',None)]:
        variable=torch.zeros_like(base_logits if name=='direct_logit_control' else mu,requires_grad=lr is not None)
        optimizer=torch.optim.Adam([variable],lr=lr) if lr is not None else None
        best=[dict(step=0,**v) for v in initial]
        unique=[{v['cds_dna']} for v in initial]
        rng=torch.Generator().manual_seed(947311)
        for step in range(1,steps+1):
            if time.monotonic()-start>3600:
                raise TimeoutError('Latent diagnostic exceeded 60-minute limit')
            grad_norm=None
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                logits=base_logits+variable if name=='direct_logit_control' else model.decode(mu+sigma*variable,tokens)
                probabilities=design_probabilities(logits,tokens)
                hard=F.one_hot(probabilities.argmax(-1),64).float()*(tokens<PAD)[...,None]
                straight_through=hard+probabilities-probabilities.detach()
                penalty=(variable.square()*active[...,None]).sum()/ (active.sum()*variable.shape[-1])
                loss=-scorer(straight_through,tokens).mean()+.01*penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite optimization objective')
                loss.backward()
                grad_norm=float(torch.nn.utils.clip_grad_norm_([variable],1.,error_if_nonfinite=True))
                optimizer.step()
                project_radius(variable,active,2. if name=='direct_logit_control' else .5)
                if name=='latent_lr_0.05':
                    radius_trace.append(standardized_rms(variable.detach(),active))
            else:
                with torch.no_grad():
                    variable.copy_(torch.randn(mu.shape,generator=rng)*active[...,None])
                    variable.mul_((radius_trace[step-1]/standardized_rms(variable,active).clamp_min(1e-12))[:,None,None])
            with torch.no_grad():
                logits=base_logits+variable if name=='direct_logit_control' else model.decode(mu+sigma*variable,tokens)
                probabilities=design_probabilities(logits,tokens)
                values=snapshot(probabilities,tokens,scorer,counts,reference)
                radius=standardized_rms(variable,active).tolist()
            rows=[]
            for index,value in enumerate(values):
                unique[index].add(value['cds_dna'])
                if value['hard_score']>best[index]['hard_score']:
                    best[index]=dict(step=step,parameter_rms=radius[index],**value)
                rows.append(dict(sequence_id=ids[index],arm=name,step=step,parameter_rms=radius[index],
                                 grad_norm=grad_norm,**{k:v for k,v in value.items() if k!='cds_dna'}))
            write_rows(out/'steps.jsonl',rows)
            if step%20==0:
                print(json.dumps(dict(arm=name,step=step,ids=ids,best_gains=[b['hard_score']-v['hard_score'] for b,v in zip(best,initial)])),flush=True)
        for i,identity in enumerate(ids):
            summary.append(dict(sequence_id=identity,arm=name,learning_rate=lr,steps=steps,
                initial=initial[i],best=best[i],final=values[i],unique_candidates=len(unique[i]),
                best_hard_gain=best[i]['hard_score']-initial[i]['hard_score'],
                original_hard_score=original_scores[i],final_hard_gain=values[i]['hard_score']-initial[i]['hard_score']))
        write_rows(out/'results.jsonl',summary[-len(ids):])
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--scorer',default='score_model')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    start=time.monotonic()
    source_hash=sha256(args.checkpoint)
    predictor_hash=sha256(Path(args.scorer)/'best_model.p')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    scorer=TranslationObjective(args.scorer)
    frozen_hashes=(state_hash(model),state_hash(scorer))
    splits,_=load_data(args.data,allow_development=True)
    records=choose(splits['val'],16,short=True)
    counts=codon_frequency(splits['train'])
    print(json.dumps(dict(event='start',records=len(records),steps=80,device='cpu',objective='HEK293T raw predictor score')),flush=True)
    all_results=[]
    for begin in range(0,len(records),4):
        all_results.extend(optimize_batch(model,scorer,records[begin:begin+4],counts,out,start,80))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert frozen_hashes==(state_hash(model),state_hash(scorer))
    assert source_hash==sha256(args.checkpoint) and predictor_hash==sha256(Path(args.scorer)/'best_model.p')
    summary={}
    for arm in sorted({r['arm'] for r in all_results}):
        rows=[r for r in all_results if r['arm']==arm]
        gains=[r['best_hard_gain'] for r in rows]
        summary[arm]=dict(mean_best_hard_gain=float(np.mean(gains)),median_best_hard_gain=float(np.median(gains)),
                         improved=sum(g>1e-5 for g in gains),sequences=len(rows),
                         mean_final_hard_gain=float(np.mean([r['final_hard_gain'] for r in rows])),
                         median_unique=float(np.median([r['unique_candidates'] for r in rows])))
    result=dict(arms=summary,checkpoint_sha256=source_hash,predictor_sha256=predictor_hash,
                model_and_scorer_unchanged=True,elapsed_seconds=time.monotonic()-start,
                steps=80,validation_records=16,condition='HEK293T',rna_count=4.5,
                optimization='straight-through one-hot, Adam, score maximization minus .01 mean square parameter penalty',
                latent_radius=.5,direct_radius=2.,latent_units='posterior standard deviations RMS at mutable positions',
                torch_version=torch.__version__,device='cpu',
                scorer_unused_training_keys=scorer.unused_checkpoint_keys)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
