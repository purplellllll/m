"""Bounded, isolated diagnostics of corpus masked-CE plateau."""
import argparse
import copy
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import AA_IDS, MASK, PAD, SYNONYMS, CodonVMAE, corrupt, synonymous_logits, vae_loss
from .replay_single import changes
from .train import batches, codon_frequency, load_data, sha256


def choose(records, number, short=False):
    eligible = [r for r in records if not short or 192 <= r['n_codons'] <= 384]
    return sorted(eligible, key=lambda r: r['sequence_id'])[:number]


def cached(records, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return [(t.to(device), *corrupt(t.to(device), .15, generator)) for t in batches(records,4)]


@torch.no_grad()
def measure(model, data):
    model.eval()
    total_ce = correct = count = clean_ce = 0.
    for tokens, inputs, selected in data:
        logits,mu,logvar = model(inputs,tokens,sample=False)
        logp = synonymous_logits(logits,tokens).log_softmax(-1)
        nll = -logp.gather(-1,tokens.clamp_max(63).unsqueeze(-1)).squeeze(-1)
        _, terms = vae_loss(logits,tokens,mu,logvar,selected,beta=.001)
        assert torch.allclose(terms['masked_ce'],nll[selected].mean(),atol=1e-6)
        total_ce += nll[selected].sum().item()
        correct += (logp.argmax(-1)[selected]==tokens[selected]).sum().item()
        count += selected.sum().item()
        clean = synonymous_logits(model(tokens,tokens,sample=False)[0],tokens).log_softmax(-1)
        clean_ce += -clean.gather(-1,tokens.clamp_max(63).unsqueeze(-1)).squeeze(-1)[selected].sum().item()
    return dict(masked_ce=total_ce/count,accuracy=correct/count,clean_on_same_sites_ce=clean_ce/count,sites=int(count))


def build_neighbor_counts(records):
    counts = np.zeros((66,66,22,64),dtype=np.int32)
    aa = AA_IDS.numpy()
    for r in records:
        tokens = np.array(r['codon_ids'],dtype=np.int64)
        left = np.r_[PAD,tokens[:-1]]
        right = np.r_[tokens[1:],PAD]
        np.add.at(counts,(left,right,aa[tokens],tokens),1)
    return counts


@torch.no_grad()
def audit(model, data, frequencies, neighbors):
    model.eval()
    synonyms = SYNONYMS[:64].to(frequencies.device)
    prior = frequencies[None,:]*synonyms
    prior = prior/prior.sum(-1,keepdim=True)
    sums = dict(sites=0,model_ce=0.,global_prior_ce=0.,sequence_prior_ce=0.,neighbor_prior_ce=0.,
                model_correct=0,global_correct=0,sequence_correct=0,neighbor_correct=0,
                clean_ce=0.,aa_only_ce=0.)
    family = {}
    for tokens,inputs,selected in data:
        logp = synonymous_logits(model(inputs,tokens,sample=False)[0],tokens).log_softmax(-1)
        clean = synonymous_logits(model(tokens,tokens,sample=False)[0],tokens).log_softmax(-1)
        aa_input = torch.where(tokens<PAD,torch.full_like(tokens,MASK),tokens)
        aa_only = synonymous_logits(model(aa_input,tokens,sample=False)[0],tokens).log_softmax(-1)
        nll = -logp.gather(-1,tokens.clamp_max(63).unsqueeze(-1)).squeeze(-1)
        sums['sites'] += int(selected.sum())
        sums['model_ce'] += nll[selected].sum().item()
        sums['model_correct'] += int((logp.argmax(-1)[selected]==tokens[selected]).sum())
        sums['clean_ce'] += -clean.gather(-1,tokens.clamp_max(63).unsqueeze(-1)).squeeze(-1)[selected].sum().item()
        sums['aa_only_ce'] += -aa_only.gather(-1,tokens.clamp_max(63).unsqueeze(-1)).squeeze(-1)[selected].sum().item()
        for size in (2,3,4,6):
            at = selected & (SYNONYMS.to(tokens.device)[tokens].sum(-1)==size)
            entry = family.setdefault(str(size),dict(sites=0,ce_sum=0.,correct=0))
            entry['sites'] += int(at.sum())
            entry['ce_sum'] += nll[at].sum().item()
            entry['correct'] += int((logp.argmax(-1)[at]==tokens[at]).sum())
        for row in range(len(tokens)):
            target = tokens[row,selected[row]]
            global_probs = prior[target]
            visible = inputs[row][inputs[row]<PAD]
            sequence_counts = torch.bincount(visible,minlength=64).float()
            seq_probs = (sequence_counts[None,:]*synonyms[target])+10.*global_probs
            seq_probs /= seq_probs.sum(-1,keepdim=True)
            left = F.pad(inputs[row],(1,0),value=PAD)[:-1][selected[row]].cpu().numpy()
            right = F.pad(inputs[row],(0,1),value=PAD)[1:][selected[row]].cpu().numpy()
            aa = AA_IDS.to(tokens.device)[target].cpu().numpy()
            local = torch.as_tensor(neighbors[left,right,aa],device=tokens.device,dtype=torch.float32)+10.*global_probs
            local /= local.sum(-1,keepdim=True)
            for name,probs in [('global',global_probs),('sequence',seq_probs),('neighbor',local)]:
                sums[name+'_prior_ce'] += -probs.gather(-1,target[:,None]).log().sum().item()
                sums[name+'_correct'] += int((probs.argmax(-1)==target).sum())
    result = {k:(v/sums['sites'] if k!='sites' else v) for k,v in sums.items()}
    result['family'] = {k:dict(sites=v['sites'],ce=v['ce_sum']/v['sites'],accuracy=v['correct']/v['sites']) for k,v in family.items() if v['sites']}
    return result


def train_arm(checkpoint, name, train_batches, fixed, probes, out, steps, lr, beta):
    device = probes['val'][0][0].device
    model = CodonVMAE(**checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    training = checkpoint['config']['training']
    assert not training['amp'] and training['batch_size']==4 and training['gradient_accumulation']==8
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=training['weight_decay'])
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    for group in optimizer.param_groups:
        group['lr'] = lr
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}
    points = [dict(step=0,probes={k:measure(model,v) for k,v in probes.items()})]
    fixed_generator = torch.Generator(device=device).manual_seed(4301)
    fixed_inputs = [corrupt(t.to(device),.15,fixed_generator) for t in train_batches] if fixed else None
    for step in range(1,steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(8):
            index = ((step-1)*8+micro)%len(train_batches)
            tokens = train_batches[index].to(device)
            torch.manual_seed(88391+(step-1)*8+micro)
            inputs,selected = fixed_inputs[index] if fixed else corrupt(tokens,.15)
            logits,mu,logvar = model(inputs,tokens)
            loss,terms = vae_loss(logits,tokens,mu,logvar,selected,beta,
                                 training['masked_loss_weight'],training.get('full_loss_weight',1.))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss/8).backward()
            losses.append(terms['masked_ce'].item())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),training['grad_clip'],error_if_nonfinite=True)
        optimizer.step()
        row = dict(arm=name,step=step,train_masked_ce=sum(losses)/8,grad_norm=norm.item(),lr=lr,beta=beta)
        if step in (40,80,120,steps):
            row['probes'] = {k:measure(model,v) for k,v in probes.items()}
            points.append(row)
            print(json.dumps(dict(event='probe',**row)),flush=True)
        elif step%10==0:
            print(json.dumps(row),flush=True)
        with (out/(name+'.jsonl')).open('a') as handle:
            handle.write(json.dumps(row)+'\n')
    delta = changes(model,initial)
    assert all(v['changed_elements']>0 for v in delta.values())
    torch.save(dict(model=model.state_dict(),diagnostic_only=True,config=checkpoint['config']),out/(name+'.model.pt'))
    return dict(lr=lr,beta=beta,steps=steps,fixed_mask=fixed,points=points,changes=delta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=160)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    source_hash=sha256(args.checkpoint)
    splits,_=load_data(args.data,allow_development=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit=choose(splits['train'],128,short=True)
    validation=choose(splits['val'],128,short=True)
    probes=dict(train_mask_A=cached(fit,device,4301),train_mask_B=cached(fit,device,9301),val=cached(validation,device,6301))
    identities={k:[r['sequence_id'] for r in rows] for k,rows in [('fit',fit),('val',validation)]}
    identities['audit_train']=[r['sequence_id'] for r in choose(splits['train'],256)]
    identities['audit_val']=[r['sequence_id'] for r in choose(splits['val'],256)]
    (out/'sample_ids.json').write_text(json.dumps(identities,indent=2)+'\n')
    model=CodonVMAE(**checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    neighbors=build_neighbor_counts(splits['train'])
    frequencies=codon_frequency(splits['train']).to(device)
    report=dict(data_sizes={k:len(v) for k,v in splits.items()},audit={},arms={})
    for split in ('train','val'):
        report['audit'][split]=audit(model,cached(choose(splits[split],256),device,173),frequencies,neighbors)
        print(json.dumps(dict(event='audit',split=split,**report['audit'][split])),flush=True)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    del model,neighbors
    if device.type=='cuda':
        torch.cuda.empty_cache()
    random.seed(20261019)
    stream=list(itertools.islice(batches(splits['train'],4,shuffle=True),args.steps*8))
    subset=list(batches(fit,4))
    for name,data,fixed,lr,beta in [
        ('stream_tail',stream,False,7.596123493895991e-7,.001),
        ('stream_reset',stream,False,1e-4,.001),
        ('stream_no_kl',stream,False,1e-4,0.),
        ('subset_fixed',subset,True,1e-4,.001),
        ('subset_dynamic',subset,False,1e-4,.001)]:
        report['arms'][name]=train_arm(checkpoint,name,data,fixed,probes,out,args.steps,lr,beta)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(event='arm_complete',arm=name,last=report['arms'][name]['points'][-1])),flush=True)
        if device.type=='cuda':
            torch.cuda.empty_cache()
    assert source_hash==sha256(args.checkpoint)
    (out/'complete.json').write_text(json.dumps(dict(source_unchanged=True,sha256=source_hash))+'\n')
    print('PLATEAU_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
