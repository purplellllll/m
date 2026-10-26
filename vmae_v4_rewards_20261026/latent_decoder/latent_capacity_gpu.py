"""Actual smaller latent architectures, initialized only from fold-specific training data."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS,Pool,emit,ce_metrics
from .latent_recovery_gpu import BASE,batch_mask,initialize,panels
from .model import CodonVMAE,PAD,mutable_positions
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import pick
from .v2_train import atomic_torch

DIMS=(16,32,64)
ARMS=('control','learned_mean','deterministic_sigma1')


@torch.no_grad()
def rank_dimensions(model,records):
    """Equal clean/hidden conditional SNR, weighted by the decoder's input sensitivity."""
    size=model.latent_dim
    pools={mode:Pool(size) for mode in ('clean','hidden')}
    noise={mode:torch.zeros(size,dtype=torch.float64) for mode in pools}
    total,n=torch.zeros(size,dtype=torch.float64),0
    device=next(model.parameters()).device
    for record in records:
        tokens,inputs,selected=batch_mask([record],173,device)
        for mode in pools:
            mu,lv=model.encode(tokens if mode=='clean' else inputs,tokens)
            measured=mutable_positions(tokens) if mode=='clean' else selected
            pools[mode].add(mu.cpu(),tokens.masked_fill(~measured,PAD).cpu())
            noise[mode]+=lv[measured].double().exp().sum(0).cpu()
            if mode=='clean':
                valid=tokens<PAD
                total+=mu[valid].double().sum(0).cpu();n+=int(valid.sum())
    score=torch.zeros(size,dtype=torch.float64)
    for mode,pool in pools.items():
        within=(pool.aa_square-pool.aa_total.square()/pool.aa_n.clamp_min(1)[:,None]).sum(0)/pool.n
        score+=.5*within.clamp_min(0)/(noise[mode]/pool.n).clamp_min(1e-12)
    score*=model.from_latent.weight.detach().double().square().sum(0).cpu()
    return score,total/n


def compress_state(state,dimensions,scores,mean):
    source_dim=state['config']['model']['latent_dim']
    assert 0<dimensions<=source_dim and len(scores)==len(mean)==source_dim
    selected=scores.topk(dimensions).indices.sort().values
    removed=torch.ones(source_dim,dtype=torch.bool);removed[selected]=False
    result=dict(config=copy.deepcopy(state['config']),model=dict(state['model']))
    result['config']['model']['latent_dim']=dimensions
    for name in ('mu.weight','mu.bias','logvar.weight','logvar.bias'):
        result['model'][name]=state['model'][name][selected].clone()
    result['model']['from_latent.weight']=state['model']['from_latent.weight'][:,selected].clone()
    result['model']['from_latent.bias']=state['model']['from_latent.bias']+state['model']['from_latent.weight'][:,removed]@mean[removed].to(state['model']['from_latent.weight'])
    result['capacity_initialization']=dict(retained_indices=selected.tolist(),
        criterion='Equal clean/hidden within-AA mu variance divided by posterior noise, times decoder column norm squared; training data only',
        discarded_mean_compensation=True,source_dimension=source_dim,target_dimension=dimensions)
    return result,selected


@torch.no_grad()
def initial_check(model,teacher,records):
    rows=[]
    device=next(model.parameters()).device
    for record in records:
        tokens,inputs,selected=batch_mask([record],173,device)
        if not selected.any():continue
        scores=[]
        for network in (teacher,model):
            logits,mu,lv=network(inputs,tokens,sample=False)
            scores.append(ce_metrics(logits,tokens,mu,lv,selected))
        rows.append(dict(sequence_id=record['sequence_id'],group=record['length_group'],
                         teacher=scores[0],compressed=scores[1]))
    return rows


def prepare(data,initial,output,fold):
    output.mkdir(parents=True,exist_ok=True)
    source=sha256(Path(__file__))
    plan=dict(fold=fold,initial_sha256=sha256(initial),source_sha256=source,
              core_sha256=sha256(Path(__file__).with_name('latent_recovery_gpu.py')),
              dimensions=list(DIMS),train_per_group=1024,val_per_group=64)
    target=output/'plan.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:save_json(target,plan)
    if (output/'complete.json').exists():return
    rows=panels(data,fold,1024,64)
    records=pick(rows['train'],8,'capacity-train-ranking-20261023')
    teacher,config=initialize(initial,BASE)
    teacher.eval().requires_grad_(False)
    score,mean=rank_dimensions(teacher,records)
    state=load_checkpoint(initial)
    save_json(output/'ranking.json',dict(plan=plan,training_ids=[r['sequence_id'] for r in records],
        scores=score.tolist(),mean_mu=mean.tolist(),outer_test_used=False))
    for dimensions in DIMS:
        target=output/f'initial_dim{dimensions}.pt'
        compressed,indices=compress_state(state,dimensions,score,mean)
        compressed['capacity_initialization'].update(teacher_sha256=plan['initial_sha256'],source_sha256=source)
        atomic_torch(target,compressed)
        model=CodonVMAE(**compressed['config']['model']).cuda().eval().requires_grad_(False)
        model.load_state_dict(compressed['model'],strict=True)
        diagnostic_records=[dict(r,length_group=('short' if r['n_codons']-1<=300 else 'medium' if r['n_codons']-1<700 else 'long')) for r in records]
        checks=initial_check(model,teacher,diagnostic_records)
        save_json(output/f'initial_dim{dimensions}_check.json',dict(rows=checks,
            initial_sha256=sha256(target),initial_compression_is_approximate=True,outer_test_used=False))
        # Relative-quarter mask is meaningful at every capacity, unlike legacy weak192.
        active=torch.zeros(dimensions)
        active[score[indices].topk(max(1,dimensions//4)).indices]=1
        atomic_torch(output/f'probe_dim{dimensions}.pt',dict(active64=active,
            definition='Top quarter of retained teacher channels by training-only score; weak means remaining three quarters'))
        emit('capacity_initialization',fold=fold,dimensions=dimensions,
             train_probe_delta_ce=sum(r['compressed']['ce']-r['teacher']['ce'] for r in checks)/len(checks))
        del model,compressed
        torch.cuda.empty_cache()
    active=torch.zeros(teacher.latent_dim);active[score.topk(teacher.latent_dim//4).indices]=1
    atomic_torch(output/'probe_dim256.pt',dict(active64=active,definition='Top quarter by same training-only score'))
    save_json(output/'complete.json',dict(prepared_dimensions=list(DIMS),teacher_unchanged=sha256(initial)==plan['initial_sha256']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    initial=args.prior/f'fold_{args.fold}'/'uniform'/'best.pt'
    prepare(args.data,initial,args.output/'inputs',args.fold)
    if args.mode=='prepare':return
    for dimensions in DIMS:
        path=args.output/'inputs'/f'initial_dim{dimensions}.pt'
        command=[sys.executable,'-u','-m','latent_decoder.latent_recovery_gpu','screen',
            '--data',str(args.data),'--initial',str(path),'--output',str(args.output/f'dim_{dimensions}'),
            '--fold',str(args.fold),'--steps','1152','--train-per','1024','--val-per','64','--arms',*ARMS]
        subprocess.run(command,check=True)
    save_json(args.output/'training_complete.json',dict(fold=args.fold,dimensions=list(DIMS),arms=list(ARMS),
        outer_test_used=False,legacy_weak192_probe_not_applicable_to_smaller_dimensions=True))


if __name__=='__main__':
    main()
