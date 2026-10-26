"""Counterfactual CDS information probes with the complete protein condition fixed."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS,Pool,ce_metrics,emit
from .latent_rank_cpu import within_geometry
from .latent_recovery_gpu import BASE,RecoveryVMAE,batch_mask,panels
from .length_diagnostics import length_group
from .model import AA_IDS,MASK,PAD,mutable_positions,validate_targets
from .paper_guided_latent_cpu import load_checkpoint
from .train import codon_frequency,sha256
from .v2_data import pick


def recode_visible(tokens,selected,kind,seed,counts):
    """Change only visible synonymous codons; protein, hidden targets and PAD stay fixed."""
    assert tokens.device.type=='cpu' and selected.shape==tokens.shape
    validate_targets(tokens)
    assert selected.dtype==torch.bool
    assert kind in ('permutation','train_frequency')
    mutable=mutable_positions(tokens)&~selected
    aa=AA_IDS[tokens]
    changed=tokens.clone()
    rng=torch.Generator().manual_seed(seed)
    for row in range(len(tokens)):
        for label in range(20):
            positions=mutable[row]&(aa[row]==label)
            values=tokens[row,positions]
            if not len(values):continue
            if kind=='permutation':
                changed[row,positions]=values[torch.randperm(len(values),generator=rng)]
            else:
                weights=counts*(AA_IDS[:64]==label)
                assert weights.sum()>0
                changed[row,positions]=torch.multinomial(weights,len(values),replacement=True,generator=rng)
    assert torch.equal(AA_IDS[changed],AA_IDS[tokens])
    assert torch.equal(changed[selected],tokens[selected])
    assert torch.equal(changed[~mutable],tokens[~mutable])
    if kind=='permutation':
        assert torch.equal(torch.bincount(changed[changed<PAD],minlength=64),
                           torch.bincount(tokens[tokens<PAD],minlength=64))
    return changed


def load_model(checkpoint,device):
    state=load_checkpoint(checkpoint)
    recipe=state.get('recipe',BASE)
    model=RecoveryVMAE(recipe,**state['config']['model']).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    return model


def projection_geometry(pool,weight,gain=1.):
    """Covariance of W*delta_mu; biases and fixed AA condition cancel exactly."""
    matrix=weight.detach().cpu().double()*gain
    center=pool.total/pool.n
    covariance=pool.cross/pool.n-torch.outer(center,center)
    within=pool.cross/pool.n-(pool.aa_total.T/pool.aa_n.clamp_min(1))@pool.aa_total/pool.n
    projected=matrix@covariance@matrix.T
    residual=matrix@within@matrix.T
    return dict(covariance_participation=float(projected.trace().square()/projected.square().sum().clamp_min(1e-30)),
        within_aa_covariance_participation=float(residual.trace().square()/residual.square().sum().clamp_min(1e-30)),
        rms=float(((projected.trace()+(matrix@center).square().sum())/len(matrix)).clamp_min(0).sqrt()),
        interpretation='Same CDS response after the actual from_latent weight and gain; fixed AA embedding and projection bias cancel')


@torch.no_grad()
def probe(model,records,counts,active64,repeats=2):
    device=next(model.parameters()).device
    pools,rows={},[]
    active64=active64.to(device)
    for record in records:
        for mode in ('clean','masked'):
            gpu_tokens,inputs,selected=batch_mask([record],175,device)
            tokens=gpu_tokens.cpu()
            selected=selected if mode=='masked' else torch.zeros_like(selected)
            inputs=inputs if mode=='masked' else gpu_tokens
            measured=selected if mode=='masked' else mutable_positions(gpu_tokens)
            assert measured.any()
            mu,lv=model.encode(inputs,gpu_tokens)
            original_ce=ce_metrics(model.decode(mu,gpu_tokens),gpu_tokens,mu,lv,measured)['ce']
            for kind in ('permutation','train_frequency'):
                for repeat in range(repeats):
                    number=int(hashlib.sha256(f'{record["sequence_id"]}:{kind}:{repeat}'.encode()).hexdigest()[:15],16)
                    changed=recode_visible(tokens,selected.cpu(),kind,number,counts)
                    altered_input=changed.masked_fill(selected.cpu(),MASK).to(device)
                    alternative_mu,_=model.encode(altered_input,gpu_tokens)
                    altered_ce=ce_metrics(model.decode(alternative_mu,gpu_tokens),gpu_tokens,mu,lv,measured)['ce']
                    # Only weak-coordinate CDS-dependent signal changes; entire protein is unchanged.
                    mixed=mu*active64+alternative_mu*(1-active64)
                    weak_ce=ce_metrics(model.decode(mixed,gpu_tokens),gpu_tokens,mu,lv,measured)['ce']
                    delta=mu-alternative_mu
                    group=length_group(record['n_codons']-1)
                    key='/'.join((group,mode,kind))
                    if key not in pools:pools[key]=Pool(model.latent_dim,covariance=True)
                    pools[key].add(delta.cpu(),tokens.masked_fill(~measured.cpu(),PAD))
                    energy=delta[measured].double().square().sum(0)
                    rows.append(dict(sequence_id=record['sequence_id'],group=group,mode=mode,kind=kind,repeat=repeat,
                        baseline_target_ce=original_ce,target_ce_after_recode=altered_ce,target_ce_delta=altered_ce-original_ce,
                        weak_coordinate_replacement_delta_ce=weak_ce-original_ce,
                        response_energy=float(energy.sum()),weak_response_energy=float((energy*(1-active64)).sum()),
                        visible_codon_change_fraction=float((changed!=tokens).sum()/(mutable_positions(tokens)&~selected.cpu()).sum().clamp_min(1)),
                        same_complete_protein=True,masked_targets_unchanged=True))
        emit('information_probe_sequence',sequence_id=record['sequence_id'])
    summary={}
    for key,pool in pools.items():
        group,mode,kind=key.split('/')
        chosen=[r for r in rows if r['group']==group and r['mode']==mode and r['kind']==kind]
        energy=sum(r['response_energy'] for r in chosen)
        summary[key]=dict(response_geometry=dict(**pool.summary(),within_aa_geometry=within_geometry(pool)),
            decoder_projection=projection_geometry(pool,model.from_latent.weight,model.recipe['gain']),
            mean_target_ce_delta=sum(r['target_ce_delta'] for r in chosen)/len(chosen),
            mean_weak_coordinate_delta_ce=sum(r['weak_coordinate_replacement_delta_ce'] for r in chosen)/len(chosen),
            weak_response_energy_fraction=sum(r['weak_response_energy'] for r in chosen)/max(energy,1e-30),
            mean_visible_change_fraction=sum(r['visible_codon_change_fraction'] for r in chosen)/len(chosen))
    return dict(summary=summary,rows=rows,
        interpretation='CDS-specific sensitivity conditional on the identical entire protein; participation is not mutual-information bits; recoding CE is a probe, not fixed validation loss',
        evaluation_positions='clean: all mutable codons; masked: the fixed hidden mutable codons; compare methods within a mode')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if args.device=='cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    chosen=panels(args.data,args.fold,1024,64)
    records=pick(chosen['val'],4,'protein-fixed-information-probe-20261023')
    counts=codon_frequency(chosen['train'])
    calibration=torch.load(args.calibration,map_location='cpu',weights_only=False)
    plan=dict(checkpoint_sha256=sha256(args.checkpoint),calibration_sha256=sha256(args.calibration),
        corpus_sha256=sha256(args.data/'corpus.jsonl.gz'),folds_sha256=sha256(args.data/'folds.json'),
        training_counts_sha256=hashlib.sha256(counts.numpy().tobytes()).hexdigest(),
        active64_indices=calibration['active64'].nonzero().flatten().tolist(),
        source_sha256=sha256(Path(__file__)),core_source_sha256=sha256(Path(__file__).with_name('latent_recovery_gpu.py')),
        fold=args.fold,device=args.device,records=[r['sequence_id'] for r in records],outer_test_used=False,
        mask_seed=175,recoding_repeats=2,counts='Fold-specific training-only Laplace-smoothed synonymous codon frequencies')
    if args.output.exists():
        assert json.loads(args.output.read_text())['plan']==plan,'Probe changed on resume'
        return
    model=load_model(args.checkpoint,args.device)
    result=probe(model,records,counts,calibration['active64'])
    result['plan']=plan
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,result)
    emit('information_probe_completed',output=str(args.output))


if __name__=='__main__':
    main()
