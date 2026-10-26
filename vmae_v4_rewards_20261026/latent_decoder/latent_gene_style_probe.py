"""Diagnostic visible-codon context bypass; this does not repair or train the VAE."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS, ce_metrics, emit
from .latent_information_probe import load_model
from .latent_recovery_gpu import batch_mask, panels
from .length_diagnostics import length_group
from .model import AA_IDS, PAD, synonymous_logits
from .train import codon_frequency, sha256
from .v2_data import read_corpus

ALPHAS = (1., 5., 20., 100.)
WEIGHTS = (0., .05, .1, .25, .5, 1.)
SETTINGS = [(a,w) for a in ALPHAS for w in WEIGHTS]


def visible_prior(tokens, hidden, counts, alpha):
    assert tokens.device.type == hidden.device.type == counts.device.type == 'cpu'
    assert tokens.ndim == 2 and len(tokens) == 1
    assert alpha > 0 and tokens.shape == hidden.shape and hidden.dtype == torch.bool
    assert (counts > 0).all() and counts.shape == (64,)
    synonyms = (AA_IDS[:64,None] == AA_IDS[None,:64]).double()
    observed = torch.bincount(tokens[(tokens < PAD) & ~hidden], minlength=64).double()
    prior = counts.double()/(synonyms@counts.double())
    return (observed+alpha*prior)/(synonyms@observed+alpha)


def macro(rows, setting):
    return {key: sum(sum(r['scores'][setting][j] for r in rows if r['group']==g)
            /sum(r['group']==g for r in rows) for g in GROUPS)/3
            for j,key in enumerate(('ce','conditional_loss'))}


@torch.no_grad()
def collect(model, records, counts, clusters):
    rows=[]
    for record in records:
        for seed in (175,176):
            tokens,inputs,selected=batch_mask([record],seed,'cuda')
            mu,lv=model.encode(inputs,tokens)
            logits=model.decode(mu,tokens)
            baseline=ce_metrics(logits,tokens,mu,lv,selected)
            probability=synonymous_logits(logits,tokens).softmax(-1).gather(-1,tokens.clamp_max(63)[...,None])[...,0]
            target=tokens[selected].cpu()
            p=probability[selected].cpu().double()
            assert abs(float(-p.log().mean())-baseline['ce'])<1e-5
            fraction=float(selected.sum()/(tokens<PAD).sum())
            distributions={a:visible_prior(tokens.cpu(),selected.cpu(),counts,a)[target] for a in ALPHAS}
            scores=[]
            for alpha,weight in SETTINGS:
                ce=float(-((1-weight)*p+weight*distributions[alpha]).clamp_min(1e-12).log().mean())
                conditional_loss=baseline['fixed_loss']+(1+fraction)*(ce-baseline['ce'])
                scores.append([ce,conditional_loss])
            rows.append(dict(sequence_id=record['sequence_id'],homology_group=clusters[record['sequence_id']],
                group=length_group(record['n_codons']-1),mask_seed=seed,scores=scores,baseline=baseline))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-for',type=int,required=True)
    args=parser.parse_args();root=args.root
    assert args.wait_for>1
    emit('waiting_for_length_profiles',pid=args.wait_for)
    while True:
        try:identity=Path(f'/proc/{args.wait_for}/cmdline').read_bytes()
        except FileNotFoundError:break
        if b'run_length_profiles.py' not in identity or str(root).encode() not in identity:break
        time.sleep(5)
    assert json.loads((root/'batch_recovery/length_profiles/status.json').read_text())['state']=='completed'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reference=json.loads((root/'v1_safe_inner_all/V1_epoch50.json').read_text())['plan']
    data=root/'source/data'
    assert sha256(data/'corpus.jsonl.gz')==reference['corpus_sha256']
    assert sha256(data/'folds.json')==reference['folds_sha256']
    identities=set(reference['records']);clusters=reference['homology_groups']
    records=[r for r in read_corpus(data/'corpus.jsonl.gz') if r['sequence_id'] in identities]
    assert len(records)==154
    unique=sorted(set(clusters.values()),key=lambda x:hashlib.sha256(('style-crossfit:'+str(x)).encode()).hexdigest())
    folds={g:i%2 for i,g in enumerate(unique)}
    assert len(unique)==62
    for fold in (0,1):
        assert all(any(folds[clusters[r['sequence_id']]]==fold and length_group(r['n_codons']-1)==g for r in records) for g in GROUPS)
    counts=codon_frequency(panels(data,0,1024,64)['train'])
    choices=[]
    for path in (root/'batch_recovery').rglob('complete.json'):
        if 'smoke' in path.parts:continue
        result=json.loads(path.read_text());choices.append((result['metrics']['fixed_loss'],path.parent/'best.pt'))
    best=min(choices,key=lambda x:x[0])[1]
    models=[('V1',root/'V1_epoch50_model_only.pt'),('V2_initial',root/'priors/fold_0/uniform/best.pt'),('V2_best_batch_profile',best)]
    output=root/'gene_style_probe';output.mkdir(exist_ok=True)
    plan=dict(source_sha256=sha256(Path(__file__)),reference_sha256=sha256(root/'v1_safe_inner_all/V1_epoch50.json'),
        records=reference['records'],homology_folds=folds,settings=SETTINGS,masks=[175,176],
        training_counts='Same1024CDS per length fold0 training pool, Laplace smoothed',
        crossfit='Two homology-group folds within INTERNAL validation; upstream model selection reused this validation',
        only_hidden_decoder_probabilities_blended=True,encoder_latent_weights_unchanged=True,
        diagnostic_context_bypass_not_a_new_vae=True,outer_test_used=False,goal_complete=False)
    save_json(output/'plan.json',plan)
    for name,path in models:
        model=load_model(path,'cuda');rows=collect(model,records,counts,clusters)
        chosen={};held=[]
        for fold in (0,1):
            tune=[r for r in rows if folds[r['homology_group']]!=fold]
            setting=min(range(len(SETTINGS)),key=lambda i:macro(tune,i)['conditional_loss'])
            chosen[str(fold)]=dict(index=setting,alpha=SETTINGS[setting][0],weight=SETTINGS[setting][1])
            held += [dict(r,scores=[r['scores'][setting],r['scores'][0]]) for r in rows if folds[r['homology_group']]==fold]
        result=dict(checkpoint_sha256=sha256(path),baseline=macro(held,1),crossfit_context=macro(held,0),
            chosen=chosen,rows=rows,interpretation='Context-bypass diagnosis; latent activity is unchanged; not V1 acceptance')
        save_json(output/(name+'.json'),result)
        emit('gene_style_probe',name=name,baseline=result['baseline'],crossfit_context=result['crossfit_context'],chosen=chosen)
        del model;torch.cuda.empty_cache()
    save_json(output/'complete.json',dict(models=[n for n,_ in models],outer_test_used=False,goal_complete=False))


if __name__=='__main__':main()
