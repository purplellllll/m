"""Fixed V3/V4-epoch6, own-score plus unchanged monotonic property rewards on CPU."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
PREVIOUS=WORK/'runs/vmae_own_full30_cpu_20261031'
sys.path.insert(0,str(WORK))
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS
VERSIONS={'v3_multi':dict(checkpoint=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt',
    expected='6592a07293da25b18f9ed71584e48dc4da23fa121b4f735f10d26ee7da26c6af',historical_method='V3_same_reward'),
    'v4_epoch6':dict(checkpoint=WORK/'deliverables/vmae_v4_epoch6_current_20261027/checkpoints/epoch6.pt',
    expected='8e95e3d3e5c62917c01a78a3add4ab45ea1afa7a1282748ce8d637f3fbea02e3',historical_method='V4_epoch6_current')}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def reward(value,reference):
    components=dict(own_score=value['own_score'],gc_reward=(value['gc_percent']-reference['gc_percent'])/5.,
        cai_reward=(value['cai']-reference['cai'])/.05,
        mfe_reward=(reference['mfe_per_nt']-value['mfe_per_nt'])/.05)
    components['combined_reward']=sum(components.values())
    return components


def inputs():
    previous=read(PREVIOUS/'inputs_locked.json')
    manifest=read(WORK/'runs/seq2ribo_multicell_comparison_20261027/candidate_manifest.json')
    lookup={(r['sequence_id'],r['method']):r for r in manifest.values()}
    rows=[]
    for r in previous:
        rows.append(dict(sequence_id=r['sequence_id'],group=r['group'],n_codons=r['n_codons'],original=r['original'],
            historical={version:lookup[r['sequence_id'],spec['historical_method']] for version,spec in VERSIONS.items()}))
    assert len(rows)==30
    return rows


def own_scorer(out):
    implementation=module('frozen_full30_driver',PREVIOUS/'run_full30.py')
    assert sha(implementation.PARAM)==read(PREVIOUS/'protocol_locked.json')['params_sha256']
    for seed in [30101,30102]:
        assert sha(implementation.PROBE/f'gated_graph_{seed}_c.pt')==read(PREVIOUS/'protocol_locked.json')['heads_sha256'][str(seed)]
    implementation.OUT=out
    class Scorer(implementation.OwnScorer):
        @torch.no_grad()
        def __call__(self,dna):
            key=hashlib.sha256(dna.encode()).hexdigest()
            old_score=PREVIOUS/'scores'/f'{key}.json';old_feature=PREVIOUS/'features'/f'{key}.npz'
            if key not in self.cache and old_score.exists():
                value=read(old_score)
                assert value['sequence_sha256']==key and value['feature_sha256']==sha(old_feature)
                value=dict(value,feature_cache_root=str(PREVIOUS))
                self.cache[key]=value;save(out/'scores'/f'{key}.json',value)
            return super().__call__(dna)
    return Scorer(),implementation


@torch.no_grad()
def one_case(model,scorer,row,version,out,steps):
    start=time.monotonic();identity=row['sequence_id'];original=scorer(row['original']['cds_dna'])
    old=scorer(row['historical'][version]['cds_dna'])
    tokens=collate_records([dict(cds_dna=original['dna'])]);active=mutable_positions(tokens)
    mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens);sigma=(.5*lv).exp()
    rng=torch.Generator().manual_seed(20261031+int(identity[4:12],16))
    def evaluate(offset,step):
        ids=synonymous_logits(model.decode(mu+sigma*offset,tokens),tokens).argmax(-1)
        ids[~active]=tokens[~active];assert torch.equal(AA_IDS[ids],AA_IDS[tokens])
        dna=''.join(CODONS[int(i)] for i in ids[0]);assert dna[:3]==original['dna'][:3] and dna[-3:]==original['dna'][-3:]
        value=scorer(dna);return dict(value,step=step,**reward(value,original))
    offset=torch.zeros_like(mu);initial=evaluate(offset,0);latent_best=initial['combined_reward']
    best=dict(original,step=-1,**reward(original,original));trace=[initial]
    if initial['combined_reward']>best['combined_reward']:best=initial
    for step in range(1,steps+1):
        trial=offset+torch.randn(mu.shape,generator=rng)*active[...,None]*max(.25,.95**(step//10))
        trial*=active[...,None];rms=trial[active].square().mean().sqrt();trial*=min(1.,8/float(rms.clamp_min(1e-12)))
        value=evaluate(trial,step);trace.append(value)
        if value['combined_reward']>latent_best:latent_best=value['combined_reward'];offset=trial
        if value['combined_reward']>best['combined_reward']:best=value
        save(out/'progress.json',dict(stage='searching',version=version,sequence_id=identity,group=row['group'],step=step,
            steps=steps,completed_cases=len(list((out/'outputs').glob('*.json'))),total_cases=30,
            best_combined_reward=best['combined_reward'],best_own_score=best['own_score'],seconds_current_case=time.monotonic()-start))
        if step%10==0:print(json.dumps(dict(version=version,sequence_id=identity,step=step,
            reward=best['combined_reward'],own_score=best['own_score'],seconds=time.monotonic()-start)),flush=True)
    return dict(version=version,sequence_id=identity,group=row['group'],n_codons=row['n_codons'],steps=steps,
        original=dict(original,**reward(original,original)),old_v3=dict(old,**reward(old,original)),
        initial_decoder=initial,best=best,trace=trace,selection='Highest own+GC+CAI+MFE combined reward, not highest own score',
        gain_from_original=best['own_score']-original['own_score'],gain_vs_old_v3=best['own_score']-old['own_score'],
        combined_gain_from_original=best['combined_reward']-original['own_score'],
        changed_codons=sum(best['dna'][j:j+3]!=original['dna'][j:j+3] for j in range(0,len(original['dna']),3)),seconds=time.monotonic()-start)


def run(version,smoke=False):
    import RNA
    torch.set_num_threads(2)
    spec=VERSIONS[version];assert sha(spec['checkpoint'])==spec['expected']
    out=ROOT/version;out.mkdir(exist_ok=True)
    for name in ['outputs','features','scores']:(out/name).mkdir(exist_ok=True)
    rows=inputs();raw=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    model=CodonVMAE(**raw['config']['model']).eval().requires_grad_(False);model.load_state_dict(raw['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==2498240
    scorer,previous=own_scorer(out)
    protocol=dict(version=version,checkpoint_sha256=sha(spec['checkpoint']),checkpoint_epoch=raw['epoch'],
        model_config=raw['config']['model'],parameters=2498240,steps=80,proposals_per_step=1,radius=8.,seed_basis=20261031,
        original_reference='Same original CDS, property rewards centered on original, not on old optimized endpoint',
        target='Own+(GC_fraction-GC_original)/0.05+(CAI-CAI_original)/0.05+(MFE_original/nt-MFE_candidate/nt)/0.05',
        property_weights={'gc':1.,'cai':1.,'mfe':1.},clip_or_threshold=False,
        search='Same black-box latent hill-climb as preceding own-only full30; select by combined reward',
        scorer_heads_sha256={str(s):sha(previous.PROBE/f'gated_graph_{s}_c.pt') for s in [30101,30102]},
        physics_parameters_sha256=sha(previous.PARAM),driver_sha256=sha(__file__),torch=torch.__version__,viennarna=RNA.__version__,
        feature_source_sha256={p.name:sha(p) for p in (previous.SOURCE/'evaluator/seq2ribo').glob('*.py')},
        source_inputs_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        precision_note='Search fold uses own-input ViennaRNA2.7.0. All90 prior endpoints had exact MFE agreement with2.7.2; final properties reevaluated with2.7.2',
        prior_own_only_is_different_objective=True,prior_feature_cache_reused_only_by_sequence_and_source_identity=True,
        checkpoint_frozen=True,no_retraining=True,independent_scorers_used_for_selection=False,
        limitations=['Own scalar head trained on bacterial mRFP; human panel OOD',
                    '30 exploratory cases reused; not an independent confirmation',
                    'Black-box80 proposals vs historical80 gradient updates: unequal query/compute budget'])
    if (out/'protocol_locked.json').exists():assert read(out/'protocol_locked.json')==protocol
    else:save(out/'protocol_locked.json',protocol);save(out/'inputs_locked.json',rows)
    results=[];begun=time.monotonic()
    for row in rows[:1] if smoke else rows:
        path=out/'outputs'/f'{row["sequence_id"]}.json'
        if path.exists():result=read(path);assert result['steps']==80
        else:
            result=one_case(model,scorer,row,version,out,2 if smoke else 80)
            save(out/'smoke.json' if smoke else path,result)
        results.append(result)
        save(out/'status.json',dict(stage='smoke_complete' if smoke else 'searching',completed=len(results),total=30,seconds=time.monotonic()-begun))
    assert sha(spec['checkpoint'])==spec['expected'] and all(p.grad is None for p in model.parameters())
    if not smoke:
        save(out/'final_search_results.json',results)
        save(out/'search_complete.json',dict(status='completed',version=version,cases=30,steps_per_case=80,
            seconds=time.monotonic()-begun,no_generator_retraining=True,checkpoint_sha256_after=sha(spec['checkpoint'])))
    del model,scorer;gc.collect()
    print(f'{version}: '+('SMOKE COMPLETE' if smoke else 'FULL30 SEARCH COMPLETE'),flush=True)


def selfcheck():
    ref=dict(own_score=8.,gc_percent=50.,cai=.7,mfe_per_nt=-.3)
    assert reward(ref,ref)['combined_reward']==8.
    altered=dict(own_score=8.2,gc_percent=55.,cai=.75,mfe_per_nt=-.35)
    r=reward(altered,ref)
    assert all(abs(r[k]-1)<1e-12 for k in ['gc_reward','cai_reward','mfe_reward'])
    assert abs(r['combined_reward']-11.2)<1e-12
    # A combined winner can have a lower own score: retain and report that tradeoff explicitly.
    tradeoff=dict(own_score=7.,gc_percent=60.,cai=.8,mfe_per_nt=-.4)
    assert reward(tradeoff,ref)['combined_reward']>reward(ref,ref)['combined_reward']
    print('PASS: unchanged centered reward units/signs; combined selection can trade off own score')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',choices=list(VERSIONS)+['selfcheck']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:
        if args.version=='selfcheck':selfcheck()
        else:run(args.version,args.smoke)
    except BaseException as error:
        save(ROOT/f'failure_{args.version}.json',dict(error=repr(error),unix=time.time()));raise
