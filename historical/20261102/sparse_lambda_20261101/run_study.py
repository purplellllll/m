"""Finite exploratory lambda search and frozen sparse-context validation, no test selection."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
sys.path.insert(0,str(WORK/'repos/riboseq_scorer_20261031'));sys.path.insert(0,str(WORK))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(name,data):
    (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    torch.set_num_threads(2)
    driver=module('locked_multi_driver',WORK/'runs/v3_v4_own_multi_cpu_20261101/run_experiment.py')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    for name in ['features','scores']:(cache/name).mkdir(exist_ok=True)
    scorer,implementation=driver.own_scorer(cache)
    lambdas=[0.,.25,.5,.75,1.]
    protocol=dict(lambdas=lambdas,objective='own_score+(1-lambda)*(gc_reward+cai_reward+mfe_reward)',
        endpoints='lambda0=current own multiobjective; lambda1=pure own score; not original RiboDecode multiobjective',
        scan='V3 and V4 frozen; 2 shortest per length group, six historical-panel exploratory cases per model;20 proposals per lambda',
        shared_seeds=True,steps=20,selection='Report all components and Pareto tradeoffs, no test-based winner selection',
        sparse='Frozen gated_graph two seeds; original219 mRFP validation only; radii16,32,64,128 plus full; no retraining',
        warning='Existing panel is exploratory, not independent test; lambda sensitivity is coefficient/unit dependent; sparse inference intervention has distribution shift',
        source_sha256=driver.sha(__file__))
    save('protocol_locked.json',protocol)
    base_reward=driver.reward
    # Candidate-pool reranking is distinct from new search trajectories.
    offline=[]
    previous=WORK/'runs/vmae_own_full30_cpu_20261031/outputs'
    for version in driver.VERSIONS:
        for path in sorted((driver.ROOT/version/'outputs').glob('*.json')):
            row=driver.read(path);pool=[row['original']]+row['trace']
            pure=previous/path.name
            if version=='v3_multi' and pure.exists():pool+=driver.read(pure)['trace']
            for lam in lambdas:
                best=max(pool,key=lambda v:base_reward(v,row['original'])['own_score']+(1-lam)*sum(base_reward(v,row['original'])[k] for k in ['gc_reward','cai_reward','mfe_reward']))
                offline.append(dict(version=version,sequence_id=row['sequence_id'],lam=lam,own_score=best['own_score'],gc_percent=best['gc_percent'],cai=best['cai'],mfe_per_nt=best['mfe_per_nt'],dna_sha256=best['sequence_sha256']))
    save('cached_pool_reranking.json',dict(scope='Available historical candidate pool only; not lambda-directed search, unequal source trajectory counts',rows=offline))
    # First inspect predictive quality with identical frozen weights, labels and features.
    sparse=module('verified_sparse',WORK/'runs/scorer_source_audit_20261101/length_probe.py')
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    data=driver.read(WORK/'runs/scalar_downstream_20261031/data/protein.json')
    rows=[r for r in data['records'] if r['split']=='validation'];assert len(rows)==219
    predictions={str(radius):[] for radius in [None,16,32,64,128]}
    for i,row in enumerate(rows):
        scorer(row['dna']);uid=row['uid'];path=cache/'features'/f'{uid}.npz'
        if not path.exists():path=driver.PREVIOUS/'features'/f'{uid}.npz'
        with np.load(path,allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
        x,g=scorer.probe.batch([row],{uid:features},'cuda')
        for radius in [None,16,32,64,128]:
            values=[]
            for model,head,meta in scorer.models:
                model.to('cuda');head.to('cuda');context=model.backbone.context
                original_forward=context.forward
                if radius is not None:
                    context.forward=types.MethodType(lambda self,z,*args,_r=radius,**kwargs:sparse.sparse_context(self,z,_r),context)
                try:
                    with torch.no_grad():
                        pooled,_=model.representations(x)
                        stats=torch.tensor(((g-np.asarray(meta['global_mean']))/np.asarray(meta['global_scale'])).astype(np.float32),device='cuda')
                        values.append(float(head(pooled,stats)[0])*meta['y_scale']+meta['y_mean'])
                finally:context.forward=original_forward;model.cpu();head.cpu()
            predictions[str(radius)].append(values)
        save('status.json',dict(stage='sparse_validation',completed=i+1,total=219))
        if (i+1)%10==0:print('sparse validation',i+1,flush=True)
    y=np.asarray([r['target'][0] for r in rows]);metrics={}
    for radius,values in predictions.items():
        p=np.asarray(values);pred=p.mean(1)
        metrics[radius]=dict(pearson=float(np.corrcoef(y,pred)[0,1]),mse=float(np.mean((y-pred)**2)),per_seed_pearson=[float(np.corrcoef(y,p[:,j])[0,1]) for j in range(2)])
    save('sparse_validation_results.json',dict(status='completed',n=219,metrics=metrics,predictions=predictions,uids=[r['uid'] for r in rows],interpretation='Validation exploration; frozen inference only; no new training and no test read'))
    selected=[];all_rows=driver.inputs()
    for group in ['short','medium','long']:selected+=sorted([r for r in all_rows if r['group']==group],key=lambda r:r['n_codons'])[:2]
    results=[]
    for version,spec in driver.VERSIONS.items():
        payload=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        model=driver.CodonVMAE(**payload['config']['model']).eval().requires_grad_(False);model.load_state_dict(payload['model'],strict=True)
        assert driver.sha(spec['checkpoint'])==spec['expected']
        for lam in lambdas:
            def reward(value,reference,_lam=lam):
                components=base_reward(value,reference)
                components['combined_reward']=components['own_score']+(1-_lam)*sum(components[k] for k in ['gc_reward','cai_reward','mfe_reward'])
                return components
            driver.reward=reward
            out=ROOT/f'{version}_lambda{lam:g}';out.mkdir(exist_ok=True);(out/'outputs').mkdir(exist_ok=True)
            for row in selected:
                path=out/'outputs'/f'{row["sequence_id"]}.json'
                if path.exists():result=driver.read(path)
                else:
                    result=driver.one_case(model,scorer,row,version,out,20);result['lambda']=lam
                    driver.save(path,result)
                results.append(result)
                save('status.json',dict(stage='lambda_search',version=version,lam=lam,completed=len(results),total=60))
        del model
    save('lambda_search_results.json',results)
    save('complete.json',dict(status='completed',searches=len(results),sparse_validation_cases=219,unix=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as error:save('failure.json',dict(error=repr(error),unix=time.time()));raise
