"""Bounded mRFP frozen-encoder branch probes; test is a separate locked phase."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.scalar_downstream import TaskScalarModel

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/scalar_downstream_20261031')
PROFILE=Path('/root/riboseq_graph_v2_20261031/runs')
CANDIDATES=['gated_graph','sequence_graph']
SEEDS=[30101,30102]
ARMS={'stats3':((0,0,0),3),'stats6':((0,0,0),6),
      'p':((1,0,0),3),'g':((0,1,0),3),'c':((0,0,1),3),
      'pg':((1,1,0),3),'pc':((1,0,1),3),'gc':((0,1,1),3),
      'pgc':((1,1,1),3),'pgc_stats6':((1,1,1),6)}
STEPS=1000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def metrics(y,p):return dict(pearson=corr(y,p),mse=float(np.mean((y-p)**2)),n=len(y))


class Probe(nn.Module):
    def __init__(self,mask,nstats,width=192):
        super().__init__();self.mask=mask;self.nstats=nstats
        self.norms=nn.ModuleList([nn.LayerNorm(2*width) for _ in range(3)])
        self.readout=nn.Sequential(nn.Linear(6*width+6,256),nn.GELU(),nn.Dropout(.1),
                                   nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,p,g):
        stats=torch.cat((g[:,:self.nstats],torch.zeros_like(g[:,self.nstats:])),1)
        return self.readout(torch.cat([norm(p[:,i])*self.mask[i] for i,norm in enumerate(self.norms)]+[stats],1)).squeeze(1)


def selfcheck():
    torch.manual_seed(31);p=torch.rand(3,3,384);g=torch.rand(3,6)
    for mask,nstats in ARMS.values():
        model=Probe(mask,nstats).eval();before=model(p,g)
        p2=p.clone();g2=g.clone();g2[:,nstats:]+=50
        for i,on in enumerate(mask):
            if not on:p2[:,i]+=50
        torch.testing.assert_close(before,model(p2,g2),atol=0,rtol=0)
    print('PASS: all probe masks and global-statistics exclusions are invariant',flush=True)


def protocol():
    return dict(task='mRFP measured protein expression scalar',candidates=CANDIDATES,seeds=SEEDS,
        backbone_training='Frozen completed LIVER profile checkpoints, no scalar tuning of encoders',
        arms=ARMS,steps=STEPS,validation_interval=100,batch_size=64,optimizer='AdamW',lr=.0003,weight_decay=.01,
        normalization='target and global statistics fit on clean train only; per-branch LayerNorm',
        selection='Highest mean per-seed validation Pearson; ties arm then candidate lexical; no test selection',
        ensemble='Average both paired backbone/readout seed predictions; single seed values also reported',
        fixed_test_scope='All predefined arms reported, primary candidate pgc vs c/gc/pc/pg and stats3; selected arm separately marked',
        test_status='Previously opened historical benchmark; no independent confirmation or SOTA claim',
        interpretation='Readout conditional increment, not full encoder ablation; context already contains graph/physics',
        feature_domain='Existing four-run mouse-proxy sTASEP; domain transfer not calibrated',
        max_total_optimizer_steps=40000,include_test_during_training=False,
        data_sha256=sha(DATA/'data/protein.json'),features_lock_sha256=sha(DATA/'protein_features_locked.json'),
        driver_sha256=sha(__file__),source_sha256={p.name:sha(p) for p in (ROOT/'physics_scorer').glob('*.py')},
        checkpoints={f'{c}_{s}':sha(PROFILE/f'{c}_{s}/best.pt') for c in CANDIDATES for s in SEEDS})


def rows_for(split):
    data=read(DATA/'data/protein.json')
    return [r for r in data['records'] if r['split']==split and not (split=='train' and r.get('exclude_from_clean_fit',False))]


def features(rows):
    lock=read(DATA/'protein_features_locked.json');assert lock['task_sha256']==sha(DATA/'data/protein.json')
    result={}
    for r in rows:
        if r['uid'] in result:continue
        path=DATA/'features'/f'{r["uid"]}.npz';assert sha(path)==lock['files'][r['uid']]
        with np.load(path,allow_pickle=False) as values:result[r['uid']]={k:values[k].copy() for k in values.files}
    return result


def batch(rows,fs,device):
    fs=[fs[r['uid']] for r in rows];b=len(rows);length=max(len(f['cod']) for f in fs)
    cod=torch.full((b,length),64,dtype=torch.long);valid=torch.zeros(b,length,dtype=torch.bool)
    phys=torch.zeros(b,length,2);struct=torch.zeros(b,length,5);edges=[];globals=[]
    for i,(r,f) in enumerate(zip(rows,fs)):
        n=len(f['cod']);assert n*3==len(r['dna'])
        assert str(f['dna_sha256'])==r['uid']
        cod[i,:n]=torch.from_numpy(f['cod']);valid[i,:n]=True
        phys[i,:n]=torch.from_numpy(f['physics']);struct[i,:n]=torch.from_numpy(f['structure'])
        edges.append(torch.from_numpy(f['edges'])+i*length)
        mean=f['simulation_mean'];sd=f['simulation_sd'];dna=r['dna']
        globals.append([np.log1p(n)/8,(dna.count('G')+dna.count('C'))/len(dna),float(f['mfe'])/len(dna),
                        np.log1p(mean.mean())/8,np.log1p(sd.mean())/8,sd.mean()/max(float(mean.mean()),1e-8)])
    x=tuple(t.to(device) for t in [cod,valid,phys,struct,torch.cat(edges,1)])
    return x,np.asarray(globals,dtype=np.float32)


@torch.no_grad()
def encode(candidate,seed,split,rows,fs):
    cache=ROOT/'cache'/f'{candidate}_{seed}_{split}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            assert z['uids'].tolist()==[r['uid'] for r in rows]
            return z['pooled'].copy(),z['globals'].copy()
    payload=torch.load(PROFILE/f'{candidate}_{seed}/best.pt',map_location='cpu',weights_only=False)
    setting=dict(payload['setting']);setting.pop('kind',None)
    backbone=ReliableGraphContext(**setting);backbone.load_state_dict(payload['state_dict'],strict=True)
    model=TaskScalarModel(backbone,'protein').cuda().eval().requires_grad_(False)
    pooled=[];global_values=[]
    for start in range(0,len(rows),16):
        x,g=batch(rows[start:start+16],fs,'cuda');p,_=model.representations(x)
        pooled.append(p.cpu().numpy());global_values.append(g)
    p=np.concatenate(pooled);g=np.concatenate(global_values)
    np.savez_compressed(cache,pooled=p,globals=g,uids=np.asarray([r['uid'] for r in rows]))
    del model;torch.cuda.empty_cache()
    return p,g


def training():
    assert not (ROOT/'selection_locked.json').exists(),'Already locked; do not retrain this run'
    spec=protocol();save(ROOT/'protocol_locked.json',spec)
    (ROOT/'cache').mkdir(exist_ok=True);(ROOT/'heads').mkdir(exist_ok=True)
    train=rows_for('train');val=rows_for('validation');assert (len(train),len(val))==(1019,219)
    fs=features(train+val);y=np.asarray([r['target'][0] for r in train],dtype=np.float32)
    vy=np.asarray([r['target'][0] for r in val],dtype=np.float32)
    ym=float(y.mean());ys=float(y.std());target=torch.from_numpy((y-ym)/ys).cuda()
    results=[];predictions={};begun=time.monotonic()
    for candidate in CANDIDATES:
        for seed in SEEDS:
            p,g=encode(candidate,seed,'train',train,fs);vp,vg=encode(candidate,seed,'validation',val,fs)
            gm=g.mean(0);gs=g.std(0);gs[gs<1e-6]=1
            pg=torch.from_numpy(p).cuda();gg=torch.from_numpy((g-gm)/gs).cuda()
            vpg=torch.from_numpy(vp).cuda();vgg=torch.from_numpy((vg-gm)/gs).cuda()
            for arm,(mask,nstats) in ARMS.items():
                name=f'{candidate}_{seed}_{arm}';path=ROOT/'heads'/f'{name}.pt'
                assert not path.exists(),f'Existing partial run {name}; inspect rather than overwrite'
                torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
                model=Probe(mask,nstats).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
                best=-float('inf');history=[]
                for step in range(1,STEPS+1):
                    model.train();ix=torch.from_numpy(rng.integers(0,len(train),64)).cuda()
                    optimizer.zero_grad(set_to_none=True);loss=(model(pg[ix],gg[ix])-target[ix]).square().mean()
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if step%100==0:
                        model.eval()
                        with torch.no_grad():v=model(vpg,vgg).cpu().numpy()*ys+ym
                        score=corr(vy,v);assert score is not None
                        history.append(dict(step=step,validation_pearson=score,train_loss=float(loss.detach())))
                        if score>best:
                            best=score;best_step=step;best_prediction=v.copy()
                            best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
                payload=dict(state_dict=best_state,mask=mask,nstats=nstats,y_mean=ym,y_scale=ys,
                    global_mean=gm.tolist(),global_scale=gs.tolist(),candidate=candidate,seed=seed,arm=arm,
                    best_step=best_step,validation_pearson=best,protocol_sha256=sha(ROOT/'protocol_locked.json'))
                torch.save(payload,path)
                item=dict(candidate=candidate,seed=seed,arm=arm,best_step=best_step,validation=metrics(vy,best_prediction),
                          path=str(path),sha256=sha(path),history=history)
                results.append(item);predictions[name]=best_prediction
                save(ROOT/'progress.json',dict(stage='training',completed=len(results),total=40,last=item,seconds=time.monotonic()-begun))
                print(json.dumps(dict(completed=len(results),candidate=candidate,seed=seed,arm=arm,validation=best)),flush=True)
                del model,optimizer
    summary=[]
    for candidate in CANDIDATES:
        for arm in ARMS:
            subset=[r for r in results if r['candidate']==candidate and r['arm']==arm]
            average=float(np.mean([r['validation']['pearson'] for r in subset]))
            ensemble=np.mean([predictions[f'{candidate}_{s}_{arm}'] for s in SEEDS],0)
            summary.append(dict(candidate=candidate,arm=arm,mean_seed_validation_pearson=average,ensemble_validation=metrics(vy,ensemble)))
    selected=sorted(summary,key=lambda r:(-r['mean_seed_validation_pearson'],r['arm'],r['candidate']))[0]
    per_candidate={c:sorted([r for r in summary if r['candidate']==c],key=lambda r:(-r['mean_seed_validation_pearson'],r['arm']))[0] for c in CANDIDATES}
    save(ROOT/'validation_results.json',dict(results=results,summary=summary,selection=selected,per_candidate_selection=per_candidate))
    np.savez_compressed(ROOT/'validation_predictions.npz',observed=vy,**predictions)
    save(ROOT/'selection_locked.json',dict(status='locked_before_test',selected=selected,per_candidate_selection=per_candidate,
        protocol_sha256=sha(ROOT/'protocol_locked.json'),validation_results_sha256=sha(ROOT/'validation_results.json'),
        heads={r['path']:r['sha256'] for r in results},test_metrics_computed=False,unix=time.time()))
    save(ROOT/'status.json',dict(stage='validation_complete_selection_locked',n_heads=40,seconds=time.monotonic()-begun))


def test():
    lock=read(ROOT/'selection_locked.json');assert lock['status']=='locked_before_test'
    assert lock['protocol_sha256']==sha(ROOT/'protocol_locked.json')
    assert lock['validation_results_sha256']==sha(ROOT/'validation_results.json')
    assert not (ROOT/'test_results.json').exists(),'Test already reported'
    for path,digest in lock['heads'].items():assert sha(path)==digest
    rows=rows_for('test');assert len(rows)==219;fs=features(rows)
    y=np.asarray([r['target'][0] for r in rows],dtype=np.float64)
    training_rows=read(DATA/'data/protein.json')['records']
    overlap={r['uid'] for r in training_rows if r.get('exclude_from_clean_fit',False)}
    clean=np.asarray([r['uid'] not in overlap for r in rows]);assert clean.sum()==217
    predictions={};summaries=[]
    for candidate in CANDIDATES:
        for seed in SEEDS:
            p,g=encode(candidate,seed,'test',rows,fs);pt=torch.from_numpy(p).cuda()
            for arm in ARMS:
                name=f'{candidate}_{seed}_{arm}';payload=torch.load(ROOT/'heads'/f'{name}.pt',map_location='cpu',weights_only=True)
                model=Probe(payload['mask'],payload['nstats']).cuda().eval();model.load_state_dict(payload['state_dict'],strict=True)
                gt=torch.from_numpy(((g-np.asarray(payload['global_mean']))/np.asarray(payload['global_scale'])).astype(np.float32)).cuda()
                with torch.no_grad():pred=model(pt,gt).cpu().numpy()*payload['y_scale']+payload['y_mean']
                assert np.isfinite(pred).all();predictions[name]=pred
                summaries.append(dict(candidate=candidate,seed=seed,arm=arm,test219=metrics(y,pred),test217=metrics(y[clean],pred[clean])))
                del model
    ensembles=[];comparisons=[];rng=np.random.default_rng(301031)
    boots={n:rng.integers(0,n,(2000,n)) for n in [219,217]}
    def bootcorr(a,b,ix):
        aa=a[ix];bb=b[ix];aa=aa-aa.mean(1,keepdims=True);bb=bb-bb.mean(1,keepdims=True)
        return (aa*bb).sum(1)/np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1)).clip(1e-20)
    for candidate in CANDIDATES:
        for arm in ARMS:
            pred=np.mean([predictions[f'{candidate}_{s}_{arm}'] for s in SEEDS],0)
            predictions[f'{candidate}_ensemble_{arm}']=pred
            ensembles.append(dict(candidate=candidate,arm=arm,test219=metrics(y,pred),test217=metrics(y[clean],pred[clean])))
        for reference in ['stats3','stats6','p','g','c','pg','pc','gc']:
            for tag,keep in [('219',np.ones(219,dtype=bool)),('217',clean)]:
                a=predictions[f'{candidate}_ensemble_pgc'][keep];b=predictions[f'{candidate}_ensemble_{reference}'][keep]
                actual=y[keep];d=bootcorr(actual,a,boots[len(actual)])-bootcorr(actual,b,boots[len(actual)])
                comparisons.append(dict(candidate=candidate,comparison='pgc minus '+reference,cohort=tag,
                    delta_pearson=corr(actual,a)-corr(actual,b),ci95=np.percentile(d,[2.5,97.5]).tolist(),
                    interpretation='Exploratory paired variant bootstrap, conditional on the two trained seeds; not multiplicity adjusted'))
    cross=[]
    for arm in ARMS:
        for tag,keep in [('219',np.ones(219,dtype=bool)),('217',clean)]:
            a=predictions[f'gated_graph_ensemble_{arm}'][keep];b=predictions[f'sequence_graph_ensemble_{arm}'][keep];actual=y[keep]
            delta=bootcorr(actual,a,boots[len(actual)])-bootcorr(actual,b,boots[len(actual)])
            cross.append(dict(arm=arm,cohort=tag,delta_pearson=corr(actual,a)-corr(actual,b),ci95=np.percentile(delta,[2.5,97.5]).tolist()))
    np.savez_compressed(ROOT/'test_predictions.npz',observed=y,clean=clean,uids=np.asarray([r['uid'] for r in rows]),**predictions)
    save(ROOT/'test_results.json',dict(status='completed',single_seeds=summaries,ensembles=ensembles,
        paired_branch_differences=comparisons,paired_candidate_differences=cross,validation_selection=lock['selected'],
        selection_lock_sha256=sha(ROOT/'selection_locked.json'),predictions_sha256=sha(ROOT/'test_predictions.npz'),
        n_test=219,n_clean_test=217,test_previously_opened=True,claims='Frozen representation transfer and readout increments only'))
    save(ROOT/'status.json',dict(stage='completed',n_heads=40,n_test=219,n_clean_test=217,unix=time.time()))
    print('TEST COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['selfcheck','train','test']);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.phase=='selfcheck':selfcheck()
    else:
        assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.35)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            if args.phase=='train':training()
            else:test()
        except BaseException as error:
            save(ROOT/'failure.json',dict(phase=args.phase,error=repr(error),unix=time.time()));raise
