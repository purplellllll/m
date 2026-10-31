"""A new, bounded codon-profile experiment on measured public mouse Ribo-seq.

Uses the published sTASEP simulator with a documented train-only surrogate fit,
not published human-cell constants or an assertion of physical rate recovery.
No sequence generation or optimization is performed.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
import torch

ARMS = {
    'full': dict(physics=True, graph=True, context=True),
    'without_physics': dict(physics=False, graph=True, context=True),
    'without_gnn': dict(physics=True, graph=False, context=True),
    'without_context': dict(physics=True, graph=True, context=False),
    'transformer_only': dict(physics=False, graph=False, context=True),
    'gnn_only': dict(physics=False, graph=True, context=False),
    'physics_only': dict(physics=True, graph=False, context=False),
    'bilstm_context': dict(physics=True, graph=True, context=True, encoder='bilstm'),
}
SEEDS = [30101,30102,30103]
ORIGINAL_CODONS = [''.join(x) for x in itertools.product('ATCG',repeat=3)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path,obj):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def native(root):
    # Import the original numerical files without loading the unrelated Mamba package.
    if '_profile_seq2ribo' not in sys.modules:
        package=types.ModuleType('_profile_seq2ribo');package.__path__=[str(Path(root)/'evaluator/seq2ribo')]
        sys.modules[package.__name__]=package
    modules=[importlib.import_module('_profile_seq2ribo.'+n) for n in ['utils','geometry','simulation']]
    return modules


def structure_job(job):
    assets, sequence=job
    import RNA
    utils, geometry, _=native(assets)
    rna=sequence.replace('T','U');n=len(rna)//3
    codons=np.array([utils.CODON2IDX[rna[i:i+3]] for i in range(0,len(rna),3)],dtype=np.int64)
    structure,mfe=RNA.fold_compound(rna).mfe()
    assert len(structure)==len(rna)
    pt=RNA.ptable(structure)
    angle=geometry._compute_vertex_angles(rna,structure)
    assert angle is not None and len(angle)==len(rna),'No failed-fold zero-feature fallback'
    paired=np.array([int(pt[i+1]>0) for i in range(len(rna))]).reshape(n,3).sum(1)
    angles=np.array(angle).reshape(n,3).sum(1)
    abin=np.clip(np.digitize(angles,utils.DEFAULT_ANGLE_BINS)-1,0,3)
    position=np.arange(n);bucket=np.where(position<n//3,0,np.where(position<2*n//3,1,2))
    contacts=sorted({(min(i//3,(pt[i+1]-1)//3),max(i//3,(pt[i+1]-1)//3))
        for i in range(len(rna)) if pt[i+1]>i+1 and i//3!=(pt[i+1]-1)//3})
    edges=np.array(contacts,dtype=np.int64).reshape(-1,2).T.copy()
    degree=np.zeros(n);longrange=np.zeros(n)
    for a,b in contacts:
        degree[[a,b]]+=1
        if b-a>4:longrange[[a,b]]+=1
    continuous=np.column_stack([paired/3,abin/3,degree/max(degree.max(),1),
                               longrange/max(longrange.max(),1),position/max(n-1,1)]).astype(np.float32)
    assert np.array_equal(np.arange(n)*3,np.arange(0,len(rna),3))
    assert ''.join(utils.CODONS[i] for i in codons)==rna
    assert edges.size==0 or (edges.min()>=0 and edges.max()<n)
    return dict(cod=codons,angles=angles,paired=paired,abin=abin,bucket=bucket,edges=edges,
                structure=continuous,mfe=float(mfe),dotbracket=structure)


def simulation_job(job):
    assets,rna,feature,params,seed=job
    utils,_,simulation=native(assets)
    random.seed(seed);np.random.seed(seed)
    codons,runs=simulation.simulate_runs(rna.replace('T','U'),feature['angles'],feature['paired'],
        .01,np.array(params['waits']),params['pair'],params['angle'],params['bucket'],4,utils.DEFAULT_ANGLE_BINS)
    assert np.array_equal([utils.CODON2IDX[c] for c in codons],feature['cod'])
    mean,std=runs.mean(0),runs.std(0);scale=float(mean.mean())
    assert runs.shape==(4,len(feature['cod'])) and np.isfinite(runs).all() and scale>0
    physics=np.column_stack([np.log1p(mean/scale),np.log1p(std/scale)]).astype(np.float32)
    return dict(simulation_mean=mean,simulation_sd=std,physics=physics,simulation_seed=seed)


def panel(source,old_panel):
    from latent_decoder.prepare_data import GENETIC_CODE
    def translate(dna):return ''.join(GENETIC_CODE[dna[i:i+3]] for i in range(0,len(dna),3))
    prior=json.loads(Path(old_panel).read_text())['records']
    excluded_genes={r['gene'] for r in prior}
    excluded_dna={''.join(ORIGINAL_CODONS[int(c)] for c in r['sequence'].strip('[]').split(',')) for r in prior}
    excluded_proteins={translate(s) for s in excluded_dna}
    frame=pd.read_csv(source)
    pool=[]
    for row in frame.itertuples(index=False):
        ids=np.fromstring(row.codon_sequence.strip('[]'),sep=',',dtype=int)
        y=np.fromstring(row.annotations.strip('[]'),sep=',',dtype=float)
        if not 80<=len(ids)<=250 or len(y)!=len(ids) or (ids<0).any() or (ids>=64).any():continue
        finite=np.isfinite(y)
        if finite.mean()<.8 or (y[finite]<0).any() or np.mean(y[finite]>0)<.3 or np.ptp(y[finite])==0:continue
        longest=current=0
        for value in y:
            current=current+1 if value==0 else 0;longest=max(longest,current)
        if longest>20:continue
        gene=str(row.gene).split('.')[0];tx=str(row.transcript)
        dna=''.join(ORIGINAL_CODONS[i] for i in ids)
        if not dna.startswith('ATG') or dna[-3:] not in {'TAA','TGA','TAG'}:continue
        protein=translate(dna)
        if '*' in protein[:-1] or gene in excluded_genes or dna in excluded_dna or protein in excluded_proteins:continue
        rank=hashlib.sha256(('aligned_profile_20261030:'+gene+':'+tx).encode()).hexdigest()
        pool.append(dict(gene=gene,transcript=tx,dna=dna,protein_sha256=hashlib.sha256(protein.encode()).hexdigest(),
            source_ids=ids.tolist(),y=[float(v) if np.isfinite(v) else None for v in y],rank=rank))
    pool.sort(key=lambda r:r['rank']);rows=[];genes=set();proteins=set();sequences=set()
    for r in pool:
        if r['gene'] in genes or r['protein_sha256'] in proteins or r['dna'] in sequences:continue
        genes.add(r['gene']);proteins.add(r['protein_sha256']);sequences.add(r['dna']);rows.append(r)
        if len(rows)==600:break
    assert len(rows)==600,f'Only {len(rows)} eligible unique records'
    for i,r in enumerate(rows):r['split']='train' if i<400 else 'validation' if i<480 else 'test'
    return dict(species='Mus musculus',tissue='liver',study='GSE73553',source_sha256=sha(source),
        source_doi='10.5281/zenodo.17100113',source_records=len(frame),eligible_records=len(pool),
        prior_300_genes_and_identical_proteins_excluded=True,near_homology_isolation=False,
        label='published normalized codon annotations; primary PCC on log1p, raw-normalized PCC secondary',
        coordinate='codon-index0 k maps to CDS nucleotide-index0 [3k,3k+3); no inferred coordinate shifts',
        absolute_rpf_or_te_available=False,counts=dict(train=400,validation=80,test=120),records=rows)


def fit_physics(rows,features):
    gram=np.zeros((75,75));rhs=np.zeros(75);n=0
    for r,f in zip(rows,features):
        if r['split']!='train':continue
        y=np.array([np.nan if v is None else v for v in r['y']]);mask=np.isfinite(y)
        target=np.clip(y[mask]/y[mask].mean(),0,10)
        design=np.zeros((len(y),75));pos=np.arange(len(y))
        for columns in [f['cod'],64+f['paired'],68+f['abin'],72+f['bucket']]:design[pos,columns]=1
        a=design[mask];gram+=a.T@a;rhs+=a.T@target;n+=len(target)
    prior=np.r_[np.ones(64),np.zeros(11)];ridge=25.
    gram+=ridge*np.eye(75);rhs+=ridge*prior
    chol=np.linalg.cholesky(gram)
    weights,residual=nnls(chol.T,np.linalg.solve(chol,rhs),maxiter=10000)
    scale=max(weights[:64].mean(),1e-6);weights/=scale
    return dict(waits=np.maximum(weights[:64],.01).tolist(),pair=weights[64:68].tolist(),
        angle=weights[68:72].tolist(),bucket=weights[72:].tolist(),ridge=ridge,n_train_positions=n,
        method='nonnegative additive dwell proxy fitted only to train relative occupancy; not kinetic rate recovery',
        published_simulator_unmodified=True,published_human_constants_used=False,n_simulations=4,init_p=.01)


def collate(examples):
    b=len(examples);length=max(len(e['cod']) for e in examples)
    cod=torch.full((b,length),64,dtype=torch.long);valid=torch.zeros(b,length,dtype=torch.bool)
    phy=torch.zeros(b,length,2);struct=torch.zeros(b,length,5);target=torch.zeros(b,length)
    observed=torch.zeros(b,length,dtype=torch.bool);edges=[]
    for i,e in enumerate(examples):
        n=len(e['cod']);cod[i,:n]=torch.from_numpy(e['cod']);valid[i,:n]=True
        phy[i,:n]=torch.from_numpy(e['physics']);struct[i,:n]=torch.from_numpy(e['structure'])
        target[i,:n]=torch.from_numpy(e['label']);observed[i,:n]=torch.from_numpy(e['observed'])
        edges.append(torch.from_numpy(e['edges'])+i*length)
    return (cod,valid,phy,struct,torch.cat(edges,1)),target,observed


@torch.no_grad()
def predict(model,examples,local=False):
    model.eval();out=[];branch={}
    for start in range(0,len(examples),8):
        batch=examples[start:start+8];inputs,_,_=collate(batch);p,aux=model(*inputs)
        for i,e in enumerate(batch):
            out.append(p[i,:len(e['cod'])].numpy().copy())
            if local:
                for key,value in aux.items():branch.setdefault(key,[]).append(value[i,:len(e['cod'])].numpy().copy())
    return (out,branch) if local else out


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(pearsonr(a,b).statistic)


def evaluate(out,rows,examples,predictions):
    per=[];summary=[];archive={};rng=np.random.default_rng(301030)
    boot=rng.integers(0,len(rows),(2000,len(rows)))
    for name,pp in predictions.items():
        ys=[];ps=[];rr=[];raw_rr=[];interior=[];mse=[]
        for i,(r,e,p) in enumerate(zip(rows,examples,pp)):
            mask=e['observed'];y=e['label'];raw=np.array([np.nan if v is None else v for v in r['y']])
            q=correlation(y[mask],p[mask]);rawq=correlation(raw[mask],np.expm1(np.clip(p[mask],0,30)))
            inside=mask.copy();inside[:5]=False;inside[-5:]=False
            iq=correlation(y[inside],p[inside]);rr.append(np.nan if q is None else q)
            raw_rr.append(np.nan if rawq is None else rawq);interior.append(np.nan if iq is None else iq)
            mse.append(float(np.mean((y[mask]-p[mask])**2)))
            per.append(dict(model=name,gene=r['gene'],transcript=r['transcript'],n_codons=len(y),n_observed=int(mask.sum()),
                pearson_log=q,pearson_raw_normalized=rawq,pearson_interior_log=iq,mse_log=mse[-1]))
            archive[f'{name}__{i}']=p
            ys.extend(y[mask]);ps.extend(p[mask])
        rr=np.array(rr);finite=np.isfinite(rr)
        ci=np.nanpercentile(np.nanmean(rr[boot],1),[2.5,97.5]) if finite.any() else [None,None]
        summary.append(dict(model=name,n_test=len(rows),n_defined=int(finite.sum()),mean_transcript_pcc_log=float(np.nanmean(rr)),
            median_transcript_pcc_log=float(np.nanmedian(rr)),ci_low=ci[0],ci_high=ci[1],
            mean_transcript_pcc_raw_normalized=float(np.nanmean(raw_rr)),mean_interior_pcc_log=float(np.nanmean(interior)),
            pooled_position_pcc_log=correlation(ys,ps),mean_transcript_mse_log=float(np.mean(mse))))
    for i,e in enumerate(examples):
        archive[f'observed_log__{i}']=e['label'];archive[f'observed_mask__{i}']=e['observed']
        archive[f'codon_ids__{i}']=e['cod'];archive[f'sim_mean__{i}']=e['simulation_mean']
    np.savez_compressed(out/'test_predictions.npz',**archive)
    pd.DataFrame(per).to_csv(out/'per_transcript_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
    differences=[]
    full=np.array([x['pearson_log'] for x in per if x['model']=='full_ensemble'],dtype=float)
    for name in ['without_physics','without_gnn','without_context','transformer_only','gnn_only','physics_only','bilstm_context']:
        other=np.array([x['pearson_log'] for x in per if x['model']==name+'_ensemble'],dtype=float)
        delta=full-other;ci=np.nanpercentile(np.nanmean(delta[boot],1),[2.5,97.5])
        differences.append(dict(comparison='full minus '+name,delta_mean_pcc=float(np.nanmean(delta)),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(differences).to_csv(out/'paired_ablation_differences.csv',index=False)
    save(out/'metric_protocol.json',dict(primary='equal-transcript arithmetic mean of within-CDS PCC on log1p supplied normalized occupancy',
        secondary=['same positions on raw-normalized scale','pooled positions log1p, length weighted','trim 5 codons at each end, log1p'],
        missing='source NaN excluded, measured zero retained; no position-shift search',
        bootstrap='2000 paired transcript resamples; unadjusted exploratory intervals',
        undefined='reported and excluded from defined-PCC mean; not replaced with zero',
        tx_level_total_pcc=None,te_pcc=None,reason='source labels are normalized occupancy, no absolute counts or paired RNA abundance',
        seed_variation='all 3 seed metrics separately; ensemble is arithmetic mean of log predictions',
        generalization_limits=['single study/tissue','short, adequately covered CDS','no near-homology isolation','surrogate physics fit']))
    return summary


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'status.json').exists(),'Choose a fresh output directory'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.nice(10)
    started=time.monotonic()
    def progress(stage,**fields):save(out/'status.json',dict(stage=stage,seconds=time.monotonic()-started,**fields))
    def deadline(*_):raise TimeoutError('Predeclared one-hour workflow deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(3600)
    try:
        sys.path.insert(0,args.assets);sys.path.insert(0,str(out))
        from physics_scorer.profile_model import AlignedProfileModel,masked_transcript_mse
        source=Path(args.ribogl)/'related_liver/LIVER.csv'
        receipt=json.loads((source.parent/'receipt.json').read_text());assert sha(source)==receipt['sha256']
        p=panel(source,Path(args.ribogl)/'liver_pilot/frozen_panel.json');save(out/'panel.json',p);rows=p['records']
        save(out/'run_manifest.json',dict(driver_sha256=sha(__file__),model_sha256=sha(out/'physics_scorer/profile_model.py'),
            graph_layer_sha256=sha(out/'physics_scorer/model.py'),panel_sha256=sha(out/'panel.json'),seeds=SEEDS,arms=ARMS,
            epochs_max=18,patience=4,batch_size=8,width=32,optimizer='AdamW lr=0.001 wd=0.01, gradclip1',
            loss='equal transcript MSE log1p plus 0.1 times mean enabled-branch auxiliary MSE',
            device='cpu',threads_per_process=1,feature_workers=2,torch=torch.__version__,
            simulator_hashes={n:sha(Path(args.assets)/'evaluator/seq2ribo'/n) for n in ['simulation.py','geometry.py','utils.py']},
            test_selection='all checkpoint choices locked before any test prediction; panel fixed before training',
            scope='new benchmark on public measured normalized mouse Ribo-seq; not original paper replication'))
        progress('structure',completed=0,total=len(rows));features=[]
        context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=2,mp_context=context) as pool:
            for i,f in enumerate(pool.map(structure_job,[(args.assets,r['dna']) for r in rows])):
                features.append(f)
                if (i+1)%20==0:progress('structure',completed=i+1,total=len(rows))
        params=fit_physics(rows,features);save(out/'physics_parameters_train_only.json',params)
        seeds=[int(hashlib.sha256(('profile_sim:'+r['gene']).encode()).hexdigest()[:8],16) for r in rows]
        progress('simulation',completed=0,total=len(rows))
        jobs=[(args.assets,r['dna'],f,params,s) for r,f,s in zip(rows,features,seeds)]
        with ProcessPoolExecutor(max_workers=2,mp_context=context) as pool:
            for i,sim in enumerate(pool.map(simulation_job,jobs)):
                features[i].update(sim)
                if (i+1)%10==0:progress('simulation',completed=i+1,total=len(rows))
        (out/'features').mkdir();alignment=[];graph_metrics=[]
        for i,(r,f) in enumerate(zip(rows,features)):
            y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
            f['observed']=np.isfinite(y);f['label']=np.log1p(np.nan_to_num(y,nan=0)).astype(np.float32)
            assert len(y)==len(f['cod'])==len(f['physics'])==len(f['structure'])
            assert r['dna']==''.join(ORIGINAL_CODONS[c] for c in r['source_ids'])
            assert ''.join(native(args.assets)[0].CODONS[c] for c in f['cod'])==r['dna'].replace('T','U')
            np.savez_compressed(out/'features'/f'{r["gene"]}.npz',**f,codon_start_nt0=np.arange(len(y))*3)
            alignment.append(dict(gene=r['gene'],transcript=r['transcript'],split=r['split'],n_codons=len(y),
                n_observed=int(f['observed'].sum()),coordinate_roundtrip=True,sequence_sha256=hashlib.sha256(r['dna'].encode()).hexdigest()))
            graph_metrics.append(dict(gene=r['gene'],split=r['split'],n_structure_contacts=f['edges'].shape[1],
                paired_fraction=float(f['paired'].mean()/3),mfe=f['mfe'],mfe_per_nt=f['mfe']/len(r['dna']),
                simulation_event_sum=float(f['simulation_mean'].sum()),simulation_mean_sd=float(f['simulation_sd'].mean())))
        pd.DataFrame(alignment).to_csv(out/'alignment_audit.csv',index=False)
        pd.DataFrame(graph_metrics).to_csv(out/'graph_simulation_metrics.csv',index=False)
        split={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','validation','test']}
        train=[features[i] for i in split['train']];val=[features[i] for i in split['validation']]
        (out/'checkpoints').mkdir();locks=[]
        for name,config in ARMS.items():
            for seed in SEEDS:
                torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                model=AlignedProfileModel(**config);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                best=float('inf');stale=0;best_state=None;best_epoch=0
                for epoch in range(18):
                    model.train();order=np.random.default_rng(seed+epoch).permutation(len(train));losses=[]
                    for start in range(0,len(train),8):
                        batch=[train[i] for i in order[start:start+8]];inputs,y,mask=collate(batch)
                        optimizer.zero_grad(set_to_none=True);pred,aux=model(*inputs)
                        loss=masked_transcript_mse(pred,y,mask)+.1*torch.stack([masked_transcript_mse(v,y,mask) for v in aux.values()]).mean()
                        assert torch.isfinite(loss);loss.backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                        optimizer.step();losses.append(float(loss.detach()))
                    pv=predict(model,val)
                    vmse=float(np.mean([np.mean((a[e['observed']]-e['label'][e['observed']])**2) for a,e in zip(pv,val)]))
                    event=dict(arm=name,seed=seed,epoch=epoch+1,train_loss=float(np.mean(losses)),validation_mse=vmse)
                    with (out/'training_history.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    if vmse<best:
                        best=vmse;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
                    else:stale+=1
                    progress('training',**event,finished_models=len(locks),total_models=24)
                    if stale>=4:break
                path=out/'checkpoints'/f'{name}_{seed}.pt'
                torch.save(dict(state_dict=best_state,config=config),path)
                locks.append(dict(arm=name,seed=seed,path=str(path),sha256=sha(path),best_epoch=best_epoch,
                    validation_mse=best,parameters=sum(v.numel() for v in model.parameters())))
                print('trained',name,seed,best_epoch,round(best,5),flush=True)
        save(out/'checkpoints_locked.json',dict(models=locks,panel_sha256=sha(out/'panel.json')))
        progress('test',models=24);test=[features[i] for i in split['test']];testrows=[rows[i] for i in split['test']]
        predictions={}
        for lock in locks:
            assert sha(lock['path'])==lock['sha256']
            ckpt=torch.load(lock['path'],map_location='cpu',weights_only=True)
            model=AlignedProfileModel(**ckpt['config']);model.load_state_dict(ckpt['state_dict'])
            if lock['arm']=='full':
                pred,local=predict(model,test,local=True)
                predictions[f'gnn_aux_{lock["seed"]}']=local['gnn']
            else:pred=predict(model,test)
            predictions[f'{lock["arm"]}_{lock["seed"]}']=pred
        for arm in ARMS:
            predictions[arm+'_ensemble']=[np.mean([predictions[f'{arm}_{s}'][i] for s in SEEDS],axis=0) for i in range(len(test))]
        predictions['gnn_aux_ensemble']=[np.mean([predictions[f'gnn_aux_{s}'][i] for s in SEEDS],axis=0) for i in range(len(test))]
        sums=np.zeros(64);counts=np.zeros(64)
        for e in train:
            m=e['observed'];np.add.at(sums,e['cod'][m],e['label'][m]);np.add.at(counts,e['cod'][m],1)
        baseline=np.divide(sums,counts,out=np.full(64,sums.sum()/counts.sum()),where=counts>0)
        predictions['codon_train_mean']=[baseline[e['cod']] for e in test]
        predictions['stasep_unpolished']=[np.log1p(e['simulation_mean']/e['simulation_mean'].mean()) for e in test]
        summary=evaluate(out,testrows,test,predictions)
        save(out/'summary.json',dict(seconds=time.monotonic()-started,n_train=400,n_validation=80,n_test=120,
            n_frozen_models=len(locks),summary=summary))
        progress('completed',models=len(locks),test_transcripts=len(test));signal.alarm(0)
        print('COMPLETED',round(time.monotonic()-started,1),flush=True)
    except BaseException as exc:
        progress('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--assets',required=True)
    parser.add_argument('--ribogl',required=True);main(parser.parse_args())
