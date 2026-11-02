"""Lock all checkpoints before inference, then score measured endpoints separately."""
from pathlib import Path
import json,csv,time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import train as tr
import features as ft
from model import MultiTaskTransformer,TASKS
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
def main():
    tr.setup();out=HERE/'evaluation';out.mkdir(exist_ok=True);selection=[]
    for arm in tr.ARMS:
        for seed in tr.SEEDS:
            root=HERE/'models'/f'{arm}_{seed}';complete=ft.read(root/'complete.json');p=root/'best.pt';assert ft.sha(p)==complete['best_sha256']
            selection.append(dict(arm=arm,seed=seed,path=str(p),sha256=ft.sha(p)))
    ft.save(HERE/'selection_locked.json',dict(models=selection,source_sha256=ft.sha(__file__),training_protocol_sha256=ft.sha(HERE/'training_protocol_locked.json'),test_labels_opened_by_this_run=False))
    inputs=ft.records(HERE/'test_inputs.jsonl');assert all('targets' not in r and 'standardized_targets' not in r for r in inputs)
    val=[r for r in ft.records(HERE/'train_validation.jsonl') if r['split']=='val'];tr.verify_inputs(inputs+val[:4]);bank=ft.load(inputs+val[:4]);pred={};reloads=[]
    for record in selection:
        raw=torch.load(record['path'],map_location='cpu',weights_only=True);model=MultiTaskTransformer(**raw['config']).cuda().eval();model.load_state_dict(raw['model'],strict=True)
        p=tr.predict(model,val[:4],bank);indices=[raw['validation_record_ids'].index(r['record_id']) for r in val[:4]];expect=raw['validation_predictions'][indices].numpy()
        error=float(np.max(np.abs(p-expect)));assert error<1e-4,error
        pred[record['arm']+'_'+str(record['seed'])]=tr.predict(model,inputs,bank);reloads.append(dict(**record,reload_max_error=error));del model;torch.cuda.empty_cache()
    np.savez_compressed(out/'predictions_locked.npz',record_ids=np.array([r['record_id'] for r in inputs]),**pred)
    for arm in tr.ARMS:pred[arm+'_ensemble']=np.mean([pred[arm+'_'+str(s)] for s in tr.SEEDS],axis=0)
    baseline_path=WORK/'runs/multitask_profile_20261102/evaluation/predictions_locked.npz'
    with np.load(baseline_path,allow_pickle=False) as z:
        assert z['record_ids'].tolist()==[r['record_id'] for r in inputs]
        pred['no_diffusion_ensemble']=np.mean([z['dynamics_'+str(s)] for s in tr.SEEDS],axis=0)
    # Test targets opened only after every arm/seed prediction is persisted.
    targets=ft.read(HERE/'sealed_test_targets.json');statistics=ft.read(HERE/'target_statistics.json');metrics=[];paired=[];long=[]
    baseline_expr=pd.read_csv(WORK/'runs/mrfp_functional_validation_20261102/predictions.csv').set_index('uid')
    baseline_human=pd.read_csv(WORK/'runs/joint_scalar_20261102/human_joint/test_predictions.csv').set_index('uid')
    baseline_sol=pd.read_csv(WORK/'runs/joint_scalar_20261102/esol_v2/test_predictions.csv').set_index('JW_ID')
    for task in tr.PRIMARY:
        ix=np.array([i for i,r in enumerate(inputs) if r['task']==task]);rows=[inputs[i] for i in ix];j=TASKS.index(task)
        y=np.array([targets[r['record_id']][task] for r in rows]);mu=statistics[task]['mean'];sd=statistics[task]['scale']
        scores={name:p[ix,j]*sd+mu for name,p in pred.items()}
        if task=='expression':
            for label,col in [('seq2ribo_native_expression','seq2ribo_native_mRFP_ipsc'),('previous_own424','ours_width424_mRFP_finetune')]:scores[label]=np.array([baseline_expr.loc[r['uid'],col] for r in rows])
        elif task=='half_life':
            for label,col in [('previous_HL_auxiliary','existing_HL_auxiliary__HL'),('seq2ribo_TE_cross_task','seq2ribo__TE'),('RiboDecode_TE_cross_task','RiboDecode_fixedRNA_TEproxy__TE')]:scores[label]=np.array([baseline_human.loc[r['canonical_rna_sha256'],col] for r in rows])
            # Previous auxiliary column is train-standardized; r is scale invariant.
        else:
            for label,col in [('CDS_Ridge_auxiliary','solubility_percent__cds_composition'),('AA_Ridge_auxiliary','solubility_percent__aa_composition')]:scores[label]=np.array([baseline_sol.loc[r['source_id'],col] for r in rows])
        groups=np.array([r['uid'] if task=='expression' else r['gene'] if task=='half_life' else r['protein_sha256'] for r in rows]);ug=np.unique(groups);gi=[np.flatnonzero(groups==g) for g in ug]
        rng=np.random.default_rng(20261102);draws=[np.concatenate([gi[k] for k in rng.integers(len(gi),size=len(gi))]) for _ in range(2000)];boot={}
        for name,p in scores.items():
            b=np.array([tr.corr(y[i],p[i]) for i in draws],dtype=float);boot[name]=b
            metrics.append(dict(task=task,model=name,n=len(y),bootstrap_groups=len(ug),pearson=tr.corr(y,p),ci95=np.nanquantile(b,[.025,.975]).tolist(),spearman=float(spearmanr(y,p).statistic),mse=float(np.mean((y-p)**2)) if name!='previous_HL_auxiliary' else None))
        refs=['no_diffusion_ensemble']+(['seq2ribo_native_expression','previous_own424'] if task=='expression' else ['previous_HL_auxiliary','seq2ribo_TE_cross_task','RiboDecode_TE_cross_task'] if task=='half_life' else ['CDS_Ridge_auxiliary','AA_Ridge_auxiliary'])
        for ref in refs:
            paired.append(dict(task=task,model='dynamics_ensemble',reference=ref,delta=tr.corr(y,scores['dynamics_ensemble'])-tr.corr(y,scores[ref]),ci95=np.nanquantile(boot['dynamics_ensemble']-boot[ref],[.025,.975]).tolist()))
        for k,r in enumerate(rows):long.append(dict(record_id=r['record_id'],uid=r['uid'],gene=r['gene'],task=task,observed=float(y[k]),**{name:float(p[k]) for name,p in scores.items()}))
    # Avoid claiming joint acceptance from separate samples/hosts.
    acceptance=dict(functional_expression_and_HL_same_sample_n=0,all_three_same_sample_n=0,joint_pearson=None,
        status='Not established: separately measured endpoint tests are not a paired functional-expression/HL/triple assay',
        physics_calibration='Host-specific observedRPF train-only relative-occupancy proxy; offsets are explicit hypotheses, not identified absolute kinetics',
        history='Previously opened benchmark datasets; all results exploratory')
    ft.save(out/'summary.json',dict(status='completed',metrics=metrics,paired=paired,reload_checks=reloads,acceptance=acceptance))
    pd.DataFrame(long).to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    ft.save(out/'complete.json',dict(status='completed',models=len(selection),test_n=len(inputs),summary_sha256=ft.sha(out/'summary.json'),unix=time.time()))
    print(json.dumps(dict(metrics=[r for r in metrics if r['model'].endswith('ensemble')],acceptance=acceptance),ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:ft.save(HERE/'evaluation_failure.json',dict(error=repr(e),unix=time.time()));raise
