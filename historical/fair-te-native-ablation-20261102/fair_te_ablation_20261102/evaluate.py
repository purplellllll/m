"""Freeze all arms before test predictions; independently recompute every summary."""
from pathlib import Path
import sys,json,csv,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent
sys.path.insert(0,str(HERE))
import experiment as e
from models import build,ARMS
def lock():
    metric_protocol=e.read(HERE/'metric_protocol_locked.json');assert metric_protocol['evaluator_sha256']==e.sha(__file__)
    records=[]
    for arm in ARMS:
        for seed in e.SEEDS:
            out=HERE/'models'/f'{arm}_{seed}';r=e.read(out/'complete.json')
            assert r['steps']==e.STEPS and r['seen']==2*e.STEPS
            assert e.sha(out/'best.pt')==r['checkpoint_sha256']
            records.append(r)
    e.save(HERE/'all_checkpoints_locked.json',dict(status='locked',models=records,
        protocol_sha256=e.sha(HERE/'protocol_locked.json'),source_sha256=e.sha(__file__),
        reused_test_scope='Historically opened903 test, short-CDS subset; explicitly exploratory'))
def predict(arm,seed):
    import torch
    e.configure();locked=e.read(HERE/'all_checkpoints_locked.json');assert locked['protocol_sha256']==e.sha(HERE/'protocol_locked.json')
    r=next(r for r in locked['models'] if r['arm']==arm and r['seed']==seed)
    path=HERE/'models'/f'{arm}_{seed}'/'best.pt';assert e.sha(path)==r['checkpoint_sha256']
    ck=torch.load(path,map_location='cpu',weights_only=True);model=build(arm).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    val=[r for r in e.read(HERE/'train_validation.json') if r['split']=='val'][:4];features=e.load(val)
    p=e.predictions(model,val,features,ck['target_mean'],ck['target_scale'])
    error=float(np.max(np.abs(p-np.array(ck['validation_predictions'][:4]))));assert error<2e-5
    rows=e.read(HERE/'test_inputs.json');assert all('target' not in r for r in rows)
    features=e.load(rows);pred=e.predictions(model,rows,features,ck['target_mean'],ck['target_scale'])
    out=HERE/'test_predictions';out.mkdir(exist_ok=True)
    e.save(out/f'{arm}_{seed}.json',dict(arm=arm,seed=seed,uids=[r['uid'] for r in rows],predictions=pred.tolist(),
        checkpoint_sha256=r['checkpoint_sha256'],reload_max_error=error,all_checkpoints_lock_sha256=e.sha(HERE/'all_checkpoints_locked.json')))
def summarize():
    from scipy.stats import pearsonr,spearmanr
    locked=e.read(HERE/'all_checkpoints_locked.json');records=e.read(HERE/'test_inputs.json')
    pred={};seed_metrics={}
    for arm in ARMS:
        arrays=[]
        for seed in e.SEEDS:
            r=e.read(HERE/'test_predictions'/f'{arm}_{seed}.json')
            assert r['uids']==[x['uid'] for x in records] and r['all_checkpoints_lock_sha256']==e.sha(HERE/'all_checkpoints_locked.json')
            arrays.append(np.array(r['predictions'],dtype=float));assert np.isfinite(arrays[-1]).all()
        pred[arm]=np.mean(arrays,axis=0);seed_metrics[arm]=arrays
    # Predictions for every arm are frozen before this script reads the reused test labels.
    protocol=e.read(HERE/'protocol_locked.json');assert e.sha(e.DATA/'sealed_test_targets.json')==protocol['previously_opened_test_targets_sha256']
    targets=e.read(e.DATA/'sealed_test_targets.json');y=np.array([targets[r['uid']] for r in records])
    rng=np.random.default_rng(20261102);resamples=rng.integers(len(y),size=(2000,len(y)))
    stats={};samples={}
    for arm,values in pred.items():
        p=e.corr(y,values)
        if p is not None:assert abs(p-float(pearsonr(y,values).statistic))<1e-12
        boot=np.array([np.nan if (v:=e.corr(y[i],values[i])) is None else v for i in resamples]);samples[arm]=boot
        defined=boot[np.isfinite(boot)];ci=np.quantile(defined,[.025,.975]).tolist() if len(defined)>=1900 else [None,None]
        seed_r=[e.corr(y,v) for v in seed_metrics[arm]]
        stats[arm]=dict(n=len(y),pearson=p,ci95=ci,bootstrap_defined=len(defined),spearman=float(spearmanr(y,values).statistic) if p is not None else None,
            mse=float(np.mean((y-values)**2)),single_seed_pearson=seed_r,single_seed_std=float(np.std(seed_r,ddof=1)) if all(v is not None for v in seed_r) else None,
            validation_mean_per_seed=float(np.mean([r['best_validation'] for r in locked['models'] if r['arm']==arm])))
    paired=[]
    for arm in ARMS:
        if arm=='full':continue
        delta=samples['full']-samples[arm]
        defined=delta[np.isfinite(delta)];point=stats['full']['pearson']-stats[arm]['pearson'] if stats['full']['pearson'] is not None and stats[arm]['pearson'] is not None else None
        paired.append(dict(comparison='full minus '+arm,delta_pearson=point,
            ci95=np.quantile(defined,[.025,.975]).tolist() if len(defined)>=1900 else [None,None],bootstrap_defined=len(defined),multiplicity_adjusted=False))
    coefficients={'full':1,'no_physics':-1,'no_graph':-1,'seq_transformer':1}
    interaction_point=sum(w*stats[a]['pearson'] for a,w in coefficients.items()) if all(stats[a]['pearson'] is not None for a in coefficients) else None
    interaction_boot=sum(w*samples[a] for a,w in coefficients.items());valid=interaction_boot[np.isfinite(interaction_boot)]
    interaction=dict(definition='r(full)-r(no_physics)-r(no_graph)+r(seq_transformer)',delta_on_pearson_scale=interaction_point,
        ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid)>=1900 else [None,None],bootstrap_defined=len(valid),
        interpretation='Exploratory2x2 physics/graph contrast with Transformer retained; not proof of biological synergy')
    e.save(HERE/'test_predictions_locked.json',dict(uids=[r['uid'] for r in records],target=y.tolist(),predictions={k:v.tolist() for k,v in pred.items()},
        source='Repeated exploratory test; already opened historically',checkpoint_lock_sha256=e.sha(HERE/'all_checkpoints_locked.json')))
    e.save(HERE/'summary.json',dict(status='completed',task='Human_HEK293_short_CDS_TE',metrics=stats,paired=paired,physics_graph_interaction_given_context=interaction,
        train_genes=1951,validation_genes=262,test_genes=len(y),seeds=e.SEEDS,max_steps=e.STEPS,
        primary_metric='Across-gene ensemble Pearson on same reused test rows',all_metrics_locally_computed=True,
        limits=protocol['limits']+['RiboGL and RiboDecode are explicitly TE-adapted architectures, not their original native-task evaluations',
            'Bootstrap is conditional on these trained models; two seeds do not establish broad training stability',
            'This controlled from-scratch benchmark is separate from the profile-pretrained full-length100000-step scorer'] ))
    with (HERE/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','n','pearson','ci_low','ci_high','spearman','mse','single_seed_std'])
        for a,r in stats.items():w.writerow([a,r['n'],r['pearson'],*r['ci95'],r['spearman'],r['mse'],r['single_seed_std']])
    with (HERE/'per_gene_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['uid','gene','target']+ARMS)
        for i,row in enumerate(records):w.writerow([row['uid'],row['gene'],y[i]]+[pred[a][i] for a in ARMS])
    lines=['# 人源TE同预算基线与三路消融','',f'1951训练、262验证、{len(y)}测试基因；80–300密码子；每组2种子；均从头训练3000更新、batch2、相同MSE与验证选择规则。测试来自已打开的903条子集，属于探索性复测。','',
        '|模型|Pearson|95%条件区间|Spearman|MSE|种子r标准差|','|---|---:|---|---:|---:|---:|']
    def fmt(value):return 'NA' if value is None else f'{value:.4f}'
    for arm,r in stats.items():lines.append(f"|{arm}|{fmt(r['pearson'])}|[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]|{fmt(r['spearman'])}|{fmt(r['mse'])}|{fmt(r['single_seed_std'])}|")
    lines+=['','所有组完整报告；配对差及区间见summary.json。这里匹配的是数据与优化预算，不是参数量、FLOPs或各架构最优训练配方。','',
        'RiboGL使用官方GCN_LSTM骨干加标量读出；seq2ribo使用官方Mamba与TE头、关闭sigmoid以拟合标准化连续标签，并输入模拟均值；RiboDecode使用官方CNN、固定RNA/环境条件，改以TE监督。均属于明确标注的本地任务适配。','',
        'no_physics去模拟物理通路；no_graph去GNN及结构表示通路；no_context绕过整个Transformer；seq_transformer同时去物理与图。主头不另加GC、CAI、MFE或长度统计。','',
        '恒定预测的Pearson按NA报告，不按0替代、不删除该组。若有效bootstrap少于1900/2000，则不报告该区间；完整计数保留在JSON。','',
        '此表不能与预训练骨干的人源100000步模型混作等条件排名，也不能据此宣称原论文SOTA或新数据上的独立优势。']
    (HERE/'公平基线与三路消融.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Independent arithmetic and checkpoint re-audit after assembling the table.
    saved=e.read(HERE/'test_predictions_locked.json');maxerr=0.
    for a,r in stats.items():
        value=e.corr(saved['target'],saved['predictions'][a])
        if r['pearson'] is None:assert value is None
        else:maxerr=max(maxerr,abs(value-r['pearson']))
    assert maxerr<1e-12
    for r in locked['models']:assert e.sha(HERE/'models'/f'{r["arm"]}_{r["seed"]}'/'best.pt')==r['checkpoint_sha256']
    e.save(HERE/'independent_audit.json',dict(status='passed',models=len(locked['models']),test_n=len(y),same_test_rows=True,
        identical_training_sample_budget=True,steps_per_model=e.STEPS,batch_size=2,max_pearson_recompute_error=maxerr,
        all_checkpoint_hashes_verified=True,all_arms_reported=True,independent_fresh_test=False,
        summary_sha256=e.sha(HERE/'summary.json'),predictions_sha256=e.sha(HERE/'test_predictions_locked.json')))
    e.save(HERE/'complete.json',dict(status='completed',fair_budget_baselines=True,three_branch_ablations=True,independent_audit=True))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['lock','predict','summarize']);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=30101);args=p.parse_args()
    try:
        if args.mode=='lock':lock()
        elif args.mode=='predict':predict(args.arm,args.seed)
        else:summarize()
    except BaseException as error:e.save(HERE/'evaluation_failure.json',dict(error=repr(error),mode=args.mode,arm=args.arm,seed=args.seed,unix=time.time()));raise
