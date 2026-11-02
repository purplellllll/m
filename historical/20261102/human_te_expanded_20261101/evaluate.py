"""One locked holdout, local predictions, locally computed Pearson for all models."""
from pathlib import Path
import csv
import hashlib
import math
import sys
import time
import numpy as np
import torch
from common import ROOT,WORK,REPO,PILOT,read,save,sha,module,verify_sources
sys.path.insert(0,str(PILOT))
from run_pilot import Head,ReliableGraphContext,TaskScalarModel
from train import setup_gpu,batches

def corr(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    return float(np.corrcoef(y,p)[0,1])

def main():
    protocol=verify_sources();selection=read(ROOT/'selection_locked.json');assert selection['test_predicted'] is False
    assert selection['protocol_sha256']==sha(ROOT/'protocol_locked.json')
    for row in selection['models']:assert sha(row['best_path'])==row['checkpoint_sha256']
    rows=read(ROOT/'test_inputs.json');assert len(rows)==903 and sha(ROOT/'test_inputs.json')==protocol['test_inputs_sha256']
    assert all('target' not in r for r in rows)
    save(ROOT/'test_open_receipt.json',dict(selection_sha256=sha(ROOT/'selection_locked.json'),n=903,unix=time.time(),all_checkpoints_verified=True))
    setup_gpu();probe=module('expanded_test_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py');features={}
    for row in rows:
        path=ROOT/'features'/f'{row["uid"]}.npz'
        with np.load(path,allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        assert str(f['dna_sha256'])==row['uid'];features[row['uid']]=f
    cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
    raw={r['uid']:np.array([(r['dna'].count('G')+r['dna'].count('C'))/len(r['dna']),float(features[r['uid']]['mfe'])/len(r['dna']),
        math.exp(np.mean([math.log(cai[r['dna'][j:j+3]]) for j in range(0,len(r['dna'])-3,3)]))]) for r in rows}
    predictions=[]
    for record in selection['selected_models']:
        checkpoint=torch.load(record['best_path'],map_location='cpu',weights_only=True)
        base=ReliableGraphContext(**checkpoint['config']);base.load_state_dict(checkpoint['backbone'],strict=True)
        model=TaskScalarModel(base,'te').cuda().eval().requires_grad_(False);head=Head().cuda().eval().requires_grad_(False)
        pred=np.empty(len(rows));gm=np.array(checkpoint['stats_mean']);gs=np.array(checkpoint['stats_scale'])
        with torch.no_grad():
            for indices in batches(rows):
                group=[rows[j] for j in indices];x,_=probe.batch(group,features,'cuda')
                g=np.stack([(raw[r['uid']]-gm)/gs for r in group]).astype(np.float32)
                if selection['arm']=='no_stats':g*=0
                p,_=model.representations(x);pred[indices]=(head(p[:,2],torch.tensor(g,device='cuda'))*checkpoint['target_scale']+checkpoint['target_mean']).cpu().numpy()
        predictions.append(pred);del model,head;torch.cuda.empty_cache()
    own=np.mean(predictions,axis=0);assert np.isfinite(own).all()
    save(ROOT/'own_test_predictions_locked.json',dict(uids=[r['uid'] for r in rows],predictions=own.tolist(),seed_predictions=[p.tolist() for p in predictions],selection_sha256=sha(ROOT/'selection_locked.json')))
    out=ROOT/'external_evaluation';out.mkdir(exist_ok=True)
    evaluator=module('expanded_official_evaluator',WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py');evaluator.OUT=out
    torch_cpu,_,_=evaluator.setup();reference=evaluator.source_protocol();save(out/'protocol_locked.json',reference)
    hard_model,condition=evaluator.ribodecode_model(torch_cpu);hard=[]
    for row in rows:
        path=out/f'{row["uid"]}_hard.json'
        if path.exists():cached=read(path);assert cached['dna_sha256']==row['dna_sha256']
        else:
            cached=dict(dna_sha256=row['dna_sha256'],score=evaluator.hard_score(torch_cpu,hard_model,condition,row['dna']));save(path,cached)
        hard.append(cached['score'])
    del hard_model,condition
    predictor,backend=evaluator.seq2_model();save(out/'backend.json',backend);seq=[]
    for i,row in enumerate(rows,1):
        path=out/f'{row["uid"]}_seq2ribo.json'
        if path.exists():cached=read(path);assert cached['dna_sha256']==row['dna_sha256']
        else:
            cached=dict(dna_sha256=row['dna_sha256'],**evaluator.seq2_score(predictor,row['dna']));save(path,cached)
        assert abs(np.mean(list(cached['seq2ribo_by_seed'].values()))-cached['seq2ribo_score'])<1e-10
        seq.append(cached['seq2ribo_score']);save(ROOT/'evaluation_status.json',dict(stage='seq2ribo',completed=i,total=903))
        if i%10==0:print('test seq2ribo',i,903,flush=True)
    del predictor
    # Open labels only after all predictions and candidate selection are fixed.
    assert sha(ROOT/'sealed_test_targets.json')==protocol['test_targets_sha256']
    targets=read(ROOT/'sealed_test_targets.json');y=np.array([targets[r['uid']] for r in rows]);assert np.isfinite(y).all()
    values=dict(ours=own,ribodecode=np.array(hard),seq2ribo=np.array(seq));summary={}
    groups={}
    for i,row in enumerate(rows):groups.setdefault(row['gene'],[]).append(i)
    group_indices=list(groups.values());rng=np.random.default_rng(20261101);samples=[]
    for _ in range(2000):samples.append(np.concatenate([group_indices[j] for j in rng.integers(0,len(group_indices),len(group_indices))]))
    draws={}
    for name,p in values.items():
        assert len(p)==903 and np.isfinite(p).all();draws[name]=np.array([corr(y[idx],p[idx]) for idx in samples])
        ci=np.percentile(draws[name],[2.5,97.5]);summary[name]=dict(n=903,genes=len(groups),pearson=corr(y,p),ci_low=float(ci[0]),ci_high=float(ci[1]))
    paired={name:dict(delta_pearson=summary['ours']['pearson']-summary[name]['pearson'],
        ci_low=float(np.percentile(draws['ours']-draws[name],2.5)),ci_high=float(np.percentile(draws['ours']-draws[name],97.5))) for name in ['ribodecode','seq2ribo']}
    table=[dict(uid=r['uid'],dna_sha256=r['dna_sha256'],transcript=r['transcript'],gene=r['gene'],n_codons=r['n_codons'],target=float(y[i]),
        own_score=float(own[i]),ribodecode_score=float(hard[i]),seq2ribo_score=float(seq[i]),gc_percent=100*float(raw[r['uid']][0]),mfe_per_nt=float(raw[r['uid']][1]),cai=float(raw[r['uid']][2])) for i,r in enumerate(rows)]
    save(ROOT/'test_predictions.json',table)
    with (ROOT/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(ROOT/'test_summary.json',dict(status='completed',task='human_HEK293_native_TE',selected_arm=selection['arm'],summary=summary,paired=paired,
        coefficients_computed_from_local_predictions=True,paper_coefficients_used=False,
        limitations=['seq2ribo published-weight training membership not certified','RiboDecode fixed-RNA score versus TE, not RPF accuracy','Native gene/exact protein split; no near-homology clustering','80-1500 codons, one human cell context','Unadjusted exploratory paired intervals']))
    lines=['# 扩大人源TE训练后的独立留出比较','', '训练7373／验证922／测试903；模型先按验证锁定，再预测测试。以下Pearson全部由本机逐样本预测与同一实测TE标签计算，没有使用论文相关系数。','',
        '|模型|N|Pearson|95%基因bootstrap区间|','|---|---:|---:|---|']
    for name,m in summary.items():lines.append(f"|{name}|903|{m['pearson']:.4f}|[{m['ci_low']:.4f}, {m['ci_high']:.4f}]|")
    lines+=['','seq2ribo作者历史权重训练交叉未认证，不能称三者都通过完全独立训练数据认证。RiboDecode固定RNA/环境输出仅作TE序列代理；不冒称其原生RPF评测。我们的新模型与本次测试在基因/CDS/完全相同蛋白上隔离，尚未做近同源隔离。','', '逐样本预测、三个seq2ribo种子和配对差值均有本地记录。']
    (ROOT/'独立测试比较.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Independent arithmetic recheck from the persisted table.
    persisted=read(ROOT/'test_predictions.json');labels=[r['target'] for r in persisted]
    for name,key in [('ours','own_score'),('ribodecode','ribodecode_score'),('seq2ribo','seq2ribo_score')]:assert abs(corr(labels,[r[key] for r in persisted])-summary[name]['pearson'])<1e-12
    save(ROOT/'evaluation_complete.json',dict(status='completed',n=903,all_local_prediction_pearsons_recomputed=True,paper_coefficients_used=False,
        test_predictions_sha256=sha(ROOT/'test_predictions.json'),selection_sha256=sha(ROOT/'selection_locked.json')))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(ROOT/'evaluation_failure.json',dict(error=repr(error),unix=time.time()));raise
