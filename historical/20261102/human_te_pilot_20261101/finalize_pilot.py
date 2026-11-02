"""Bounded follow-through: validation baselines and independent human-head audit."""
from pathlib import Path
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.ptp(b)>0 else 0.

def main():
    deadline=time.monotonic()+8*3600
    while not (ROOT/'complete.json').exists():
        if (ROOT/'failure.json').exists():raise RuntimeError(read(ROOT/'failure.json'))
        if time.monotonic()>deadline:raise TimeoutError('Eight-hour pilot deadline reached')
        time.sleep(5)
    rows=read(ROOT/'selected_development_rows.json');train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    y=np.array([r['target'] for r in val]);yt=np.array([r['target'] for r in train]);results=read(ROOT/'results.json')
    assert len(results)==12 and len(train)==256 and len(val)==96
    summary=[]
    for arm in sorted({r['arm'] for r in results}):
        subset=[r for r in results if r['arm']==arm];assert len(subset)==2
        for r in subset:
            assert abs(corr(y,r['validation_predictions'])-r['validation_pearson'])<1e-6
            assert abs(max(z['validation_pearson'] for z in r['history'])-r['validation_pearson'])<1e-12
            p=ROOT/'checkpoints'/f'{arm}_{r["seed"]}.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
        p=np.array([r['validation_predictions'] for r in subset]).mean(0)
        summary.append(dict(arm=arm,mean_seed_validation_pearson=statistics.mean(r['validation_pearson'] for r in subset),ensemble_validation_pearson=corr(y,p),mse=float(np.mean((y-p)**2)),parameters=subset[0]['parameters']))
    chosen=max(summary,key=lambda r:r['mean_seed_validation_pearson'])
    save(ROOT/'selection_locked.json',dict(arm=chosen['arm'],rule='Highest mean per-seed validation Pearson; development only; not full deployment certification',test_predicted=False))
    # Official native predictor for descriptive comparison; no claim about training membership.
    source=WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py'
    spec=importlib.util.spec_from_file_location('locked_human_baseline',source);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);e.OUT=ROOT/'baselines';e.OUT.mkdir(exist_ok=True)
    torch,_,_=e.setup();protocol=e.source_protocol();save(e.OUT/'protocol.json',protocol)
    hard_model,condition=e.ribodecode_model(torch);hard=[e.hard_score(torch,hard_model,condition,r['dna']) for r in val];del hard_model,condition
    predictor,backend=e.seq2_model();save(e.OUT/'backend.json',backend);seq=[]
    for i,r in enumerate(val):
        path=e.OUT/f'{r["uid"]}.json'
        if path.exists():score=read(path)
        else:score=e.seq2_score(predictor,r['dna']);save(path,score)
        seq.append(score['seq2ribo_score']);save(ROOT/'followthrough_status.json',dict(stage='native_seq2ribo_validation',completed=i+1,total=96))
    save(ROOT/'baseline_predictions.json',dict(uids=[r['uid'] for r in val],target=y.tolist(),seq2ribo=seq,ribodecode=hard))
    baselines=[dict(method='native_seq2ribo_paper_HEK293',validation_pearson=corr(y,seq),limitation='Corrected native cohort versus historical paper weights; training overlap not certified, descriptive only'),
               dict(method='RiboDecode_fixed_RNA_proxy',validation_pearson=corr(y,hard),limitation='Fixed RNA/environment score versus TE; not native TE head')]
    # Ridge controls: exact same development rows, train normalization only.
    cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json');codons=sorted(cai)
    def features(r,kind):
        dna=r['dna'];cs=[dna[i:i+3] for i in range(0,len(dna)-3,3)]
        if kind=='length_only':return [np.log1p(len(dna)//3)]
        if kind=='codon_frequency':return [cs.count(c)/len(cs) for c in codons]
        with np.load(ROOT/'features'/f'{r["uid"]}.npz',allow_pickle=False) as z:mfe=float(z['mfe'])
        return [(dna.count('G')+dna.count('C'))/len(dna),mfe/len(dna),np.exp(np.mean([np.log(cai[c]) for c in cs]))]
    for kind in ['length_only','codon_frequency','GC_MFE_CAI']:
        a=np.array([features(r,kind) for r in train]);b=np.array([features(r,kind) for r in val]);mean=a.mean(0);scale=a.std(0).clip(1e-6)
        a=np.c_[np.ones(len(a)),(a-mean)/scale];b=np.c_[np.ones(len(b)),(b-mean)/scale]
        weight=np.linalg.solve(a.T@a+10*np.diag([0]+[1]*(a.shape[1]-1)),a.T@yt);p=b@weight
        baselines.append(dict(method=kind+'_ridge10',validation_pearson=corr(y,p)))
    save(ROOT/'validation_summary.json',dict(models=summary,baselines=baselines,selected=chosen['arm'],n=96,task='human_HEK293_TE',test_predicted=False))
    save(ROOT/'audit_complete.json',dict(status='passed',n_models=12,checkpoint_hashes_verified=True,all_model_pearsons_recomputed=True,validation_only=True))
    lines=['# 人源HEK293 TE有限开发结果','', '256训练、96验证，80–500密码子，两个种子；未预测测试集。','', '|模型|验证Pearson|','|---|---:|']
    for r in summary:lines.append(f"|{r['arm']}两种子集成|{r['ensemble_validation_pearson']:.4f}|")
    for r in baselines:lines.append(f"|{r['method']}|{r['validation_pearson']:.4f}|")
    lines+=['','作者seq2ribo历史权重与当前原生对齐划分的训练交叉尚未认证，只作描述性比较。RiboDecode固定RNA分数不是专门训练的TE头。六支线都在验证集开发，不能当独立确认。扩大训练和确认集评估前，不替换旧实验权重或冒称已实现全面人源校准。']
    (ROOT/'人源TE开发结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'followthrough_complete.json',dict(status='completed',validation_n=96,test_predicted=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(ROOT/'followthrough_failure.json',dict(error=repr(error),unix=time.time()));raise
