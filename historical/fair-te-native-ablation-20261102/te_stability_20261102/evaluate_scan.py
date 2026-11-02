"""Post-lock endpoint scoring and audit. Never starts further TE training."""
from pathlib import Path
import sys,time,json,hashlib,csv,statistics,math
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent;OUT=ROOT/'lambda_scan'
sys.path.insert(0,str(WORK))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    deadline=time.monotonic()+7200
    while not (OUT/'search_complete.json').exists():
        if (OUT/'failure.json').exists():raise RuntimeError(read(OUT/'failure.json'))
        if time.monotonic()>deadline:raise TimeoutError('search exceeded two-hour limit')
        time.sleep(5)
    rows=read(OUT/'results.json');assert len(rows)==45
    protocol=read(OUT/'protocol_locked.json')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('new_scan_external',WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py');e=module_from_spec(spec);spec.loader.exec_module(e)
    e.OUT=OUT/'external';e.OUT.mkdir(exist_ok=True)
    torch,np,RNA=e.setup();reference=e.source_protocol()
    for k in ['seq2ribo','ribodecode','properties']:assert protocol['external_protocol'][k]==reference[k]
    reference.update(own_scorer_outside_mRFP_domain=False,task='Human HEK293 continued TE model development lambda scan',source_sha256=sha(__file__))
    save(e.OUT/'protocol_locked.json',reference)
    bank={}
    for row in rows:
        assert len(row['trace'])==12
        assert abs(row['best_reward']-max([0.]+[r['reward'] for r in row['trace']]))<1e-10
        for r in [row['original'],row['best']]:bank[r['sequence_sha256']]=r
    save(OUT/'endpoints_locked.json',dict(results_sha256=sha(OUT/'results.json'),sequence_sha256=sorted(bank),count=len(bank)))
    model,condition=e.ribodecode_model(torch)
    hard={uid:e.hard_score(torch,model,condition,r['dna']) for uid,r in bank.items()}
    assert all(abs(hard[uid]-r['ribodecode_score'])<2e-5 for uid,r in bank.items())
    del model,condition
    predictor,backend=e.seq2_model();save(e.OUT/'backend.json',backend)
    scores={}
    for index,(uid,r) in enumerate(bank.items(),1):
        path=e.OUT/f'{uid}.json'
        if path.exists():s=read(path)
        else:s=e.seq2_score(predictor,r['dna']);save(path,s)
        assert abs(statistics.mean(s['seq2ribo_by_seed'].values())-s['seq2ribo_score'])<1e-10
        _,mfe=RNA.fold(r['dna'].replace('T','U'))
        scores[uid]=dict(seq2ribo_score=s['seq2ribo_score'],ribodecode_score=hard[uid],evaluation_mfe=float(mfe),evaluation_mfe_per_nt=float(mfe)/len(r['dna']))
        save(OUT/'evaluation_status.json',dict(stage='external_seq2ribo',completed=index,total=len(bank)))
        print('external',index,len(bank),flush=True)
    table=[];originals={}
    for row in rows:
        z=row['best'];table.append(dict(case_uid=row['case_uid'],arm=row['arm'],lambda_value=row['lambda'],n_codons=row['n_codons'],
            **{k:v for k,v in z.items() if k not in ['ribodecode_score','arm']},**scores[z['sequence_sha256']]))
        z=row['original'];originals[row['case_uid']]=dict(case_uid=row['case_uid'],arm='original',lambda_value=None,n_codons=row['n_codons'],
            **{k:v for k,v in z.items() if k not in ['ribodecode_score','arm']},**scores[z['sequence_sha256']])
    table+=list(originals.values());save(OUT/'endpoint_metrics.json',table)
    fields=['own_score','ribodecode_score','seq2ribo_score','evaluation_mfe_per_nt','stability_log_score','gc_percent','cai']
    summaries=[]
    for arm in ['original','own_multi','ribodecode_multi','ribodecode_mfe']:
        for lam in ([None] if arm=='original' else protocol['lambdas']):
            group=[r for r in table if r['arm']==arm and r['lambda_value']==lam];assert len(group)==3
            summaries.append(dict(arm=arm,lambda_value=lam,n=3,**{k:statistics.mean(r[k] for r in group) for k in fields}))
    def dominates(a,b):
        av=[a[k] for k in fields[:3]]+[-a['evaluation_mfe_per_nt'],a['stability_log_score']]
        bv=[b[k] for k in fields[:3]]+[-b['evaluation_mfe_per_nt'],b['stability_log_score']]
        return all(x>=y-1e-8 for x,y in zip(av,bv)) and any(x>y+1e-8 for x,y in zip(av,bv))
    checks=[]
    for lam in protocol['lambdas']:
        own=next(r for r in summaries if r['arm']=='own_multi' and r['lambda_value']==lam)
        for baseline in ['ribodecode_multi','ribodecode_mfe']:
            b=next(r for r in summaries if r['arm']==baseline and r['lambda_value']==lam)
            wins=0
            for uid in originals:
                a=next(r for r in table if r['case_uid']==uid and r['arm']=='own_multi' and r['lambda_value']==lam)
                bb=next(r for r in table if r['case_uid']==uid and r['arm']==baseline and r['lambda_value']==lam)
                wins+=int(dominates(a,bb))
            checks.append(dict(lambda_value=lam,baseline=baseline,mean_five_metric_dominance=dominates(own,b),case_dominance_count=wins,n=3))
    report=dict(status='completed',summary=summaries,dominance_checks=checks,unique_external_sequences=len(bank),
        caveats=['No native RiboDecode optimizer comparison in this run', 'All candidate endpoints are predictions, not measured TE or half-life',
                 'Three development cases cannot establish general superiority','Half-life scorer is separate kmer ridge validated on586genes; score is transformed target, not hours',
                 'No new independent TE test evaluation; prior903 already opened','GC/CAI/MFE optimization does not guarantee cellular stability'])
    save(OUT/'summary.json',report)
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    lines=['# 人源TE与mRNA稳定性开发扫描','', '3条开发集CDS；V3生成器固定；每支线×λ×CDS 12次提案。所有候选指标为预测值。RiboDecode支线为评分器引导V3，不是原生优化器。','',
        '|支线|λ|TE评分|RiboDecode|seq2ribo|MFE/nt|半衰期对数评分|GC%|CAI|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|'+str(r['arm'])+'|'+str(r['lambda_value'])+'|'+'|'.join(f'{r[k]:.4f}' for k in fields)+'|')
    lines+=['','λ=1只优化翻译评分；λ=0只优化相应性质组合。多目标支线保留GC、CAI、MFE，并加入独立辅助半衰期评分。MFE更负表示更低折叠自由能，不能替代细胞内半衰期。',
        '', 'TE新模型的验证提升不能与旧模型的测试集基线直接排名；本次不读取旧903条测试集。', '',
        '是否同时支配五项指标，详见summary.json中dominance_checks；不根据联合奖励更高宣称所有分项都更高。']
    (OUT/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    selection=read(ROOT/'te_continuation/selection_locked.json')
    plan=dict(status='arranged_not_started',prerequisites_completed=['continued_TE_weights_locked_and_reload_checked','human_half_life_gene_validation','matched_lambda_scan','endpoint_external_scores'],
        te_training_mean_validation_pearson=selection['validation_mean_per_seed']['no_stats'],
        next_training='Separate new run: extend two no_stats seeds to at most24000 total steps, validation every500, stop after6 non-improvements; retain7373/922 split and never tune on903.',
        learning_rate='Continue from current low final learning rate; any restart must be declared as a separate intervention.',
        required_interpretation='More TE steps cannot train half-life capability into the TE head. Future multi-task stability head requires its own gene-separated training/evaluation.',
        native_baseline_remaining='Protein-specific native RiboDecode optimizer not reproduced; score-guided comparison alone cannot establish superiority over full tool.',
        execution='No TE continuation launched by this evaluator. Complete reviewable comparison first, as requested.',
        artifacts_sha256={'scan_summary':sha(OUT/'summary.json'),'stability_summary':sha(ROOT/'stability/summary.json'),'scorer_lock':sha(ROOT/'scorer_locked.json')})
    save(ROOT/'next_training_arrangement.json',plan)
    save(OUT/'complete.json',dict(status='completed',external_scores=True,training_arranged=True,training_started=False))
    print('comparison complete; further training arranged, not started',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:save(OUT/'evaluation_failure.json',dict(error=repr(error),unix=time.time()));raise
