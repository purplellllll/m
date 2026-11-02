"""Wait for both actual native runs and own search, then cross-score frozen outputs."""
from pathlib import Path
import sys,json,hashlib,time,subprocess,statistics,csv
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;OUT=HERE/'native_compare'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    import importlib.util
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare():
    import torch
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(WORK))
    from stability import StabilityScorer
    from latent_decoder.prepare_data import GENETIC_CODE
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    m=module('final24000_scorer',HERE/'locked_scorer_24000.py');scorer=m.LockedScorer('cuda');stability=StabilityScorer()
    expected=[r['checkpoint_sha256'] for r in read(HERE/'te_24000/selection_locked.json')['selected_models']]
    assert [r[3] for r in scorer.models]==expected
    assert read(OUT/'own_search/protocol_locked.json')['own_checkpoint_sha256']==expected
    inputs=read(HERE/'lambda_scan/inputs_locked.json');records=[]
    for i,row in enumerate(inputs,1):
        own=read(OUT/f'own_search/case{i}.json');native=read(OUT/f'case{i}_native/completion_receipt.json')
        assert native['state']=='completed' and native['proposals']==2400 and native['scorer_weights_unchanged']
        candidates=[('original',row['dna'],None),('own_lambda075',own['best']['dna'],None),('RiboDecode_native_final',native['native_returned_cds'],3)]
        native_rounds=native['all_round_selected']
        if isinstance(native_rounds,dict):native_rounds=[native_rounds[k] for k in sorted(native_rounds,key=int)]
        assert len(native_rounds)==3
        candidates += [('RiboDecode_native_round',dna,j) for j,dna in enumerate(native_rounds,1)]
        for method,dna,rnd in candidates:
            dna=dna.upper().replace('U','T');assert len(dna)==len(row['dna']) and set(dna)<=set('ACGT')
            assert all(GENETIC_CODE[dna[j:j+3]]==GENETIC_CODE[row['dna'][j:j+3]] for j in range(0,len(dna),3))
            v=scorer(dna)
            if method=='own_lambda075':assert max(abs(a-b) for a,b in zip(v['seed_scores'],own['best']['seed_scores']))<2e-5
            records.append(dict(case_uid=row['uid'],case_index=i,method=method,round=rnd,n_codons=row['n_codons'],
                **{k:x for k,x in v.items() if k!='arm'},stability_log_score=stability(dna,row['utr5'],row['utr3']),
                start_preserved=dna[:3]==row['dna'][:3],stop_preserved=dna[-3:]==row['dna'][-3:],
                changed_codons=sum(dna[j:j+3]!=row['dna'][j:j+3] for j in range(0,len(dna),3))))
    assert len(records)==18
    save(OUT/'endpoint_own_metrics_locked.json',dict(records=records,own_protocol_sha256=sha(OUT/'own_search/protocol_locked.json'),
        native_protocol_sha256=sha(OUT/'protocol_locked.json'),source_sha256=sha(__file__),all_proteins_preserved=True))
def evaluate():
    sys.path.insert(0,str(WORK))
    e=module('native_final_external',WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py');e.OUT=OUT/'endpoint_external';e.OUT.mkdir(exist_ok=True)
    torch,np,RNA=e.setup();p=e.source_protocol();p.update(own_scorer_outside_mRFP_domain=False,reference='Actual bounded native optimizer vs own V3, frozen endpoints')
    native_protocol=read(OUT/'protocol_locked.json');prior=read(HERE/'lambda_scan/external/protocol_locked.json')
    for key in ['seq2ribo','ribodecode','properties']:assert p[key]==prior[key]
    save(e.OUT/'protocol_locked.json',p)
    locked=read(OUT/'endpoint_own_metrics_locked.json');records=locked['records'];bank={r['sequence_sha256']:r for r in records}
    model,cond=e.ribodecode_model(torch);hard={uid:e.hard_score(torch,model,cond,r['dna']) for uid,r in bank.items()};del model,cond
    predictor,backend=e.seq2_model();save(e.OUT/'backend.json',backend);scores={}
    for i,(uid,row) in enumerate(bank.items(),1):
        path=e.OUT/f'{uid}.json';old=HERE/'lambda_scan/external'/f'{uid}.json'
        if path.exists():s=read(path)
        elif old.exists():s=read(old);save(path,s)
        else:s=e.seq2_score(predictor,row['dna']);save(path,s)
        assert abs(statistics.mean(s['seq2ribo_by_seed'].values())-s['seq2ribo_score'])<1e-10
        _,mfe=RNA.fold(row['dna'].replace('T','U'))
        scores[uid]=dict(ribodecode_score=hard[uid],seq2ribo_score=s['seq2ribo_score'],evaluation_mfe=float(mfe),evaluation_mfe_per_nt=float(mfe)/len(row['dna']))
        save(OUT/'evaluation_status.json',dict(stage='external_scoring',completed=i,total=len(bank)))
        print('native comparison external',i,len(bank),flush=True)
    table=[dict(**r,**scores[r['sequence_sha256']]) for r in records];save(OUT/'endpoint_metrics.json',table)
    fields=['own_score','ribodecode_score','seq2ribo_score','evaluation_mfe_per_nt','stability_log_score','gc_percent','cai']
    means=[]
    for method in ['original','own_lambda075','RiboDecode_native_final']:
        group=[r for r in table if r['method']==method];assert len(group)==3
        means.append(dict(method=method,n=3,**{k:statistics.mean(r[k] for r in group) for k in fields}))
    def dominates(a,b):
        x=[a[k] for k in fields[:3]]+[-a['evaluation_mfe_per_nt'],a['stability_log_score']]
        y=[b[k] for k in fields[:3]]+[-b['evaluation_mfe_per_nt'],b['stability_log_score']]
        return all(v>=w-1e-8 for v,w in zip(x,y)) and any(v>w+1e-8 for v,w in zip(x,y))
    own=next(r for r in means if r['method']=='own_lambda075');native=next(r for r in means if r['method']=='RiboDecode_native_final')
    casewins=sum(dominates(next(r for r in table if r['case_index']==i and r['method']=='own_lambda075'),
        next(r for r in table if r['case_index']==i and r['method']=='RiboDecode_native_final')) for i in range(1,4))
    resources=[]
    for i in range(1,4):
        n=read(OUT/f'case{i}_native/completion_receipt.json');o=read(OUT/f'own_search/case{i}.json')
        resources.append(dict(case=i,native_nominal_proposals=n['proposals'],native_optimizer_updates=n['optimizer_updates'],native_target_negative_queries=n['native_target_negative_queries'],
            native_true_fold_sequences=n['true_fold_sequence_evaluations'],native_seconds=n['elapsed_seconds'],own_attempts=o['attempts'],own_unique_proposals=o['unique_proposals'],own_seconds=o['seconds']))
    report=dict(status='completed',methods=means,mean_five_metric_dominance=dominates(own,native),case_dominance_count=casewins,
        resources=resources,own_selection=read(HERE/'te_24000/selection_locked.json'),
        caveats=native_protocol['limits']+['Proposal-matched, not total-compute-matched','Native last-round output used without selecting the best round by external scores',
            'Auxiliary stability prediction is not experimental candidate half-life or a half-life-trained TE head'],
        endpoints_sha256=sha(OUT/'endpoint_own_metrics_locked.json'),all_metrics_locally_computed=True)
    report['native_changed_terminal_stop_cases']=[r['case_index'] for r in table if r['method']=='RiboDecode_native_final' and not r['stop_preserved']]
    if report['native_changed_terminal_stop_cases']:
        report['caveats'].append('Native optimizer permits synonymous terminal stop changes; own search fixes the original stop. Protein constraints agree but exact terminal codon constraints differ. Native outputs were not silently rewritten.')
    recovery=OUT/'case3_recovery_execution.json'
    if recovery.exists():
        report['native_recovery']=read(recovery)
        report['caveats'].append('Native case3 replayed after mounted-drive I/O failure; discarded attempt adds compute beyond nominal proposal budget.')
    own_recovery=OUT/'own_search/memory_recovery.json'
    if own_recovery.exists():
        report['own_search_recovery']=read(own_recovery)
        report['caveats'].append('Own case2 replayed after native memory-allocation exit using the same seed and frozen score cache; per-case seconds exclude the interrupted attempt.')
    save(OUT/'summary.json',report)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
    lines=['# 续训后与RiboDecode原生优化对照','', '原生RiboDecode1.3.0的有限预算运行：每条3轮×800提案，MFE权重0.7。各条独立训练MFE代理，保留末轮原生返回结果；不是默认40000×20完整预算。自有V3固定λ0.75，每条2400提案。','',
        '|方法|N|自有TE|RiboDecode|seq2ribo|MFE/nt|半衰期对数评分|GC%|CAI|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:lines.append('|'+r['method']+'|3|'+'|'.join(f'{r[k]:.4f}' for k in fields)+'|')
    lines+=['',f"五项均值严格支配原生终点：{report['mean_five_metric_dominance']}；逐条支配：{casewins}/3。",'',
        '三条开发CDS，λ来自此前开发扫描；无新的独立测试确认。候选TE和半衰期均无实测标签，不能把预测分数称作真实改善或候选Pearson。HEK293T原生条件与HEK293评分条件不同。','',
        '提案数量匹配，梯度更新、评分调用、真实折叠次数、唯一序列数和耗时不匹配，详见summary.json的resources。','',
        'MFE统一以ViennaRNA2.7.2重算；原生内部RNAfold2.6.4保留并披露。已打开的903条TE测试集未用于续训、搜索或选择。']
    (OUT/'原生优化对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'complete.json',dict(status='completed',native_actual_optimizer=True,full_default=False,cases=3,all_endpoints_cross_scored=True))
    arrangement=read(HERE/'next_training_arrangement.json')
    active_30000=(HERE/'te_30000/protocol_locked.json').exists() and not (HERE/'te_30000/training_complete.json').exists()
    active_100000=(HERE/'te_100000/requested_plan.json').exists() and not (HERE/'te_100000/training_complete.json').exists()
    status='native_comparison_completed_training100000_queued_or_running' if active_100000 else ('native_comparison_completed_training30000_running' if active_30000 else 'executed_and_native_comparison_completed')
    arrangement.update(status=status,
        native_comparison_training_result=str(HERE/'te_24000/selection_locked.json'),native_comparison_result=str(OUT/'summary.json'))
    save(HERE/'next_training_arrangement.json',arrangement)
def main():
    if '--prepare' in sys.argv:return prepare()
    if '--evaluate' in sys.argv:return evaluate()
    deadline=time.monotonic()+8*3600
    prerequisites=[OUT/'native_complete.json',OUT/'own_search/complete.json']
    failures=[OUT/'queue_failure.json',OUT/'own_search/failure.json',HERE/'te_24000/training_failure.json']
    while not all(p.exists() for p in prerequisites):
        for p in failures:
            if p.exists():raise RuntimeError(str(p)+': '+str(read(p)))
        if time.monotonic()>deadline:raise TimeoutError('Native comparison supervisor eight-hour deadline')
        time.sleep(10)
    cuda=WORK/'.runtime/scorer_cuda/Scripts/python.exe'
    for python,mode in [(cuda,'--prepare'),(Path(sys.executable),'--evaluate')]:
        with (OUT/(mode[2:]+'.log')).open('ab') as log:
            result=subprocess.run([str(python),'-u',str(Path(__file__).resolve()),mode],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,(mode,result.returncode)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);save(OUT/'comparison_failure.json',dict(error=repr(error),unix=time.time()));raise
