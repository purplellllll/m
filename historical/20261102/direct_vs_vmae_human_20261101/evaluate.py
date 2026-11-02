"""Automatically cross-score paired search endpoints; never touch expanded heldout data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import statistics
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
PILOT=WORK/'runs/human_te_pilot_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    deadline=time.monotonic()+7200
    while not (ROOT/'search_complete.json').exists():
        if (ROOT/'failure.json').exists():raise RuntimeError(read(ROOT/'failure.json'))
        if time.monotonic()>deadline:raise TimeoutError('Two-hour finite search deadline')
        time.sleep(5)
    searches=read(ROOT/'search_results.json');assert len(searches)==6
    records=[];originals={}
    for row in searches:
        records.append(dict(case_uid=row['case_uid'],method=row['method'],n_codons=row['n_codons'],**row['best']))
        originals[row['case_uid']]=dict(case_uid=row['case_uid'],method='original',n_codons=row['n_codons'],**row['original'])
    records+=list(originals.values());bank={r['sequence_sha256']:r for r in records}
    source=WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py'
    spec=importlib.util.spec_from_file_location('direct_external_scores',source);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    e.OUT=ROOT/'external';e.OUT.mkdir(exist_ok=True);protocol=e.source_protocol();torch,_,_=e.setup();save(e.OUT/'protocol_locked.json',protocol)
    old_protocol=read(PILOT/'baselines/protocol.json')
    for key in ['seq2ribo','ribodecode','properties']:assert old_protocol[key]==protocol[key]
    model,condition=e.ribodecode_model(torch);hard={key:e.hard_score(torch,model,condition,r['dna']) for key,r in bank.items()};del model,condition
    predictor,backend=e.seq2_model();save(e.OUT/'backend.json',backend);seq={}
    for i,(key,row) in enumerate(bank.items(),1):
        path=e.OUT/f'{key}.json';native_uid=hashlib.sha256(row['dna'].replace('T','U').encode()).hexdigest();old=PILOT/'baselines'/f'{native_uid}.json'
        if path.exists():value=read(path)
        elif old.exists():value=dict(**read(old),provenance='Reused original development sequence, same frozen evaluator sources');save(path,value)
        else:value=dict(**e.seq2_score(predictor,row['dna']),provenance='Local CPU recomputed');save(path,value)
        assert abs(statistics.mean(value['seq2ribo_by_seed'].values())-value['seq2ribo_score'])<1e-10
        seq[key]=value['seq2ribo_score'];save(ROOT/'evaluation_status.json',dict(stage='seq2ribo',completed=i,total=len(bank)))
    table=[dict(**r,ribodecode_score=hard[r['sequence_sha256']],seq2ribo_score=seq[r['sequence_sha256']]) for r in records]
    save(ROOT/'endpoint_metrics.json',table);summary=[]
    for method in ['original','direct_synonymous','vmae_latent']:
        group=[r for r in table if r['method']==method];assert len(group)==3
        summary.append(dict(method=method,n=3,**{k:statistics.mean(r[k] for r in group) for k in ['own_score','ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_per_nt']}))
    # Verify highest scoring fallback/trace selection independently.
    for row in searches:
        assert len(row['trace'])==20
        assert abs(row['best']['own_score']-max(r['own_score'] for r in row['trace']+[row['original']]))<1e-10
    save(ROOT/'summary.json',dict(status='completed',methods=summary,unique_external_sequences=len(bank),all_metrics_locally_computed=True,
        limits='Three development CDS, frozen small human TE pilot; no measured expression for changed sequences, no expanded test read'))
    lines=['# 人源评分器直接同义搜索与VMAE对照','', '同3条开发CDS、同人源pilot头、每方法20次提案；目标仅为自有TE分数。终点锁定后本地外部评分。','',
        '|方法|N|自有TE评分|RiboDecode|seq2ribo TE|GC%|CAI|MFE/nt|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['method']}|3|{r['own_score']:.4f}|{r['ribodecode_score']:.4f}|{r['seq2ribo_score']:.4f}|{r['gc_percent']:.2f}|{r['cai']:.4f}|{r['mfe_per_nt']:.4f}|")
    lines+=['','直接搜索在当前最优CDS上更改1/2/4个同义密码子，接受自有分数改善；VMAE在潜变量扰动后解码。两者都保留原始序列，氨基酸和起止固定。匹配的是提案次数，不是独特查询数、编辑幅度或总耗时。','',
        '这些是候选预测均值，不是Pearson。没有生成候选的实测标签，不能据此证明真实表达提升。人源pilot使用256训练/96验证，扩大的7373训练及903留出测试仍独立运行。']
    (ROOT/'直接搜索对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'complete.json',dict(status='completed',cases=3,search_methods=2,amino_acid_constraints=True,winners_verified=True))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(ROOT/'evaluation_failure.json',dict(error=repr(error),unix=time.time()));raise
