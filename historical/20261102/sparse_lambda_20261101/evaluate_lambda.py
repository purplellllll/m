"""Complete external scores for all 60 lambda endpoints under the locked CPU protocol."""
import csv
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
OUT=ROOT/'external_evaluation'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    source=WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py'
    spec=importlib.util.spec_from_file_location('locked_external_evaluator',source)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator);evaluator.OUT=OUT
    save=evaluator.save
    searches=read(ROOT/'lambda_search_results.json');assert len(searches)==60
    records=[];originals={}
    for row in searches:
        value=row['best'];dna=value['dna'];identity=hashlib.sha256(dna.encode()).hexdigest()
        assert identity==value['sequence_sha256'] and row['steps']==20
        records.append(dict(version=row['version'],lam=row['lambda'],sequence_id=row['sequence_id'],group=row['group'],
            dna=dna,sequence_sha256=identity,own_score=value['own_score']))
        originals[row['sequence_id']]=dict(version='original',lam=None,sequence_id=row['sequence_id'],group=row['group'],
            dna=row['original']['dna'],sequence_sha256=row['original']['sequence_sha256'],own_score=row['original']['own_score'])
    assert len(originals)==6
    records+=list(originals.values());bank={r['sequence_sha256']:r for r in records}
    protocol=evaluator.source_protocol();torch,np,RNA=evaluator.setup()
    protocol.update(torch=torch.__version__,viennarna=RNA.__version__,scope='60 lambda endpoints plus6 original references; exact-sequence deduplication',
        endpoint_source_sha256=sha(ROOT/'lambda_search_results.json'),evaluation_driver_sha256=sha(__file__),base_evaluator_sha256=sha(source),
        source_inputs_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())
    lock=OUT/'protocol_locked.json'
    if lock.exists():assert read(lock)==protocol
    else:save(lock,protocol)
    digest=sha(lock);begun=time.monotonic()
    for folder in ['ribodecode','seq2ribo']:(OUT/folder).mkdir(exist_ok=True)
    # Reuse only complete, matching, immutable-source evaluation caches.
    reused={'ribodecode':0,'seq2ribo':0}
    for source_dir in [WORK/'runs/vmae_own_full30_cpu_20261031',WORK/'runs/v3_v4_own_multi_cpu_20261101/v3_multi',WORK/'runs/v3_v4_own_multi_cpu_20261101/v4_epoch6']:
        source_lock=source_dir/'evaluation_protocol_locked.json'
        if not source_lock.exists():continue
        old=read(source_lock)
        for key in ['seq2ribo','ribodecode','properties']:assert protocol[key]==old[key],key
        for identity in bank:
            for folder in reused:
                origin=source_dir/'evaluation'/folder/f'{identity}.json';target=OUT/folder/f'{identity}.json'
                if target.exists() or not origin.exists():continue
                value=read(origin);assert value['sequence_sha256']==identity and value['protocol_sha256']==sha(source_lock)
                value['protocol_sha256']=digest
                value['provenance']=dict(kind='reused_identical_sequence_verified_protocol',source=str(origin),source_sha256=sha(origin))
                save(target,value);reused[folder]+=1
    missing=[(key,row) for key,row in bank.items() if not (OUT/'ribodecode'/f'{key}.json').exists()]
    if missing:
        model,condition=evaluator.ribodecode_model(torch)
        for key,row in missing:
            value=evaluator.hard_score(torch,model,condition,row['dna'])
            save(OUT/'ribodecode'/f'{key}.json',dict(sequence_sha256=key,protocol_sha256=digest,ribodecode_score=value,provenance=dict(kind='local_CPU_recomputed')))
        del model,condition;gc.collect()
    missing=[(key,row) for key,row in bank.items() if not (OUT/'seq2ribo'/f'{key}.json').exists()]
    save(OUT/'status.json',dict(stage='seq2ribo',unique_sequences=len(bank),missing=len(missing),completed=0,reused=reused))
    if missing:
        predictor,backend=evaluator.seq2_model();save(OUT/'backend.json',backend)
        for number,(key,row) in enumerate(missing,1):
            values=evaluator.seq2_score(predictor,row['dna'])
            save(OUT/'seq2ribo'/f'{key}.json',dict(sequence_sha256=key,protocol_sha256=digest,**values,provenance=dict(kind='local_CPU_recomputed')))
            save(OUT/'status.json',dict(stage='seq2ribo',unique_sequences=len(bank),missing=len(missing),completed=number,seconds=time.monotonic()-begun,reused=reused))
            print(f'seq2ribo {number}/{len(missing)}',flush=True)
        del predictor;gc.collect()
    table=[];cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
    for row in records:
        key=row['sequence_sha256'];dna=row['dna'];hard=read(OUT/'ribodecode'/f'{key}.json');seq=read(OUT/'seq2ribo'/f'{key}.json')
        assert hard['protocol_sha256']==seq['protocol_sha256']==digest
        assert len(seq['seq2ribo_by_seed'])==3 and abs(statistics.mean(seq['seq2ribo_by_seed'].values())-seq['seq2ribo_score'])<1e-10
        md=RNA.md();md.temperature=37.;md.dangles=2;_,mfe=RNA.fold_compound(dna.replace('T','U'),md).mfe()
        table.append(dict(**row,ribodecode_score=hard['ribodecode_score'],seq2ribo_score=seq['seq2ribo_score'],
            seq2ribo_by_seed=seq['seq2ribo_by_seed'],gc_percent=100*(dna.count('G')+dna.count('C'))/len(dna),
            cai=math.exp(statistics.mean(math.log(cai[dna[i:i+3]]) for i in range(0,len(dna)-3,3))),mfe_per_nt=float(mfe)/len(dna)))
    save(OUT/'per_case_metrics.json',table)
    keys=['own_score','ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_per_nt'];summary=[]
    groups=[('original',None)]+[(v,l) for v in ['v3_multi','v4_epoch6'] for l in [0.,.25,.5,.75,1.]]
    for version,lam in groups:
        rows=[r for r in table if r['version']==version and r['lam']==lam];assert len(rows)==6
        summary.append(dict(version=version,lam=lam,n=6,**{k:statistics.mean(r[k] for r in rows) for k in keys}))
    save(OUT/'summary.json',summary)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    lines=['# lambda全部支线：补齐外部评分','', '同6条CDS，每lambda20步搜索。RiboDecode固定logRNA=4.5；seq2ribo固定HEK293/CDS-only、每种子8次模拟、三个锁定种子平均；性质用ViennaRNA2.7.2重算。','',
        '|模型|lambda|自有评分|RiboDecode|seq2ribo TE|GC%|CAI|MFE/nt|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['version']}|{r['lam']}|{r['own_score']:.4f}|{r['ribodecode_score']:.4f}|{r['seq2ribo_score']:.4f}|{r['gc_percent']:.2f}|{r['cai']:.4f}|{r['mfe_per_nt']:.4f}|")
    lines+=['','全部是预测分数，不是实测表达或相关系数。外部评分不参与原搜索终点选择。样本为已用探索面板，不提供独立确认。GC/CAI/MFE不是无条件越高或越负越好。']
    (ROOT/'lambda完整外部评分对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'complete.json',dict(status='completed',endpoint_rows=60,original_rows=6,unique_sequences=len(bank),seconds=time.monotonic()-begun,
        all_seed_means_verified=True,protocol_sha256=digest,metrics_sha256=sha(OUT/'per_case_metrics.json'),reused=reused))
    save(OUT/'status.json',dict(stage='completed',endpoint_rows=60,original_rows=6))
    print('ALL60 LAMBDA EXTERNAL EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);(OUT/'failure.json').write_text(json.dumps(dict(error=repr(error),unix=time.time())),encoding='utf-8');raise
