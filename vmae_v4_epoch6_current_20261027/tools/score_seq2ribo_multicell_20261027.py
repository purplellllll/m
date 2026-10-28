"""Score locked existing TEST candidates only; never train, search or select sequences."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

BASE=Path('/root/gpufree-data/vmae_transfer_assessment_20261024')
ASSETS=BASE.parent/'seq2ribo_multicell_20261027'
CURRENT=BASE.parent/'vmae_v4_epoch6_current_20261027'
OUT=BASE.parent/'seq2ribo_multicell_comparison_20261027'
CELLS=('lcl','rpe','ipsc')
PROPERTIES=('ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol','mfe_per_nt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def discover():
    """Use the comparison's canonical method names and resolve their original DNA."""
    ids={r['sequence_id'] for r in json.loads((CURRENT/'test_panel.json').read_text())}
    result={}
    with (CURRENT/'all_model_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        for entry in csv.DictReader(stream):
            path=Path(entry['source_file'])
            if path.suffix=='.json':
                row=json.loads(path.read_text())
            else:
                with path.open(encoding='utf-8-sig',newline='') as source:
                    row=next(r for r in csv.DictReader(source) if r['sequence_id']==entry['sequence_id'] and r['method']==entry['method'])
            dna=row['cds_dna']
            assert row['sequence_id'] in ids
            assert hashlib.sha256(dna.encode()).hexdigest()==row['sequence_sha256']
            key=entry['sequence_id']+'|'+entry['method']
            assert key not in result
            result[key]=dict(sequence_id=entry['sequence_id'],method=entry['method'],cds_dna=dna,
                sequence_sha256=row['sequence_sha256'],source_file=str(path),
                **{k:float(row[k]) for k in PROPERTIES})
    return result


def summarize(rows):
    """Always pair by CDS identity; never compare unequal panels as equal samples."""
    methods=sorted({r['method'] for r in rows})
    lookup={(r['sequence_id'],r['method']):r for r in rows}
    assert len(lookup)==len(rows)
    means={}
    paired={}
    for method in methods:
        selected=[r for r in rows if r['method']==method]
        means[method]=dict(n=len(selected),means={k:statistics.mean(r[k] for r in selected)
            for k in (*PROPERTIES,*CELLS)})
        for reference in ('original','V4_epoch6_current'):
            if reference==method:
                continue
            ids=sorted(i for i,m in lookup if m==method and (i,reference) in lookup)
            if ids:
                delta={c:[lookup[i,method][c]-lookup[i,reference][c] for i in ids] for c in CELLS}
                paired[method+' minus '+reference]=dict(n=len(ids),sequence_ids=ids,
                    mean_delta={c:statistics.mean(v) for c,v in delta.items()},
                    higher_count={c:sum(x>0 for x in v) for c,v in delta.items()})
    return dict(methods=means,paired=paired)


def report(manifest):
    rows=[]
    for path in sorted((OUT/'metrics').glob('*/*.json')):
        result=json.loads(path.read_text())
        source=manifest[result['sequence_id']+'|'+result['method']]
        assert source['sequence_sha256']==result['sequence_sha256']
        rows.append(dict(sequence_id=result['sequence_id'],method=result['method'],
            **{k:source[k] for k in PROPERTIES},**{c:result['scores'][c]['mean'] for c in CELLS}))
    summary=summarize(rows)
    save(OUT/'summary.json',dict(summary,generated_unix=time.time(),optimization_started=False))
    if rows:
        with (OUT/'metrics.csv.tmp').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (OUT/'metrics.csv.tmp').replace(OUT/'metrics.csv')
    text='# 跨环境seq2ribo评分对比（仅评分）\n\n同一批30条TEST身份，固定已有候选，未训练、未搜索、未按评分重选序列。\n\n'
    text+='| 方法 | N/30 | HEK293原评分 | LCL | RPE-1 | iPSC | GC% | CAI | MFE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for method,value in summary['methods'].items():
        text+='| '+method+' | '+str(value['n'])+' | '+' | '.join(f"{value['means'][k]:.6f}" for k in ('seq2ribo_score',*CELLS,'gc_percent','cai','mfe_kcal_mol'))+' |\n'
    text+='\nHEK293列复用原锁定评价结果；新增三环境均为三个固定种子、各8次sTASEP、CDS-only TE逆变换输出。跨环境数值尺度不同，不直接横向排名。\n\n'
    text+='| 同身份配对差值 | N | ΔLCL | ΔRPE-1 | ΔiPSC |\n|---|---:|---:|---:|---:|\n'
    for label,value in summary['paired'].items():
        text+='| '+label+' | '+str(value['n'])+' | '+' | '.join(f"{value['mean_delta'][c]:+.6f}" for c in CELLS)+' |\n'
    text+='\n未完成N不与30条完整均值直接排名。RNop权重不可获取；LinearDesign和完整RiboDecode按启动时已完成的候选评分，第6轮新候选随现有任务产出后补评。\n'
    text+='\nGC、CAI、MFE沿用原统一设置；预测评分不是实测表达。\n'
    (OUT/'COMPARISON_ZH.md.tmp').write_text(text,encoding='utf-8')
    (OUT/'COMPARISON_ZH.md.tmp').replace(OUT/'COMPARISON_ZH.md')
    return summary


def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'workflow.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=json.loads((ASSETS/'scorer_lock.json').read_text())
        assert json.loads((ASSETS/'status.json').read_text())['state']=='ready'
        assert sha(ASSETS/'assets.json')==lock['assets_sha256']
        assert sha(ASSETS/'environment_parameters.json')==lock['environment_parameters_sha256']
        assets=json.loads((ASSETS/'assets.json').read_text())['assets']
        for name,value in assets.items():
            assert sha(ASSETS/'weights'/name)==value['sha256']
        repository=BASE/'tmp/seq2ribo_source_20261020/seq2ribo-main'
        for name,expected in lock['source_sha256'].items():
            assert sha(repository/name)==expected
        sys.path.insert(0,str(BASE/'.runtime/mamba_cuda_v2_2_4/packages'));sys.path.insert(0,str(repository))
        import numpy as np
        import torch
        from seq2ribo import Seq2Ribo
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        manifest_path=OUT/'candidate_manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else discover()
        save(manifest_path,manifest)
        plan=dict(task='Score existing locked TEST candidates only',cell_lines=list(CELLS),seeds=lock['seeds'],n_stasep_runs=8,
            scorer_lock_sha256=sha(ASSETS/'scorer_lock.json'),source_sha256=sha(__file__),
            follow_new_candidates='V4_epoch6_current only; do not launch optimization',optimization_started=False)
        if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
        else:save(OUT/'plan.json',plan)
        predictors={c:Seq2Ribo(cell_line=c,device='cuda',weights_dir=str(ASSETS/'weights'),cache_dir=str(ASSETS/'geometry')) for c in CELLS}
        for predictor in predictors.values():predictor._load_model('te',use_utr=False).eval().requires_grad_(False)
        while True:
            for key,row in discover().items():
                if row['method']=='V4_epoch6_current':
                    if key in manifest:assert manifest[key]==row
                    else:manifest[key]=row
            save(manifest_path,manifest)
            for key,row in sorted(manifest.items(),key=lambda pair:(len(pair[1]['cds_dna']),pair[0])):
                destination=OUT/'metrics'/row['sequence_id']/(row['method']+'.json')
                if destination.exists():
                    assert json.loads(destination.read_text())['sequence_sha256']==row['sequence_sha256']
                    continue
                scores={}
                for cell,predictor in predictors.items():
                    save(OUT/'status.json',dict(state='scoring',method=row['method'],sequence_id=row['sequence_id'],cell_line=cell,
                        completed_cases=len(list((OUT/'metrics').glob('*/*.json'))),available_cases=len(manifest),heartbeat_unix=time.time(),optimization_started=False))
                    cache=OUT/'score_cache'/cell/(row['sequence_sha256']+'.json')
                    if cache.exists():
                        score=json.loads(cache.read_text());assert score['scorer_lock_sha256']==plan['scorer_lock_sha256']
                    else:
                        with torch.inference_mode():
                            values={str(seed):float(predictor.predict(row['cds_dna'].replace('T','U'),task='te',use_utr=False,
                                return_scaled_te=False,n_stasep_runs=8,init_p=.01,seed=seed,batch_size=8)[0]) for seed in lock['seeds']}
                        assert np.isfinite(list(values.values())).all()
                        score=dict(by_seed=values,mean=statistics.mean(values.values()),seed_sd=statistics.pstdev(values.values()),
                            scorer_lock_sha256=plan['scorer_lock_sha256'])
                        save(cache,score)
                    scores[cell]=score
                save(destination,dict(sequence_id=row['sequence_id'],method=row['method'],sequence_sha256=row['sequence_sha256'],scores=scores,
                    candidate_locked_before_scoring=True,optimization_started=False))
                report(manifest)
                print('SCORED',key,flush=True)
            report(manifest)
            current_state=json.loads((CURRENT/'status.json').read_text())['state']
            epoch6=sum(r['method']=='V4_epoch6_current' for r in manifest.values())
            if current_state=='completed' and epoch6==30:
                save(OUT/'complete.json',dict(completed=True,epoch6_cases=30,scored_candidates=len(manifest),cell_lines=list(CELLS),optimization_started=False))
                save(OUT/'status.json',dict(state='completed',completed_cases=len(manifest),heartbeat_unix=time.time(),optimization_started=False))
                return
            save(OUT/'status.json',dict(state='waiting_for_existing_epoch6_results' if current_state!='failed' else 'upstream_failed',
                completed_cases=len(manifest),epoch6_cases=epoch6,heartbeat_unix=time.time(),optimization_started=False))
            if current_state=='failed':raise RuntimeError('Existing epoch6 evaluation failed; completed scores preserved')
            time.sleep(60)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(OUT/'status.json',dict(state='failed',error=repr(error),heartbeat_unix=time.time(),optimization_started=False))
        raise
