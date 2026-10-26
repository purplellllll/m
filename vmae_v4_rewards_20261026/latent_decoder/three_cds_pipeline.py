"""Fresh CPU runs of frozen B5-JS and the shipped RiboDecode return pipeline."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .analyze_random_validation import cai_weights,sequence_metrics,synonym_changes,fold_candidate,write_csv
from .budget_comparison import ROOT
from .budget_js_extension import read_rows,run_trial,save_json
from .dual_space_ablation import CountedScorer,masks
from .js_constraint import mutable_js
from .latent_optimization import TranslationObjective,design_probabilities,state_hash
from .model import AA_IDS,CODONS,CodonVMAE,collate_records,encode_cds
from .train import sha256

ARMS=('B5_JS','RiboDecode_native')
PACKAGE=ROOT/'external/ribodecode_1_3_0/RiboDecode'
CHECKPOINT=ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'


def load_model():
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=CodonVMAE(**saved['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    return model


def validate_dna(original,dna):
    before=encode_cds(original)
    after=encode_cds(dna)
    assert len(before)==len(after)
    assert torch.equal(AA_IDS[torch.tensor(before)],AA_IDS[torch.tensor(after)])
    assert dna[:3]==original[:3]
    return dict(changed_codons=sum(a!=b for a,b in zip(before,after)),
                changed_nucleotides=sum(a!=b for a,b in zip(original,dna)),
                protein_preserved=True,start_preserved=True,stop_preserved=dna[-3:]==original[-3:])


def b5_worker(input_path,run):
    from tools.windows_torch_io import enable_unicode_save
    enable_unicode_save()
    started=time.perf_counter()
    task=json.loads(input_path.read_text())
    run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model=load_model()
    scorer=CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    frozen=state_hash(model),state_hash(scorer)
    tokens=collate_records([dict(cds_dna=task['cds_dna'])])
    with torch.no_grad():
        mu,logvar=model.encode(tokens,tokens)
        sigma=(.5*logvar).exp()
        reference=design_probabilities(model.decode(mu,tokens),tokens)
        baseline=float(scorer(F.one_hot(tokens,64).float(),tokens))
    setup=time.perf_counter()-started
    row=run_trial(model,scorer,tokens,mu,sigma,reference,task['config'],
        dict(sequence_id=task['sequence_id'],seed=task['seed'],budget_percent=100,js_limit=.05),
        baseline,run,None,False)
    assert row['best']['soft_js']<=.05+1e-7
    assert validate_dna(task['cds_dna'],row['best']['cds_dna'])['stop_preserved']
    assert frozen==(state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert scorer.forward_sequences==161
    row.update(setup_seconds=setup,worker_seconds=time.perf_counter()-started,weights_unchanged=True)
    save_json(run/'result.json',row)
    (run/'final.fasta').write_text('>'+task['sequence_id']+'|B5_JS|'+str(task['seed'])+'\n'+row['best']['cds_dna']+'\n')


def summarize(output,rows,completed=False):
    summary=dict(completed=completed,rows=len(rows),planned=18,per_cds=[],overall={})
    for identity in sorted({r['sequence_id'] for r in rows}):
        group=[r for r in rows if r['sequence_id']==identity]
        if len(group)!=6:
            continue
        item=dict(sequence_id=identity,n_aa=group[0]['n_aa'])
        for arm in ARMS:
            values=[r for r in group if r['arm']==arm]
            item[arm]={k:float(np.mean([r[k] for r in values])) for k in
                ('hard_score','hard_gain','all_codon_edit_fraction','gc_fraction','gc_delta_pp','train_reference_cai',
                 'mfe','mfe_per_nt','cold_generation_seconds','common_evaluation_seconds','output_hard_js')}
        summary['per_cds'].append(item)
    if len(summary['per_cds'])==3:
        for arm in ARMS:
            summary['overall'][arm]={k:float(np.mean([r[arm][k] for r in summary['per_cds']])) for k in summary['per_cds'][0][arm]}
    save_json(output/'summary.json',summary)
    text='# 三条CDS：B5-JS与原版RiboDecode完整推理流程\n\n'
    text+=f"{'完成' if completed else '运行中'}，统一评价{len(rows)}/18组；73/466/825 aa，每条3个种子。均不限制修改率。\n\n"
    text+='B5-JS保留JS优化及最终JS≤.05，原版保留自身返回选择规则、无JS限制。原版为1外轮/80次更新/batch1/纯翻译；不是发行版默认40000候选的多轮MFE训练。共同评分器HEK293T、mRNA=4.5，评分去终止密码子；原版优化目标仍按发行版包含终止密码子。RNAfold全长、CAI训练集参考，均只用于事后评价。\n\n'
    text+='生成耗时为同一CPU环境中每组独立进程的冷启动墙钟时间，含导入、加载、搜索、候选选择及导出；共同评价耗时另列。VMAE预训练成本不包含在推理耗时。JS硬序列指标与优化时soft JS定义不同。\n\n'
    text+='| aa | 方法 | 平均Hard Score | Hard Gain | 修改率 | GC | CAI | MFE/nt | 冷启动生成/秒 | 共同评价/秒 |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for item in sorted(summary['per_cds'],key=lambda x:x['n_aa']):
        for arm in ARMS:
            m=item[arm]
            text+=f"| {item['n_aa']} | {arm} | {m['hard_score']:.6f} | {m['hard_gain']:+.6f} | {m['all_codon_edit_fraction']:.2%} | {m['gc_fraction']:.2%} | {m['train_reference_cai']:.4f} | {m['mfe_per_nt']:.4f} | {m['cold_generation_seconds']:.2f} | {m['common_evaluation_seconds']:.2f} |\n"
    text+='\n原版每次160个原生目标/负条件评分，另对80候选作共同硬评分；B5-JS每次160个搜索评分和1个原始序列评分，另作共同赢家复核。不同方法不等查询/FLOPs；只用3条已使用过的开发CDS，不作总体或真实表达效果推断。原生RNAfold日志计算在纯翻译模式下延后到共同全长评价，其缺失值保持null。\n'
    (output/'report.md').write_text(text,encoding='utf-8')
    return summary


def evaluate(output,plan,timings):
    started=time.perf_counter()
    torch.set_num_threads(4)
    model=load_model()
    scorer=CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes=state_hash(model),state_hash(scorer)
    for relative,digest in plan['cai_training_sha256'].items():
        assert sha256(ROOT/'dataset'/relative)==digest
    weights=cai_weights(ROOT/'dataset')
    save_json(output/'cai_weights.json',weights)
    tree=ast.parse((PACKAGE/'dataset.py').read_text())
    vocabulary=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='my_vocab' for t in n.targets))
    order=[vocabulary[c] for c in CODONS]
    folds={}
    def properties(dna):
        key=hashlib.sha256(dna.encode()).hexdigest()
        if key not in folds:
            folds[key]=fold_candidate(dna)
        return dict(sequence_metrics(dna,weights),**folds[key])
    init_seconds=time.perf_counter()-started
    rows=[]
    for record in plan['records']:
        identity=record['sequence_id']
        original=record['cds_dna']
        tokens=collate_records([dict(cds_dna=original)])
        active,_=masks(tokens)
        with torch.no_grad():
            mu,_=model.encode(tokens,tokens)
            reference=design_probabilities(model.decode(mu,tokens),tokens)
            baseline=float(scorer(F.one_hot(tokens,64).float(),tokens))
        before=properties(original)
        save_json(output/'originals'/f'{identity}.json',dict(sequence_id=identity,hard_score=baseline,cds_dna=original,**before))
        for seed in plan['seeds']:
            for arm in ARMS:
                begun=time.perf_counter()
                timer=next(t for t in timings if (t['sequence_id'],t['seed'],t['arm'])==(identity,seed,arm))
                run=Path(timer['run_path'])
                if arm=='B5_JS':
                    internal=json.loads((run/'result.json').read_text())
                    best=internal['best']
                    trace=read_rows(run/'steps.jsonl')
                    assert len(trace)==80
                    dna=best['cds_dna']
                    with torch.no_grad():
                        selected=collate_records([dict(cds_dna=dna)])
                        hard_score=float(scorer(F.one_hot(selected,64).float(),tokens))
                        raw=collate_records([dict(cds_dna=internal['best_budget_only']['cds_dna'])])
                        raw_score=float(scorer(F.one_hot(raw,64).float(),tokens))
                    assert abs(hard_score-best['hard_score'])<1e-5
                    assert abs(raw_score-internal['best_budget_only']['hard_score'])<1e-5
                    assert best['soft_js']<=.05+1e-7
                    details=dict(search_soft_js=best['soft_js'],raw_archive_score=raw_score,
                        filter_gap=raw_score-hard_score,search_seconds=internal['search_seconds'],
                        search_setup_seconds=internal['setup_seconds'],native_execution_seconds=None,
                        search_scorer_queries=160,native_negative_scorer_queries=0,common_rescore_queries=2,
                        unique_search_candidates=len({r['cds_dna'] for r in trace}),returned_step=best['step'],
                        matched_native_soft_js_min=None,matched_native_soft_js_max=None)
                else:
                    receipt=json.loads((run/'completion_receipt.json').read_text())
                    assert receipt['completed'] and receipt['scorer_weights_unchanged']
                    raw_path=run/'results_natural/HEK293T_optim_mfe_dist-optim/0/samples/optim_results.txt'
                    lines=[s.split('\t') for s in raw_path.read_text().splitlines()]
                    candidates=[r[0] for r in lines]
                    distributions=np.load(run/'native_distributions.npz')['probabilities']
                    assert len(candidates)==80 and distributions.shape==(80,65,tokens.shape[1])
                    values=[]
                    js=[]
                    with torch.no_grad():
                        for k,candidate in enumerate(candidates):
                            validate_dna(original,candidate)
                            t=collate_records([dict(cds_dna=candidate)])
                            p=torch.from_numpy(distributions[k,order,:].T.copy())[None]
                            assert torch.equal(p.argmax(-1),t)
                            values.append(float(scorer(F.one_hot(t,64).float(),tokens)))
                            js.append(float(mutable_js(p,reference,active)))
                    dna=receipt['native_returned_cds']
                    matches=[i for i,d in enumerate(candidates) if d==dna]
                    assert matches
                    hard_score=values[matches[0]]
                    save_json(run/'common_archive.json',dict(candidates=[dict(step=i+1,cds_dna=d,hard_score=v,
                        soft_js=j,native_rpf_proxy=float(lines[i][1])) for i,(d,v,j) in enumerate(zip(candidates,values,js))]))
                    details=dict(search_soft_js=None,raw_archive_score=max([baseline]+values),filter_gap=None,
                        search_seconds=None,search_setup_seconds=None,native_execution_seconds=receipt['elapsed_seconds'],
                        search_scorer_queries=80,native_negative_scorer_queries=80,common_rescore_queries=80,
                        unique_search_candidates=len(set(candidates)),returned_step=None,
                        matched_native_soft_js_min=min(js[i] for i in matches),matched_native_soft_js_max=max(js[i] for i in matches))
                validation=validate_dna(original,dna)
                prop=properties(dna)
                _,tv=synonym_changes(original,dna)
                selected=collate_records([dict(cds_dna=dna)])
                with torch.no_grad():
                    hard_js=float(mutable_js(F.one_hot(selected,64).float(),reference,active))
                substitutions=Counter(CODONS[a]+'>'+CODONS[b] for a,b in zip(encode_cds(original),encode_cds(dna)) if a!=b)
                row=dict(sequence_id=identity,n_aa=record['n_codons']-1,n_nt=len(dna),seed=seed,arm=arm,
                    cds_dna=dna,hard_score=hard_score,original_hard_score=baseline,hard_gain=hard_score-baseline,
                    rpf_proxy=.2*math.expm1(hard_score),all_codon_edit_fraction=validation['changed_codons']/tokens.shape[1],
                    mutable_edit_fraction=int(((selected!=tokens)&active).sum())/int(active.sum()),
                    nucleotide_edit_fraction=validation['changed_nucleotides']/len(dna),
                    gc_delta_pp=100*(prop['gc_fraction']-before['gc_fraction']),cai_delta=prop['train_reference_cai']-before['train_reference_cai'],
                    mfe_delta=prop['mfe']-before['mfe'],mfe_per_nt_delta=prop['mfe_per_nt']-before['mfe_per_nt'],
                    synonymous_usage_tv=tv,output_hard_js=hard_js,substitution_counts=dict(substitutions),
                    cold_generation_seconds=timer['cold_generation_seconds'],**validation,**prop,**details)
                row['common_evaluation_seconds']=time.perf_counter()-begun
                rows.append(row)
                save_json(output/'metrics'/f'{identity}_{seed}_{arm}.json',row)
                summarize(output,rows)
                print(json.dumps(dict(event='evaluated',completed=len(rows),arm=arm,n_aa=row['n_aa'],score=hard_score)),flush=True)
    assert len(rows)==18 and scorer.forward_sequences==3+9*2+9*80
    assert hashes==(state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    save_json(output/'fold_cache.json',folds)
    scalar_rows=[{k:v for k,v in r.items() if k!='substitution_counts'} for r in rows]
    write_csv(output/'per_seed_metrics.csv',scalar_rows)
    fasta=''.join('>'+r['sequence_id']+'|'+r['arm']+'|seed='+str(r['seed'])+'\n'+r['cds_dna']+'\n' for r in rows)
    (output/'final_sequences.fasta').write_text(fasta)
    winners=[]
    for record in plan['records']:
        for arm in ARMS:
            winner=max([r for r in rows if r['sequence_id']==record['sequence_id'] and r['arm']==arm],key=lambda r:r['hard_score'])
            winners.append(winner)
    save_json(output/'best_of_three_seeds.json',winners)
    (output/'best_of_three_seeds.fasta').write_text(''.join('>'+r['sequence_id']+'|'+r['arm']+'|seed='+str(r['seed'])+'\n'+r['cds_dna']+'\n' for r in winners))
    summarize(output,rows,completed=True)
    return dict(completed=True,metrics=18,weights_unchanged=True,common_scorer_queries=scorer.forward_sequences,
        common_initialization_seconds=init_seconds,common_evaluation_total_seconds=time.perf_counter()-started,
        unique_folded_sequences=len(folds),viennarna_version=next(iter(folds.values()))['viennarna_version'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-input',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.worker_input:
        b5_worker(args.worker_input,args.output)
        return
    started=time.perf_counter()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=args.resume)
    old=json.loads((ROOT/'runs/unrestricted_js_comparison_20261022/plan.json').read_text())
    selected=[r for r in old['records'] if r['n_codons']-1 in (73,466,825)]
    originals={r['sequence_id']:r['initial']['cds_dna'] for r in read_rows(ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl')}
    records=[dict(r,cds_dna=originals[r['sequence_id']]) for r in selected]
    assert len(records)==3
    assert sha256(CHECKPOINT)==old['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p')==old['scorer_sha256']==sha256(PACKAGE/'score_model/best_model.p')
    for name,digest in old['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name))==digest
    sources={str(Path('latent_decoder')/name):digest for name,digest in old['source_sha256'].items()}
    for relative in ['latent_decoder/three_cds_pipeline.py','latent_decoder/analyze_random_validation.py',
        'tools/run_three_cds_native.py','tools/run_original_ribodecode_small.py','tools/windows_torch_io.py']:
        sources[relative]=sha256(ROOT/relative)
    for relative in ['train_optim_sim.py','dataset.py','models.py','check.py','model_config.py','score_model/inference.py']:
        path=PACKAGE/relative
        sources[str(path.relative_to(ROOT))]=sha256(path)
    manifest=json.loads((ROOT/'dataset/manifest.json').read_text())
    plan=dict(records=records,seeds=old['seeds'],arms=list(ARMS),steps=80,config=old['config'],
        checkpoint_sha256=old['checkpoint_sha256'],scorer_sha256=old['scorer_sha256'],source_sha256=sources,
        native_source_sha256=sha256(PACKAGE/'train_optim_sim.py'),native_inference_sha256=sha256(PACKAGE/'score_model/inference.py'),
        cai_training_sha256={k:v for k,v in manifest['files_sha256'].items() if k.startswith('train/')},
        no_edit_budget=True,b5_js_optimization_and_final_limit=.05,gc_filter=False,
        native_protocol='unchanged shipped return selection, one outer round,80 updates,batch1,mfe_weight0',
        timing='fresh process per CDS/seed/method; cold generation separately from common posthoc metrics; pretrained weights reused',
        device='cpu',threads=4,python=sys.version,torch_version=torch.__version__,planned_runs=18)
    if args.resume:
        assert json.loads((output/'plan.json').read_text())==plan
    else:
        save_json(output/'plan.json',plan)
        for directory in ('inputs','logs','metrics','originals','source'):
            (output/directory).mkdir()
        for relative in sources:
            destination=output/'source'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
        (output/'input_sequences.fasta').write_text(''.join('>'+r['sequence_id']+'\n'+r['cds_dna']+'\n' for r in records))
    timings=[]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for record in records:
        for seed in plan['seeds']:
            for arm in ARMS:
                name=f"{record['n_codons']-1}aa_{seed}_{arm}"
                run=output/'runs'/name
                timing_path=output/'logs'/(name+'.timing.json')
                if timing_path.exists():
                    timings.append(json.loads(timing_path.read_text()))
                    continue
                assert not run.exists(), 'Incomplete worker directory retained; inspect before resuming'
                input_path=output/'inputs'/(name+'.json')
                save_json(input_path,dict(record,seed=seed,config=old['config']))
                if arm=='B5_JS':
                    command=[sys.executable,'-u','-m','latent_decoder.three_cds_pipeline','--worker-input',str(input_path),'--output',str(run)]
                else:
                    command=[sys.executable,'-u',str(ROOT/'tools/run_three_cds_native.py'),'--package',str(PACKAGE),
                        '--output',str(run),'--cds',record['cds_dna'],'--seed',str(seed)]
                print(json.dumps(dict(event='worker_start',arm=arm,n_aa=record['n_codons']-1,seed=seed,completed=len(timings))),flush=True)
                begin=time.perf_counter()
                with (output/'logs'/(name+'.log')).open('w') as log:
                    subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
                timer=dict(sequence_id=record['sequence_id'],seed=seed,arm=arm,run_path=str(run),cold_generation_seconds=time.perf_counter()-begin)
                save_json(timing_path,timer)
                timings.append(timer)
                save_json(output/'status.json',dict(stage='generation',completed=len(timings),planned=18,elapsed_seconds=time.perf_counter()-started))
                print(json.dumps(dict(event='worker_complete',completed=len(timings),arm=arm,seconds=timer['cold_generation_seconds'])),flush=True)
    save_json(output/'timings.json',timings)
    save_json(output/'status.json',dict(stage='common_evaluation',completed=18,planned=18))
    receipt=evaluate(output,plan,timings)
    assert sha256(CHECKPOINT)==plan['checkpoint_sha256']
    assert sha256(PACKAGE/'score_model/best_model.p')==plan['scorer_sha256']
    assert all(sha256(ROOT/name)==digest for name,digest in plan['source_sha256'].items())
    receipt['wall_seconds_this_session']=time.perf_counter()-started
    receipt['summed_cold_generation_seconds']=sum(t['cold_generation_seconds'] for t in timings)
    save_json(output/'completion_receipt.json',receipt)
    save_json(output/'status.json',dict(stage='complete',completed=18,planned=18))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
