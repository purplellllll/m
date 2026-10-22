"""Paired validation of JS-aware B5 against frozen, already completed baselines."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .b6_reproducibility import frozen_config, verify_reference
from .dual_space_ablation import CountedScorer, optimize_batch
from .latent_optimization import TranslationObjective, state_hash
from .length_diagnostics import encoder_row, length_group
from .model import CodonVMAE, collate_records
from .train import codon_frequency, load_data, sha256

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(rho=1.)


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def run_metrics(run):
    result = next(r for r in read_rows(run/'results.jsonl') if r['arm']=='B5')
    diag = read_rows(run/'length_search_summary.jsonl')[0]
    steps = read_rows(run/'length_search_steps.jsonl')
    search = read_rows(run/'steps.jsonl')
    best = result['best']
    assert len(steps)==len(search)==80
    assert result['scorer_forward_sequences']==240 and result['backward_sequence_evaluations']==80
    assert best['feasible'] and best['protein_preserved']
    assert best['js_from_initial']<=.05+1e-7 and best['mutable_hamming']<=.2+1e-7
    assert abs(best['gc_fraction']-result['initial']['gc_fraction'])<=.05+1e-7
    assert best['cds_dna'][:3]==result['initial']['cds_dna'][:3] and best['cds_dna'][-3:]==result['initial']['cds_dna'][-3:]
    return dict(legal_gain=result['best_hard_gain'], raw_gain=diag['best_before_filter']['hard_score']-diag['baseline'],
        filter_gap=diag['filter_loss'], legal_js=best['js_from_initial'],legal_edit=best['mutable_hamming'],
        feasible_step_fraction=sum(r['feasible'] for r in steps)/80,
        js_rejection_fraction=sum('js' in r['rejected_by'] for r in steps)/80,
        max_js=max(r['js'] for r in steps),final_js=steps[-1]['js'],best_step=best['step'],
        max_dual=max(r.get('js_dual_after',0.) for r in search),
        initial_score=result['initial']['hard_score'],initial_cds=result['initial']['cds_dna'],
        random_start_hash=read_rows(run/'random_initial.jsonl')[0]['latent_sha256'])


def resume_rows(prior, tasks, source_hashes, baseline_hashes):
    plan=json.loads((prior/'plan.json').read_text())
    assert plan['tasks']==tasks and plan['settings']==SETTINGS
    assert plan['baseline_files_sha256']==baseline_hashes
    for name,digest in plan['source_sha256'].items():
        assert sha256(prior/'source'/name)==digest
        if name!='js_constraint_validation.py':
            assert source_hashes[name]==digest, 'Search algorithm changed; cannot merge runs'
    rows=json.loads((prior/'summary.json').read_text())['per_run']
    assert 0<len(rows)<=len(tasks)
    inherited_hashes={str(prior/'plan.json'):sha256(prior/'plan.json')}
    for task,row in zip(tasks,rows):
        assert all(task[k]==row[k] for k in ('stage','sequence_id','seed'))
        run=Path(row.get('run_path',prior/'runs'/row['stage']/row['sequence_id']/str(row['seed'])))
        assert run_metrics(run)==row['constrained']
        assert run_metrics(Path(task['baseline']))==row['baseline']
        row['run_path']=str(run.resolve())
        for name in ('config.json','results.jsonl','length_search_summary.jsonl','length_search_steps.jsonl','random_initial.jsonl','steps.jsonl'):
            inherited_hashes[str(run/name)]=sha256(run/name)
    return rows,inherited_hashes


def report(output, rows):
    stages = {}
    for stage in ('development','long_test'):
        complete = [r for r in rows if r['stage']==stage]
        sequences = []
        for identity in sorted({r['sequence_id'] for r in complete}):
            runs = [r for r in complete if r['sequence_id']==identity]
            expected = 1 if stage=='development' else 3
            if len(runs)!=expected: continue
            sequences.append(dict(sequence_id=identity,
                **{side:{k:float(np.mean([r[side][k] for r in runs])) for k in
                    ('legal_gain','raw_gain','filter_gap','legal_js','legal_edit','feasible_step_fraction','js_rejection_fraction')}
                   for side in ('baseline','constrained')}))
        if not sequences: continue
        aggregate = {side:{k:float(np.mean([r[side][k] for r in sequences])) for k in sequences[0][side]}
                     for side in ('baseline','constrained')}
        contrasts = {}
        for key in aggregate['baseline']:
            delta = np.array([r['constrained'][key]-r['baseline'][key] for r in sequences])
            rng = np.random.default_rng(20261021)
            draws = delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
            contrasts[key] = dict(mean=float(delta.mean()),exploratory_ci95=np.quantile(draws,[.025,.975]).tolist())
        stages[stage]=dict(complete_cds=len(sequences),means=aggregate,paired_deltas=contrasts,per_cds=sequences,
            descriptive_success=contrasts['legal_gain']['mean']>0 and contrasts['filter_gap']['mean']<0,
            both_exploratory_intervals_support=contrasts['legal_gain']['exploratory_ci95'][0]>0 and contrasts['filter_gap']['exploratory_ci95'][1]<0)
    summary = dict(completed=len(rows),planned=27,stages=stages,per_run=rows,
        interpretation='Exploratory; reused diagnostic test cohort; parameters fixed before development results; no corrected confirmatory inference')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text = '# JS 约束纳入 B5 优化的配对验证\n\n'
    text += f'完成 {len(rows)}/27 组新搜索。JS≤.05、修改率≤.2、GC变化≤.05全部保留；原模型、学习率、缩放及三项正则不变。\n\n'
    text += '方法：对可变位平均 JS 使用规范化违反量 g=JS/.05−1，加入 PHR 不等式增广拉格朗日项；rho=1，初始乘子0，逐步非负更新。新方法与原方法各80次更新、240次搜索评分，初始化另计6次；JS计算不新增评分调用。候选仍由原来的全部硬约束筛选。\n\n'
    text += '| 阶段 | 完整CDS | 原合法Gain | 新合法Gain | 原Filter Gap | 新Filter Gap | 原可行步比例 | 新可行步比例 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for stage,data in stages.items():
        a,b=data['means']['baseline'],data['means']['constrained']
        text+=f"| {stage} | {data['complete_cds']} | {a['legal_gain']:.6f} | {b['legal_gain']:.6f} | {a['filter_gap']:.6f} | {b['filter_gap']:.6f} | {a['feasible_step_fraction']:.2%} | {b['feasible_step_fraction']:.2%} |\n"
    text += '\n判据：长序列同CDS/seed配对后，平均合法Hard Gain提高且Filter Gap下降；同时保留筛前Gain，避免只用降低筛前分数来宣称成功。Filter Gap定义为轨迹最高离散分数（含原序列）减合法档案最高分，不是soft-hard差。三种子先在CDS内平均，再进行8条CDS的配对bootstrap；区间探索性，未做多重确认性校正。\n'
    text += '\n开发集沿用Rz实验的3条CDS与seed20261023；长序列验证沿用长度诊断的8条≥700 aa CDS及全部三个seed，属于复用测试样本的机制验证，不是新鲜确认性测试。基线直接使用已完成且哈希校验的原B5结果，逐run核对原序列、初始化分数与随机潜变量哈希。\n'
    text += '\n增广惩罚允许搜索中暂时违反JS，不能保证每一步可行；最终输出仍严格按原门槛选择。没有自动替换冻结主配置或进行独立表达验证。\n'
    (output/'report.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume-from',help='Reuse verified complete runs in a new output directory; incomplete runs are restarted')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    reference=ROOT/'deliverables/b0_b5_frozen_20261020'
    manifest=verify_reference(reference)
    original,base=frozen_config(reference,['B5'])
    checkpoint_path=ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path)==manifest['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p')==manifest['scorer_sha256']
    assert sha256(ROOT/'dataset/manifest.json')==original['dataset_manifest_sha256']
    rz=ROOT/'runs/rz_relaxation_20261021';diagnostic=ROOT/'runs/length_diagnostics_20261021'
    for prior in (rz,diagnostic):
        receipt=json.loads((prior/'completion_receipt.json').read_text())
        assert receipt['completed']
    rz_plan=json.loads((rz/'plan.json').read_text())
    old_plan=json.loads((diagnostic/'plan.json').read_text())
    assert rz_plan['checkpoint_sha256']==old_plan['checkpoint_sha256']==manifest['checkpoint_sha256']
    splits,_=load_data(ROOT/'dataset',allow_development=True)
    by_id={r['sequence_id']:r for r in splits['val']+splits['test']}
    tasks=[]
    for r in rz_plan['search_records']:
        tasks.append(dict(stage='development',sequence_id=r['sequence_id'],seed=rz_plan['seed'],baseline=str(rz/r['sequence_id']/'0.1')))
    for r in old_plan['search_records']:
        if length_group(r['n_codons']-1)!='long':continue
        for seed in old_plan['seeds']:
            tasks.append(dict(stage='long_test',sequence_id=r['sequence_id'],seed=seed,
                              baseline=str(diagnostic/'runs'/r['sequence_id']/str(seed))))
    assert len(tasks)==27 and len({t['sequence_id'] for t in tasks if t['stage']=='long_test'})==8
    base.update(batch_size=1,random_start_rms=.05,timeout_seconds=24*3600,length_diagnostics=True)
    sources=['js_constraint_validation.py','js_constraint.py','dual_space_ablation.py','length_diagnostics.py']
    (output/'source').mkdir()
    for name in sources:shutil.copyfile(Path(__file__).with_name(name),output/'source'/name)
    source_hashes={name:sha256(output/'source'/name) for name in sources}
    baseline_hashes={str(Path(t['baseline'])/name):sha256(Path(t['baseline'])/name)
        for t in tasks for name in ('results.jsonl','length_search_summary.jsonl','length_search_steps.jsonl','random_initial.jsonl')}
    rows=[];inherited_hashes={}
    if args.resume_from:
        rows,inherited_hashes=resume_rows(Path(args.resume_from).resolve(),tasks,source_hashes,baseline_hashes)
    plan=dict(version='js-aware-b5-20261021',origin_skill='academic-research-suite/experiment-agent',
        tasks=tasks,settings=SETTINGS,original_config=original,base_config=base,
        policy='fixed rho=1 and dual update eta=rho; no development-result tuning; all declared long-test runs execute',
        criterion='paired mean legal Hard Gain increases AND Filter Gap decreases; report raw gain and uncertainty',
        checkpoint_sha256=manifest['checkpoint_sha256'],scorer_sha256=manifest['scorer_sha256'],
        source_sha256=source_hashes,baseline_files_sha256=baseline_hashes,
        frozen_main_config_changed=False,validation='existing diagnostic long-test cohort; exploratory mechanism check',
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        inherited_complete_runs=len(rows),inherited_files_sha256=inherited_hashes)
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if rows: report(output,rows)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    scorer=CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes=state_hash(model),state_hash(scorer)
    counts=codon_frequency(splits['train']);start=time.monotonic()
    inherited_count=len(rows)
    for index,task in enumerate(tasks):
        if index<inherited_count:continue
        record=by_id[task['sequence_id']]
        baseline_path=Path(task['baseline'])
        if task['stage']=='development':
            old_config=json.loads((baseline_path/'config.json').read_text())
        else:
            old_config=copy.deepcopy(old_plan['actual_config'])
        for key in ('lr','latent_scale','direct_scale','direct_lr_factor','regularization','max_js','max_mutable_hamming','max_gc_change','steps','arm_lr_multipliers'):
            assert base[key]==old_config[key],key
        tokens=collate_records([record])
        with torch.no_grad():mu,logvar=model.encode(tokens,tokens)
        config=copy.deepcopy(base)
        config.update(random_start_seed=task['seed'],diagnostic_mu_rms=encoder_row(mu,logvar,tokens)['mu_rms'],js_constraint=SETTINGS)
        run=output/'runs'/task['stage']/record['sequence_id']/str(task['seed']);run.mkdir(parents=True,exist_ok=False)
        (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        print(json.dumps(dict(event='trial_start',completed=index,total=27,n_aa=record['n_codons']-1,**task)),flush=True)
        torch.manual_seed(task['seed'])
        optimize_batch(model,scorer,[record],counts,config,run,start)
        before,after=run_metrics(baseline_path),run_metrics(run)
        assert before['random_start_hash']==after['random_start_hash']
        assert before['initial_cds']==after['initial_cds']==record['cds_dna']
        assert abs(before['initial_score']-after['initial_score'])<1e-6
        row=dict(stage=task['stage'],sequence_id=record['sequence_id'],n_aa=record['n_codons']-1,seed=task['seed'],baseline=before,constrained=after,run_path=str(run.resolve()))
        rows.append(row);report(output,rows)
        print(json.dumps(dict(event='trial_complete',completed=len(rows),total=27,stage=task['stage'],
            legal_gain_delta=after['legal_gain']-before['legal_gain'],filter_gap_delta=after['filter_gap']-before['filter_gap'],seconds=time.monotonic()-start)),flush=True)
    assert scorer.forward_sequences==(27-inherited_count)*246
    assert hashes==(state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert source_hashes=={name:sha256(Path(__file__).with_name(name)) for name in sources}
    assert all(sha256(path)==digest for path,digest in baseline_hashes.items())
    assert all(sha256(path)==digest for path,digest in inherited_hashes.items())
    verify_reference(reference)
    (output/'completion_receipt.json').write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start,
        weights_unchanged=True,baseline_files_unchanged=True,matched_random_starts=True,scorer_calls=scorer.forward_sequences+inherited_count*246,
        this_process_scorer_calls=scorer.forward_sequences,inherited_complete_runs=inherited_count,
        thresholds_unchanged=True,main_frozen_config_unchanged=True)),encoding='utf-8')


if __name__=='__main__':main()
