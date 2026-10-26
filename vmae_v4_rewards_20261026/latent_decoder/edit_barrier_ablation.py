"""Paired single-term ablations of the existing frozen B5-JS search."""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .budget_comparison import ROOT
from .budget_js_extension import read_rows, run_trial, save_json
from .dual_space_ablation import CountedScorer
from .latent_optimization import TranslationObjective, design_probabilities, state_hash
from .model import CodonVMAE, collate_records
from .train import sha256
from .unrestricted_js_comparison import validate_winner

ARMS = ('control', 'no_js_penalty', 'no_decoder_kl', 'no_direct_l2')


def settings(config, arm):
    assert arm in ARMS
    config = copy.deepcopy(config)
    if arm == 'no_decoder_kl':
        config['regularization']['decoder_kl'] = 0.
    if arm == 'no_direct_l2':
        config['regularization']['direct_l2'] = 0.
    # JS <= ln(2): this ceiling makes the unchanged controller identically inactive.
    limit = math.log(2)+1e-6 if arm == 'no_js_penalty' else .05
    return config, limit


def crossing_metrics(trace, original):
    assert [r['step'] for r in trace] == list(range(1, 81))
    codons = [original[i:i+3] for i in range(0, len(original), 3)]
    ever = set()
    first = {str(t): None for t in (0, .05, .10, .20)}
    for row in trace:
        dna = row['cds_dna']
        assert len(dna) == len(original)
        changed = {i for i, c in enumerate(codons) if c != dna[3*i:3*i+3]}
        assert len(changed) == row['changed_codons']
        ever.update(changed)
        for threshold in (0, .05, .10, .20):
            if first[str(threshold)] is None and len(changed)/len(codons) > threshold:
                first[str(threshold)] = row['step']
    return dict(first_step_exceeding_fraction=first,
                peak_edited_fraction=max(r['changed_codons'] for r in trace)/len(codons),
                ever_edited_fraction=len(ever)/len(codons),
                endpoint_edited_fraction=trace[-1]['changed_codons']/len(codons),
                js05_feasible_step_fraction=sum(r['soft_js'] <= .05+1e-7 for r in trace)/80)


def summarize(output, rows, seeds, completed=False):
    identities = sorted({r['sequence_id'] for r in rows})
    paired = [i for i in identities if {(r['seed'],r['arm']) for r in rows if r['sequence_id']==i}
              == {(s,a) for s in seeds for a in ARMS}]
    means = {}
    for arm in ARMS:
        current = [r for r in rows if r['sequence_id'] in paired and r['arm']==arm]
        if not current:
            continue
        means[arm] = {key: float(np.mean([r[key] for r in current])) for key in
            ('raw_score','legal_score','raw_gain','legal_gain','raw_edited_fraction','legal_edited_fraction',
             'peak_edited_fraction','ever_edited_fraction','endpoint_edited_fraction','js05_feasible_step_fraction')}
    summary = dict(completed=completed,completed_runs=len(rows),planned_runs=36,complete_paired_cds=len(paired),
                   means=means,per_cds=[])
    for identity in paired:
        current = [r for r in rows if r['sequence_id']==identity]
        item = dict(sequence_id=identity,n_aa=current[0]['n_aa'])
        for arm in ARMS:
            selected = [r for r in current if r['arm']==arm]
            item[arm] = {key: float(np.mean([r[key] for r in selected])) for key in
                ('raw_score','legal_score','raw_edited_fraction','legal_edited_fraction','peak_edited_fraction','ever_edited_fraction')}
        summary['per_cds'].append(item)
    save_json(output/'summary.json',summary)
    text = '# B5-JS 单项消融：密码子切换门槛\n\n'
    text += f"{'完成' if completed else '运行中'}：{len(rows)}/36组，完整配对{len(paired)}/3条CDS。每条三个相同种子；9组对照复用、27组新增。\n\n"
    text += '无修改预算或GC筛选；所有组以相同JS≤.05统计合法成绩，同时保留不筛JS的最高分。模型、学习率、缩放、起点、步数保持一致。JS惩罚消融通过将控制器阈值置于JS理论上限ln(2)之上实现，并逐步断言惩罚和乘子恒为零；没有改变用于比较的合法阈值.05。\n\n'
    if means:
        text += '| 组别 | 无筛选Score | 合法Score | 无筛选赢家修改率 | 合法赢家修改率 | 搜索峰值修改率 | 曾修改位置比例 |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for arm,m in means.items():
            text += f"| {arm} | {m['raw_score']:.6f} | {m['legal_score']:.6f} | {m['raw_edited_fraction']:.2%} | {m['legal_edited_fraction']:.2%} | {m['peak_edited_fraction']:.2%} | {m['ever_edited_fraction']:.2%} |\n"
        for item in sorted(summary['per_cds'],key=lambda x:x['n_aa']):
            text += f"\n{item['n_aa']} aa：\n\n| 组别 | 无筛选Score | 合法Score | 无筛选赢家修改率 | 合法赢家修改率 |\n|---|---:|---:|---:|---:|\n"
            for arm in ARMS:
                m=item[arm]
                text += f"| {arm} | {m['raw_score']:.6f} | {m['legal_score']:.6f} | {m['raw_edited_fraction']:.2%} | {m['legal_edited_fraction']:.2%} |\n"
    text += '\n探索性机制试验，仅3条已使用过的开发序列，不能泛化为总体优势。单项消融不能排除各项之间的交互作用；修改更多不代表合法评分更高。保留其他旧实验的暂停状态。\n'
    (output/'report.md').write_text(text,encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(4)
    prior_dir=ROOT/'runs/unrestricted_js_comparison_20261022'
    prior=json.loads((prior_dir/'plan.json').read_text())
    records=[r for r in prior['records'] if r['n_codons']-1 in (73,466,825)]
    assert len(records)==3
    seeds=prior['seeds']
    controls=[json.loads(p.read_text()) for p in (prior_dir/'results').glob('*B5_JS*.json')]
    controls={(r['sequence_id'],r['seed']):r for r in controls if r['n_aa'] in (73,466,825)}
    assert len(controls)==9
    for name,digest in prior['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name))==digest
    checkpoint=ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint)==prior['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p')==prior['scorer_sha256']
    sources=dict(prior['source_sha256'])
    sources[Path(__file__).name]=sha256(Path(__file__))
    plan=dict(records=records,seeds=seeds,arms=list(ARMS),steps=80,scorer_queries_per_new_trial=160,
        checkpoint_sha256=prior['checkpoint_sha256'],scorer_sha256=prior['scorer_sha256'],source_sha256=sources,
        configs={arm:settings(prior['config'],arm)[0] for arm in ARMS},
        js_controller_limits={arm:settings(prior['config'],arm)[1] for arm in ARMS},common_legal_js_limit=.05,
        no_edit_budget=True,gc_filter=False,planned_runs=36,new_runs=27,reused_controls=9,
        device='cpu',torch_threads=4,torch_version=torch.__version__,
        original_cds_sha256=prior['original_cds_sha256'],control_plan_sha256=sha256(prior_dir/'plan.json'),
        control_result_sha256={p.name:sha256(p) for p in (prior_dir/'results').glob('*B5_JS*.json')
                              if (json.loads(p.read_text())['sequence_id'],json.loads(p.read_text())['seed']) in controls})
    if args.resume:
        assert json.loads((output/'plan.json').read_text())==plan
    else:
        save_json(output/'plan.json',plan)
        (output/'source').mkdir()
        for name in sources:
            shutil.copyfile(Path(__file__).with_name(name),output/'source'/name)
    (output/'results').mkdir(exist_ok=True)
    rows=[json.loads(p.read_text()) for p in (output/'results').glob('*.json')]
    done={(r['sequence_id'],r['seed'],r['arm']) for r in rows}
    assert len(done)==len(rows)
    original_path=ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl'
    assert sha256(original_path)==plan['original_cds_sha256']
    originals={r['sequence_id']:r['initial']['cds_dna'] for r in read_rows(original_path)}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=CodonVMAE(**saved['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    del saved
    scorer=CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes=state_hash(model),state_hash(scorer)
    started=time.monotonic()
    new_runs=0
    for record in records:
        identity=record['sequence_id']
        dna=originals[identity]
        tokens=collate_records([dict(cds_dna=dna)])
        with torch.no_grad():
            mu,logvar=model.encode(tokens,tokens)
            sigma=(.5*logvar).exp()
            reference=design_probabilities(model.decode(mu,tokens),tokens)
            baseline=float(scorer(F.one_hot(tokens,64).float(),tokens))
        for seed in seeds:
            for arm in ARMS:
                if (identity,seed,arm) in done:
                    continue
                print(json.dumps(dict(event='run_start',arm=arm,n_aa=tokens.shape[1]-1,seed=seed,completed=len(rows))),flush=True)
                control=controls[identity,seed]
                assert abs(control['baseline']-baseline)<1e-6
                if arm=='control':
                    internal=control
                else:
                    config,js_limit=settings(prior['config'],arm)
                    run=output/'trajectories'/f'{identity}_{seed}_{arm}'
                    run.mkdir(parents=True,exist_ok=True)
                    (run/'steps.jsonl').write_text('',encoding='utf-8')
                    internal=run_trial(model,scorer,tokens,mu,sigma,reference,config,
                        dict(sequence_id=identity,seed=seed,budget_percent=100,js_limit=js_limit),baseline,run,None,False)
                    new_runs+=1
                for name,digest in internal['artifact_sha256'].items():
                    assert sha256(Path(internal['run_path'])/name)==digest
                assert internal['initial_latent_sha256']==control['initial_latent_sha256']
                trace=read_rows(Path(internal['run_path'])/'steps.jsonl')
                if arm=='no_js_penalty':
                    assert all(t['js_optimization_penalty']==0. and t['js_dual_after']==0. for t in trace)
                raw,legal=internal['best_budget_only'],internal['best_js05']
                assert legal['soft_js']<=.05+1e-7
                assert raw['hard_score']==max([baseline]+[t['hard_score'] for t in trace])
                row=dict(internal,arm=arm,reused_control=arm=='control',best=raw,
                    raw_score=raw['hard_score'],legal_score=legal['hard_score'],raw_gain=raw['hard_score']-baseline,
                    legal_gain=legal['hard_score']-baseline,raw_edited_fraction=raw['changed_codons']/tokens.shape[1],
                    legal_edited_fraction=legal['changed_codons']/tokens.shape[1],**crossing_metrics(trace,dna))
                validate_winner(row,tokens)
                save_json(output/'results'/f'{identity}_{seed}_{arm}.json',row)
                rows.append(row)
                summarize(output,rows,seeds)
                save_json(output/'status.json',dict(completed=len(rows),planned=36,new_runs_this_session=new_runs,
                    elapsed_seconds=time.monotonic()-started,last_arm=arm,last_n_aa=tokens.shape[1]-1))
                print(json.dumps(dict(event='run_complete',completed=len(rows),arm=arm,raw_score=row['raw_score'],
                    legal_score=row['legal_score'],raw_edit=row['raw_edited_fraction'],legal_edit=row['legal_edited_fraction'])),flush=True)
    assert len(rows)==36 and summarize(output,rows,seeds)['complete_paired_cds']==3
    assert hashes==(state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert scorer.forward_sequences==3+160*new_runs
    save_json(output/'completion_receipt.json',dict(completed=True,runs=36,weights_unchanged=True,
        session_seconds=time.monotonic()-started,new_runs_this_session=new_runs,session_scorer_queries=scorer.forward_sequences))
    summarize(output,rows,seeds,completed=True)


if __name__=='__main__':
    main()
