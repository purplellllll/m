"""Remove the edit cap and final JS filter, retaining B5-JS's optimization penalty."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .budget_comparison import ROOT
from .budget_js_extension import paired_stats, read_rows, run_trial, save_json
from .dual_space_ablation import CountedScorer
from .latent_optimization import TranslationObjective, design_probabilities, state_hash
from .length_diagnostics import length_group
from .model import AA_IDS, CodonVMAE, collate_records, encode_cds
from .train import sha256
from .v2_search import native

ARMS = ('B5_JS_soft_only', 'RiboDecode')


def validate_winner(row, tokens):
    selected = torch.tensor([encode_cds(row['best']['cds_dna'])])
    assert selected.shape == tokens.shape
    assert torch.equal(AA_IDS[selected], AA_IDS[tokens])
    assert torch.equal(selected[:, [0, -1]], tokens[:, [0, -1]])
    assert int((selected != tokens).sum()) == row['best']['changed_codons']
    assert row['best']['hard_score'] >= row['baseline']-1e-7
    assert row['scorer_queries'] == 160


def aggregate(rows, seeds):
    per_cds = []
    for identity in sorted({r['sequence_id'] for r in rows}):
        current = [r for r in rows if r['sequence_id'] == identity]
        if {(r['arm'], r['seed']) for r in current} != {(a,s) for a in ARMS for s in seeds}:
            continue
        assert len(current) == 2*len(seeds)
        item = dict(sequence_id=identity, n_aa=current[0]['n_aa'], group=length_group(current[0]['n_aa']),
                    baseline=float(np.mean([r['baseline'] for r in current])))
        for arm in ARMS:
            selected = [r for r in current if r['arm']==arm]
            item[arm] = dict(score=float(np.mean([r['best']['hard_score'] for r in selected])),
                gain=float(np.mean([r['gain'] for r in selected])),
                changed_fraction=float(np.mean([r['changed_fraction'] for r in selected])))
        selected = [r for r in current if r['arm']=='B5_JS_soft_only']
        item['B5_JS_soft_only'].update(mean_soft_js=float(np.mean([r['best']['soft_js'] for r in selected])),
            js05_violation_runs=sum(r['best']['soft_js']>.05+1e-7 for r in selected),
            mean_score_if_js05_filter=float(np.mean([r['best_if_js05']['hard_score'] for r in selected])))
        item['difference'] = item['B5_JS_soft_only']['score']-item['RiboDecode']['score']
        per_cds.append(item)
    means = {a:{key:float(np.mean([r[a][key] for r in per_cds])) for key in ('score','gain','changed_fraction')} for a in ARMS} if per_cds else {}
    return dict(completed_runs=len(rows), planned_runs=48, complete_cds=len(per_cds), means=means, per_cds=per_cds,
                paired_difference=paired_stats([r['difference'] for r in per_cds]) if per_cds else None,
                interpretation='Seeds averaged within CDS; reused development cohort; intervals exploratory, not independent validation')


def report(output, rows, seeds, completed=False):
    summary = aggregate(rows, seeds)
    summary['completed'] = completed
    save_json(output/'summary.json', summary)
    text = '# B5-JS取消修改率上限及最终JS硬筛选的对照\n\n'
    text += '状态：'+('完成' if completed else '运行中')+f"；{len(rows)}/48组，完整配对{summary['complete_cds']}/8条CDS。\n\n"
    text += 'B5-JS保留优化过程中的JS≤.05增广拉格朗日惩罚，rho=1；不限制修改率，最终直接取最高Hard Score。两方法均保持同义修改、固定起止密码子，不加GC筛选。旧VMAE epoch50、原学习率/缩放/正则不变。相同8条CDS、3种子、80步/160次搜索评分；不是V2训练结果。\n\n'
    if summary['per_cds']:
        text += '| 方法 | 平均Hard Score | 平均Hard Gain | 平均修改率（全部codon分母） |\n|---|---:|---:|---:|\n'
        for arm, value in summary['means'].items():
            text += f"| {arm} | {value['score']:.6f} | {value['gain']:.6f} | {value['changed_fraction']:.2%} |\n"
        d=summary['paired_difference']
        text += f"\nB5-JS减原版：{d['mean']:+.6f}；领先{d['wins']}/{d['n_cds']}条；探索性CDS bootstrap 95%区间：{d['exploratory_ci95']}。\n\n"
        text += '| aa长度 | B5-JS Score | 原版Score | 差值 | B5-JS修改率 | 原版修改率 | B5-JS平均soft JS |\n|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in sorted(summary['per_cds'],key=lambda x:x['n_aa']):
            a,b=r['B5_JS_soft_only'],r['RiboDecode']
            text += f"| {r['n_aa']} | {a['score']:.6f} | {b['score']:.6f} | {r['difference']:+.6f} | {a['changed_fraction']:.2%} | {b['changed_fraction']:.2%} | {a['mean_soft_js']:.6f} |\n"
    text += '\nJS衡量投影前soft分布相对于clean decoder分布，不能将其直接解释为最终DNA的JS。保留JS优化惩罚与彻底移除所有JS机制是不同实验；本轮不能据此判定VMAE的无约束能力上限。最大修改预算100%仅是代码中实现“无修改率上限”的方式，可修改位点不超过总codon数。评分器为冻结代理模型，未做真实表达实验。\n'
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
    old=ROOT/'runs/budget_comparison_20261021_v2'
    previous=json.loads((old/'plan.json').read_text())
    prior_js=json.loads((ROOT/'runs/budget_js_extension_20261021/plan.json').read_text())
    config=previous['frozen_b5_config']
    records,seeds=previous['records'],previous['seeds']
    originals=read_rows(ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl')
    checkpoint=ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint)==previous['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p')==previous['scorer_sha256']
    for name,digest in prior_js['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name))==digest
    source_names=sorted(set(prior_js['source_sha256'])|{'unrestricted_js_comparison.py','v2_search.py'})
    plan=dict(records=records,seeds=seeds,arms=ARMS,config=config,checkpoint_sha256=previous['checkpoint_sha256'],
        scorer_sha256=previous['scorer_sha256'],device='cpu',torch_version=torch.__version__,
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in source_names},
        original_cds_sha256=sha256(ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl'),
        steps=80,search_scorer_queries=160,planned_runs=48,js_optimization_limit=.05,rho=1.,
        output_edit_rate_cap=None,final_js_filter=False,gc_filter=False,
        primary='Same historical eight CDS/three seeds; no edit cap for either method; retain B5-JS optimization penalty. Final selection by hard score only.',
        source_checkpoint='Original 50-epoch VMAE, not incomplete V2',
        native='Shipped 65-channel generator, AdamW .1 wd1e-4, native target100 loss/scheduler; same scorer and 80-step archive',
        setup='B5 clean reference/baseline cached once per CDS; native one baseline per run, outside matched 160 search queries')
    plan=json.loads(json.dumps(plan))
    if args.resume:
        assert json.loads((output/'plan.json').read_text())==plan
    else:
        save_json(output/'plan.json',plan)
        (output/'source').mkdir()
        for n in source_names:
            shutil.copyfile(Path(__file__).with_name(n),output/'source'/n)
    (output/'results').mkdir(exist_ok=True)
    rows=[json.loads(p.read_text()) for p in sorted((output/'results').glob('*.json'))]
    done={(r['sequence_id'],r['seed'],r['arm']) for r in rows}
    assert len(done)==len(rows)
    for row in rows:
        if row['arm']=='B5_JS_soft_only':
            for name,digest in row['artifact_sha256'].items():
                assert sha256(Path(row['run_path'])/name)==digest
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=CodonVMAE(**saved['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    del saved
    scorer=CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes=state_hash(model),state_hash(scorer)
    started=time.monotonic()
    for record in records:
        identity=record['sequence_id']
        dna=next(r['initial']['cds_dna'] for r in originals if r['sequence_id']==identity)
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
                begin=time.monotonic()
                if arm=='B5_JS_soft_only':
                    run=output/'trajectories'/f'{identity}_{seed}'
                    run.mkdir(parents=True,exist_ok=True)
                    # On interrupted resume, replace the incomplete trace instead of appending duplicate steps.
                    (run/'steps.jsonl').write_text('',encoding='utf-8')
                    task=dict(sequence_id=identity,seed=seed,budget_percent=100,js_limit=.05)
                    internal=run_trial(model,scorer,tokens,mu,sigma,reference,config,task,baseline,run,None,False)
                    row=dict(internal)
                    row.update(best=internal['best_budget_only'],best_if_js05=internal['best'],
                               gain=internal['best_budget_only']['hard_score']-baseline,
                               filter_gap=0.,
                               removed_js_filter_gain=internal['best_budget_only']['hard_score']-internal['best']['hard_score'])
                else:
                    row=native(scorer,tokens,identity,seed,100)
                    assert abs(row['baseline']-baseline)<1e-6
                row.update(arm=arm,n_aa=tokens.shape[1]-1,changed_fraction=row['best']['changed_codons']/tokens.shape[1],
                           elapsed_seconds=time.monotonic()-begin,final_js_filter=False,edit_rate_cap=None)
                validate_winner(row,tokens)
                save_json(output/'results'/f'{identity}_{seed}_{arm}.json',row)
                rows.append(row)
                summary=report(output,rows,seeds)
                save_json(output/'status.json',dict(completed=len(rows),planned=48,elapsed_seconds=time.monotonic()-started,
                          last_arm=arm,last_n_aa=tokens.shape[1]-1,last_seconds=row['elapsed_seconds']))
                print(json.dumps(dict(event='run_complete',arm=arm,completed=len(rows),score=row['best']['hard_score'],
                                      gain=row['gain'],changed_fraction=row['changed_fraction'],seconds=row['elapsed_seconds'])),flush=True)
    assert len(rows)==48 and aggregate(rows,seeds)['complete_cds']==8
    assert hashes==(state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    save_json(output/'completion_receipt.json',dict(completed=True,runs=48,weights_unchanged=True,
               session_seconds=time.monotonic()-started,session_scorer_queries=scorer.forward_sequences))
    report(output,rows,seeds,completed=True)


if __name__=='__main__':
    main()
