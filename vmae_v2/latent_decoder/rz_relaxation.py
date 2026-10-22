"""Quick validation-set-only ablation of B5 latent L2, with all other settings frozen."""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch

from .b6_reproducibility import frozen_config, verify_reference
from .dual_space_ablation import CountedScorer, optimize_batch
from .latent_optimization import TranslationObjective, state_hash
from .length_diagnostics import encoder_row, length_group, select_records
from .model import CodonVMAE, collate_records
from .train import codon_frequency, load_data, sha256

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [.1, .01, 0.]
SEED = 20261023


def trial_config(base, weight):
    config = copy.deepcopy(base)
    config['regularization']['latent_l2'] = weight
    for key in base:
        if key != 'regularization':
            assert config[key] == base[key]
    for key in ('direct_l2','decoder_kl'):
        assert config['regularization'][key] == base['regularization'][key]
    return config


def report(output, rows):
    summary = []
    for weight in WEIGHTS:
        group = [r for r in rows if r['latent_l2_weight'] == weight]
        if group:
            summary.append(dict(latent_l2_weight=weight, cds=len(group),
                **{key:sum(r[key] for r in group)/len(group) for key in
                   ('feasible_gain','raw_gain','filter_loss','feasible_edit_fraction','raw_edit_fraction',
                    'first20_actual_z_update_rms','max_latent_rms','js_rejection_fraction','total_loss_gradient_dims99')}))
    (output/'summary.json').write_text(json.dumps(dict(completed=len(rows),planned=9,means=summary,per_cds=rows),indent=2),encoding='utf-8')
    text = '# 潜变量 L2 放宽快速开发集实验\n\n'
    text += f'完成 {len(rows)}/9 组；短、中、长各1条验证集CDS，固定seed={SEED}。仅改变 latent_l2 权重，0.1→0.01→0；原模型、学习率、缩放、其他正则及可行性门槛均保留。\n\n'
    text += '| 权重 | 档位 | aa | 合法增益 | 筛前增益 | 筛选损失 | 合法修改率 | 最大latent RMS | 前20步Δz RMS |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    text += '\n'.join(f"| {r['latent_l2_weight']} | {r['group']} | {r['n_aa']} | {r['feasible_gain']:.6f} | {r['raw_gain']:.6f} | {r['filter_loss']:.6f} | {r['feasible_edit_fraction']:.2%} | {r['max_latent_rms']:.4f} | {r['first20_actual_z_update_rms']:.6f} |" for r in rows)
    text += '\n\nR_z 是优化坐标v的均方，损失项是lambda_z*R_z；提高lambda_z会加重惩罚。当前RMS上限16是数值保护，已有测试轨迹最大约0.82，未触限。放宽lambda_z不改变冻结编码器的AU数量，也不保证增加有用信息；必须查看同样约束下的离散评分。\n'
    text += '\n本次样本极小，只作快速机制排查，不选择正式测试参数，不替换冻结主配置。诊断总损失梯度能量维数不等于信息维度。完整配置与逐步候选见各run；基础配置快照及源码保留。\n'
    (output/'report.md').write_text(text,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    reference = ROOT/'deliverables/b0_b5_frozen_20261020'
    manifest = verify_reference(reference)
    original, base = frozen_config(reference,['B5'])
    checkpoint_path = ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path) == manifest['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p') == manifest['scorer_sha256']
    assert sha256(ROOT/'dataset/manifest.json') == original['dataset_manifest_sha256']
    splits, _ = load_data(ROOT/'dataset',allow_development=True)
    selected = select_records(splits['val'],1)
    base.update(batch_size=1,random_start_rms=.05,random_start_seed=SEED,
                timeout_seconds=24*3600,length_diagnostics=True)
    source = output/'source';source.mkdir()
    names = ['rz_relaxation.py','length_diagnostics.py','dual_space_ablation.py']
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    source_hashes = {name:sha256(source/name) for name in names}
    plan = dict(version='rz-relaxation-validation-20261021',origin_skill='academic-research-suite/experiment-agent',
        split='val',seed=SEED,weights=WEIGHTS,search_records=selected,base_config=base,original_config=original,
        selection='score-blind salted hash, one unique component per length bin',
        changed='only lambda_z; not R_z radius, not direct-residual L2',
        main_config_changed=False, source_sha256=source_hashes,checkpoint_sha256=manifest['checkpoint_sha256'])
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes = state_hash(model),state_hash(scorer)
    counts = codon_frequency(splits['train'])
    rows = [];starts = {};start = time.monotonic()
    for record in selected:
        tokens = collate_records([record])
        with torch.no_grad():
            mu, logvar = model.encode(tokens,tokens)
            base['diagnostic_mu_rms'] = encoder_row(mu,logvar,tokens)['mu_rms']
        for weight in WEIGHTS:
            torch.manual_seed(SEED)
            config = trial_config(base,weight)
            run = output/record['sequence_id']/str(weight);run.mkdir(parents=True,exist_ok=False)
            (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            print(json.dumps(dict(event='trial_start',group=length_group(record['n_codons']-1),n_aa=record['n_codons']-1,weight=weight)),flush=True)
            results = optimize_batch(model,scorer,[record],counts,config,run,start)
            random_start = json.loads((run/'random_initial.jsonl').read_text())['latent_sha256']
            identity = record['sequence_id']
            if identity in starts:
                assert starts[identity] == random_start
            starts[identity] = random_start
            diag = json.loads((run/'length_search_summary.jsonl').read_text())
            steps = [json.loads(s) for s in (run/'length_search_steps.jsonl').read_text().splitlines()]
            best = results[-1]['best'];raw = diag['best_before_filter']
            assert best['feasible'] and best['protein_preserved']
            row = dict(sequence_id=identity,group=length_group(record['n_codons']-1),n_aa=record['n_codons']-1,
                latent_l2_weight=weight,feasible_gain=results[-1]['best_hard_gain'],raw_gain=raw['hard_score']-diag['baseline'],
                filter_loss=diag['filter_loss'],feasible_edit_fraction=best['mutable_hamming'],raw_edit_fraction=raw['mutable_hamming'],
                max_latent_rms=max(math.sqrt(r['latent_l2']) for r in steps),
                first20_actual_z_update_rms=sum(r['actual_z_update_rms'] for r in steps[:20])/20,
                js_rejection_fraction=sum('js' in r['rejected_by'] for r in steps)/80,
                total_loss_gradient_dims99=diag['dims_for_99pct_total_loss_gradient_energy'])
            rows.append(row);report(output,rows)
            print(json.dumps(dict(event='trial_complete',completed=len(rows),total=9,**row)),flush=True)
    assert scorer.forward_sequences == 9*246
    assert hashes == (state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert source_hashes == {name:sha256(Path(__file__).with_name(name)) for name in names}
    verify_reference(reference)
    (output/'completion_receipt.json').write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start,
        frozen_weights_unchanged=True,main_config_unchanged=True,matched_random_starts=True,scorer_calls=scorer.forward_sequences)),encoding='utf-8')


if __name__ == '__main__':
    main()
